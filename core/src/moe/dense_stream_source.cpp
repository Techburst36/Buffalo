#include "dense_stream_source.h"
#include "bmoe/log.h"
#include "tensor_names.h"
#include "virtual_host_buft.h"
#include "../io/platform_io.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace bmoe {

namespace {

HttpProviderOptions http_options_from_cfg(const MoeStreamConfig & cfg) {
    HttpProviderOptions opt;
    opt.cache_dir = cfg.http_cache_dir;
    opt.cache_bytes = cfg.http_cache_mb > 0 ? (uint64_t) cfg.http_cache_mb << 20 : 0;
    opt.lanes = cfg.http_threads > 0 ? cfg.http_threads : std::max(cfg.io_threads, MoeStreamConfig::http_threads_default);
    return opt;
}

} // namespace

DenseStreamSource::~DenseStreamSource() {
    shutdown();
}

bool DenseStreamSource::init(const std::vector<std::string> & shard_paths,
                             std::vector<DenseTensorRef> tensors,
                             int n_layer,
                             const MoeStreamConfig & cfg,
                             std::unique_ptr<BlockProvider> existing_provider) {
    shutdown();
    if (tensors.empty()) return true;
    if (shard_paths.empty()) {
        log_printf(LogLevel::Error, "dense-stream: no shard paths provided");
        return false;
    }

    n_layer_ = std::max(0, n_layer);
    embd_group_ = n_layer_;
    head_group_ = n_layer_ + 1;
    tied_head_ = false;
    groups_.clear();
    groups_.resize((size_t) (n_layer_ + 2));
    for (int i = 0; i < (int) groups_.size(); ++i)
        groups_[(size_t) i].id = i;

    const size_t page = pio::vm_page();
    const size_t tail = pio::tensor_tail_slack();
    const ggml_backend_buffer_type_t vhost = bmoe_virtual_host_buffer_type();
    const bool remote = is_remote_url(shard_paths[0]);

    uint64_t max_tensor_bytes = 0;
    uint64_t dropped_mmap = 0;
    size_t n_tensors = 0;

    auto rollback = [&]() {
        for (Group & g : groups_) {
            for (ManagedTensor & mt : g.tensors) {
                if (mt.ref.tensor) mt.ref.tensor->data = mt.orig_data;
                for (size_t a = 0; a < mt.ref.aliases.size() && a < mt.alias_orig_data.size(); ++a) {
                    if (mt.ref.aliases[a]) mt.ref.aliases[a]->data = mt.alias_orig_data[a];
                }
                if (mt.owns_vm && mt.base && mt.span_bytes) pio::vm_release(mt.base, mt.span_bytes);
            }
        }
        for (RingSlot & rs : ring_slots_) {
            if (rs.base && rs.alloc_bytes) pio::vm_release(rs.base, rs.alloc_bytes);
        }
        ring_slots_.clear();
        use_ring_ = false;
        groups_.clear();
        tensor_to_group_.clear();
        provider_.reset();
        active_ = false;
        total_weight_bytes_ = 0;
        budget_bytes_ = 0;
    };

    for (DenseTensorRef & d : tensors) {
        if (!d.tensor || d.size == 0) continue;
        const std::string name = d.tensor->name;
        const int bl = tensor_names::block_layer(name.c_str());
        int gid = head_group_;
        if (bl >= 0 && bl < n_layer_) {
            gid = bl;
        } else if (tensor_names::is_token_table(name) || tensor_names::is_embedding_table(name.c_str())) {
            gid = embd_group_;
            if (!d.aliases.empty()) tied_head_ = true;
        } else {
            gid = head_group_;
        }

        const size_t span = (((size_t) d.size + tail + page - 1) / page) * page;
        ManagedTensor mt;
        mt.ref = d;
        mt.orig_data = d.tensor->data;
        mt.alias_orig_data.reserve(d.aliases.size());
        for (ggml_tensor * al : d.aliases)
            mt.alias_orig_data.push_back(al ? al->data : nullptr);
        mt.span_bytes = span;

        // Reuse the model's existing page-aligned virtual-host reservation when present (remote
        // HTTP boot); otherwise allocate a private virtual reservation and rebind tensor->data
        // (and all tied aliases) so evicted pages stay completely disjoint from neighbors.
        const bool already_vhost = d.tensor->buffer && ggml_backend_buffer_get_type(d.tensor->buffer) == vhost &&
                                   d.tensor->data != nullptr && (((uintptr_t) d.tensor->data % page) == 0);
        if (already_vhost) {
            mt.base = d.tensor->data;
            mt.owns_vm = false;
            for (ggml_tensor * al : d.aliases)
                if (al) al->data = mt.base;
            pio::vm_evict(mt.base, mt.span_bytes);
        } else {
            void * base = pio::vm_reserve(span);
            if (!base) {
                log_printf(LogLevel::Error, "dense-stream: vm_reserve(%zu) failed for %s", span, name.c_str());
                rollback();
                return false;
            }
            mt.base = base;
            mt.owns_vm = true;
            d.tensor->data = base;
            for (ggml_tensor * al : d.aliases)
                if (al) al->data = base;

            // Drop now-unused local mmap pages (aligned inward to whole pages so neighboring
            // resident tensors on shared boundary pages are untouched).
            if (!remote && mt.orig_data) {
                const uintptr_t a0 = ((uintptr_t) mt.orig_data + page - 1) & ~(uintptr_t) (page - 1);
                const uintptr_t a1 = ((uintptr_t) mt.orig_data + (size_t) d.size) & ~(uintptr_t) (page - 1);
                if (a1 > a0) {
                    pio::vm_drop_file_pages((void *) a0, (size_t) (a1 - a0));
                    dropped_mmap += (uint64_t) (a1 - a0);
                }
            }
        }

        Group & g = groups_[(size_t) gid];
        g.total_bytes += d.size;
        g.span_bytes += span;
        tensor_to_group_[d.tensor] = gid;
        for (ggml_tensor * al : d.aliases)
            if (al) tensor_to_group_[al] = gid;
        g.tensors.push_back(std::move(mt));
        total_weight_bytes_ += d.size;
        if (d.size > max_tensor_bytes) max_tensor_bytes = d.size;
        ++n_tensors;
    }

    if (n_tensors == 0) {
        groups_.clear();
        return true;
    }

    chunk_bytes_ = std::min<size_t>(std::max<size_t>((size_t) max_tensor_bytes, 64ull << 10), k_max_chunk_bytes);

    if (existing_provider && existing_provider->is_open()) {
        provider_ = std::move(existing_provider);
    } else {
        const HttpProviderOptions http_opt = http_options_from_cfg(cfg);
        const int lanes = remote ? http_opt.lanes : std::max(1, std::min(cfg.io_threads, MoeStreamConfig::io_threads_max));
        std::string perr;
        provider_ = open_block_provider(shard_paths, lanes, cfg.o_direct, chunk_bytes_, cfg.io_backend, http_opt, &perr);
        if (!provider_) {
            log_printf(LogLevel::Error, "dense-stream: failed to open block provider (%s)",
                       perr.empty() ? "unknown error" : perr.c_str());
            rollback();
            return false;
        }
    }

    const size_t tensor_align =
        std::max<size_t>(std::max<size_t>(page, provider_->buffer_align()), size_t(64));
    chunk_bytes_ = ((chunk_bytes_ + tensor_align - 1) / tensor_align) * tensor_align;
    if (chunk_bytes_ > k_max_chunk_bytes) chunk_bytes_ = k_max_chunk_bytes;

    // Size the RAM window: explicit --dense-cache-mb, or auto (minimal 2-layer look-ahead window).
    uint64_t top1 = 0, top2 = 0;
    size_t n_groups = 0;
    for (const Group & g : groups_) {
        if (g.tensors.empty()) continue;
        ++n_groups;
        if (g.total_bytes >= top1) {
            top2 = top1;
            top1 = g.total_bytes;
        } else if (g.total_bytes > top2) {
            top2 = g.total_bytes;
        }
    }
    if (cfg.dense_cache_mb > 0) {
        budget_bytes_ = (uint64_t) cfg.dense_cache_mb << 20;
    } else {
        budget_bytes_ = top1 + top2;
        if (budget_bytes_ == 0) budget_bytes_ = top1;
    }

    // Cyclic-scan anti-thrashing policy: when total_weight_bytes_ > budget_bytes_, pure LRU over
    // a cyclic layer scan (0 -> 1 -> ... -> N-1 -> 0) gets a 0% hit rate because later layers
    // evict layer 0 before the next token loops back. Reserve the working window needed by the
    // rotating tail (2 * max_group_bytes for 1-layer look-ahead, or 1 * max_group_bytes in
    // starvation mode) and permanently static-pin the earliest groups in execution order that fit
    // in the remaining pin_budget. Static-pinned groups load once on token 1 and are never evicted.
    static_pinned_groups_ = 0;
    static_pinned_bytes_ = 0;
    if (total_weight_bytes_ > budget_bytes_ && top1 > 0) {
        const uint64_t stream_window = (budget_bytes_ >= 2 * top1) ? (2 * top1) : top1;
        if (budget_bytes_ > stream_window) {
            const uint64_t pin_budget = budget_bytes_ - stream_window;
            std::vector<int> exec_order;
            exec_order.reserve(n_groups);
            if (group_has_tensors(embd_group_)) exec_order.push_back(embd_group_);
            for (int il = 0; il < n_layer_; ++il)
                if (group_has_tensors(il)) exec_order.push_back(il);
            if (group_has_tensors(head_group_)) exec_order.push_back(head_group_);

            for (int gid : exec_order) {
                Group & g = groups_[(size_t) gid];
                if (static_pinned_bytes_ + g.total_bytes > pin_budget) break;
                g.static_pinned = true;
                static_pinned_bytes_ += g.total_bytes;
                ++static_pinned_groups_;
            }
        }
    }

    // Hint THP on permanent (static-pinned or fits-in-budget) spans >= 2 MiB to cut TLB misses.
    for (Group & g : groups_) {
        if (g.tensors.empty()) continue;
        if (total_weight_bytes_ <= budget_bytes_ || g.static_pinned) {
            for (ManagedTensor & mt : g.tensors) {
                if (mt.base && mt.span_bytes >= k_hugepage_bytes) pio::vm_advise_hugepage(mt.base, mt.span_bytes);
            }
        }
    }

    // Pre-faulted physical ring slots for rotating tail groups: instead of calling MADV_DONTNEED
    // on every layer eviction (which forces the kernel to unmap, re-allocate, zero-fill, and
    // fault every 4 KiB page on the next token sweep), allocate 1-2 physical slots aligned to
    // 2 MiB with MADV_HUGEPAGE and pre-fault them once at init. Rotating tail groups recycle
    // these hot physical buffers and retarget tensor->data on the eval thread before compute,
    // incurring zero minor page faults on steady-state sweeps.
    use_ring_ = false;
    ring_slots_.clear();
    if (total_weight_bytes_ > budget_bytes_) {
        size_t n_rotating = 0;
        uint64_t max_tail_slot_bytes = 0;
        uint64_t max_tail_weight_bytes = 0;
        for (Group & g : groups_) {
            if (g.tensors.empty() || g.static_pinned) continue;
            ++n_rotating;
            size_t cur_off = 0;
            for (ManagedTensor & mt : g.tensors) {
                mt.slot_offset = cur_off;
                const size_t packed = (((size_t) mt.ref.size + tail + tensor_align - 1) / tensor_align) * tensor_align;
                cur_off += packed;
            }
            g.slot_bytes = (uint64_t) cur_off;
            if (g.slot_bytes > max_tail_slot_bytes) max_tail_slot_bytes = g.slot_bytes;
            if (g.total_bytes > max_tail_weight_bytes) max_tail_weight_bytes = g.total_bytes;
        }
        if (n_rotating > 0 && max_tail_slot_bytes > 0) {
            const int num_slots =
                (n_rotating >= 2 && budget_bytes_ >= static_pinned_bytes_ + 2 * max_tail_weight_bytes) ? 2 : 1;
            const size_t slot_align = max_tail_slot_bytes >= k_hugepage_bytes ? k_hugepage_bytes : page;
            const size_t alloc_bytes =
                (((size_t) max_tail_slot_bytes + slot_align - 1) / slot_align) * slot_align;
            bool ring_ok = true;
            ring_slots_.reserve((size_t) num_slots);
            for (int s = 0; s < num_slots; ++s) {
                void * base = pio::vm_reserve_aligned(alloc_bytes, slot_align);
                if (!base) {
                    ring_ok = false;
                    break;
                }
                if (alloc_bytes >= k_hugepage_bytes) pio::vm_advise_hugepage(base, alloc_bytes);
                if (!pio::vm_commit(base, alloc_bytes)) {
                    pio::vm_release(base, alloc_bytes);
                    ring_ok = false;
                    break;
                }
                // Pre-fault every page once so steady-state O_DIRECT / io_uring / pread writes
                // hit hot resident PTEs/PMDs with zero minor page faults.
                auto * touch = static_cast<volatile uint8_t *>(base);
                for (size_t off = 0; off < alloc_bytes; off += page)
                    touch[off] = 0;
                ring_slots_.push_back(RingSlot{base, alloc_bytes, -1});
            }
            if (ring_ok && !ring_slots_.empty()) {
                use_ring_ = true;
            } else {
                for (RingSlot & rs : ring_slots_)
                    if (rs.base && rs.alloc_bytes) pio::vm_release(rs.base, rs.alloc_bytes);
                ring_slots_.clear();
                log_printf(LogLevel::Warn,
                           "dense-stream: could not allocate %d x %.2f MiB physical ring slots — falling back to VM evict",
                           num_slots, (double) alloc_bytes / (1024.0 * 1024.0));
            }
        }
    }

    fatal_.store(false, std::memory_order_relaxed);
    prefetch_stop_ = false;
    prefetch_target_ = -1;
    lru_tick_ = 0;
    resident_bytes_ = 0;
    peak_resident_bytes_ = 0;
    bytes_read_ = 0;
    groups_loaded_ = 0;
    groups_evicted_ = 0;
    prefetch_hits_ = 0;
    io_errors_ = 0;
    active_ = true;

    prefetch_thread_ = std::thread(&DenseStreamSource::prefetch_worker_main, this);

    log_printf(LogLevel::Info,
               "dense-stream: %zu tensor(s) in %zu group(s), %.2f MiB total, budget %.2f MiB (%s), "
               "static-pinned %.2f MiB (%llu group(s)), ring=%zux%.2f MiB, provider=%s, %llu MiB mmap dropped",
               n_tensors, n_groups, (double) total_weight_bytes_ / (1024.0 * 1024.0),
               (double) budget_bytes_ / (1024.0 * 1024.0), cfg.dense_cache_mb > 0 ? "explicit" : "auto 2-layer",
               (double) static_pinned_bytes_ / (1024.0 * 1024.0), (unsigned long long) static_pinned_groups_,
               ring_slots_.size(),
               ring_slots_.empty() ? 0.0 : (double) ring_slots_[0].alloc_bytes / (1024.0 * 1024.0),
               provider_->backend_name(), (unsigned long long) (dropped_mmap >> 20));
    return true;
}

void DenseStreamSource::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        prefetch_stop_ = true;
        prefetch_target_ = -1;
    }
    prefetch_cv_.notify_all();
    if (prefetch_thread_.joinable()) prefetch_thread_.join();

    std::lock_guard<std::mutex> lk(mtx_);
    for (Group & g : groups_) {
        if (g.state == GroupState::Resident && g.slot_idx < 0) evict_group_vm(g);
        g.slot_idx = -1;
        for (ManagedTensor & mt : g.tensors) {
            // Unconditionally restore original tensor->data (and aliases) before releasing ring
            // slots or private VM reservations so llama_model_free() sees valid original pointers.
            if (mt.ref.tensor) mt.ref.tensor->data = mt.orig_data;
            for (size_t a = 0; a < mt.ref.aliases.size() && a < mt.alias_orig_data.size(); ++a) {
                if (mt.ref.aliases[a]) mt.ref.aliases[a]->data = mt.alias_orig_data[a];
            }
            if (mt.owns_vm && mt.base && mt.span_bytes) pio::vm_release(mt.base, mt.span_bytes);
        }
    }
    for (RingSlot & rs : ring_slots_) {
        if (rs.base && rs.alloc_bytes) pio::vm_release(rs.base, rs.alloc_bytes);
    }
    ring_slots_.clear();
    use_ring_ = false;
    groups_.clear();
    tensor_to_group_.clear();
    provider_.reset();
    active_ = false;
    resident_bytes_ = 0;
    total_weight_bytes_ = 0;
    budget_bytes_ = 0;
    static_pinned_groups_ = 0;
    static_pinned_bytes_ = 0;
}

bool DenseStreamSource::reopen_readers() {
    std::lock_guard<std::mutex> io_lk(io_mtx_);
    return provider_ ? provider_->reopen() : true;
}

int DenseStreamSource::group_of(const ggml_tensor * leaf) const {
    if (!active_ || !leaf) return -1;
    auto it = tensor_to_group_.find(leaf);
    return it == tensor_to_group_.end() ? -1 : it->second;
}

bool DenseStreamSource::group_has_tensors(int g) const {
    return g >= 0 && g < (int) groups_.size() && !groups_[(size_t) g].tensors.empty();
}

int DenseStreamSource::next_group_after(int g, bool is_embd_gather) const {
    if (!active_) return -1;
    if (g == embd_group_) {
        // At inp_embd (GET_ROWS), advance into the first streamed block layer (or head_group_);
        // at result_output (tied head MATMUL), this is the end of the forward pass.
        if (!is_embd_gather) return -1;
        for (int il = 0; il < n_layer_; ++il)
            if (group_has_tensors(il)) return il;
        if (group_has_tensors(head_group_)) return head_group_;
        return -1;
    }
    if (g >= 0 && g < n_layer_) {
        for (int il = g + 1; il < n_layer_; ++il)
            if (group_has_tensors(il)) return il;
        if (group_has_tensors(head_group_)) return head_group_;
        if (tied_head_ && group_has_tensors(embd_group_)) return embd_group_;
        return -1;
    }
    return -1;
}

void DenseStreamSource::evict_group_vm(Group & g) {
    for (ManagedTensor & mt : g.tensors) {
        if (mt.base && mt.span_bytes) pio::vm_evict(mt.base, mt.span_bytes);
    }
}

void DenseStreamSource::evict_group_locked(Group & vg) {
    if (vg.slot_idx >= 0) {
        // Logical ring-slot eviction: keep the pre-faulted physical pages hot and mapped so the
        // next rotating layer overwrites them with zero MADV_DONTNEED and zero minor page faults.
        RingSlot & rs = ring_slots_[(size_t) vg.slot_idx];
        if (rs.owner_group == vg.id) rs.owner_group = -1;
        vg.slot_idx = -1;
    } else {
        evict_group_vm(vg);
    }
    vg.state = GroupState::Evicted;
    vg.prefetched = false;
    resident_bytes_ = resident_bytes_ >= vg.total_bytes ? (resident_bytes_ - vg.total_bytes) : 0;
    ++groups_evicted_;
}

void DenseStreamSource::bind_group_ptrs_locked(Group & g) {
    // Always called on the eval thread before compute for `g` begins (never from the prefetch thread).
    char * slot_base = (g.slot_idx >= 0 && g.slot_idx < (int) ring_slots_.size())
                           ? static_cast<char *>(ring_slots_[(size_t) g.slot_idx].base)
                           : nullptr;
    for (ManagedTensor & mt : g.tensors) {
        void * ptr = slot_base ? static_cast<void *>(slot_base + mt.slot_offset) : mt.base;
        if (mt.ref.tensor) mt.ref.tensor->data = ptr;
        for (ggml_tensor * al : mt.ref.aliases)
            if (al) al->data = ptr;
    }
}

int DenseStreamSource::find_evictable_slot_locked() const {
    for (int s = 0; s < (int) ring_slots_.size(); ++s) {
        if (ring_slots_[(size_t) s].owner_group < 0) return s;
    }
    int best_slot = -1;
    uint64_t oldest = UINT64_MAX;
    for (int s = 0; s < (int) ring_slots_.size(); ++s) {
        const int og = ring_slots_[(size_t) s].owner_group;
        if (og < 0 || og >= (int) groups_.size()) return s;
        const Group & g = groups_[(size_t) og];
        if (g.state == GroupState::Resident && !g.pinned && !g.static_pinned && g.lru_tick < oldest) {
            oldest = g.lru_tick;
            best_slot = s;
        }
    }
    return best_slot;
}

bool DenseStreamSource::any_slot_loading_locked() const {
    for (const RingSlot & rs : ring_slots_) {
        if (rs.owner_group >= 0 && rs.owner_group < (int) groups_.size() &&
            groups_[(size_t) rs.owner_group].state == GroupState::Loading) {
            return true;
        }
    }
    return false;
}

int DenseStreamSource::acquire_ring_slot_locked(Group & g) {
    const int s = find_evictable_slot_locked();
    if (s < 0) return -1;
    const int prev = ring_slots_[(size_t) s].owner_group;
    if (prev >= 0 && prev < (int) groups_.size()) {
        Group & vg = groups_[(size_t) prev];
        if (vg.state == GroupState::Resident) {
            evict_group_locked(vg);
        } else {
            ring_slots_[(size_t) s].owner_group = -1;
            vg.slot_idx = -1;
        }
    }
    ring_slots_[(size_t) s].owner_group = g.id;
    g.slot_idx = s;
    return s;
}

uint64_t DenseStreamSource::unevictable_bytes_locked() const {
    uint64_t sum = 0;
    for (const Group & g : groups_) {
        if (g.pinned || g.state == GroupState::Loading || (g.static_pinned && g.state == GroupState::Resident))
            sum += g.total_bytes;
    }
    return sum;
}

void DenseStreamSource::evict_unpinned_locked(uint64_t need_bytes) {
    while (resident_bytes_ + need_bytes > budget_bytes_) {
        int victim = -1;
        uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < (int) groups_.size(); ++i) {
            const Group & g = groups_[(size_t) i];
            if (g.state == GroupState::Resident && !g.pinned && !g.static_pinned && g.lru_tick < oldest) {
                oldest = g.lru_tick;
                victim = i;
            }
        }
        if (victim < 0) break; // only pinned/static-pinned/loading groups remain; never evict them
        evict_group_locked(groups_[(size_t) victim]);
    }
}

bool DenseStreamSource::load_group_io(Group & g, int slot_idx, uint64_t & out_bytes_read) {
    out_bytes_read = 0;
    if (!provider_) return false;

    char * slot_base = (slot_idx >= 0 && slot_idx < (int) ring_slots_.size())
                           ? static_cast<char *>(ring_slots_[(size_t) slot_idx].base)
                           : nullptr;

    std::vector<BlockReadReq> reqs;
    reqs.reserve(g.tensors.size());

    for (ManagedTensor & mt : g.tensors) {
        char * dst_base = nullptr;
        if (slot_base) {
            dst_base = slot_base + mt.slot_offset;
        } else {
            if (!mt.base || mt.span_bytes == 0) continue;
            if (!pio::vm_commit(mt.base, mt.span_bytes)) {
                log_printf(LogLevel::Error, "dense-stream: vm_commit(%zu) failed for %s", mt.span_bytes,
                           mt.ref.tensor ? mt.ref.tensor->name : "?");
                return false;
            }
            dst_base = static_cast<char *>(mt.base);
        }
        for (uint64_t off = 0; off < mt.ref.size;) {
            const uint64_t len = std::min<uint64_t>(chunk_bytes_, mt.ref.size - off);
            BlockReadReq r;
            r.shard = mt.ref.file_idx;
            r.off = mt.ref.file_off + off;
            r.nbytes = len;
            r.dst = dst_base + off;
            reqs.push_back(r);
            off += len;
        }
    }

    if (reqs.empty()) return true;

    std::lock_guard<std::mutex> io_lk(io_mtx_);
    const bool ok = provider_->read_batch(reqs.data(), reqs.size());
    for (const BlockReadReq & r : reqs) {
        if (r.bytes_moved > 0) out_bytes_read += (uint64_t) r.bytes_moved;
    }
    return ok;
}

bool DenseStreamSource::ensure_group_pinned(int g) {
    if (!active_ || g < 0 || g >= (int) groups_.size()) return true;
    if (fatal_.load(std::memory_order_relaxed)) return false;

    int slot_idx = -1;
    {
        std::unique_lock<std::mutex> lk(mtx_);
        Group & grp = groups_[(size_t) g];
        if (grp.tensors.empty()) return true;

        for (;;) {
            // Wait if an async prefetch for this group is currently in flight.
            while (grp.state == GroupState::Loading) {
                cv_.wait(lk);
            }

            if (grp.state == GroupState::Resident) {
                if (grp.prefetched) {
                    ++prefetch_hits_;
                    grp.prefetched = false;
                }
                grp.pinned = true;
                grp.lru_tick = ++lru_tick_;
                bind_group_ptrs_locked(grp);
                return true;
            }

            if (use_ring_ && !grp.static_pinned) {
                slot_idx = acquire_ring_slot_locked(grp);
                if (slot_idx < 0 && any_slot_loading_locked()) {
                    // Wait for an in-flight prefetch of another group to finish and become evictable.
                    cv_.wait(lk);
                    continue;
                }
            }

            // Synchronous miss: evict oldest unpinned groups, mark Loading + pinned, release lock for I/O.
            grp.state = GroupState::Loading;
            grp.pinned = true;
            grp.prefetched = false;
            grp.slot_idx = slot_idx;
            evict_unpinned_locked(grp.total_bytes);
            break;
        }
    }

    uint64_t read_b = 0;
    const bool ok = load_group_io(groups_[(size_t) g], slot_idx, read_b);

    {
        std::lock_guard<std::mutex> lk(mtx_);
        Group & grp = groups_[(size_t) g];
        if (ok) {
            grp.state = GroupState::Resident;
            grp.lru_tick = ++lru_tick_;
            resident_bytes_ += grp.total_bytes;
            if (resident_bytes_ > peak_resident_bytes_) peak_resident_bytes_ = resident_bytes_;
            bytes_read_ += read_b;
            ++groups_loaded_;
            bind_group_ptrs_locked(grp);
        } else {
            if (slot_idx >= 0) {
                ring_slots_[(size_t) slot_idx].owner_group = -1;
                grp.slot_idx = -1;
            } else {
                evict_group_vm(grp);
            }
            grp.state = GroupState::Evicted;
            grp.pinned = false;
            ++io_errors_;
            fatal_.store(true, std::memory_order_relaxed);
            log_printf(LogLevel::Error, "dense-stream: I/O failed loading group %d", g);
        }
    }
    cv_.notify_all();
    return ok;
}

void DenseStreamSource::unpin_group(int g) {
    if (!active_ || g < 0 || g >= (int) groups_.size()) return;
    std::lock_guard<std::mutex> lk(mtx_);
    Group & grp = groups_[(size_t) g];
    if (!grp.pinned) return;
    grp.pinned = false;
    grp.lru_tick = ++lru_tick_;
    // Enforce budget immediately: under extreme 1-layer starvation (budget < 1 group), this
    // evicts the just-finished group right away before the next layer begins.
    if (resident_bytes_ > budget_bytes_) evict_unpinned_locked(0);
}

void DenseStreamSource::unpin_all() {
    if (!active_) return;
    std::lock_guard<std::mutex> lk(mtx_);
    bool any = false;
    for (Group & g : groups_) {
        if (g.pinned) {
            g.pinned = false;
            g.lru_tick = ++lru_tick_;
            any = true;
        }
    }
    if (any && resident_bytes_ > budget_bytes_) evict_unpinned_locked(0);
}

void DenseStreamSource::prefetch_group_async(int next_g) {
    if (!active_ || fatal_.load(std::memory_order_relaxed)) return;
    if (next_g < 0 || next_g >= (int) groups_.size()) return;

    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (prefetch_stop_ || prefetch_target_ >= 0) return;
        Group & grp = groups_[(size_t) next_g];
        if (grp.tensors.empty() || grp.state != GroupState::Evicted) return;
        // Only enqueue look-ahead when the next group fits alongside all currently unevictable
        // (active pinned, in-flight, and resident static-pinned) groups inside budget_bytes_.
        // Under extreme 1-layer budgets this cleanly disables look-ahead instead of thrashing.
        if (unevictable_bytes_locked() + grp.total_bytes > budget_bytes_) return;

        int slot_idx = -1;
        if (use_ring_ && !grp.static_pinned) {
            slot_idx = acquire_ring_slot_locked(grp);
            if (slot_idx < 0) return; // all ring slots currently pinned/loading; skip look-ahead
        }

        grp.state = GroupState::Loading;
        grp.prefetched = true;
        grp.slot_idx = slot_idx;
        evict_unpinned_locked(grp.total_bytes);
        prefetch_target_ = next_g;
    }
    prefetch_cv_.notify_one();
}

void DenseStreamSource::prefetch_worker_main() {
    for (;;) {
        int target = -1;
        int slot_idx = -1;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            prefetch_cv_.wait(lk, [this] { return prefetch_stop_ || prefetch_target_ >= 0; });
            if (prefetch_stop_) break;
            target = prefetch_target_;
            prefetch_target_ = -1;
            slot_idx = groups_[(size_t) target].slot_idx;
        }

        uint64_t read_b = 0;
        const bool ok = load_group_io(groups_[(size_t) target], slot_idx, read_b);

        {
            std::lock_guard<std::mutex> lk(mtx_);
            Group & grp = groups_[(size_t) target];
            if (ok) {
                grp.state = GroupState::Resident;
                grp.lru_tick = ++lru_tick_;
                resident_bytes_ += grp.total_bytes;
                if (resident_bytes_ > peak_resident_bytes_) peak_resident_bytes_ = resident_bytes_;
                bytes_read_ += read_b;
                ++groups_loaded_;
            } else {
                if (slot_idx >= 0) {
                    ring_slots_[(size_t) slot_idx].owner_group = -1;
                    grp.slot_idx = -1;
                } else {
                    evict_group_vm(grp);
                }
                grp.state = GroupState::Evicted;
                grp.prefetched = false;
                ++io_errors_;
                fatal_.store(true, std::memory_order_relaxed);
                log_printf(LogLevel::Error, "dense-stream: async prefetch I/O failed for group %d", target);
            }
        }
        cv_.notify_all();
    }
}

DenseStreamStats DenseStreamSource::stats() const {
    DenseStreamStats s;
    std::lock_guard<std::mutex> lk(mtx_);
    s.total_streamed_weight_bytes = total_weight_bytes_;
    s.budget_bytes = budget_bytes_;
    s.static_pinned_groups = static_pinned_groups_;
    s.static_pinned_bytes = static_pinned_bytes_;
    s.bytes_read = bytes_read_;
    s.groups_loaded = groups_loaded_;
    s.groups_evicted = groups_evicted_;
    s.prefetch_hits = prefetch_hits_;
    s.peak_resident_bytes = peak_resident_bytes_;
    s.current_resident_bytes = resident_bytes_;
    s.io_errors = io_errors_;
    return s;
}

} // namespace bmoe
