// Layer-wise dense weight streamer (--dense-stream).
//
// Streams host-resident dense tensors (attention, dense FFNs, shared experts, SSM projections,
// output head, non-row-streamed embeddings) on demand over any BlockProvider (Posix, io_uring,
// or HttpRangeProvider) inside a bounded RAM budget, with async 1-layer look-ahead prefetch.
//
// Each static-pinned (or fits-in-budget) tensor keeps a stable, disjoint, OS-page-aligned virtual
// address for the lifetime of the session. Rotating unpinned tail groups share a small pool of
// pre-faulted, 2 MiB-aligned THP-backed physical ring slots sized to the rotating window (1-2 x
// max tail group): evicting a tail group logically recycles its ring slot and retargets tensor->data
// on the eval thread before compute, eliminating per-sweep MADV_DONTNEED and 4 KiB page-fault churn.
#pragma once

#include "../io/block_provider.h"
#include "bmoe/config.h"
#include "bmoe/metrics.h"
#include "dense_weights.h" // DenseTensorRef

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct ggml_tensor;

namespace bmoe {

class DenseStreamSource {
public:
    DenseStreamSource() = default;
    ~DenseStreamSource();

    DenseStreamSource(const DenseStreamSource &) = delete;
    DenseStreamSource & operator=(const DenseStreamSource &) = delete;

    // Bind `tensors` (host-resident dense weights >= dense_min_bytes) into layer groups and open
    // (or adopt `existing_provider`) a BlockProvider over `shard_paths`.
    // Returns true with active() == false when `tensors` is empty.
    // On failure, restores any partially rebound tensors and returns false.
    bool init(const std::vector<std::string> & shard_paths,
              std::vector<DenseTensorRef> tensors,
              int n_layer,
              const MoeStreamConfig & cfg,
              std::unique_ptr<BlockProvider> existing_provider = nullptr);

    // Join the prefetch worker, evict resident pages, release private VM reservations, and
    // restore original tensor->data pointers before llama_model_free().
    void shutdown();

    bool active() const { return active_; }
    bool fatal() const { return fatal_.load(std::memory_order_relaxed); }

    // Reopen underlying file descriptors (called after Windows release_file_mappings).
    bool reopen_readers();

    // O(1) lookup of which dense group owns `leaf` (primary tensor or tied alias), or -1.
    int group_of(const ggml_tensor * leaf) const;

    int n_layer() const { return n_layer_; }
    int embd_group() const { return embd_group_; }
    int head_group() const { return head_group_; }
    bool tied_head() const { return tied_head_; }
    bool group_has_tensors(int g) const;

    // Next non-empty group in forward execution order after `g`, or -1 at the end of a pass.
    // `is_embd_gather` disambiguates a tied token_embd group at inp_embd vs. result_output.
    int next_group_after(int g, bool is_embd_gather) const;

    // Guarantee group `g` is resident and pinned in RAM before its compute nodes execute.
    // Blocks if an async prefetch for `g` is in flight; evicts oldest unpinned groups if needed;
    // never evicts a pinned or in-flight group. Returns false only on fatal I/O error.
    bool ensure_group_pinned(int g);

    // Clear the pin on group `g` after its last node finishes. Immediately evicts unpinned LRU
    // groups if resident bytes exceed budget_bytes_ (including `g` itself under <2-layer budgets).
    void unpin_group(int g);

    // Safety cleanup called after llama_decode() returns: unpins any still-pinned groups and
    // enforces budget_bytes_.
    void unpin_all();

    // Enqueue background look-ahead load for `next_g` when `next_g` is evicted and fits alongside
    // all currently pinned/loading groups inside budget_bytes_. No-op when budget is too tight.
    void prefetch_group_async(int next_g);

    DenseStreamStats stats() const;

private:
    struct ManagedTensor {
        DenseTensorRef ref;
        void * base = nullptr;
        void * orig_data = nullptr;
        std::vector<void *> alias_orig_data;
        size_t span_bytes = 0;
        size_t slot_offset = 0; // byte offset inside a physical ring slot (when rotating)
        bool owns_vm = false;
    };

    enum class GroupState { Evicted, Loading, Resident };

    struct Group {
        int id = -1;
        std::vector<ManagedTensor> tensors;
        uint64_t total_bytes = 0;
        uint64_t span_bytes = 0;
        uint64_t slot_bytes = 0; // packed ring-slot footprint (with per-tensor alignment & tail slack)
        int slot_idx = -1;       // assigned physical ring slot index, or -1 if using permanent mt.base
        GroupState state = GroupState::Evicted;
        bool pinned = false;
        bool static_pinned = false; // permanently pinned prefix group; loaded once, never evicted
        bool prefetched = false;
        uint64_t lru_tick = 0;
    };

    struct RingSlot {
        void * base = nullptr;
        size_t alloc_bytes = 0;
        int owner_group = -1;
    };

    static constexpr size_t k_max_chunk_bytes = 4ull << 20; // 4 MiB max read chunk / bounce size
    static constexpr size_t k_hugepage_bytes = 2ull << 20;  // 2 MiB THP alignment granule on Linux

    void evict_group_vm(Group & g);
    void evict_group_locked(Group & g);
    void evict_unpinned_locked(uint64_t need_bytes);
    void bind_group_ptrs_locked(Group & g);
    int find_evictable_slot_locked() const;
    bool any_slot_loading_locked() const;
    int acquire_ring_slot_locked(Group & g);
    uint64_t unevictable_bytes_locked() const;
    bool load_group_io(Group & g, int slot_idx, uint64_t & out_bytes_read);
    void prefetch_worker_main();

    bool active_ = false;
    bool use_ring_ = false;
    int n_layer_ = 0;
    int embd_group_ = -1;
    int head_group_ = -1;
    bool tied_head_ = false;
    size_t chunk_bytes_ = k_max_chunk_bytes;

    std::unique_ptr<BlockProvider> provider_;
    std::vector<Group> groups_;
    std::vector<RingSlot> ring_slots_;
    std::unordered_map<const ggml_tensor *, int> tensor_to_group_;

    uint64_t total_weight_bytes_ = 0;
    uint64_t budget_bytes_ = 0;
    uint64_t static_pinned_groups_ = 0;
    uint64_t static_pinned_bytes_ = 0;

    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::mutex io_mtx_; // serializes provider_->read_batch() outside mtx_

    // Dedicated single background worker thread for 1-layer look-ahead prefetch.
    std::thread prefetch_thread_;
    std::condition_variable prefetch_cv_;
    bool prefetch_stop_ = false;
    int prefetch_target_ = -1;

    uint64_t lru_tick_ = 0;
    uint64_t resident_bytes_ = 0;
    uint64_t peak_resident_bytes_ = 0;
    uint64_t bytes_read_ = 0;
    uint64_t groups_loaded_ = 0;
    uint64_t groups_evicted_ = 0;
    uint64_t prefetch_hits_ = 0;
    uint64_t io_errors_ = 0;
    std::atomic<bool> fatal_{false};
};

} // namespace bmoe
