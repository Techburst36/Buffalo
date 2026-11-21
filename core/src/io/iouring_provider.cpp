#include "iouring_provider.h"

#ifdef BMOE_HAS_IO_URING

#include "bmoe/log.h"

#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>

namespace bmoe {

namespace {

using clock_t_ = std::chrono::steady_clock;

inline uint32_t load_acquire(const uint32_t * p) {
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}

inline void store_release(uint32_t * p, uint32_t v) {
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}

uint32_t next_pow2(uint32_t v) {
    if (v <= 1) return 1;
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

int sys_io_uring_setup(uint32_t entries, struct io_uring_params * p) {
#ifdef __NR_io_uring_setup
    return (int) ::syscall(__NR_io_uring_setup, entries, p);
#else
    (void) entries;
    (void) p;
    errno = ENOSYS;
    return -1;
#endif
}

int sys_io_uring_enter(int fd, uint32_t to_submit, uint32_t min_complete, uint32_t flags) {
#ifdef __NR_io_uring_enter
    return (int) ::syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, nullptr, (size_t) 0);
#else
    (void) fd;
    (void) to_submit;
    (void) min_complete;
    (void) flags;
    errno = ENOSYS;
    return -1;
#endif
}

// Per-slot in-flight state for one BlockReadReq inside read_batch().
struct InFlight {
    size_t req_idx = 0;
    int shard = 0;
    bool use_bounce = false;
    uint64_t a0 = 0;       // aligned window start (equals req.off when !use_bounce)
    uint64_t bulk_end = 0; // direct-readable aligned end
    uint64_t read_end = 0; // full window end (may include sub-alignment EOF tail)
    uint64_t cur_off = 0;  // file offset of the currently issued SQE
    size_t cur_len = 0;    // byte length of the currently issued SQE
    char * cur_buf = nullptr;
    bool in_tail = false;  // true when issuing sub-alignment EOF tail via buffered fd
    clock_t_::time_point t0{};
};

} // namespace

IoUringBlockProvider::~IoUringBlockProvider() {
    close();
}

bool IoUringBlockProvider::setup_ring(uint32_t entries, int * out_ring_errno) {
    struct io_uring_params p;
    std::memset(&p, 0, sizeof(p));

    const int rfd = sys_io_uring_setup(entries, &p);
    if (rfd < 0) {
        if (out_ring_errno) *out_ring_errno = errno;
        return false;
    }

    size_t sq_sz = p.sq_off.array + p.sq_entries * sizeof(uint32_t);
    size_t cq_sz = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    const bool single_mmap = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
    if (single_mmap) {
        if (cq_sz > sq_sz) sq_sz = cq_sz;
        cq_sz = sq_sz;
    }

    void * sq_ptr = ::mmap(nullptr, sq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, rfd, IORING_OFF_SQ_RING);
    if (sq_ptr == MAP_FAILED) {
        if (out_ring_errno) *out_ring_errno = errno;
        ::close(rfd);
        return false;
    }

    void * cq_ptr = sq_ptr;
    if (!single_mmap) {
        cq_ptr = ::mmap(nullptr, cq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, rfd, IORING_OFF_CQ_RING);
        if (cq_ptr == MAP_FAILED) {
            if (out_ring_errno) *out_ring_errno = errno;
            ::munmap(sq_ptr, sq_sz);
            ::close(rfd);
            return false;
        }
    }

    const size_t sqes_sz = (size_t) p.sq_entries * sizeof(struct io_uring_sqe);
    void * sqes = ::mmap(nullptr, sqes_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, rfd, IORING_OFF_SQES);
    if (sqes == MAP_FAILED) {
        if (out_ring_errno) *out_ring_errno = errno;
        if (!single_mmap) ::munmap(cq_ptr, cq_sz);
        ::munmap(sq_ptr, sq_sz);
        ::close(rfd);
        return false;
    }

    ring_fd_ = rfd;
    qd_ = p.sq_entries;
    sq_ptr_ = sq_ptr;
    sq_ring_sz_ = sq_sz;
    cq_ptr_ = cq_ptr;
    cq_ring_sz_ = cq_sz;
    sqes_ = static_cast<io_uring_sqe *>(sqes);
    sqes_sz_ = sqes_sz;

    auto * sq_bytes = static_cast<char *>(sq_ptr_);
    auto * cq_bytes = static_cast<char *>(cq_ptr_);
    sq_head_ = reinterpret_cast<uint32_t *>(sq_bytes + p.sq_off.head);
    sq_tail_ = reinterpret_cast<uint32_t *>(sq_bytes + p.sq_off.tail);
    sq_mask_ = *reinterpret_cast<uint32_t *>(sq_bytes + p.sq_off.ring_mask);
    sq_array_ = reinterpret_cast<uint32_t *>(sq_bytes + p.sq_off.array);

    cq_head_ = reinterpret_cast<uint32_t *>(cq_bytes + p.cq_off.head);
    cq_tail_ = reinterpret_cast<uint32_t *>(cq_bytes + p.cq_off.tail);
    cq_mask_ = *reinterpret_cast<uint32_t *>(cq_bytes + p.cq_off.ring_mask);
    cqes_ = reinterpret_cast<io_uring_cqe *>(cq_bytes + p.cq_off.cqes);
    return true;
}

void IoUringBlockProvider::teardown_ring() {
    if (sqes_) {
        ::munmap(sqes_, sqes_sz_);
        sqes_ = nullptr;
        sqes_sz_ = 0;
    }
    if (cq_ptr_ && cq_ptr_ != sq_ptr_) {
        ::munmap(cq_ptr_, cq_ring_sz_);
    }
    cq_ptr_ = nullptr;
    cq_ring_sz_ = 0;
    if (sq_ptr_) {
        ::munmap(sq_ptr_, sq_ring_sz_);
        sq_ptr_ = nullptr;
        sq_ring_sz_ = 0;
    }
    if (ring_fd_ >= 0) {
        ::close(ring_fd_);
        ring_fd_ = -1;
    }
    sq_head_ = sq_tail_ = sq_array_ = nullptr;
    cq_head_ = cq_tail_ = nullptr;
    cqes_ = nullptr;
    sq_mask_ = cq_mask_ = qd_ = 0;
}

bool IoUringBlockProvider::open(const std::vector<std::string> & paths,
                                int lanes,
                                bool direct,
                                size_t max_read,
                                int * out_ring_errno) {
    if (out_ring_errno) *out_ring_errno = 0;
    if (is_open() || paths.empty()) return false;

    const uint32_t want_qd = next_pow2(std::min<uint32_t>(256u, std::max<uint32_t>(32u, (uint32_t) std::max(1, lanes) * 4u)));
    if (!setup_ring(want_qd, out_ring_errno)) return false;

    paths_ = paths;
    lanes_ = lanes;
    direct_req_ = direct;
    max_read_ = max_read;

    const size_t page = pio::vm_page();
    direct_ = true;
    align_ = pio::direct_align_fallback;
    align_reported_ = true;
    buffer_align_ = page;

    shards_.reserve(paths.size());
    for (const std::string & p : paths) {
        Shard s;
        s.path = p;
        s.direct = direct;
        s.fd = pio::open_read(p.c_str(), direct, &s.direct);
        if (!pio::fd_ok(s.fd) && direct) {
            s.direct = false;
            s.fd = pio::open_read(p.c_str(), false);
        }
        if (!pio::fd_ok(s.fd)) {
            log_printf(LogLevel::Error, "io_uring: open %s failed", p.c_str());
            close();
            return false;
        }
        s.size = pio::file_size(s.fd);
        const size_t stated = pio::direct_alignment(s.fd);
        s.align_reported = stated != 0;
        s.align = s.align_reported ? stated : pio::direct_align_fallback;
        s.aligned_reads = s.direct && pio::direct_needs_alignment();

        // Same open-time O_DIRECT corruption verify as FileReader (guards FUSE/emulated volumes).
        if (s.aligned_reads && s.size >= (uint64_t) s.align) {
            uint64_t voff = (s.size / 2) & ~(uint64_t) (s.align - 1);
            if (voff + s.align > s.size) voff = 0;
            void * dbuf = pio::alloc_aligned(s.align, s.align);
            pio::fd_t vfd = pio::open_read(p.c_str(), false);
            if (dbuf && pio::fd_ok(vfd)) {
                std::vector<char> bbuf(s.align);
                const long long want = (long long) s.align;
                const long long gd = pio::pread_at(s.fd, dbuf, s.align, voff);
                const long long gb = pio::pread_at(vfd, bbuf.data(), s.align, voff);
                if (gd != want || gb != want || std::memcmp(dbuf, bbuf.data(), s.align) != 0) {
                    log_printf(LogLevel::Warn,
                               "io_uring: O_DIRECT returns wrong data on %s — using buffered I/O", p.c_str());
                    s.direct = false;
                    pio::close_fd(s.fd);
                    s.fd = pio::open_read(p.c_str(), false);
                }
            }
            if (dbuf) pio::aligned_free(dbuf);
            if (pio::fd_ok(vfd)) pio::close_fd(vfd);
            if (!pio::fd_ok(s.fd)) {
                log_printf(LogLevel::Error, "io_uring: reopen after O_DIRECT check failed");
                close();
                return false;
            }
            s.aligned_reads = s.aligned_reads && s.direct;
        }

        if (s.aligned_reads) {
            s.fd_buf = pio::open_read(p.c_str(), false);
            if (!pio::fd_ok(s.fd_buf))
                log_printf(LogLevel::Warn, "io_uring: buffered tail fd unavailable for %s — EOF-tail reads will fail",
                           p.c_str());
        }

        direct_ = direct_ && s.direct;
        align_ = std::max(align_, s.align);
        align_reported_ = align_reported_ && s.align_reported;
        buffer_align_ = std::max(buffer_align_, std::max(s.align, page));
        shards_.push_back(std::move(s));
    }

    // Bounce slots are allocated lazily on first use in ensure_bounce() so physical memory stays
    // bounded to peak concurrent in-flight requests rather than the full SQ ring depth.
    bounces_.resize(qd_);
    return true;
}

void IoUringBlockProvider::close() {
    for (auto & b : bounces_) {
        if (b.buf && b.cap) pio::vm_release(b.buf, b.cap);
        b.buf = nullptr;
        b.cap = 0;
    }
    bounces_.clear();
    for (auto & s : shards_) {
        if (pio::fd_ok(s.fd)) pio::close_fd(s.fd);
        if (pio::fd_ok(s.fd_buf)) pio::close_fd(s.fd_buf);
        s.fd = s.fd_buf = pio::fd_invalid;
    }
    shards_.clear();
    teardown_ring();
    direct_ = false;
    align_ = pio::direct_align_fallback;
    align_reported_ = false;
    buffer_align_ = pio::vm_page();
}

bool IoUringBlockProvider::reopen() {
    if (!is_open()) return false;
    std::lock_guard<std::mutex> lk(ring_mtx_);
    const std::vector<std::string> paths = paths_;
    const int lanes = lanes_;
    const bool direct = direct_req_;
    const size_t max_read = max_read_;
    close();
    return open(paths, lanes, direct, max_read);
}

uint64_t IoUringBlockProvider::shard_size(int shard) const {
    if (shard < 0 || shard >= (int) shards_.size()) return 0;
    return shards_[(size_t) shard].size;
}

bool IoUringBlockProvider::ensure_bounce(uint32_t slot, size_t need) {
    BounceSlot & b = bounces_[slot];
    if (b.cap >= need) return true;
    if (b.buf && b.cap) pio::vm_release(b.buf, b.cap);
    b.buf = nullptr;
    b.cap = 0;
    // Grow directly to at least max_read_ + 2 * align_ when known so a slot never reallocates
    // across projections of a layer.
    const size_t target = max_read_ > 0 ? std::max(need, max_read_ + 2 * align_) : need;
    size_t alloc_sz = 0;
    b.buf = pio::alloc_thp_buffer(target, align_, target, &alloc_sz);
    if (!b.buf) {
        log_printf(LogLevel::Error, "io_uring: bounce alloc %zu failed (slot %u)", target, slot);
        return false;
    }
    b.cap = alloc_sz;
    return true;
}

long long IoUringBlockProvider::read(int /*lane*/, int shard, void * dst, uint64_t off, uint64_t nbytes) {
    if (nbytes == 0) return 0;
    BlockReadReq req;
    req.shard = shard;
    req.off = off;
    req.nbytes = nbytes;
    req.dst = dst;
    read_batch(&req, 1);
    return req.bytes_moved;
}

bool IoUringBlockProvider::read_batch(BlockReadReq * reqs,
                                      size_t count,
                                      const std::function<void(size_t idx, bool ok)> & on_complete,
                                      const std::function<void()> & on_submitted) {
    if (count == 0) {
        if (on_submitted) on_submitted();
        return true;
    }
    if (!reqs || !is_open()) return false;

    std::lock_guard<std::mutex> lk(ring_mtx_);
    const auto batch_t0 = clock_t_::now();

    std::vector<InFlight> slots(qd_);
    std::vector<uint32_t> free_slots;
    free_slots.reserve(qd_);
    for (uint32_t i = qd_; i > 0; --i)
        free_slots.push_back(i - 1);

    // Slots that need another SQE (short read, EINTR/EAGAIN, or phase-2 buffered EOF tail).
    std::vector<uint32_t> resubmit;
    resubmit.reserve(qd_);

    bool all_ok = true;
    bool submitted_fired = false;
    size_t next_req = 0;
    uint32_t in_flight = 0;
    uint32_t sq_tail = *sq_tail_;

    auto push_sqe = [&](uint32_t slot_id, int fd, void * buf, size_t len, uint64_t off) {
        const uint32_t idx = sq_tail & sq_mask_;
        io_uring_sqe & sqe = sqes_[idx];
        std::memset(&sqe, 0, sizeof(sqe));
        sqe.opcode = IORING_OP_READ;
        sqe.fd = fd;
        sqe.off = off;
        sqe.addr = (uint64_t) (uintptr_t) buf;
        sqe.len = (uint32_t) len;
        sqe.user_data = slot_id;
        sq_array_[idx] = idx;
        ++sq_tail;
    };

    auto fail_req = [&](size_t idx) {
        reqs[idx].bytes_moved = -1;
        reqs[idx].latency_ns = 0;
        all_ok = false;
        if (on_complete) on_complete(idx, false);
    };

    while (next_req < count || in_flight > 0 || !resubmit.empty()) {
        uint32_t to_submit = 0;

        // 1. Re-submit any pending continuations (short reads / buffered EOF tails).
        for (uint32_t slot_id : resubmit) {
            InFlight & st = slots[slot_id];
            const Shard & sh = shards_[(size_t) st.shard];
            const int fd = st.in_tail ? sh.fd_buf : sh.fd;
            push_sqe(slot_id, fd, st.cur_buf, st.cur_len, st.cur_off);
            ++to_submit;
        }
        resubmit.clear();

        // 2. Fill free slots with new requests up to ring capacity.
        while (next_req < count && !free_slots.empty()) {
            const size_t ridx = next_req++;
            BlockReadReq & r = reqs[ridx];
            if (r.nbytes == 0) {
                r.bytes_moved = 0;
                r.latency_ns = 0;
                if (on_complete) on_complete(ridx, true);
                continue;
            }
            if (r.shard < 0 || r.shard >= (int) shards_.size()) {
                log_printf(LogLevel::Error, "io_uring: invalid shard %d of %zu", r.shard, shards_.size());
                fail_req(ridx);
                continue;
            }
            const Shard & sh = shards_[(size_t) r.shard];
            if (sh.size && (r.off > sh.size || r.nbytes > sh.size - r.off)) {
                log_printf(LogLevel::Error, "io_uring: read of %llu bytes at %llu runs past the end of %s (%llu bytes)",
                           (unsigned long long) r.nbytes, (unsigned long long) r.off, sh.path.c_str(),
                           (unsigned long long) sh.size);
                fail_req(ridx);
                continue;
            }
            // IORING_OP_READ takes a 32-bit length; cap single SQE chunk at 1 GiB (remainder resubmits).
            constexpr uint64_t k_max_sqe = 1ull << 30;

            const uint32_t slot_id = free_slots.back();
            InFlight & st = slots[slot_id];
            st = InFlight{};
            st.req_idx = ridx;
            st.shard = r.shard;
            st.t0 = clock_t_::now();

            const bool dst_aligned = ((uintptr_t) r.dst & (sh.align - 1)) == 0;
            const bool off_aligned = (r.off & (sh.align - 1)) == 0;
            const bool len_aligned = (r.nbytes & (sh.align - 1)) == 0;

            if (!sh.aligned_reads || (dst_aligned && off_aligned && len_aligned)) {
                // Zero-copy direct-to-destination path: plain buffered read, or fully aligned O_DIRECT read.
                st.use_bounce = false;
                st.a0 = r.off;
                st.bulk_end = r.off + r.nbytes;
                st.read_end = st.bulk_end;
                st.cur_off = st.a0;
                st.cur_len = (size_t) std::min<uint64_t>(r.nbytes, k_max_sqe);
                st.cur_buf = static_cast<char *>(r.dst);
                st.in_tail = false;
            } else {
                const uint64_t mask = (uint64_t) (sh.align - 1);
                const uint64_t a0 = r.off & ~mask;
                const uint64_t a1 = (r.off + r.nbytes + mask) & ~mask;
                const uint64_t read_end = (sh.size && a1 > sh.size) ? sh.size : a1;
                const uint64_t bulk_end = read_end & ~mask;
                const size_t win_len = (size_t) (a1 - a0);

                if (bulk_end < read_end && !pio::fd_ok(sh.fd_buf)) {
                    log_printf(LogLevel::Error, "io_uring: EOF tail at %llu needs buffered fd, which failed to open",
                               (unsigned long long) bulk_end);
                    fail_req(ridx);
                    continue;
                }
                if (!ensure_bounce(slot_id, win_len)) {
                    fail_req(ridx);
                    continue;
                }
                st.use_bounce = true;
                st.a0 = a0;
                st.bulk_end = bulk_end;
                st.read_end = read_end;
                if (a0 < bulk_end) {
                    st.in_tail = false;
                    st.cur_off = a0;
                    st.cur_len = (size_t) std::min<uint64_t>(bulk_end - a0, k_max_sqe);
                    st.cur_buf = static_cast<char *>(bounces_[slot_id].buf);
                } else {
                    st.in_tail = true;
                    st.cur_off = bulk_end;
                    st.cur_len = (size_t) std::min<uint64_t>(read_end - bulk_end, k_max_sqe);
                    st.cur_buf = static_cast<char *>(bounces_[slot_id].buf) + (bulk_end - a0);
                }
            }

            free_slots.pop_back();
            ++in_flight;
            const int fd = st.in_tail ? sh.fd_buf : sh.fd;
            push_sqe(slot_id, fd, st.cur_buf, st.cur_len, st.cur_off);
            ++to_submit;
        }

        if (in_flight == 0) break;

        if (to_submit > 0) {
            store_release(sq_tail_, sq_tail);
        }

        // If the caller supplied an on_submitted hook, kick off the NVMe SQEs immediately with
        // min_complete=0, run the caller's overlapped CPU work while storage reads execute, and
        // then enter again to reap completions.
        if (on_submitted && !submitted_fired) {
            submitted_fired = true;
            if (to_submit > 0) {
                int sret = 0;
                do {
                    sret = sys_io_uring_enter(ring_fd_, to_submit, 0, 0);
                    if (sret >= 0) {
                        to_submit = (uint32_t) sret < to_submit ? to_submit - (uint32_t) sret : 0;
                    }
                } while (sret < 0 && errno == EINTR);
                if (sret < 0) {
                    log_printf(LogLevel::Error, "io_uring_enter(submit) failed: %s (errno=%d)", std::strerror(errno),
                               errno);
                    on_submitted();
                    all_ok = false;
                    for (uint32_t i = 0; i < qd_; ++i) {
                        if (std::find(free_slots.begin(), free_slots.end(), i) == free_slots.end()) {
                            fail_req(slots[i].req_idx);
                            free_slots.push_back(i);
                        }
                    }
                    while (next_req < count) fail_req(next_req++);
                    in_flight = 0;
                    break;
                }
            }
            on_submitted();
        }

        // Submit any remaining queued SQEs and wait for at least 1 completion in one syscall.
        int ret = 0;
        do {
            ret = sys_io_uring_enter(ring_fd_, to_submit, 1, IORING_ENTER_GETEVENTS);
            if (ret >= 0) {
                to_submit = (uint32_t) ret < to_submit ? to_submit - (uint32_t) ret : 0;
            }
        } while (ret < 0 && errno == EINTR);

        if (ret < 0) {
            log_printf(LogLevel::Error, "io_uring_enter failed: %s (errno=%d)", std::strerror(errno), errno);
            // Fail all remaining requests cleanly.
            all_ok = false;
            for (uint32_t i = 0; i < qd_; ++i) {
                if (std::find(free_slots.begin(), free_slots.end(), i) == free_slots.end()) {
                    fail_req(slots[i].req_idx);
                    free_slots.push_back(i);
                }
            }
            while (next_req < count) fail_req(next_req++);
            in_flight = 0;
            break;
        }

        // 3. Reap every available CQE.
        uint32_t cq_head = *cq_head_;
        const uint32_t cq_tail = load_acquire(cq_tail_);
        while (cq_head != cq_tail) {
            const io_uring_cqe & cqe = cqes_[cq_head & cq_mask_];
            const uint32_t slot_id = (uint32_t) cqe.user_data;
            const int32_t res = cqe.res;
            ++cq_head;

            InFlight & st = slots[slot_id];
            BlockReadReq & r = reqs[st.req_idx];
            const Shard & sh = shards_[(size_t) st.shard];
            constexpr uint64_t k_max_sqe = 1ull << 30;

            if (res == -EINTR || res == -EAGAIN) {
                resubmit.push_back(slot_id);
                continue;
            }
            if (res <= 0) {
                log_printf(LogLevel::Error, "io_uring read failed on %s at %llu (res=%d)", sh.path.c_str(),
                           (unsigned long long) st.cur_off, res);
                --in_flight;
                free_slots.push_back(slot_id);
                fail_req(st.req_idx);
                continue;
            }

            const uint64_t got = (uint64_t) res;
            st.cur_off += got;
            st.cur_buf += got;
            const uint64_t phase_end = st.in_tail ? st.read_end : st.bulk_end;
            if (st.cur_off < phase_end) {
                // Short read or >1 GiB chunk continuation.
                st.cur_len = (size_t) std::min<uint64_t>(phase_end - st.cur_off, k_max_sqe);
                resubmit.push_back(slot_id);
                continue;
            }
            if (!st.in_tail && st.bulk_end < st.read_end) {
                // Transition from aligned direct bulk to buffered sub-alignment EOF tail.
                st.in_tail = true;
                st.cur_off = st.bulk_end;
                st.cur_buf = static_cast<char *>(bounces_[slot_id].buf) + (st.bulk_end - st.a0);
                st.cur_len = (size_t) std::min<uint64_t>(st.read_end - st.bulk_end, k_max_sqe);
                resubmit.push_back(slot_id);
                continue;
            }

            // Full window finished.
            if (st.use_bounce) {
                std::memcpy(r.dst, static_cast<const char *>(bounces_[slot_id].buf) + (r.off - st.a0),
                            (size_t) r.nbytes);
            }
            const long long moved = (long long) (st.read_end - st.a0);
            r.bytes_moved = moved;
            r.latency_ns =
                (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t_::now() - st.t0).count();
            read_bytes_.fetch_add(moved, std::memory_order_relaxed);

            --in_flight;
            free_slots.push_back(slot_id);
            if (on_complete) on_complete(st.req_idx, true);
        }
        store_release(cq_head_, cq_head);
    }

    if (on_submitted && !submitted_fired) on_submitted();

    io_ns_.fetch_add((long long) std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t_::now() - batch_t0).count(),
                     std::memory_order_relaxed);
    return all_ok;
}

} // namespace bmoe

#endif // BMOE_HAS_IO_URING
