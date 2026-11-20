// Raw-syscall Linux io_uring BlockProvider (zero liburing dependency).
//
// Submits batched IORING_OP_READ requests across all shards in a single io_uring_enter syscall
// using <linux/io_uring.h> and direct mmap of SQ/CQ/SQE rings. Preserves O_DIRECT alignment,
// open-time FUSE verification, buffered sub-alignment EOF tails, and short-read resubmission.
#pragma once

#ifdef BMOE_HAS_IO_URING

#include "block_provider.h"
#include "platform_io.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct io_uring_sqe;
struct io_uring_cqe;

namespace bmoe {

class IoUringBlockProvider final : public BlockProvider {
public:
    IoUringBlockProvider() = default;
    ~IoUringBlockProvider() override;

    // Open `paths` on a fresh io_uring with queue depth >= max(32, lanes * 4).
    // If io_uring_setup fails, sets *out_ring_errno to the syscall errno (e.g. ENOSYS/EPERM/EACCES)
    // and returns false without touching any file descriptor.
    bool open(const std::vector<std::string> & paths,
              int lanes,
              bool direct,
              size_t max_read,
              int * out_ring_errno = nullptr);

    bool is_open() const override { return ring_fd_ >= 0 && !shards_.empty(); }
    void close() override;
    bool reopen() override;

    int shard_count() const override { return (int) shards_.size(); }
    uint64_t shard_size(int shard) const override;
    bool direct() const override { return direct_; }
    size_t align() const override { return align_; }
    bool align_reported() const override { return align_reported_; }
    size_t buffer_align() const override { return buffer_align_; }
    bool prefers_batch() const override { return true; }

    long long read(int lane, int shard, void * dst, uint64_t off, uint64_t nbytes) override;
    bool read_batch(BlockReadReq * reqs,
                    size_t count,
                    const std::function<void(size_t idx, bool ok)> & on_complete = nullptr) override;

    long long read_bytes() const override { return read_bytes_.load(std::memory_order_relaxed); }
    long long io_ns() const override { return io_ns_.load(std::memory_order_relaxed); }
    const char * backend_name() const override { return "io_uring"; }

private:
    struct Shard {
        std::string path;
        pio::fd_t fd = pio::fd_invalid;
        pio::fd_t fd_buf = pio::fd_invalid;
        uint64_t size = 0;
        size_t align = pio::direct_align_fallback;
        bool align_reported = false;
        bool direct = false;
        bool aligned_reads = false;
    };

    struct BounceSlot {
        void * buf = nullptr;
        size_t cap = 0;
    };

    bool setup_ring(uint32_t entries, int * out_ring_errno);
    void teardown_ring();
    bool ensure_bounce(uint32_t slot, size_t need);

    int ring_fd_ = -1;
    uint32_t qd_ = 0;

    // Mapped kernel ring regions
    void * sq_ptr_ = nullptr;
    size_t sq_ring_sz_ = 0;
    void * cq_ptr_ = nullptr;
    size_t cq_ring_sz_ = 0;
    io_uring_sqe * sqes_ = nullptr;
    size_t sqes_sz_ = 0;

    uint32_t * sq_head_ = nullptr;
    uint32_t * sq_tail_ = nullptr;
    uint32_t * sq_array_ = nullptr;
    uint32_t sq_mask_ = 0;

    uint32_t * cq_head_ = nullptr;
    uint32_t * cq_tail_ = nullptr;
    io_uring_cqe * cqes_ = nullptr;
    uint32_t cq_mask_ = 0;

    std::vector<Shard> shards_;
    std::vector<BounceSlot> bounces_;

    // Open parameters saved for reopen()
    std::vector<std::string> paths_;
    int lanes_ = 0;
    bool direct_req_ = false;
    size_t max_read_ = 0;

    bool direct_ = false;
    size_t align_ = pio::direct_align_fallback;
    bool align_reported_ = false;
    size_t buffer_align_ = 4096;

    std::mutex ring_mtx_; // serializes ring access if read()/read_batch() are called cross-thread
    std::atomic<long long> read_bytes_{0};
    std::atomic<long long> io_ns_{0};
};

} // namespace bmoe

#endif // BMOE_HAS_IO_URING
