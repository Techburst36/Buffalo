// Pluggable multi-shard positioned block reader.
//
// Abstracts the underlying storage I/O engine (cross-platform FileReader, Linux io_uring, or
// future remote HTTP range streaming) behind one interface so ExpertStreamSource, DenseWeights,
// RowStream, and DeviceExpertArena never depend directly on a specific OS read primitive.
//
// One BlockProvider owns all shards of a model (0..shard_count()-1). Consumers still construct
// their own providers so O_DIRECT, queue depth, and bounce-buffer sizing remain independent across
// the expert stream, dense loader, and row stream.
#pragma once

#include "bmoe/config.h" // IoBackend

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace bmoe {

class FileReader;

struct BlockReadReq {
    int shard = 0;            // shard index in [0, shard_count())
    uint64_t off = 0;         // byte offset in shard
    uint64_t nbytes = 0;      // exact bytes requested
    void * dst = nullptr;     // caller-owned committed destination buffer
    long long bytes_moved = 0; // actual bytes pulled from storage/network (>= nbytes for aligned window), or < 0 on error
    uint64_t latency_ns = 0;  // per-request latency (used by IoTraceRow)
};

class BlockProvider {
public:
    virtual ~BlockProvider() = default;

    BlockProvider(const BlockProvider &) = delete;
    BlockProvider & operator=(const BlockProvider &) = delete;

    virtual bool is_open() const = 0;
    virtual void close() = 0;
    // Close and reopen underlying descriptors with the same configuration (required on Windows
    // after releasing the model's file mapping; no-op for network providers).
    virtual bool reopen() { return true; }

    virtual int shard_count() const = 0;
    virtual uint64_t shard_size(int shard) const = 0;
    virtual bool direct() const = 0;        // effective cache-bypass mode across all shards
    virtual size_t align() const = 0;       // coarsest direct-read granule across all shards
    virtual bool align_reported() const = 0; // true when the platform reported the granule
    virtual size_t buffer_align() const = 0; // max(align(), vm_page())

    // True when the provider executes read_batch() asynchronously/batched (e.g. io_uring or HTTP),
    // signalling callers like ExpertStreamSource to submit all layer jobs in one read_batch() call
    // instead of fanning out one blocking read() per worker thread.
    virtual bool prefers_batch() const { return false; }

    // Positioned read of `nbytes` at `off` in `shard` into `dst`, on `lane`.
    // Thread-safe across distinct lanes. Returns bytes moved from storage (>= nbytes for aligned
    // direct windows), 0 when nbytes == 0, or -1 on error / out-of-bounds range.
    virtual long long read(int lane, int shard, void * dst, uint64_t off, uint64_t nbytes) = 0;

    // Batched read of `count` requests across shards. If `on_complete` is provided, it is invoked
    // once per request as each completes (with its request index and success flag). If
    // `on_submitted` is provided, it is invoked once after the initial wave of asynchronous reads
    // has been submitted to the OS/network so the caller can overlap CPU work with I/O latency.
    // Default implementation executes sequentially via read(0, ...).
    virtual bool read_batch(BlockReadReq * reqs,
                            size_t count,
                            const std::function<void(size_t idx, bool ok)> & on_complete = nullptr,
                            const std::function<void()> & on_submitted = nullptr);

    virtual long long read_bytes() const = 0;
    virtual long long io_ns() const = 0;
    virtual const char * backend_name() const = 0;

protected:
    BlockProvider() = default;
};

// Cross-platform local file provider wrapping one FileReader per shard.
class PosixBlockProvider final : public BlockProvider {
public:
    PosixBlockProvider();
    ~PosixBlockProvider() override;

    bool open(const std::vector<std::string> & paths, int lanes, bool direct, size_t max_read);

    bool is_open() const override { return !readers_.empty(); }
    void close() override;
    bool reopen() override;

    int shard_count() const override { return (int) readers_.size(); }
    uint64_t shard_size(int shard) const override;
    bool direct() const override { return direct_; }
    size_t align() const override { return align_; }
    bool align_reported() const override { return align_reported_; }
    size_t buffer_align() const override { return buffer_align_; }

    long long read(int lane, int shard, void * dst, uint64_t off, uint64_t nbytes) override;

    long long read_bytes() const override;
    long long io_ns() const override;
    const char * backend_name() const override { return "posix"; }

private:
    void refresh_geometry();

    std::vector<std::unique_ptr<FileReader>> readers_;
    bool direct_ = false;
    size_t align_ = 4096;
    bool align_reported_ = false;
    size_t buffer_align_ = 4096;
};

// Options specific to remote HTTP Range streaming.
// When cache_dir is empty or cache_bytes == 0, HttpRangeProvider operates in strict 0-disk mode.
struct HttpProviderOptions {
    std::string cache_dir;
    uint64_t cache_bytes = 0;
    int lanes = 0; // 0 = use caller's lane count (or default minimum for HTTP)
};

// URL scheme helpers for remote HTTP GGUF streaming.
bool is_http_url(const std::string & path);
bool is_https_url(const std::string & path);
bool is_remote_url(const std::string & path);

// Expand a split URL `<prefix>-00001-of-NNNNN.gguf[?query]` into all N shard URLs (preserving any
// query string). Returns {url} for a non-split URL; returns empty vector if url names shard > 1.
std::vector<std::string> expand_split_urls(const std::string & url);

// Open a local-file BlockProvider honoring `pref` (auto | posix | io_uring).
// When io_uring is compiled in and requested (auto or io_uring), attempts IoUringBlockProvider
// first; if io_uring_setup fails due to kernel/sandbox policy (ENOSYS/EPERM/EACCES/ENOMEM), logs
// a single warning and falls back cleanly to PosixBlockProvider.
std::unique_ptr<BlockProvider> open_local_block_provider(const std::vector<std::string> & paths,
                                                         int lanes,
                                                         bool direct,
                                                         size_t max_read,
                                                         IoBackend pref = IoBackend::Auto);

// Open either a local BlockProvider or an HttpRangeProvider when `paths[0]` is a remote URL.
// If `error` is non-null, writes an actionable diagnostic on failure.
std::unique_ptr<BlockProvider> open_block_provider(const std::vector<std::string> & paths,
                                                   int lanes,
                                                   bool direct,
                                                   size_t max_read,
                                                   IoBackend pref = IoBackend::Auto,
                                                   const HttpProviderOptions & http_opt = {},
                                                   std::string * error = nullptr);

} // namespace bmoe
