// Remote HTTP Range BlockProvider with keep-alive lane pool and optional bounded local disk spill cache.
//
// Streams tensor bytes directly over HTTP 206 Partial Content Range requests using vendored
// cpp-httplib (zero OpenSSL/libcurl dependencies for plain http://).
//
// When cache_dir is empty or cache_bytes == 0 (the default), operates in strict 0-byte local disk
// mode: no directories, cache files, or temporary files are ever touched.
#pragma once

#include "block_provider.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace httplib {
class Client;
}

namespace bmoe {

class HttpRangeProvider final : public BlockProvider {
public:
    // Fixed chunk granularity for the optional bounded local disk spill cache.
    static constexpr uint64_t k_spill_chunk_bytes = 256ull << 10; // 256 KiB

    HttpRangeProvider();
    ~HttpRangeProvider() override;

    // Probe each shard URL via HEAD (falling back to GET Range: bytes=0-0), follow redirects
    // (301/302/303/307/308), confirm HTTP 206 Range support, initialize the keep-alive lane pool,
    // and (only if cache_dir is non-empty and cache_bytes > 0) initialize the bounded spill cache.
    bool open(const std::vector<std::string> & urls,
              int lanes,
              const HttpProviderOptions & opt = {},
              std::string * error = nullptr);

    bool is_open() const override { return open_; }
    void close() override;
    bool reopen() override { return open_; }

    int shard_count() const override { return (int) shards_.size(); }
    uint64_t shard_size(int shard) const override;
    bool direct() const override { return false; }
    size_t align() const override { return 1; }
    bool align_reported() const override { return true; }
    size_t buffer_align() const override { return buffer_align_; }
    bool prefers_batch() const override { return true; }

    long long read(int lane, int shard, void * dst, uint64_t off, uint64_t nbytes) override;

    bool read_batch(BlockReadReq * reqs,
                    size_t count,
                    const std::function<void(size_t idx, bool ok)> & on_complete = nullptr) override;

    // Total HTTP payload bytes transferred over the wire (excludes local disk spill cache hits).
    long long read_bytes() const override { return (long long) read_bytes_.load(std::memory_order_relaxed); }
    long long io_ns() const override { return (long long) io_ns_.load(std::memory_order_relaxed); }
    const char * backend_name() const override { return "http_range"; }

    // Spill cache observability (0 when spill cache is disabled).
    uint64_t spill_cache_hits() const { return spill_hits_.load(std::memory_order_relaxed); }
    uint64_t spill_cache_misses() const { return spill_misses_.load(std::memory_order_relaxed); }

private:
    struct Shard {
        std::string orig_url;
        std::string scheme_host_port; // resolved final authority after redirects
        std::string path_query;       // resolved final path + query
        uint64_t size = 0;
        uint64_t url_hash = 0;        // stable key prefix for optional disk spill cache
    };

    struct Lane {
        std::mutex mtx;
        std::unordered_map<std::string, std::unique_ptr<httplib::Client>> clients;
    };

    struct SpillEntry {
        std::string key;
        std::string path;
        uint64_t size = 0;
    };

    // Fetch exact [off, off + nbytes) directly over HTTP on `lane` into `dst`.
    // Returns nbytes on success, or -1 on error. Updates read_bytes_ with HTTP bytes received.
    long long fetch_http_exact(int lane, int shard, void * dst, uint64_t off, uint64_t nbytes);

    // Read through the optional 256 KiB chunk LRU local disk spill cache.
    long long read_with_spill(int lane, int shard, void * dst, uint64_t off, uint64_t nbytes);

    bool init_spill_cache(std::string * error);
    void spill_evict_locked(uint64_t need_bytes);
    bool spill_read_chunk(const std::string & key, uint64_t inner_off, void * dst, uint64_t len);
    void spill_write_chunk(const std::string & key, const void * data, uint64_t len);

    httplib::Client * lane_client(Lane & lane, const std::string & scheme_host_port, bool reset = false);

    // Persistent worker pool for concurrent read_batch() without per-batch thread churn.
    void start_workers();
    void stop_workers();
    void batch_worker(int lane);
    void execute_batch_slot(int lane);

    bool open_ = false;
    int lanes_ = 1;
    size_t buffer_align_ = 4096;
    std::vector<Shard> shards_;
    std::vector<std::unique_ptr<Lane>> lane_pool_;

    // Optional bounded disk spill cache (inactive when cache_dir_.empty() || cache_bytes_ == 0).
    std::string cache_dir_;
    uint64_t cache_bytes_ = 0;
    std::mutex cache_mtx_;
    std::list<SpillEntry> spill_lru_; // oldest at front, newest at back
    std::unordered_map<std::string, std::list<SpillEntry>::iterator> spill_map_;
    uint64_t spill_used_bytes_ = 0;
    std::atomic<uint64_t> spill_tmp_seq_{0};

    // Persistent batch dispatch state (serialized by batch_call_mtx_).
    std::mutex batch_call_mtx_;
    std::mutex worker_mtx_;
    std::condition_variable worker_cv_;
    std::condition_variable batch_done_cv_;
    std::vector<std::thread> workers_;
    bool worker_stop_ = false;
    uint64_t batch_gen_ = 0;
    BlockReadReq * cur_reqs_ = nullptr;
    size_t cur_count_ = 0;
    const std::function<void(size_t idx, bool ok)> * cur_cb_ = nullptr;
    std::atomic<size_t> batch_next_{0};
    std::atomic<bool> batch_all_ok_{true};
    int active_workers_ = 0;

    std::atomic<uint64_t> read_bytes_{0};
    std::atomic<uint64_t> io_ns_{0};
    std::atomic<uint64_t> spill_hits_{0};
    std::atomic<uint64_t> spill_misses_{0};
};

} // namespace bmoe
