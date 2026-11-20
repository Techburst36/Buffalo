// Unit and end-to-end tests for Milestone 2b:
//   - Virtual-reservation host buffer type (bmoe_virtual_host_buffer_type)
//   - HttpRangeProvider (read, concurrent read_batch, redirects, HEAD->GET fallback,
//     retry on 503, Range rejection, HTTPS error, strict 0-disk mode, bounded LRU spill cache)
//   - Zero-disk remote single-file GGUF boot + prefill/decode byte-identity (CPU & CUDA device cache)
//   - Zero-disk remote 4-shard split GGUF boot + prefill/decode byte-identity + probe parity
//
// Usage: bmoe_http_range_test <tiny-moe-qwen3moe.gguf> <tiny-moe-split-00001-of-00004.gguf>

#include "bmoe/config.h"
#include "bmoe/expert_layout.h"
#include "bmoe/metrics.h"
#include "bmoe/probe.h"
#include "bmoe/session.h"
#include "block_provider.h"
#include "gguf_offsets.h"
#include "http_range_provider.h"
#include "platform_io.h"
#include "virtual_host_buft.h"

#include "cpp-httplib/httplib.h"
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace bmoe;

namespace {

int g_failures = 0;

void check(bool ok, const std::string & what) {
    if (ok) {
        std::printf("[PASS] %s\n", what.c_str());
    } else {
        std::printf("[FAIL] %s\n", what.c_str());
        ++g_failures;
    }
}

bool read_file_all(const std::string & path, std::vector<uint8_t> & out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    const std::streamsize sz = in.tellg();
    if (sz < 0) return false;
    in.seekg(0, std::ios::beg);
    out.resize((size_t) sz);
    if (sz > 0 && !in.read(reinterpret_cast<char *>(out.data()), sz)) return false;
    return true;
}

struct ByteRangeLog {
    std::string path;
    uint64_t start = 0;
    uint64_t end = 0; // exclusive
};

// In-process tracking HTTP server supporting RFC 7233 HEAD + Range GET, redirects,
// HEAD-disabled fallback, non-Range 200 error testing, and transient 503 retry testing.
class TrackingHttpServer {
public:
    TrackingHttpServer() = default;
    ~TrackingHttpServer() { stop(); }

    void add_blob(const std::string & route, std::vector<uint8_t> data) {
        blobs_[route] = std::move(data);
    }

    bool add_file(const std::string & route, const std::string & disk_path) {
        std::vector<uint8_t> bytes;
        if (!read_file_all(disk_path, bytes)) return false;
        blobs_[route] = std::move(bytes);
        return true;
    }

    bool start() {
        // cpp-httplib dispatches both GET and HEAD to Get() handlers; inspect req.method == "HEAD".
        server_.Get("/redirect-blob", [](const httplib::Request &, httplib::Response & res) {
            res.status = 302;
            res.set_header("Location", "/blob.bin");
        });

        server_.Get("/no-head-blob", [this](const httplib::Request & req, httplib::Response & res) {
            if (req.method == "HEAD") {
                res.status = 405;
                return;
            }
            serve_range("/blob.bin", req, res);
        });

        server_.Get("/no-range-head", [this](const httplib::Request &, httplib::Response & res) {
            auto it = blobs_.find("/blob.bin");
            res.status = 200;
            res.set_header("Accept-Ranges", "none");
            res.set_header("Content-Length", std::to_string(it != blobs_.end() ? it->second.size() : 0));
        });

        server_.Get("/no-range-blob", [this](const httplib::Request & req, httplib::Response & res) {
            auto it = blobs_.find("/blob.bin");
            if (it == blobs_.end()) {
                res.status = 404;
                return;
            }
            if (req.method == "HEAD") {
                // Force client to fall back to GET Range: bytes=0-0 (cpp-httplib adds
                // Accept-Ranges: bytes to 200 HEAD responses unless explicitly overridden).
                res.status = 405;
                return;
            }
            // Return 200 OK via chunked provider (ignores Range, never auto-converted to 206 by cpp-httplib).
            res.status = 200;
            res.set_header("Accept-Ranges", "none");
            const auto * data = &it->second;
            res.set_chunked_content_provider(
                "application/octet-stream",
                [data](size_t offset, httplib::DataSink & sink) {
                    if (offset >= data->size()) {
                        sink.done();
                        return true;
                    }
                    const size_t n = std::min<size_t>(4096, data->size() - offset);
                    sink.write(reinterpret_cast<const char *>(data->data() + offset), n);
                    return true;
                });
        });

        server_.Get("/flaky-blob", [this](const httplib::Request & req, httplib::Response & res) {
            if (req.method == "HEAD") {
                auto it = blobs_.find("/blob.bin");
                if (it == blobs_.end()) {
                    res.status = 404;
                    return;
                }
                res.status = 200;
                res.set_header("Accept-Ranges", "bytes");
                res.set_header("Content-Length", std::to_string(it->second.size()));
                return;
            }
            if (flaky_fail_once_.exchange(false, std::memory_order_relaxed)) {
                res.status = 503;
                return;
            }
            serve_range("/blob.bin", req, res);
        });

        // Generic handler for all registered blobs/shards.
        server_.Get(R"(/.*)", [this](const httplib::Request & req, httplib::Response & res) {
            if (req.method == "HEAD") {
                auto it = blobs_.find(req.path);
                if (it == blobs_.end()) {
                    res.status = 404;
                    return;
                }
                res.status = 200;
                res.set_header("Accept-Ranges", "bytes");
                res.set_header("Content-Length", std::to_string(it->second.size()));
                return;
            }
            serve_range(req.path, req, res);
        });

        port_ = server_.bind_to_any_port("127.0.0.1");
        if (port_ <= 0) return false;
        thread_ = std::thread([this]() { server_.listen_after_bind(); });
        for (int i = 0; i < 200 && !server_.is_running(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return server_.is_running();
    }

    void stop() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
        port_ = -1;
    }

    std::string url(const std::string & route) const {
        return "http://127.0.0.1:" + std::to_string(port_) + route;
    }

    void reset_stats() {
        std::lock_guard<std::mutex> lk(mtx_);
        ranges_.clear();
        bytes_served_ = 0;
        get_requests_ = 0;
    }

    void arm_flaky_once() { flaky_fail_once_.store(true, std::memory_order_relaxed); }

    uint64_t bytes_served() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return bytes_served_;
    }

    size_t get_requests() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return get_requests_;
    }

    // Sum of overlap bytes between all recorded GET Range requests on `route` and `[a, b)`.
    uint64_t overlap_bytes(const std::string & route, uint64_t a, uint64_t b) const {
        std::lock_guard<std::mutex> lk(mtx_);
        uint64_t sum = 0;
        for (const ByteRangeLog & r : ranges_) {
            if (r.path != route) continue;
            const uint64_t lo = std::max(r.start, a);
            const uint64_t hi = std::min(r.end, b);
            if (hi > lo) sum += (hi - lo);
        }
        return sum;
    }

private:
    void serve_range(const std::string & route, const httplib::Request & req, httplib::Response & res) {
        auto it = blobs_.find(route);
        if (it == blobs_.end()) {
            res.status = 404;
            return;
        }
        const auto * data = &it->second;
        const size_t total = data->size();
        res.set_header("Accept-Ranges", "bytes");
        // Use set_content_provider with the full blob size so cpp-httplib handles RFC 7233
        // Range parsing, 206 Partial Content, and Content-Range cleanly without double-slicing,
        // while our callback records the exact byte intervals actually served over the wire.
        if (req.has_header("Range")) {
            res.status = 206;
        } else {
            res.status = 200;
        }
        auto logged = std::make_shared<bool>(false);
        res.set_content_provider(
            total, "application/octet-stream",
            [this, route, data, logged](size_t offset, size_t length, httplib::DataSink & sink) {
                if (offset >= data->size()) return false;
                const size_t n = std::min(length, data->size() - offset);
                if (!sink.write(reinterpret_cast<const char *>(data->data() + offset), n)) return false;
                std::lock_guard<std::mutex> lk(mtx_);
                ranges_.push_back({route, (uint64_t) offset, (uint64_t) (offset + n)});
                bytes_served_ += (uint64_t) n;
                if (!*logged) {
                    *logged = true;
                    ++get_requests_;
                }
                return true;
            });
    }

    httplib::Server server_;
    std::thread thread_;
    int port_ = -1;
    std::unordered_map<std::string, std::vector<uint8_t>> blobs_;
    std::atomic<bool> flaky_fail_once_{false};
    mutable std::mutex mtx_;
    std::vector<ByteRangeLog> ranges_;
    uint64_t bytes_served_ = 0;
    size_t get_requests_ = 0;
};

struct LogprobSink : IMetricsSink {
    std::vector<float> logprobs;
    std::vector<std::vector<TokenLogprob>> top_logprobs;
    void on_token(const TokenMetrics & m) override {
        logprobs.push_back(m.logprob);
        top_logprobs.push_back(m.top_logprobs);
    }
    void on_summary(const RunSummary &) override {}
};

bool logprobs_exact_equal(const LogprobSink & a, const LogprobSink & b) {
    if (a.logprobs.size() != b.logprobs.size() || a.logprobs.empty()) return false;
    if (a.top_logprobs.size() != b.top_logprobs.size()) return false;
    for (size_t i = 0; i < a.logprobs.size(); ++i) {
        if (a.logprobs[i] != b.logprobs[i]) return false;
        const auto & ta = a.top_logprobs[i];
        const auto & tb = b.top_logprobs[i];
        if (ta.size() != tb.size()) return false;
        for (size_t j = 0; j < ta.size(); ++j) {
            if (ta[j].id != tb[j].id || ta[j].logprob != tb[j].logprob) return false;
        }
    }
    return true;
}

std::string find_discrete_gpu() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        const char * name = ggml_backend_dev_name(dev);
        if (!name) continue;
        // Skip RPC pseudo-devices; pick a real local GPU (e.g. CUDA0).
        if (std::strncmp(name, "RPC", 3) == 0) continue;
        return name;
    }
    return "";
}

struct RoutedRange {
    int shard = 0;
    uint64_t start = 0;
    uint64_t end = 0;
};

std::vector<RoutedRange> collect_routed_ranges(const GgufMeta & meta, uint64_t & total_routed_bytes) {
    total_routed_bytes = 0;
    std::vector<RoutedRange> out;
    const ExpertLayout lay = discover_expert_layout(meta.offsets.extents, meta.info.n_expert);
    if (!lay.ok) return out;
    for (const TensorExtent & ext : meta.offsets.extents) {
        int elayer = 0;
        if (lay.slot_of(ext.name.c_str(), elayer) < 0) continue;
        const uint64_t off = meta.offsets.off_by_name.at(ext.name);
        const uint64_t sz = meta.offsets.size_by_name.at(ext.name);
        const int shard = meta.offsets.file_by_name.at(ext.name);
        out.push_back({shard, off, off + sz});
        total_routed_bytes += sz;
    }
    return out;
}

uint64_t dir_total_bytes(const std::filesystem::path & dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::exists(dir, ec)) return 0;
    uint64_t total = 0;
    for (const auto & e : fs::directory_iterator(dir, ec)) {
        if (e.is_regular_file(ec)) total += (uint64_t) e.file_size(ec);
    }
    return total;
}

// ── Test 0: Virtual-reservation host buffer type ───────────────────────────────────
void test_virtual_host_buft() {
    llama_backend_init();

    ggml_backend_buffer_type_t buft = bmoe_virtual_host_buffer_type();
    check(buft != nullptr, "virtual_host_buft: non-null singleton");
    check(ggml_backend_buft_is_host(buft), "virtual_host_buft: ggml_backend_buft_is_host() == true");

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    check(cpu_dev != nullptr && ggml_backend_dev_supports_buft(cpu_dev, buft),
          "virtual_host_buft: CPU device supports buft");

    // Zero-size dummy allocation (used by llama-model-loader weight_buft_supported).
    ggml_backend_buffer_t zero_buf = ggml_backend_buft_alloc_buffer(buft, 0);
    check(zero_buf != nullptr, "virtual_host_buft: alloc_buffer(0) succeeds");
    if (zero_buf) ggml_backend_buffer_free(zero_buf);

    // 512 MiB reservation must commit ~0 physical RAM even after clear() and WEIGHTS usage.
    pio::ProcessMemory pm_before{}, pm_after{};
    const bool have_pm = pio::process_memory(&pm_before);
    constexpr size_t k_reserve = 512ull << 20;
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, k_reserve);
    check(buf != nullptr && ggml_backend_buffer_get_base(buf) != nullptr,
          "virtual_host_buft: 512 MiB vm_reserve succeeds");
    if (buf) {
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        check(ggml_backend_buffer_is_host(buf) &&
                  ggml_backend_buffer_get_usage(buf) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
              "virtual_host_buft: satisfies llama_moe_cache_is_host_weight() predicates");
        ggml_backend_buffer_clear(buf, 0);
        if (have_pm && pio::process_memory(&pm_after)) {
            const uint64_t rss_delta =
                pm_after.rss_bytes > pm_before.rss_bytes ? pm_after.rss_bytes - pm_before.rss_bytes : 0;
            check(rss_delta < (16ull << 20),
                  "virtual_host_buft: 512 MiB reserve + clear commits < 16 MiB RSS (delta=" +
                      std::to_string(rss_delta) + " B)");
        }
        ggml_backend_buffer_free(buf);
    }
}

// ── Test 1: HttpRangeProvider unit tests ───────────────────────────────────────────
void test_http_range_provider(TrackingHttpServer & srv) {
    // 3 MiB deterministic blob + two 512 KiB split shards.
    constexpr size_t k_blob_sz = 3ull << 20;
    std::vector<uint8_t> blob(k_blob_sz);
    for (size_t i = 0; i < k_blob_sz; ++i)
        blob[i] = (uint8_t) ((i * 131u + 17u) & 0xFFu);

    constexpr size_t k_shard_sz = 512ull << 10;
    std::vector<uint8_t> shard0(k_shard_sz), shard1(k_shard_sz);
    for (size_t i = 0; i < k_shard_sz; ++i) {
        shard0[i] = (uint8_t) ((i * 37u + 3u) & 0xFFu);
        shard1[i] = (uint8_t) ((i * 53u + 91u) & 0xFFu);
    }

    srv.add_blob("/blob.bin", blob);
    srv.add_blob("/split-00001-of-00002.gguf", shard0);
    srv.add_blob("/split-00002-of-00002.gguf", shard1);

    // 1a. Basic single-file read() + strict 0-disk check.
    namespace fs = std::filesystem;
    const fs::path base_tmp = fs::temp_directory_path() / ("bmoe_http_test_" + std::to_string(std::rand()));
    std::error_code ec;
    fs::remove_all(base_tmp, ec);

    {
        HttpProviderOptions opt;
        opt.cache_dir = (base_tmp / "must_not_exist").string();
        opt.cache_bytes = 0; // 0-disk mode
        HttpRangeProvider prov;
        std::string err;
        check(prov.open({srv.url("/blob.bin")}, 4, opt, &err), "http_provider: open single URL (" + err + ")");
        check(prov.shard_count() == 1 && prov.shard_size(0) == k_blob_sz, "http_provider: exact shard size");
        check(prov.prefers_batch() && !prov.direct() && prov.align() == 1 &&
                  std::strcmp(prov.backend_name(), "http_range") == 0,
              "http_provider: metadata properties");

        std::vector<uint8_t> dst(4096);
        check(prov.read(0, 0, dst.data(), 137, dst.size()) == (long long) dst.size() &&
                  std::memcmp(dst.data(), blob.data() + 137, dst.size()) == 0,
              "http_provider: unaligned read matches ground truth");
        check(prov.read(0, 0, dst.data(), k_blob_sz - 77, 77) == 77 &&
                  std::memcmp(dst.data(), blob.data() + k_blob_sz - 77, 77) == 0,
              "http_provider: tail read matches ground truth");
        check(prov.read(0, 0, dst.data(), k_blob_sz, 1) < 0, "http_provider: out-of-bounds read fails");
        check(!fs::exists(base_tmp / "must_not_exist", ec),
              "http_provider: strict 0-disk mode never creates cache_dir");
    }

    // 1b. Concurrent read_batch() + on_complete across multi-shard expanded URLs.
    {
        const std::vector<std::string> urls = expand_split_urls(srv.url("/split-00001-of-00002.gguf"));
        check(urls.size() == 2 && urls[1] == srv.url("/split-00002-of-00002.gguf"),
              "http_provider: expand_split_urls expands 2 shards");
        HttpRangeProvider prov;
        std::string err;
        check(prov.open(urls, 8, {}, &err), "http_provider: open multi-shard URLs");

        constexpr size_t k_nreq = 32;
        std::vector<std::vector<uint8_t>> bufs(k_nreq);
        std::vector<BlockReadReq> reqs(k_nreq);
        uint64_t expected_bytes = 0;
        for (size_t i = 0; i < k_nreq; ++i) {
            const int sh = (int) (i & 1u);
            const uint64_t off = (uint64_t) (i * 12347u % (k_shard_sz - 8192u));
            const uint64_t len = 1024u + (uint64_t) ((i * 577u) % 6144u);
            bufs[i].resize((size_t) len);
            reqs[i].shard = sh;
            reqs[i].off = off;
            reqs[i].nbytes = len;
            reqs[i].dst = bufs[i].data();
            expected_bytes += len;
        }
        std::vector<std::atomic<int>> cb_seen(k_nreq);
        for (auto & x : cb_seen)
            x.store(0);
        std::atomic<bool> cb_all_ok{true};
        const bool ok = prov.read_batch(reqs.data(), reqs.size(), [&](size_t idx, bool rok) {
            cb_seen[idx].fetch_add(1, std::memory_order_relaxed);
            if (!rok) cb_all_ok.store(false, std::memory_order_relaxed);
        });
        bool bytes_ok = ok && cb_all_ok.load();
        for (size_t i = 0; i < k_nreq; ++i) {
            if (cb_seen[i].load() != 1) bytes_ok = false;
            const auto & src = reqs[i].shard == 0 ? shard0 : shard1;
            if (std::memcmp(bufs[i].data(), src.data() + reqs[i].off, (size_t) reqs[i].nbytes) != 0) bytes_ok = false;
        }
        check(bytes_ok && (uint64_t) prov.read_bytes() == expected_bytes && prov.io_ns() > 0,
              "http_provider: concurrent 32-req multi-shard read_batch + on_complete exact");
    }

    // 1c. Redirects (302), HEAD->GET 206 fallback (405), and retry on transient 503.
    {
        HttpRangeProvider prov;
        std::string err;
        check(prov.open({srv.url("/redirect-blob")}, 2, {}, &err) && prov.shard_size(0) == k_blob_sz,
              "http_provider: follows 302 redirect on HEAD/probe");
        std::vector<uint8_t> dst(512);
        check(prov.read(0, 0, dst.data(), 4096, 512) == 512 &&
                  std::memcmp(dst.data(), blob.data() + 4096, 512) == 0,
              "http_provider: reads exact bytes after redirect");
    }
    {
        HttpRangeProvider prov;
        std::string err;
        check(prov.open({srv.url("/no-head-blob")}, 2, {}, &err) && prov.shard_size(0) == k_blob_sz,
              "http_provider: falls back from HEAD 405 to GET Range: bytes=0-0");
    }
    {
        srv.arm_flaky_once();
        HttpRangeProvider prov;
        std::string err;
        std::vector<uint8_t> dst(1024);
        check(prov.open({srv.url("/flaky-blob")}, 2, {}, &err) &&
                  prov.read(0, 0, dst.data(), 2048, 1024) == 1024 &&
                  std::memcmp(dst.data(), blob.data() + 2048, 1024) == 0,
              "http_provider: retries transient HTTP 503 and recovers cleanly");
    }

    // 1d. Error handling: server rejecting Range (Accept-Ranges: none & 200 OK) & HTTPS without TLS.
    {
        HttpRangeProvider prov;
        std::string err;
        const bool opened = prov.open({srv.url("/no-range-head")}, 2, {}, &err);
        check(!opened && !err.empty(), "http_provider: rejects Accept-Ranges: none on HEAD (" + err + ")");
    }
    {
        HttpRangeProvider prov;
        std::string err;
        const bool opened = prov.open({srv.url("/no-range-blob")}, 2, {}, &err);
        check(!opened && err.find("200") != std::string::npos,
              "http_provider: rejects non-Range server returning 200 OK (" + err + ")");
    }
    {
        HttpRangeProvider prov;
        std::string err;
        const bool opened = prov.open({"https://127.0.0.1:443/model.gguf"}, 2, {}, &err);
        check(!opened && err.find("HTTPS") != std::string::npos,
              "http_provider: rejects https:// without TLS cleanly (" + err + ")");
    }

    // 1e. Optional bounded local disk spill cache (--http-cache-dir + --http-cache-mb=1).
    {
        const fs::path cache_dir = base_tmp / "spill_cache";
        constexpr uint64_t k_cap = 1ull << 20; // 1 MiB (4 chunks of 256 KiB)
        HttpProviderOptions opt;
        opt.cache_dir = cache_dir.string();
        opt.cache_bytes = k_cap;

        HttpRangeProvider prov;
        std::string err;
        check(prov.open({srv.url("/blob.bin")}, 4, opt, &err), "http_spill: open with 1 MiB spill cache");

        // Turn 1: read 3 distinct ranges inside chunks 0, 1, 2 (768 KiB <= 1 MiB cap).
        srv.reset_stats();
        std::vector<uint8_t> a(16384), b(16384), c(16384);
        const bool r1 = prov.read(0, 0, a.data(), 1000, a.size()) == (long long) a.size() &&
                        prov.read(1, 0, b.data(), (256ull << 10) + 500, b.size()) == (long long) b.size() &&
                        prov.read(2, 0, c.data(), (512ull << 10) + 900, c.size()) == (long long) c.size();
        const uint64_t turn1_http = srv.bytes_served();
        const uint64_t turn1_disk = dir_total_bytes(cache_dir);
        check(r1 && turn1_http == 3 * HttpRangeProvider::k_spill_chunk_bytes && turn1_disk <= k_cap &&
                  std::memcmp(a.data(), blob.data() + 1000, a.size()) == 0 &&
                  std::memcmp(b.data(), blob.data() + (256ull << 10) + 500, b.size()) == 0 &&
                  std::memcmp(c.data(), blob.data() + (512ull << 10) + 900, c.size()) == 0,
              "http_spill: Turn 1 populates 3 chunks (http=" + std::to_string(turn1_http) +
                  " B, disk=" + std::to_string(turn1_disk) + " B)");

        // Turn 2: re-read the exact same ranges -> must serve 0 bytes over HTTP!
        srv.reset_stats();
        const long long prev_prov_bytes = prov.read_bytes();
        std::vector<uint8_t> a2(16384), b2(16384), c2(16384);
        const bool r2 = prov.read(0, 0, a2.data(), 1000, a2.size()) == (long long) a2.size() &&
                        prov.read(1, 0, b2.data(), (256ull << 10) + 500, b2.size()) == (long long) b2.size() &&
                        prov.read(2, 0, c2.data(), (512ull << 10) + 900, c2.size()) == (long long) c2.size();
        check(r2 && srv.bytes_served() == 0 && prov.read_bytes() == prev_prov_bytes && prov.spill_cache_hits() >= 3 &&
                  a2 == a && b2 == b && c2 == c,
              "http_spill: Turn 2 serves 0 HTTP bytes from local spill cache (exact match)");

        // Eviction test: read across 8 distinct 256 KiB chunks (2 MiB > 1 MiB cap) and verify strict disk bound.
        std::vector<uint8_t> tmp(4096);
        bool evict_ok = true;
        for (int ch = 0; ch < 8; ++ch) {
            const uint64_t off = (uint64_t) ch * HttpRangeProvider::k_spill_chunk_bytes + 123;
            if (prov.read(ch % 4, 0, tmp.data(), off, tmp.size()) != (long long) tmp.size() ||
                std::memcmp(tmp.data(), blob.data() + off, tmp.size()) != 0)
                evict_ok = false;
        }
        const uint64_t disk_after = dir_total_bytes(cache_dir);
        check(evict_ok && disk_after <= k_cap && disk_after > 0,
              "http_spill: LRU eviction strictly bounds disk usage <= 1 MiB (used=" + std::to_string(disk_after) +
                  " B)");
    }

    fs::remove_all(base_tmp, ec);
}

// ── Test 2: End-to-End Single-File Remote GGUF (CPU + optional CUDA device cache) ──
void test_remote_single_gguf(TrackingHttpServer & srv, const std::string & local_path) {
    const std::string route = "/tiny-moe-qwen3moe.gguf";
    check(srv.add_file(route, local_path), "remote_single: load fixture into HTTP server");

    const GgufMeta local_meta = read_gguf_meta(local_path.c_str());
    check(local_meta.ok, "remote_single: read local GgufMeta");
    uint64_t total_routed_bytes = 0;
    const std::vector<RoutedRange> routed = collect_routed_ranges(local_meta, total_routed_bytes);
    check(!routed.empty() && total_routed_bytes > 0, "remote_single: discovered routed expert ranges");

    const uint64_t file_size = (uint64_t) std::filesystem::file_size(local_path);
    const std::string prompt = "Hello world, this is a remote HTTP MoE streaming verification prompt.";

    auto make_cfg = [&](const std::string & path, bool stream, int cache_mb) {
        RunConfig c;
        c.model_path = path;
        c.prompt = prompt;
        c.n_ctx = 512;
        c.n_ubatch = 256;
        c.n_predict = 24;
        c.n_threads = 2;
        c.device_use = DeviceUse::CpuOnly;
        c.moe.enabled = stream;
        c.moe.cache_mb = cache_mb;
        c.moe.force_cache = cache_mb > 0;
        c.moe.http_cache_mb = 0;
        c.moe.http_threads = 4;
        return c;
    };

    // Local unstreamed CPU reference.
    std::string err;
    RunResult ref_res;
    LogprobSink ref_sink;
    {
        const RunConfig ref_cfg = make_cfg(local_path, false, 0);
        auto sess = Session::open(session_config_from(ref_cfg), err);
        check(sess != nullptr, "remote_single: open local unstreamed reference (" + err + ")");
        if (!sess) return;
        GenerateRequest req;
        req.prompt = ref_cfg.prompt;
        req.n_predict = ref_cfg.n_predict;
        req.n_probs = 4;
        ref_res = sess->generate(req, nullptr, &ref_sink);
        check(ref_res.ok, "remote_single: local reference generate (" + ref_res.error + ")");
    }

    // Remote streamed CPU session with strict 0-disk mode.
    const std::filesystem::path forbidden_dir =
        std::filesystem::temp_directory_path() / ("bmoe_zero_disk_single_" + std::to_string(std::rand()));
    std::error_code ec;
    std::filesystem::remove_all(forbidden_dir, ec);

    srv.reset_stats();
    RunConfig rem_cfg = make_cfg(srv.url(route), true, 1);
    rem_cfg.moe.http_cache_dir = forbidden_dir.string();
    rem_cfg.moe.http_cache_mb = 0;

    auto rem_sess = Session::open(session_config_from(rem_cfg), err);
    check(rem_sess != nullptr, "remote_single: open remote streamed session (" + err + ")");
    if (!rem_sess) return;

    const uint64_t boot_http_bytes = srv.bytes_served();
    uint64_t boot_routed_bytes = 0;
    for (const RoutedRange & r : routed)
        boot_routed_bytes += srv.overlap_bytes(route, r.start, r.end);

    std::printf("[PROOF] Single-file remote boot: file=%llu B, routed_experts=%llu B | boot_http_bytes=%llu B, "
                "boot_routed_expert_bytes=%llu B\n",
                (unsigned long long) file_size, (unsigned long long) total_routed_bytes,
                (unsigned long long) boot_http_bytes, (unsigned long long) boot_routed_bytes);

    check(!std::filesystem::exists(forbidden_dir, ec), "remote_single: 0 local cache files/dirs created");
    check(boot_routed_bytes == 0,
          "remote_single: 0 routed expert bytes requested over HTTP during boot (got " +
              std::to_string(boot_routed_bytes) + " B)");
    check(boot_http_bytes > 0 && boot_http_bytes < file_size,
          "remote_single: boot fetched only metadata + dense weights (" + std::to_string(boot_http_bytes) + " / " +
              std::to_string(file_size) + " B)");

    // Prefill + decode over HTTP and verify exact token & per-token logprob byte-identity.
    srv.reset_stats();
    LogprobSink rem_sink;
    GenerateRequest req;
    req.prompt = rem_cfg.prompt;
    req.n_predict = rem_cfg.n_predict;
    req.n_probs = 4;
    const RunResult rem_res = rem_sess->generate(req, nullptr, &rem_sink);
    uint64_t gen_routed_bytes = 0;
    for (const RoutedRange & r : routed)
        gen_routed_bytes += srv.overlap_bytes(route, r.start, r.end);

    check(rem_res.ok, "remote_single: remote generate succeeded (" + rem_res.error + ")");
    check(gen_routed_bytes > 0,
          "remote_single: generation streamed routed experts on demand (" + std::to_string(gen_routed_bytes) + " B)");
    check(rem_res.generated_text == ref_res.generated_text,
          "remote_single: exact token output matches local unstreamed reference");
    check(logprobs_exact_equal(ref_sink, rem_sink),
          "remote_single: exact per-token chosen + top-4 logprobs byte-identical to local reference");

    // Optional CUDA / discrete GPU device-cache test over HTTP!
    const std::string gpu = find_discrete_gpu();
    if (gpu.empty()) {
        std::printf("[SKIP] remote_single GPU device-cache test: no discrete GPU available\n");
    } else {
        const int n_offload = local_meta.info.n_block + 1;
        auto make_gpu_cfg = [&](const std::string & path, bool stream) {
            RunConfig c;
            c.model_path = path;
            c.prompt = prompt;
            c.n_ctx = 512;
            c.n_ubatch = 256;
            c.n_predict = 24;
            c.n_threads = 2;
            c.n_gpu_layers = n_offload;
            c.device_split = {{gpu, n_offload}};
            c.buft_overrides = {"\\.ffn_(up|down|gate|gate_up)_(ch|)exps"};
            c.moe.enabled = stream;
            c.moe.cache_mb = stream ? 1 : 0;
            c.moe.force_cache = stream;
            c.moe.device_cache_mb = 4;
            c.moe.http_cache_mb = 0;
            c.moe.http_threads = 4;
            return c;
        };

        RunResult gpu_ref_res;
        LogprobSink gpu_ref_sink;
        {
            const RunConfig c = make_gpu_cfg(local_path, false);
            auto s = Session::open(session_config_from(c), err);
            check(s != nullptr, "remote_single_gpu: open local GPU reference (" + err + ")");
            if (s) gpu_ref_res = s->generate(req, nullptr, &gpu_ref_sink);
        }

        srv.reset_stats();
        const RunConfig gpu_rem_cfg = make_gpu_cfg(srv.url(route), true);
        auto gpu_rem_sess = Session::open(session_config_from(gpu_rem_cfg), err);
        check(gpu_rem_sess != nullptr, "remote_single_gpu: open remote GPU device-cache session (" + err + ")");
        if (gpu_rem_sess) {
            uint64_t gpu_boot_routed = 0;
            for (const RoutedRange & r : routed)
                gpu_boot_routed += srv.overlap_bytes(route, r.start, r.end);
            std::printf("[PROOF] Single-file remote GPU (%s + device-cache 4 MiB) boot: boot_http_bytes=%llu B, "
                        "boot_routed_expert_bytes=%llu B\n",
                        gpu.c_str(), (unsigned long long) srv.bytes_served(), (unsigned long long) gpu_boot_routed);
            check(gpu_boot_routed == 0, "remote_single_gpu: 0 routed expert bytes requested during GPU boot");

            LogprobSink gpu_rem_sink;
            const RunResult gpu_rem_res = gpu_rem_sess->generate(req, nullptr, &gpu_rem_sink);
            check(gpu_rem_res.ok && gpu_rem_res.generated_text == gpu_ref_res.generated_text &&
                      logprobs_exact_equal(gpu_ref_sink, gpu_rem_sink),
                  "remote_single_gpu: exact token + top-4 logprob parity with local GPU reference (" + gpu + ")");
        }
    }
}

// ── Test 3: End-to-End 4-Shard Split Remote GGUF ───────────────────────────────────
void test_remote_split_gguf(TrackingHttpServer & srv, const std::string & local_shard0) {
    const GgufMeta local_meta = read_gguf_meta(local_shard0.c_str());
    check(local_meta.ok && local_meta.offsets.shard_paths.size() == 4,
          "remote_split: read local 4-shard GgufMeta");
    if (!local_meta.ok) return;

    std::vector<std::string> shard_routes;
    uint64_t total_file_bytes = 0;
    for (size_t i = 0; i < local_meta.offsets.shard_paths.size(); ++i) {
        const std::string & sp = local_meta.offsets.shard_paths[i];
        const std::string route = "/" + std::filesystem::path(sp).filename().string();
        check(srv.add_file(route, sp), "remote_split: serve shard " + route);
        shard_routes.push_back(route);
        total_file_bytes += (uint64_t) std::filesystem::file_size(sp);
    }

    uint64_t total_routed_bytes = 0;
    const std::vector<RoutedRange> routed = collect_routed_ranges(local_meta, total_routed_bytes);
    check(!routed.empty() && total_routed_bytes > 0, "remote_split: discovered routed expert ranges across shards");

    const std::string remote_shard0 = srv.url(shard_routes[0]);

    // Verify planner probe_model() & model_fingerprint() work over HTTP and match local disk.
    const ModelProfile local_prof = probe_model(local_shard0.c_str());
    const ModelProfile remote_prof = probe_model(remote_shard0.c_str());
    check(remote_prof.ok && remote_prof.arch == local_prof.arch && remote_prof.n_layer == local_prof.n_layer &&
              remote_prof.n_expert == local_prof.n_expert && remote_prof.expert_bytes == local_prof.expert_bytes &&
              remote_prof.dense_bytes == local_prof.dense_bytes,
          "remote_split: probe_model(http://...) matches local probe_model()");
    check(model_fingerprint(remote_shard0.c_str()) == model_fingerprint(local_shard0.c_str()),
          "remote_split: model_fingerprint(http://...) matches local fingerprint");

    const std::string prompt = "Split remote GGUF streaming test across four shards over localhost HTTP.";
    auto make_cfg = [&](const std::string & path, bool stream) {
        RunConfig c;
        c.model_path = path;
        c.prompt = prompt;
        c.n_ctx = 512;
        c.n_ubatch = 256;
        c.n_predict = 24;
        c.n_threads = 2;
        c.device_use = DeviceUse::CpuOnly;
        c.moe.enabled = stream;
        c.moe.cache_mb = stream ? 1 : 0;
        c.moe.force_cache = stream;
        c.moe.http_cache_mb = 0;
        c.moe.http_threads = 4;
        return c;
    };

    std::string err;
    GenerateRequest req;
    req.prompt = prompt;
    req.n_predict = 24;
    req.n_probs = 4;

    RunResult ref_res;
    LogprobSink ref_sink;
    {
        const RunConfig ref_cfg = make_cfg(local_shard0, false);
        auto s = Session::open(session_config_from(ref_cfg), err);
        check(s != nullptr, "remote_split: open local 4-shard reference (" + err + ")");
        if (!s) return;
        ref_res = s->generate(req, nullptr, &ref_sink);
        check(ref_res.ok, "remote_split: local 4-shard reference generate (" + ref_res.error + ")");
    }

    srv.reset_stats();
    const RunConfig rem_cfg = make_cfg(remote_shard0, true);
    auto rem_sess = Session::open(session_config_from(rem_cfg), err);
    check(rem_sess != nullptr, "remote_split: open remote 4-shard streamed session (" + err + ")");
    if (!rem_sess) return;

    const uint64_t boot_http_bytes = srv.bytes_served();
    uint64_t boot_routed_bytes = 0;
    for (const RoutedRange & r : routed)
        boot_routed_bytes += srv.overlap_bytes(shard_routes[(size_t) r.shard], r.start, r.end);

    std::printf("[PROOF] 4-shard split remote boot: total_files=%llu B, routed_experts=%llu B | boot_http_bytes=%llu B, "
                "boot_routed_expert_bytes=%llu B\n",
                (unsigned long long) total_file_bytes, (unsigned long long) total_routed_bytes,
                (unsigned long long) boot_http_bytes, (unsigned long long) boot_routed_bytes);

    check(boot_routed_bytes == 0,
          "remote_split: 0 routed expert bytes requested across all 4 shards during boot (got " +
              std::to_string(boot_routed_bytes) + " B)");

    srv.reset_stats();
    LogprobSink rem_sink;
    const RunResult rem_res = rem_sess->generate(req, nullptr, &rem_sink);
    uint64_t gen_routed_bytes = 0;
    for (const RoutedRange & r : routed)
        gen_routed_bytes += srv.overlap_bytes(shard_routes[(size_t) r.shard], r.start, r.end);

    check(rem_res.ok, "remote_split: remote 4-shard generate succeeded (" + rem_res.error + ")");
    check(gen_routed_bytes > 0,
          "remote_split: generation streamed routed experts across shards (" + std::to_string(gen_routed_bytes) + " B)");
    check(rem_res.generated_text == ref_res.generated_text,
          "remote_split: exact token output matches local 4-shard disk reference");
    check(logprobs_exact_equal(ref_sink, rem_sink),
          "remote_split: exact per-token chosen + top-4 logprobs byte-identical to local 4-shard disk reference");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <tiny-moe-qwen3moe.gguf> <tiny-moe-split-00001-of-00004.gguf>\n", argv[0]);
        return 2;
    }

    test_virtual_host_buft();

    TrackingHttpServer srv;
    if (!srv.start()) {
        std::fprintf(stderr, "[FAIL] could not bind localhost HTTP server on 127.0.0.1:0\n");
        return 1;
    }

    test_http_range_provider(srv);
    test_remote_single_gguf(srv, argv[1]);
    test_remote_split_gguf(srv, argv[2]);

    srv.stop();

    if (g_failures > 0) {
        std::printf("\nFAILED: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("\nALL HTTP RANGE & REMOTE GGUF TESTS PASSED\n");
    return 0;
}
