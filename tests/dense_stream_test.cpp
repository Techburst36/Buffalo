// End-to-end and unit verification suite for Milestone 3 ("Smash the Dense Wall"):
//   0. Page-aligned bmoe_virtual_host_buffer_type() eviction isolation
//   1. Pure Dense (qwen3): Local Posix & io_uring (--dense-stream --dense-cache-mb 6,
//      auto 2-layer window, and --row-stream) — bounded peak RAM, real evictions, prefetch hits
//   2. Pure Dense (qwen3): Extreme 1-layer starvation (--dense-cache-mb 1) — zero deadlock,
//      look-ahead disabled cleanly, immediate post-unpin eviction
//   3. Pure Dense (qwen3): Partial GPU offload (--gpu-layers 3 on CUDA0) — host layers streamed
//      with op_offload=false while GPU layers stay resident in VRAM
//   4. MoE + Dense Combined ("Full Kimi Mode") across qwen3moe, gemma4 (interleaved dense +
//      shared expert + tied embeddings), and nemotron_h_moe (Mamba2 SSM + Attn + MoE + shexp)
//   5. Zero-disk HTTP (--http-cache-mb 0) for Pure Dense (qwen3) AND MoE+Dense Combined
//      (qwen3moe & gemma4): proves 0 bytes of streamed dense layers AND 0 bytes of routed
//      experts fetched at boot, 0 local files created, and exact token/logprob parity.
//
// Usage:
//   bmoe_dense_stream_test <tiny-dense-qwen3.gguf> <tiny-moe-qwen3moe.gguf>
//                          <tiny-moe-gemma4.gguf> <tiny-moe-nemotron_h_moe.gguf>

#include "bmoe/config.h"
#include "bmoe/expert_layout.h"
#include "bmoe/metrics.h"
#include "bmoe/session.h"
#include "gguf_offsets.h"
#include "platform_io.h"
#include "tensor_names.h"
#include "virtual_host_buft.h"

#include "cpp-httplib/httplib.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
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

class TrackingHttpServer {
public:
    ~TrackingHttpServer() { stop(); }

    bool add_file(const std::string & route, const std::string & disk_path) {
        std::vector<uint8_t> bytes;
        if (!read_file_all(disk_path, bytes)) return false;
        blobs_[route] = std::move(bytes);
        return true;
    }

    bool start() {
        server_.Get(R"(/.*)", [this](const httplib::Request & req, httplib::Response & res) {
            auto it = blobs_.find(req.path);
            if (it == blobs_.end()) {
                res.status = 404;
                return;
            }
            const auto & data = it->second;
            if (req.method == "HEAD") {
                res.status = 200;
                res.set_header("Accept-Ranges", "bytes");
                res.set_header("Content-Length", std::to_string(data.size()));
                return;
            }
            serve_range(req.path, data, req, res);
        });

        port_ = server_.bind_to_any_port("127.0.0.1");
        if (port_ <= 0) return false;
        thread_ = std::thread([this]() { server_.listen_after_bind(); });
        for (int i = 0; i < 200 && !server_.is_running(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return server_.is_running();
    }

    void stop() {
        if (port_ > 0) {
            server_.stop();
            if (thread_.joinable()) thread_.join();
            port_ = 0;
        }
    }

    std::string url(const std::string & route) const {
        return "http://127.0.0.1:" + std::to_string(port_) + route;
    }

    void clear_logs() {
        std::lock_guard<std::mutex> lk(log_mtx_);
        logs_.clear();
    }

    std::vector<ByteRangeLog> take_logs() const {
        std::lock_guard<std::mutex> lk(log_mtx_);
        return logs_;
    }

private:
    void serve_range(const std::string & route,
                     const std::vector<uint8_t> & data,
                     const httplib::Request & req,
                     httplib::Response & res) {
        const auto * bytes = &data;
        const size_t total = bytes->size();
        res.set_header("Accept-Ranges", "bytes");
        res.status = req.has_header("Range") ? 206 : 200;
        res.set_content_provider(
            total, "application/octet-stream",
            [this, route, bytes](size_t offset, size_t length, httplib::DataSink & sink) {
                if (offset >= bytes->size()) return false;
                const size_t n = std::min(length, bytes->size() - offset);
                if (!sink.write(reinterpret_cast<const char *>(bytes->data() + offset), n)) return false;
                std::lock_guard<std::mutex> lk(log_mtx_);
                logs_.push_back({route, (uint64_t) offset, (uint64_t) (offset + n)});
                return true;
            });
    }

    httplib::Server server_;
    std::thread thread_;
    int port_ = 0;
    std::unordered_map<std::string, std::vector<uint8_t>> blobs_;
    mutable std::mutex log_mtx_;
    std::vector<ByteRangeLog> logs_;
};

struct GenCapture {
    bool ok = false;
    std::string error;
    std::string text;
    std::vector<std::string> pieces;
    std::vector<float> chosen_logprobs;
    std::vector<std::vector<TokenLogprob>> top_logprobs;
    DenseStreamStats dense_stats{};
};

GenCapture run_generation(const SessionConfig & cfg, const std::string & prompt, int n_predict) {
    GenCapture out;
    auto sess = Session::open(cfg, out.error);
    if (!sess) return out;

    GenerateRequest req;
    req.prompt = prompt;
    req.n_predict = n_predict;
    req.clear_kv = true;
    req.render_text = false;
    req.n_probs = 4;

    RunResult r = sess->generate(req, [&](const TokenMetrics & m) {
        out.pieces.push_back(m.piece);
        if (m.has_logprobs) {
            out.chosen_logprobs.push_back(m.logprob);
            out.top_logprobs.push_back(m.top_logprobs);
        }
    });
    out.ok = r.ok;
    out.error = r.error;
    out.text = r.generated_text;
    out.dense_stats = sess->dense_stream_stats();
    return out;
}

bool compare_captures(const GenCapture & ref,
                      const GenCapture & got,
                      float max_logprob_diff,
                      std::string & diag) {
    if (!ref.ok || !got.ok) {
        diag = "run failed (ref.ok=" + std::to_string(ref.ok) + " got.ok=" + std::to_string(got.ok) +
               " err='" + got.error + "')";
        return false;
    }
    if (ref.text != got.text || ref.pieces != got.pieces) {
        diag = "text mismatch: ref='" + ref.text + "' vs got='" + got.text + "'";
        return false;
    }
    if (ref.top_logprobs.size() != got.top_logprobs.size() || ref.top_logprobs.empty()) {
        diag = "logprob count mismatch: " + std::to_string(ref.top_logprobs.size()) + " vs " +
               std::to_string(got.top_logprobs.size());
        return false;
    }
    float worst = 0.0f;
    for (size_t i = 0; i < ref.top_logprobs.size(); ++i) {
        const float cd = std::fabs(ref.chosen_logprobs[i] - got.chosen_logprobs[i]);
        if (cd > worst) worst = cd;
        if (ref.top_logprobs[i].size() != got.top_logprobs[i].size()) {
            diag = "top_logprobs width mismatch at step " + std::to_string(i);
            return false;
        }
        for (size_t k = 0; k < ref.top_logprobs[i].size(); ++k) {
            const auto & a = ref.top_logprobs[i][k];
            const auto & b = got.top_logprobs[i][k];
            if (a.id != b.id) {
                diag = "top-" + std::to_string(k) + " id mismatch at step " + std::to_string(i) + ": " +
                       std::to_string(a.id) + " vs " + std::to_string(b.id);
                return false;
            }
            const float d = std::fabs(a.logprob - b.logprob);
            if (d > worst) worst = d;
        }
    }
    if (worst > max_logprob_diff) {
        diag = "max logprob diff " + std::to_string(worst) + " > " + std::to_string(max_logprob_diff);
        return false;
    }
    diag = "exact tokens (" + std::to_string(got.pieces.size()) + "), max |dlogp|=" + std::to_string(worst);
    return true;
}

uint64_t overlap_bytes(const std::vector<ByteRangeLog> & logs,
                       const std::string & route,
                       const std::vector<std::pair<uint64_t, uint64_t>> & ranges) {
    uint64_t total = 0;
    for (const ByteRangeLog & log : logs) {
        if (log.path != route) continue;
        for (const auto & r : ranges) {
            const uint64_t lo = std::max(log.start, r.first);
            const uint64_t hi = std::min(log.end, r.second);
            if (hi > lo) total += (hi - lo);
        }
    }
    return total;
}

bool has_discrete_gpu() {
    llama_backend_init();
    bool found = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            found = true;
            break;
        }
    }
    llama_backend_free();
    return found;
}

// ── Test 0: Page-aligned bmoe_virtual_host_buffer_type() eviction isolation ─────────
void test_virtual_host_page_isolation() {
    std::printf("\n=== Test 0: Page-aligned bmoe_virtual_host_buffer_type() ===\n");
    const size_t page = pio::vm_page();
    ggml_backend_buffer_type_t vhost = bmoe_virtual_host_buffer_type();
    const size_t align = ggml_backend_buft_get_alignment(vhost);
    check(align >= page && (align % page) == 0, "vhost alignment >= vm_page (" + std::to_string(align) + ")");

    ggml_init_params gp = {1024 * 1024, nullptr, true};
    ggml_context * gctx = ggml_init(gp);
    // Deliberately choose non-page-multiple tensor sizes (3000 bytes, 5100 bytes).
    ggml_tensor * t0 = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 750);  // 3000 bytes
    ggml_tensor * t1 = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 1275); // 5100 bytes

    const size_t s0 = ggml_backend_buft_get_alloc_size(vhost, t0);
    const size_t s1 = ggml_backend_buft_get_alloc_size(vhost, t1);
    check(s0 >= ggml_nbytes(t0) + pio::tensor_tail_slack() && (s0 % page) == 0,
          "vhost get_alloc_size(t0) rounded up with tail slack (" + std::to_string(s0) + ")");
    check(s1 >= ggml_nbytes(t1) + pio::tensor_tail_slack() && (s1 % page) == 0,
          "vhost get_alloc_size(t1) rounded up with tail slack (" + std::to_string(s1) + ")");

    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(vhost, s0 + s1);
    check(buf != nullptr, "vhost alloc_buffer succeeded");
    ggml_tallocr talloc = ggml_tallocr_new(buf);
    check(ggml_tallocr_alloc(&talloc, t0) == GGML_STATUS_SUCCESS, "tallocr_alloc(t0) ok");
    check(ggml_tallocr_alloc(&talloc, t1) == GGML_STATUS_SUCCESS, "tallocr_alloc(t1) ok");
    check((((uintptr_t) t0->data) % page) == 0 && (((uintptr_t) t1->data) % page) == 0,
          "both tensors are page-aligned");
    check((uintptr_t) t1->data >= (uintptr_t) t0->data + s0, "t0 and t1 sit on disjoint page spans");

    check(pio::vm_commit(t0->data, s0) && pio::vm_commit(t1->data, s1), "vm_commit both spans ok");
    std::memset(t0->data, 0xA5, ggml_nbytes(t0));
    std::memset(t1->data, 0x5A, ggml_nbytes(t1));

    // Evict t0; t1 must remain 100% untouched.
    pio::vm_evict(t0->data, s0);
    bool t1_intact = true;
    const auto * p1 = static_cast<const uint8_t *>(t1->data);
    for (size_t i = 0; i < ggml_nbytes(t1); ++i) {
        if (p1[i] != 0x5A) {
            t1_intact = false;
            break;
        }
    }
    check(t1_intact, "vm_evict(t0) leaves neighboring tensor t1 bytes 100% intact");

    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
}

// ── Test 1 & 2: Pure Dense (qwen3) Posix, io_uring, Auto, RowStream, and 1 MiB Starvation ──
void test_pure_dense_local(const std::string & qwen3_path, GenCapture & out_baseline) {
    std::printf("\n=== Test 1: Pure Dense (qwen3) — Local Posix & io_uring ===\n");
    const std::string prompt = "The quick brown fox jumps over the lazy dog and streams dense layers.";
    constexpr int n_predict = 12;

    SessionConfig base_cfg;
    base_cfg.model_path = qwen3_path;
    base_cfg.n_ctx = 256;
    base_cfg.n_batch = 128;
    base_cfg.n_threads = 4;
    base_cfg.device_use = DeviceUse::CpuOnly;
    base_cfg.moe.enabled = false;
    base_cfg.moe.dense_stream = false;

    out_baseline = run_generation(base_cfg, prompt, n_predict);
    check(out_baseline.ok && out_baseline.pieces.size() == (size_t) n_predict,
          "qwen3 resident baseline generated " + std::to_string(out_baseline.pieces.size()) + " tokens");

    // 1a. Posix --dense-stream --dense-cache-mb 6
    uint64_t posix_6mb_read = 0;
    {
        SessionConfig cfg = base_cfg;
        cfg.moe.dense_stream = true;
        cfg.moe.dense_cache_mb = 6;
        cfg.moe.io_backend = IoBackend::Posix;
        GenCapture got = run_generation(cfg, prompt, n_predict);
        std::string diag;
        const bool ok = compare_captures(out_baseline, got, 1e-6f, diag);
        check(ok, "qwen3 Posix dense-cache-mb=6 parity: " + diag);
        const DenseStreamStats & st = got.dense_stats;
        posix_6mb_read = st.bytes_read;
        std::printf("  [Posix 6MiB] total=%.2f MiB, budget=%.2f MiB, static_pinned=%.2f MiB (%llu groups), "
                    "peak=%.2f MiB, read=%.2f MiB, loaded=%llu, evicted=%llu, prefetch_hits=%llu\n",
                    st.total_streamed_weight_bytes / 1048576.0, st.budget_bytes / 1048576.0,
                    st.static_pinned_bytes / 1048576.0, (unsigned long long) st.static_pinned_groups,
                    st.peak_resident_bytes / 1048576.0, st.bytes_read / 1048576.0,
                    (unsigned long long) st.groups_loaded, (unsigned long long) st.groups_evicted,
                    (unsigned long long) st.prefetch_hits);
        check(st.total_streamed_weight_bytes > (12ull << 20), "Posix: total streamed dense > 12 MiB");
        check(st.peak_resident_bytes <= (6ull << 20) && st.peak_resident_bytes < st.total_streamed_weight_bytes,
              "Posix: peak resident bounded <= 6 MiB and < total dense bytes");
        check(st.groups_evicted > 0, "Posix: real LRU evictions occurred");
        check(st.prefetch_hits > 0, "Posix: async 1-layer look-ahead prefetch hits occurred");
        check(st.io_errors == 0, "Posix: zero I/O errors");
    }

    // 1b. io_uring --dense-stream --dense-cache-mb 6
    uint64_t iouring_6mb_read = 0;
    {
        SessionConfig cfg = base_cfg;
        cfg.moe.dense_stream = true;
        cfg.moe.dense_cache_mb = 6;
        cfg.moe.io_backend = IoBackend::IoUring;
        GenCapture got = run_generation(cfg, prompt, n_predict);
        std::string diag;
        const bool ok = compare_captures(out_baseline, got, 1e-6f, diag);
        check(ok, "qwen3 io_uring dense-cache-mb=6 parity: " + diag);
        const DenseStreamStats & st = got.dense_stats;
        iouring_6mb_read = st.bytes_read;
        std::printf("  [io_uring 6MiB] total=%.2f MiB, budget=%.2f MiB, static_pinned=%.2f MiB (%llu groups), "
                    "peak=%.2f MiB, read=%.2f MiB, loaded=%llu, evicted=%llu, prefetch_hits=%llu\n",
                    st.total_streamed_weight_bytes / 1048576.0, st.budget_bytes / 1048576.0,
                    st.static_pinned_bytes / 1048576.0, (unsigned long long) st.static_pinned_groups,
                    st.peak_resident_bytes / 1048576.0, st.bytes_read / 1048576.0,
                    (unsigned long long) st.groups_loaded, (unsigned long long) st.groups_evicted,
                    (unsigned long long) st.prefetch_hits);
        check(st.peak_resident_bytes <= (6ull << 20) && st.groups_evicted > 0 && st.prefetch_hits > 0,
              "io_uring: bounded peak RAM, real evictions, and prefetch hits");
    }

    // 1c. Static-pinning anti-thrash proof (--dense-cache-mb 10 on 16.26 MiB 6-layer qwen3)
    //     + Linux zero-mmap verification on local --dense-stream boot.
    for (IoBackend backend : {IoBackend::Posix, IoBackend::IoUring}) {
        const char * bname = backend == IoBackend::Posix ? "Posix" : "io_uring";
        const uint64_t ref_6mb_read = backend == IoBackend::Posix ? posix_6mb_read : iouring_6mb_read;

        SessionConfig cfg = base_cfg;
        cfg.moe.dense_stream = true;
        cfg.moe.dense_cache_mb = 10;
        cfg.moe.io_backend = backend;

        std::string err;
        auto sess = Session::open(cfg, err);
        check(sess != nullptr, std::string("qwen3 ") + bname + " 10MiB opened cleanly" + (err.empty() ? "" : ": " + err));
        if (sess) {
#if defined(__linux__)
            // Confirm local --dense-stream booted via llama_model_init_from_user() with ZERO mmap of the GGUF.
            std::ifstream maps("/proc/self/maps");
            std::string line;
            bool mapped = false;
            const std::string fname = std::filesystem::path(qwen3_path).filename().string();
            while (std::getline(maps, line)) {
                if (line.find(fname) != std::string::npos) {
                    mapped = true;
                    break;
                }
            }
            check(!mapped, std::string("qwen3 ") + bname + " local --dense-stream: ZERO mmap of GGUF in /proc/self/maps");
#endif
            GenerateRequest req;
            req.prompt = prompt;
            req.n_predict = n_predict;
            req.clear_kv = true;
            req.render_text = false;
            req.n_probs = 4;

            GenCapture got;
            RunResult r = sess->generate(req, [&](const TokenMetrics & m) {
                got.pieces.push_back(m.piece);
                if (m.has_logprobs) {
                    got.chosen_logprobs.push_back(m.logprob);
                    got.top_logprobs.push_back(m.top_logprobs);
                }
            });
            got.ok = r.ok;
            got.error = r.error;
            got.text = r.generated_text;
            got.dense_stats = sess->dense_stream_stats();

            std::string diag;
            const bool ok = compare_captures(out_baseline, got, 1e-6f, diag);
            check(ok, std::string("qwen3 ") + bname + " dense-cache-mb=10 static-pin parity: " + diag);

            const DenseStreamStats & st = got.dense_stats;
            const double cyclic_full_mib = (13.0 * st.total_streamed_weight_bytes) / 1048576.0;
            const double saved_vs_cyclic_mib = cyclic_full_mib - (st.bytes_read / 1048576.0);
            const double saved_vs_6mb_mib = (ref_6mb_read > st.bytes_read) ? ((ref_6mb_read - st.bytes_read) / 1048576.0) : 0.0;
            std::printf("  [%s 10MiB static-pin] total=%.2f MiB, budget=%.2f MiB, static_pinned=%.2f MiB "
                        "(%llu groups), peak=%.2f MiB, read=%.2f MiB (vs 6MiB=%.2f MiB, cyclic=%.2f MiB, "
                        "saved=%.2f MiB vs 6MiB / %.2f MiB vs cyclic), loaded=%llu, evicted=%llu, prefetch_hits=%llu\n",
                        bname, st.total_streamed_weight_bytes / 1048576.0, st.budget_bytes / 1048576.0,
                        st.static_pinned_bytes / 1048576.0, (unsigned long long) st.static_pinned_groups,
                        st.peak_resident_bytes / 1048576.0, st.bytes_read / 1048576.0, ref_6mb_read / 1048576.0,
                        cyclic_full_mib, saved_vs_6mb_mib, saved_vs_cyclic_mib,
                        (unsigned long long) st.groups_loaded, (unsigned long long) st.groups_evicted,
                        (unsigned long long) st.prefetch_hits);
            check(st.static_pinned_groups >= 2 && st.static_pinned_bytes > (2ull << 20),
                  std::string(bname) + " 10MiB: prefix layers static-pinned (>= 2 groups, > 2 MiB)");
            check(st.peak_resident_bytes <= (10ull << 20), std::string(bname) + " 10MiB: peak resident <= 10 MiB");
            check(st.bytes_read < ref_6mb_read && st.bytes_read + (25ull << 20) < 13ull * st.total_streamed_weight_bytes,
                  std::string(bname) + " 10MiB: static pinning strictly reduced bytes_read vs 6 MiB and cyclic scan");
            check(st.groups_evicted > 0 && st.prefetch_hits > 0 && st.io_errors == 0,
                  std::string(bname) + " 10MiB: rotating tail evicted & prefetched cleanly with 0 errors");

#if defined(__linux__)
            // Steady-state minor page fault proof: ring slots and static-pinned prefix layers are
            // now hot from the run above. Run a second 12-token sweep (~175+ MiB of rotating reads,
            // which previously triggered ~45,000+ 4 KiB minor faults via MADV_DONTNEED) and verify
            // minor page faults remain near zero.
            const uint64_t mf_before = pio::minor_faults();
            const uint64_t bytes_before = sess->dense_stream_stats().bytes_read;
            RunResult r2 = sess->generate(req, [](const TokenMetrics &) {});
            const uint64_t mf_delta = pio::minor_faults() - mf_before;
            const uint64_t bytes_delta = sess->dense_stream_stats().bytes_read - bytes_before;
            std::printf("  [%s 10MiB steady-state] streamed=%.2f MiB, minor_faults=%llu\n", bname,
                        bytes_delta / 1048576.0, (unsigned long long) mf_delta);
            check(r2.ok && bytes_delta > (100ull << 20) && mf_delta < 512,
                  std::string(bname) + " 10MiB: steady-state rotating sweeps incur ~0 minor page faults (" +
                      std::to_string(mf_delta) + ")");
#endif
        }
    }

    // 1d. Auto 2-layer window (--dense-cache-mb 0) + --row-stream on pure dense model
    {
        SessionConfig cfg = base_cfg;
        cfg.moe.dense_stream = true;
        cfg.moe.dense_cache_mb = 0; // auto 2-layer look-ahead window
        cfg.moe.row_stream = true;
        GenCapture got = run_generation(cfg, prompt, n_predict);
        std::string diag;
        const bool ok = compare_captures(out_baseline, got, 1e-6f, diag);
        check(ok, "qwen3 auto 2-layer window + row-stream parity: " + diag);
        const DenseStreamStats & st = got.dense_stats;
        check(st.peak_resident_bytes <= st.budget_bytes && st.groups_evicted > 0 && st.prefetch_hits > 0,
              "qwen3 auto 2-layer window: peak <= auto budget, evictions > 0, prefetch_hits > 0");
    }

    std::printf("\n=== Test 2: Pure Dense (qwen3) — Extreme 1-Layer Starvation (--dense-cache-mb 1) ===\n");
    {
        SessionConfig cfg = base_cfg;
        cfg.moe.dense_stream = true;
        cfg.moe.dense_cache_mb = 1; // 1 MiB < single 2.625 MiB layer
        GenCapture got = run_generation(cfg, prompt, n_predict);
        std::string diag;
        const bool ok = compare_captures(out_baseline, got, 1e-6f, diag);
        check(ok, "qwen3 1 MiB starvation parity (zero deadlock): " + diag);
        const DenseStreamStats & st = got.dense_stats;
        std::printf("  [Starvation 1MiB] total=%.2f MiB, budget=%.2f MiB, peak=%.2f MiB, current=%.2f MiB, "
                    "loaded=%llu, evicted=%llu, prefetch_hits=%llu\n",
                    st.total_streamed_weight_bytes / 1048576.0, st.budget_bytes / 1048576.0,
                    st.peak_resident_bytes / 1048576.0, st.current_resident_bytes / 1048576.0,
                    (unsigned long long) st.groups_loaded, (unsigned long long) st.groups_evicted,
                    (unsigned long long) st.prefetch_hits);
        // 6 block layers (2.625 MiB each) exceed the 1 MiB cap so ZERO block-layer look-ahead
        // prefetches occur; only the 0.25 MiB output head fits inside the 1 MiB cap (<= 13 passes).
        check(st.prefetch_hits <= (uint64_t) (n_predict + 1),
              "1 MiB starvation: block-layer look-ahead cleanly disabled (prefetch_hits <= passes)");
        check(st.groups_evicted > 0 && st.current_resident_bytes <= st.budget_bytes,
              "1 MiB starvation: oversized layers evicted immediately (current_resident <= 1 MiB)");
        check(st.peak_resident_bytes <= (3ull << 20),
              "1 MiB starvation: peak resident held to 1 layer at a time (<= 3 MiB)");
    }
}

// ── Test 3: Pure Dense (qwen3) Partial GPU Offload on CUDA0 ─────────────────────────
void test_pure_dense_partial_gpu(const std::string & qwen3_path, uint64_t full_cpu_streamed_bytes) {
    std::printf("\n=== Test 3: Pure Dense (qwen3) — Partial GPU Offload (--gpu-layers 3) ===\n");
    if (!has_discrete_gpu()) {
        std::printf("  [SKIP] no discrete GPU backend available on this host\n");
        return;
    }
    const std::string prompt = "The quick brown fox jumps over the lazy dog and streams dense layers.";
    constexpr int n_predict = 12;

    SessionConfig gpu_base;
    gpu_base.model_path = qwen3_path;
    gpu_base.n_ctx = 256;
    gpu_base.n_batch = 128;
    gpu_base.n_threads = 4;
    gpu_base.n_gpu_layers = 3;
    gpu_base.device_use = DeviceUse::Auto;
    gpu_base.moe.enabled = false;
    gpu_base.moe.dense_stream = false;

    GenCapture base = run_generation(gpu_base, prompt, n_predict);
    check(base.ok && base.pieces.size() == (size_t) n_predict,
          "qwen3 GPU baseline (--gpu-layers 3) generated " + std::to_string(base.pieces.size()) + " tokens");

    SessionConfig gpu_stream = gpu_base;
    gpu_stream.moe.dense_stream = true;
    gpu_stream.moe.dense_cache_mb = 4;

    GenCapture got = run_generation(gpu_stream, prompt, n_predict);
    std::string diag;
    // Allow tiny FP32 GEMM order tolerance (1e-4) between prefill op_offload=true (baseline)
    // and op_offload=false (streamed) on mixed CPU+CUDA layers; tokens must match identically.
    const bool ok = compare_captures(base, got, 1e-4f, diag);
    check(ok, "qwen3 GPU partial offload (--gpu-layers 3) parity: " + diag);
    const DenseStreamStats & st = got.dense_stats;
    std::printf("  [GPU-3 + Host 4MiB] streamed=%.2f MiB (vs full CPU %.2f MiB), peak=%.2f MiB, "
                "loaded=%llu, evicted=%llu\n",
                st.total_streamed_weight_bytes / 1048576.0, full_cpu_streamed_bytes / 1048576.0,
                st.peak_resident_bytes / 1048576.0, (unsigned long long) st.groups_loaded,
                (unsigned long long) st.groups_evicted);
    check(st.total_streamed_weight_bytes > 0 && st.total_streamed_weight_bytes < full_cpu_streamed_bytes,
          "GPU partial offload: GPU layers stayed in VRAM, only host layers streamed");
    check(st.peak_resident_bytes <= (4ull << 20) && st.groups_evicted > 0,
          "GPU partial offload: host dense peak <= 4 MiB with real evictions");
}

// ── Test 4: MoE + Dense Combined ("Full Kimi Mode") across all 3 MoE architectures ──
void test_moe_plus_dense_combined(const std::string & name,
                                  const std::string & model_path,
                                  bool expect_evictions,
                                  GenCapture * out_baseline = nullptr) {
    std::printf("\n=== Test 4 (%s): MoE + Dense Combined (Full Kimi Mode) ===\n", name.c_str());
    const std::string prompt = "Hello world from Buffalo dense plus expert streaming!";
    constexpr int n_predict = 10;

    SessionConfig base_cfg;
    base_cfg.model_path = model_path;
    base_cfg.n_ctx = 256;
    base_cfg.n_batch = 128;
    base_cfg.n_threads = 4;
    base_cfg.device_use = DeviceUse::CpuOnly;
    base_cfg.moe.enabled = false;
    base_cfg.moe.dense_stream = false;

    GenCapture base = run_generation(base_cfg, prompt, n_predict);
    check(base.ok && base.pieces.size() == (size_t) n_predict,
          name + " resident baseline generated " + std::to_string(base.pieces.size()) + " tokens");
    if (out_baseline) *out_baseline = base;

    SessionConfig combined = base_cfg;
    combined.moe.enabled = true;
    combined.moe.cache_mb = 2;
    combined.moe.force_cache = true;
    combined.moe.dense_stream = true;
    combined.moe.dense_cache_mb = 1;
    // Tiny MoE fixtures use N_EMBD=128 (attn/shexp matrices are 32-128 KiB), so lower threshold
    // to 16 KiB to stream all attention, dense FFN, SSM, shared-expert, and embedding matrices
    // while keeping tiny 1-D norms, biases, and router gates resident.
    combined.moe.dense_min_bytes = 16384;

    GenCapture got = run_generation(combined, prompt, n_predict);
    std::string diag;
    const bool ok = compare_captures(base, got, 1e-6f, diag);
    check(ok, name + " MoE+Dense combined parity: " + diag);
    const DenseStreamStats & st = got.dense_stats;
    std::printf("  [%s MoE+Dense] streamed_dense=%.2f MiB, budget=%.2f MiB, peak=%.2f MiB, "
                "loaded=%llu, evicted=%llu, prefetch_hits=%llu\n",
                name.c_str(), st.total_streamed_weight_bytes / 1048576.0, st.budget_bytes / 1048576.0,
                st.peak_resident_bytes / 1048576.0, (unsigned long long) st.groups_loaded,
                (unsigned long long) st.groups_evicted, (unsigned long long) st.prefetch_hits);
    check(st.total_streamed_weight_bytes > 0 && st.groups_loaded > 0 && st.io_errors == 0,
          name + " MoE+Dense: dense weights actively streamed with 0 errors");
    if (expect_evictions) {
        check(st.groups_evicted > 0 && st.peak_resident_bytes <= st.budget_bytes,
              name + " MoE+Dense: real dense evictions occurred under 1 MiB budget");
    }
}

// ── Test 5: Zero-Disk HTTP Streaming (Pure Dense + MoE+Dense Combined) ──────────────
void test_http_zero_disk(const std::string & qwen3_path,
                         const GenCapture & qwen3_base,
                         const std::string & qwen3moe_path,
                         const GenCapture & qwen3moe_base,
                         const std::string & gemma4_path,
                         const GenCapture & gemma4_base) {
    std::printf("\n=== Test 5: Zero-Disk HTTP Streaming (Pure Dense & MoE+Dense Combined) ===\n");
    TrackingHttpServer srv;
    check(srv.add_file("/qwen3.gguf", qwen3_path), "HTTP server loaded tiny-dense-qwen3.gguf");
    check(srv.add_file("/qwen3moe.gguf", qwen3moe_path), "HTTP server loaded tiny-moe-qwen3moe.gguf");
    check(srv.add_file("/gemma4.gguf", gemma4_path), "HTTP server loaded tiny-moe-gemma4.gguf");
    check(srv.start(), "HTTP server listening on loopback");

    const std::filesystem::path spill_probe =
        std::filesystem::temp_directory_path() / "bmoe_m3_zero_disk_probe";
    std::error_code ec;
    std::filesystem::remove_all(spill_probe, ec);

    // 5a. Pure Dense (qwen3) over HTTP (--dense-stream --dense-cache-mb 6 --http-cache-mb 0)
    {
        const GgufMeta meta = read_gguf_meta(qwen3_path.c_str());
        std::vector<std::pair<uint64_t, uint64_t>> dense_ranges;
        uint64_t total_streamable = 0;
        for (const auto & kv : meta.offsets.size_by_name) {
            if (kv.second >= (uint64_t) MoeStreamConfig::dense_min_bytes_default) {
                const uint64_t off = meta.offsets.off_by_name.at(kv.first);
                dense_ranges.push_back({off, off + kv.second});
                total_streamable += kv.second;
            }
        }

        SessionConfig cfg;
        cfg.model_path = srv.url("/qwen3.gguf");
        cfg.n_ctx = 256;
        cfg.n_batch = 128;
        cfg.n_threads = 4;
        cfg.device_use = DeviceUse::CpuOnly;
        cfg.moe.enabled = false;
        cfg.moe.dense_stream = true;
        cfg.moe.dense_cache_mb = 6;
        cfg.moe.http_cache_dir = spill_probe.string();
        cfg.moe.http_cache_mb = 0; // strict 0-disk!

        srv.clear_logs();
        std::string err;
        auto sess = Session::open(cfg, err);
        check(sess != nullptr, "HTTP Pure Dense qwen3 opened cleanly" + (err.empty() ? "" : ": " + err));
        const auto boot_logs = srv.take_logs();
        const uint64_t boot_dense_bytes = overlap_bytes(boot_logs, "/qwen3.gguf", dense_ranges);
        std::printf("  [HTTP Pure Dense boot] streamable_dense=%.2f MiB, boot_dense_bytes=%llu B\n",
                    total_streamable / 1048576.0, (unsigned long long) boot_dense_bytes);
        check(boot_dense_bytes == 0, "HTTP Pure Dense boot: ZERO streamed dense bytes requested over HTTP");

        if (sess) {
            srv.clear_logs();
            GenerateRequest req;
            req.prompt = "The quick brown fox jumps over the lazy dog and streams dense layers.";
            req.n_predict = 12;
            req.clear_kv = true;
            req.render_text = false;
            req.n_probs = 4;

            GenCapture got;
            RunResult r = sess->generate(req, [&](const TokenMetrics & m) {
                got.pieces.push_back(m.piece);
                if (m.has_logprobs) {
                    got.chosen_logprobs.push_back(m.logprob);
                    got.top_logprobs.push_back(m.top_logprobs);
                }
            });
            got.ok = r.ok;
            got.error = r.error;
            got.text = r.generated_text;
            got.dense_stats = sess->dense_stream_stats();

            std::string diag;
            const bool ok = compare_captures(qwen3_base, got, 1e-6f, diag);
            check(ok, "HTTP Pure Dense qwen3 byte-identity vs local disk: " + diag);
            const auto gen_logs = srv.take_logs();
            const uint64_t gen_dense_bytes = overlap_bytes(gen_logs, "/qwen3.gguf", dense_ranges);
            std::printf("  [HTTP Pure Dense gen] http_dense_read=%.2f MiB, peak=%.2f MiB, evicted=%llu, "
                        "prefetch_hits=%llu\n",
                        gen_dense_bytes / 1048576.0, got.dense_stats.peak_resident_bytes / 1048576.0,
                        (unsigned long long) got.dense_stats.groups_evicted,
                        (unsigned long long) got.dense_stats.prefetch_hits);
            check(gen_dense_bytes > 0 && got.dense_stats.groups_evicted > 0 && got.dense_stats.prefetch_hits > 0,
                  "HTTP Pure Dense gen: streamed over HTTP with real evictions & prefetch hits");
            check(got.dense_stats.peak_resident_bytes <= (6ull << 20),
                  "HTTP Pure Dense gen: peak resident <= 6 MiB");
            check(!std::filesystem::exists(spill_probe), "HTTP Pure Dense: 0 local spill files/dirs created");
        }
    }

    // 5b. MoE + Dense Combined over HTTP (qwen3moe & gemma4): 0 expert bytes AND 0 streamed dense bytes at boot!
    auto run_http_combined = [&](const std::string & label,
                                 const std::string & route,
                                 const std::string & disk_path,
                                 const GenCapture & ref_cap) {
        const GgufMeta meta = read_gguf_meta(disk_path.c_str());
        const ExpertLayout lay = discover_expert_layout(meta.offsets.extents, meta.info.n_expert);
        std::vector<std::pair<uint64_t, uint64_t>> expert_ranges, dense_ranges;
        constexpr uint64_t min_bytes = 16384;
        for (const auto & kv : meta.offsets.size_by_name) {
            const uint64_t off = meta.offsets.off_by_name.at(kv.first);
            int il = -1;
            if (lay.slot_of(kv.first.c_str(), il) >= 0) {
                expert_ranges.push_back({off, off + kv.second});
            } else if (kv.second >= min_bytes) {
                dense_ranges.push_back({off, off + kv.second});
            }
        }

        SessionConfig cfg;
        cfg.model_path = srv.url(route);
        cfg.n_ctx = 256;
        cfg.n_batch = 128;
        cfg.n_threads = 4;
        cfg.device_use = DeviceUse::CpuOnly;
        cfg.moe.enabled = true;
        cfg.moe.cache_mb = 2;
        cfg.moe.force_cache = true;
        cfg.moe.dense_stream = true;
        cfg.moe.dense_cache_mb = 1;
        cfg.moe.dense_min_bytes = (int) min_bytes;
        cfg.moe.http_cache_dir = spill_probe.string();
        cfg.moe.http_cache_mb = 0; // strict 0-disk!

        srv.clear_logs();
        std::string err;
        auto sess = Session::open(cfg, err);
        check(sess != nullptr, "HTTP MoE+Dense " + label + " opened cleanly" + (err.empty() ? "" : ": " + err));
        const auto boot_logs = srv.take_logs();
        const uint64_t boot_exp = overlap_bytes(boot_logs, route, expert_ranges);
        const uint64_t boot_den = overlap_bytes(boot_logs, route, dense_ranges);
        std::printf("  [HTTP %s boot] boot_expert_bytes=%llu B, boot_streamed_dense_bytes=%llu B\n",
                    label.c_str(), (unsigned long long) boot_exp, (unsigned long long) boot_den);
        check(boot_exp == 0 && boot_den == 0,
              "HTTP MoE+Dense " + label + " boot: 0 routed expert bytes AND 0 streamed dense bytes requested");

        if (sess) {
            GenerateRequest req;
            req.prompt = "Hello world from Buffalo dense plus expert streaming!";
            req.n_predict = 10;
            req.clear_kv = true;
            req.render_text = false;
            req.n_probs = 4;

            GenCapture got;
            RunResult r = sess->generate(req, [&](const TokenMetrics & m) {
                got.pieces.push_back(m.piece);
                if (m.has_logprobs) {
                    got.chosen_logprobs.push_back(m.logprob);
                    got.top_logprobs.push_back(m.top_logprobs);
                }
            });
            got.ok = r.ok;
            got.error = r.error;
            got.text = r.generated_text;
            got.dense_stats = sess->dense_stream_stats();

            std::string diag;
            const bool ok = compare_captures(ref_cap, got, 1e-6f, diag);
            check(ok, "HTTP MoE+Dense " + label + " byte-identity vs local disk: " + diag);
            check(!std::filesystem::exists(spill_probe),
                  "HTTP MoE+Dense " + label + ": 0 local spill files/dirs created");
        }
    };

    run_http_combined("qwen3moe", "/qwen3moe.gguf", qwen3moe_path, qwen3moe_base);
    run_http_combined("gemma4", "/gemma4.gguf", gemma4_path, gemma4_base);
    srv.stop();
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
                     "usage: %s <tiny-dense-qwen3.gguf> <tiny-moe-qwen3moe.gguf> "
                     "<tiny-moe-gemma4.gguf> <tiny-moe-nemotron_h_moe.gguf>\n",
                     argv[0]);
        return 2;
    }

    const std::string qwen3_path = argv[1];
    const std::string qwen3moe_path = argv[2];
    const std::string gemma4_path = argv[3];
    const std::string nemo_path = argv[4];

    test_virtual_host_page_isolation();

    GenCapture qwen3_base;
    test_pure_dense_local(qwen3_path, qwen3_base);

    // Measure full-CPU streamed dense bytes once from qwen3 metadata for Test 3 comparison.
    uint64_t full_cpu_streamed = 0;
    {
        const GgufMeta m = read_gguf_meta(qwen3_path.c_str());
        for (const auto & kv : m.offsets.size_by_name)
            if (kv.second >= (uint64_t) MoeStreamConfig::dense_min_bytes_default) full_cpu_streamed += kv.second;
    }
    test_pure_dense_partial_gpu(qwen3_path, full_cpu_streamed);

    GenCapture qwen3moe_base, gemma4_base;
    test_moe_plus_dense_combined("qwen3moe", qwen3moe_path, /*expect_evictions=*/false, &qwen3moe_base);
    test_moe_plus_dense_combined("gemma4", gemma4_path, /*expect_evictions=*/true, &gemma4_base);
    test_moe_plus_dense_combined("nemotron_h_moe", nemo_path, /*expect_evictions=*/false, nullptr);

    test_http_zero_disk(qwen3_path, qwen3_base, qwen3moe_path, qwen3moe_base, gemma4_path, gemma4_base);

    if (g_failures == 0) {
        std::printf("\nALL M3 DENSE STREAM TESTS PASSED\n");
        return 0;
    }
    std::printf("\n%d M3 DENSE STREAM TEST(S) FAILED\n", g_failures);
    return 1;
}
