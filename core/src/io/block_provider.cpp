#include "block_provider.h"
#include "bmoe/log.h"
#include "file_reader.h"
#include "http_range_provider.h"
#include "platform_io.h"

#ifdef BMOE_HAS_IO_URING
#include "iouring_provider.h"
#include <cerrno>
#include <cstring>
#include <mutex>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>

namespace bmoe {

using clock_t_ = std::chrono::steady_clock;

bool BlockProvider::read_batch(BlockReadReq * reqs,
                               size_t count,
                               const std::function<void(size_t idx, bool ok)> & on_complete,
                               const std::function<void()> & on_submitted) {
    if (on_submitted) on_submitted();
    if (count == 0) return true;
    if (!reqs) return false;
    bool all_ok = true;
    for (size_t i = 0; i < count; ++i) {
        BlockReadReq & r = reqs[i];
        const auto t0 = clock_t_::now();
        const long long got = read(0, r.shard, r.dst, r.off, r.nbytes);
        r.latency_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t_::now() - t0).count();
        r.bytes_moved = got;
        const bool ok = got >= 0;
        if (!ok) all_ok = false;
        if (on_complete) on_complete(i, ok);
    }
    return all_ok;
}

PosixBlockProvider::PosixBlockProvider() = default;

PosixBlockProvider::~PosixBlockProvider() {
    close();
}

void PosixBlockProvider::refresh_geometry() {
    const size_t page = pio::vm_page();
    direct_ = !readers_.empty();
    align_ = pio::direct_align_fallback;
    align_reported_ = !readers_.empty();
    buffer_align_ = page;
    for (const auto & r : readers_) {
        direct_ = direct_ && r->direct();
        align_ = std::max(align_, r->align());
        align_reported_ = align_reported_ && r->align_reported();
        buffer_align_ = std::max(buffer_align_, r->buffer_align());
    }
}

bool PosixBlockProvider::open(const std::vector<std::string> & paths, int lanes, bool direct, size_t max_read) {
    if (is_open() || paths.empty()) return false;
    readers_.reserve(paths.size());
    for (const std::string & p : paths) {
        auto r = std::make_unique<FileReader>();
        if (!r->open(p, lanes, direct, max_read)) {
            close();
            return false;
        }
        readers_.push_back(std::move(r));
    }
    refresh_geometry();
    return true;
}

void PosixBlockProvider::close() {
    readers_.clear();
    direct_ = false;
    align_ = pio::direct_align_fallback;
    align_reported_ = false;
    buffer_align_ = pio::vm_page();
}

bool PosixBlockProvider::reopen() {
    if (!is_open()) return false;
    bool ok = true;
    for (auto & r : readers_)
        ok = r->reopen() && ok;
    if (ok) refresh_geometry();
    return ok;
}

uint64_t PosixBlockProvider::shard_size(int shard) const {
    if (shard < 0 || shard >= (int) readers_.size()) return 0;
    return readers_[(size_t) shard]->file_size();
}

long long PosixBlockProvider::read(int lane, int shard, void * dst, uint64_t off, uint64_t nbytes) {
    if (shard < 0 || shard >= (int) readers_.size()) return -1;
    return readers_[(size_t) shard]->read(lane, dst, off, nbytes);
}

long long PosixBlockProvider::read_bytes() const {
    long long total = 0;
    for (const auto & r : readers_)
        total += r->read_bytes();
    return total;
}

long long PosixBlockProvider::io_ns() const {
    long long total = 0;
    for (const auto & r : readers_)
        total += r->syscall_ns();
    return total;
}

std::unique_ptr<BlockProvider> open_local_block_provider(const std::vector<std::string> & paths,
                                                         int lanes,
                                                         bool direct,
                                                         size_t max_read,
                                                         IoBackend pref) {
    if (paths.empty()) return nullptr;

#ifdef BMOE_HAS_IO_URING
    if (pref == IoBackend::Auto || pref == IoBackend::IoUring) {
        auto ring = std::make_unique<IoUringBlockProvider>();
        int ring_errno = 0;
        if (ring->open(paths, lanes, direct, max_read, &ring_errno)) return ring;
        // Only fall back when the kernel/sandbox refused io_uring_setup; do not mask file errors
        // (e.g. ENOENT) by silently retrying open unless ring setup itself was refused.
        if (ring_errno == ENOSYS || ring_errno == EPERM || ring_errno == EACCES || ring_errno == ENOMEM) {
            static std::once_flag warn_once;
            std::call_once(warn_once, [ring_errno]() {
                log_printf(LogLevel::Warn,
                           "io_uring unavailable (%s, errno=%d) — falling back to posix block provider",
                           std::strerror(ring_errno), ring_errno);
            });
        } else {
            return nullptr;
        }
    }
#else
    if (pref == IoBackend::IoUring) {
        static std::once_flag warn_once;
        std::call_once(warn_once, []() {
            log_printf(LogLevel::Warn,
                       "io_uring is not compiled into this build — falling back to posix block provider");
        });
    }
#endif

    auto posix = std::make_unique<PosixBlockProvider>();
    if (!posix->open(paths, lanes, direct, max_read)) return nullptr;
    return posix;
}

static bool starts_with_ci(const std::string & s, const char * prefix) {
    for (size_t i = 0; prefix[i] != '\0'; ++i) {
        if (i >= s.size()) return false;
        if (std::tolower((unsigned char) s[i]) != std::tolower((unsigned char) prefix[i])) return false;
    }
    return true;
}

bool is_http_url(const std::string & path) {
    return starts_with_ci(path, "http://");
}

bool is_https_url(const std::string & path) {
    return starts_with_ci(path, "https://");
}

bool is_remote_url(const std::string & path) {
    return is_http_url(path) || is_https_url(path);
}

std::vector<std::string> expand_split_urls(const std::string & url) {
    // Strip optional ?query / #fragment before inspecting the split suffix so presigned URLs work.
    size_t cut = url.find_first_of("?#");
    const std::string base = cut == std::string::npos ? url : url.substr(0, cut);
    const std::string query =
        (cut != std::string::npos && url[cut] == '?') ? url.substr(cut, url.find('#', cut) - cut) : std::string();

    // <prefix>-DDDDD-of-DDDDD.gguf -> 19 chars after the separating dash
    static constexpr size_t tail_len = 5 + 4 + 5 + 5;
    if (base.size() < tail_len + 1) return {url};
    const size_t tail = base.size() - tail_len;
    auto five_digits = [&](size_t at) {
        for (size_t i = at; i < at + 5; ++i)
            if (base[i] < '0' || base[i] > '9') return false;
        return true;
    };
    if (base[tail - 1] != '-' || !five_digits(tail) || base.compare(tail + 5, 4, "-of-") != 0 ||
        !five_digits(tail + 9) || base.compare(tail + 14, 5, ".gguf") != 0)
        return {url};

    const int no = std::stoi(base.substr(tail, 5));
    const int count = std::stoi(base.substr(tail + 9, 5));
    if (count <= 1) return {url};
    if (no != 1) {
        log_printf(LogLevel::Error, "%s is shard %d of a split model — pass the first shard (-00001-of-)", url.c_str(),
                   no);
        return {};
    }

    const std::string prefix = base.substr(0, tail - 1);
    std::vector<std::string> out;
    out.reserve((size_t) count);
    for (int i = 1; i <= count; ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "-%05d-of-%05d.gguf", i, count);
        out.push_back(prefix + buf + query);
    }
    return out;
}

std::unique_ptr<BlockProvider> open_block_provider(const std::vector<std::string> & paths,
                                                   int lanes,
                                                   bool direct,
                                                   size_t max_read,
                                                   IoBackend pref,
                                                   const HttpProviderOptions & http_opt,
                                                   std::string * error) {
    if (paths.empty()) {
        if (error) *error = "no model paths provided";
        return nullptr;
    }
    if (is_remote_url(paths[0])) {
        auto http = std::make_unique<HttpRangeProvider>();
        const int eff_lanes = http_opt.lanes > 0 ? http_opt.lanes : std::max(1, lanes);
        std::string err;
        if (!http->open(paths, eff_lanes, http_opt, &err)) {
            if (error) *error = err;
            log_printf(LogLevel::Error, "http block provider failed: %s", err.c_str());
            return nullptr;
        }
        return http;
    }
    auto local = open_local_block_provider(paths, lanes, direct, max_read, pref);
    if (!local && error) *error = "could not open local model shard(s): " + paths[0];
    return local;
}

} // namespace bmoe
