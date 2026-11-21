#include "http_range_provider.h"
#include "bmoe/log.h"
#include "platform_io.h"

#include "cpp-httplib/httplib.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <utility>

namespace bmoe {

namespace {

using clock_t_ = std::chrono::steady_clock;

constexpr int k_max_redirects = 8;
constexpr int k_max_retries = 3;
constexpr int k_connect_timeout_s = 10;
constexpr int k_read_timeout_s = 30;
constexpr int k_write_timeout_s = 10;

struct ParsedUrl {
    std::string scheme;           // "http" or "https" (lowercase)
    std::string scheme_host_port; // e.g. "http://127.0.0.1:8080"
    std::string path_query;       // e.g. "/model.gguf?sig=1"
};

bool parse_url(const std::string & raw, ParsedUrl & out, std::string & err) {
    const size_t sep = raw.find("://");
    if (sep == std::string::npos) {
        err = "invalid URL (missing scheme): " + raw;
        return false;
    }
    std::string scheme = raw.substr(0, sep);
    for (char & c : scheme)
        c = (char) std::tolower((unsigned char) c);

    if (scheme == "https") {
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
        err = "HTTPS URL '" + raw +
              "' requires TLS support, which is disabled in this build (LLAMA_OPENSSL=OFF); "
              "serve the GGUF over plain http:// (e.g. LAN/localhost HTTP server) or rebuild with TLS enabled";
        return false;
#endif
    } else if (scheme != "http") {
        err = "unsupported URL scheme '" + scheme + "://' (expected http://): " + raw;
        return false;
    }

    // Strip any #fragment before splitting authority and path+query.
    const size_t frag = raw.find('#', sep + 3);
    const std::string no_frag = frag == std::string::npos ? raw : raw.substr(0, frag);
    const size_t auth_start = sep + 3;
    const size_t path_pos = no_frag.find_first_of("/?", auth_start);
    const std::string authority =
        path_pos == std::string::npos ? no_frag.substr(auth_start) : no_frag.substr(auth_start, path_pos - auth_start);

    if (authority.empty()) {
        err = "invalid URL (empty host): " + raw;
        return false;
    }

    // Validate host[:port], supporting bracketed IPv6 literals ([::1]:8080).
    std::string host;
    std::string port_str;
    if (authority.front() == '[') {
        const size_t rb = authority.find(']');
        if (rb == std::string::npos) {
            err = "invalid URL (unterminated IPv6 host): " + raw;
            return false;
        }
        host = authority.substr(0, rb + 1);
        if (rb + 1 < authority.size()) {
            if (authority[rb + 1] != ':') {
                err = "invalid URL authority: " + raw;
                return false;
            }
            port_str = authority.substr(rb + 2);
        }
    } else {
        const size_t colon = authority.rfind(':');
        if (colon != std::string::npos) {
            host = authority.substr(0, colon);
            port_str = authority.substr(colon + 1);
        } else {
            host = authority;
        }
    }
    if (host.empty()) {
        err = "invalid URL (empty host): " + raw;
        return false;
    }
    if (!port_str.empty()) {
        char * end = nullptr;
        const long port = std::strtol(port_str.c_str(), &end, 10);
        if (!end || *end != '\0' || port <= 0 || port > 65535) {
            err = "invalid URL port '" + port_str + "': " + raw;
            return false;
        }
    }

    out.scheme = scheme;
    out.scheme_host_port = scheme + "://" + authority;
    if (path_pos == std::string::npos) {
        out.path_query = "/";
    } else if (no_frag[path_pos] == '?') {
        out.path_query = "/" + no_frag.substr(path_pos);
    } else {
        out.path_query = no_frag.substr(path_pos);
    }
    return true;
}

std::string resolve_redirect_url(const ParsedUrl & cur, const std::string & location) {
    if (location.empty()) return "";
    if (is_remote_url(location)) return location;
    if (location.size() >= 2 && location[0] == '/' && location[1] == '/') return cur.scheme + ":" + location;
    if (location[0] == '/') return cur.scheme_host_port + location;
    // Relative path against current directory.
    const size_t q = cur.path_query.find('?');
    const std::string cur_path = q == std::string::npos ? cur.path_query : cur.path_query.substr(0, q);
    const size_t slash = cur_path.rfind('/');
    const std::string dir = slash == std::string::npos ? "/" : cur_path.substr(0, slash + 1);
    return cur.scheme_host_port + dir + location;
}

bool is_redirect_status(int status) {
    return status == 301 || status == 302 || status == 303 || status == 307 || status == 308;
}

bool is_retryable_status(int status) {
    return status == 429 || status == 500 || status == 502 || status == 503 || status == 504;
}

bool parse_u64_strict(const std::string & s, uint64_t & out) {
    if (s.empty()) return false;
    size_t i = 0;
    while (i < s.size() && std::isspace((unsigned char) s[i]))
        ++i;
    if (i >= s.size() || !std::isdigit((unsigned char) s[i])) return false;
    uint64_t val = 0;
    for (; i < s.size(); ++i) {
        if (std::isspace((unsigned char) s[i])) break;
        if (!std::isdigit((unsigned char) s[i])) return false;
        const uint64_t d = (uint64_t) (s[i] - '0');
        if (val > (UINT64_MAX - d) / 10) return false;
        val = val * 10 + d;
    }
    while (i < s.size()) {
        if (!std::isspace((unsigned char) s[i])) return false;
        ++i;
    }
    out = val;
    return true;
}

// Parse `Content-Range: bytes <start>-<end>/<total>` -> returns true and sets `total` (> 0).
bool parse_content_range_total(const std::string & cr, uint64_t & total) {
    // Expect "bytes <start>-<end>/<total>" (case-insensitive "bytes").
    size_t p = 0;
    while (p < cr.size() && std::isspace((unsigned char) cr[p]))
        ++p;
    if (cr.size() - p < 6) return false;
    std::string unit = cr.substr(p, 5);
    for (char & c : unit)
        c = (char) std::tolower((unsigned char) c);
    if (unit != "bytes" || !std::isspace((unsigned char) cr[p + 5])) return false;
    const size_t slash = cr.find('/', p + 6);
    if (slash == std::string::npos || slash + 1 >= cr.size()) return false;
    const std::string total_str = cr.substr(slash + 1);
    if (total_str == "*") return false;
    return parse_u64_strict(total_str, total) && total > 0;
}

std::string to_lower_copy(std::string s) {
    for (char & c : s)
        c = (char) std::tolower((unsigned char) c);
    return s;
}

uint64_t fnv1a_64(const void * data, size_t len) {
    uint64_t h = 1469598103934665603ull;
    const auto * p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

std::unique_ptr<httplib::Client> make_http_client(const std::string & scheme_host_port) {
    auto cli = std::make_unique<httplib::Client>(scheme_host_port);
    cli->set_keep_alive(true);
    cli->set_decompress(false);
    cli->set_follow_location(false);
    cli->set_connection_timeout(k_connect_timeout_s, 0);
    cli->set_read_timeout(k_read_timeout_s, 0);
    cli->set_write_timeout(k_write_timeout_s, 0);
    return cli;
}

} // namespace

HttpRangeProvider::HttpRangeProvider() : buffer_align_(pio::vm_page()) {}

HttpRangeProvider::~HttpRangeProvider() {
    close();
}

httplib::Client * HttpRangeProvider::lane_client(Lane & lane, const std::string & scheme_host_port, bool reset) {
    if (reset) lane.clients.erase(scheme_host_port);
    auto it = lane.clients.find(scheme_host_port);
    if (it == lane.clients.end() || !it->second) {
        auto cli = make_http_client(scheme_host_port);
        httplib::Client * raw = cli.get();
        lane.clients[scheme_host_port] = std::move(cli);
        return raw;
    }
    return it->second.get();
}

bool HttpRangeProvider::open(const std::vector<std::string> & urls,
                             int lanes,
                             const HttpProviderOptions & opt,
                             std::string * error) {
    if (open_) close();
    auto fail = [&](std::string msg) {
        close();
        if (error) *error = std::move(msg);
        return false;
    };
    if (urls.empty()) return fail("no shard URLs provided");

    lanes_ = std::max(1, std::min(MoeStreamConfig::http_threads_max, lanes));
    buffer_align_ = pio::vm_page();
    cache_dir_ = opt.cache_bytes > 0 ? opt.cache_dir : std::string();
    cache_bytes_ = cache_dir_.empty() ? 0 : opt.cache_bytes;
    shards_.clear();
    shards_.reserve(urls.size());

    // Probe every shard URL: follow redirects (301/302/303/307/308), check HEAD first, and fall
    // back to GET Range: bytes=0-0 (parsing Content-Range and verifying HTTP 206 Partial Content).
    for (size_t i = 0; i < urls.size(); ++i) {
        std::string cur_url = urls[i];
        ParsedUrl parsed;
        std::string perr;
        if (!parse_url(cur_url, parsed, perr)) return fail(perr);

        uint64_t size = 0;
        bool confirmed_206 = false;
        bool need_range_probe = true;

        // 1. HEAD with manual redirect following.
        for (int hop = 0; hop <= k_max_redirects; ++hop) {
            auto cli = make_http_client(parsed.scheme_host_port);
            auto res = cli->Head(parsed.path_query);
            if (!res) {
                // Transport error on HEAD: fail fast with the transport error.
                return fail("HTTP HEAD failed for " + cur_url + ": " + httplib::to_string(res.error()));
            }
            if (is_redirect_status(res->status)) {
                if (hop == k_max_redirects) return fail("too many HTTP redirects for " + urls[i]);
                const std::string loc = res->get_header_value("Location");
                cur_url = resolve_redirect_url(parsed, loc);
                if (cur_url.empty()) return fail("HTTP redirect missing Location header for " + urls[i]);
                if (!parse_url(cur_url, parsed, perr)) return fail(perr);
                continue;
            }
            if (res->status == 200) {
                const std::string accept_ranges = to_lower_copy(res->get_header_value("Accept-Ranges"));
                if (accept_ranges == "none") {
                    return fail("HTTP server rejected byte ranges (Accept-Ranges: none) for " + cur_url);
                }
                uint64_t cl = 0;
                if (res->has_header("Content-Length") && parse_u64_strict(res->get_header_value("Content-Length"), cl) &&
                    cl > 0) {
                    size = cl;
                }
                if (size > 0 && accept_ranges.find("bytes") != std::string::npos) {
                    need_range_probe = false;
                    confirmed_206 = true;
                }
                break;
            }
            if (res->status == 405 || res->status == 501) {
                // HEAD not supported; fall through to GET Range: bytes=0-0.
                break;
            }
            return fail("HTTP HEAD returned status " + std::to_string(res->status) + " for " + cur_url);
        }

        // 2. Fallback (or confirmation when HEAD omitted Accept-Ranges: bytes / Content-Length):
        // GET Range: bytes=0-0 and require HTTP 206 Partial Content + Content-Range.
        if (need_range_probe) {
            for (int hop = 0; hop <= k_max_redirects; ++hop) {
                auto cli = make_http_client(parsed.scheme_host_port);
                httplib::Headers hdrs = {{"Range", "bytes=0-0"}};
                int seen_status = 0;
                std::string seen_location;
                std::string seen_content_range;
                size_t body_bytes = 0;

                auto res = cli->Get(
                    parsed.path_query, hdrs,
                    [&](const httplib::Response & r) -> bool {
                        seen_status = r.status;
                        seen_location = r.get_header_value("Location");
                        seen_content_range = r.get_header_value("Content-Range");
                        // Abort immediately on non-206 so a server ignoring Range never streams GBs.
                        return r.status == 206;
                    },
                    [&](const char *, size_t len) -> bool {
                        body_bytes += len;
                        return body_bytes <= 1; // bytes=0-0 must deliver at most 1 byte
                    });

                if (seen_status == 0) {
                    return fail("HTTP Range probe failed for " + cur_url + ": " + httplib::to_string(res.error()));
                }
                if (is_redirect_status(seen_status)) {
                    if (hop == k_max_redirects) return fail("too many HTTP redirects for " + urls[i]);
                    cur_url = resolve_redirect_url(parsed, seen_location);
                    if (cur_url.empty()) return fail("HTTP redirect missing Location header for " + urls[i]);
                    if (!parse_url(cur_url, parsed, perr)) return fail(perr);
                    continue;
                }
                if (seen_status == 200) {
                    return fail("HTTP server does not support Range requests (returned 200 OK instead of 206 Partial "
                                "Content) for " +
                                cur_url);
                }
                if (seen_status != 206) {
                    return fail("HTTP Range probe returned status " + std::to_string(seen_status) + " for " + cur_url);
                }
                uint64_t total = 0;
                if (!parse_content_range_total(seen_content_range, total)) {
                    if (size == 0) {
                        return fail("HTTP 206 response missing valid Content-Range total for " + cur_url + " (got '" +
                                    seen_content_range + "')");
                    }
                } else {
                    size = total;
                }
                confirmed_206 = true;
                break;
            }
        }

        if (!confirmed_206 || size == 0) {
            return fail("could not determine remote shard size and HTTP Range support for " + urls[i]);
        }

        Shard s;
        s.orig_url = urls[i];
        s.scheme_host_port = parsed.scheme_host_port;
        s.path_query = parsed.path_query;
        s.size = size;
        const std::string key_seed = s.scheme_host_port + s.path_query + "#" + std::to_string(s.size);
        s.url_hash = fnv1a_64(key_seed.data(), key_seed.size());
        shards_.push_back(std::move(s));
    }

    // Initialize optional local disk spill cache ONLY when explicitly requested.
    if (!cache_dir_.empty() && cache_bytes_ > 0) {
        std::string cerr;
        if (!init_spill_cache(&cerr)) return fail(cerr);
    }

    lane_pool_.clear();
    lane_pool_.reserve((size_t) lanes_);
    for (int l = 0; l < lanes_; ++l)
        lane_pool_.push_back(std::make_unique<Lane>());

    read_bytes_.store(0, std::memory_order_relaxed);
    io_ns_.store(0, std::memory_order_relaxed);
    spill_hits_.store(0, std::memory_order_relaxed);
    spill_misses_.store(0, std::memory_order_relaxed);

    open_ = true;
    start_workers();
    return true;
}

void HttpRangeProvider::close() {
    stop_workers();
    open_ = false;
    lane_pool_.clear();
    shards_.clear();
    {
        std::lock_guard<std::mutex> lk(cache_mtx_);
        spill_lru_.clear();
        spill_map_.clear();
        spill_used_bytes_ = 0;
    }
    cache_dir_.clear();
    cache_bytes_ = 0;
}

uint64_t HttpRangeProvider::shard_size(int shard) const {
    if (shard < 0 || shard >= (int) shards_.size()) return 0;
    return shards_[(size_t) shard].size;
}

bool HttpRangeProvider::init_spill_cache(std::string * error) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(cache_dir_, ec);
    if (ec) {
        if (error) *error = "cannot create HTTP cache directory '" + cache_dir_ + "': " + ec.message();
        return false;
    }

    struct Found {
        std::string key;
        std::string path;
        uint64_t size = 0;
        fs::file_time_type mtime{};
    };
    std::vector<Found> existing;
    for (const auto & entry : fs::directory_iterator(cache_dir_, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() < 5 || name.compare(name.size() - 4, 4, ".bin") != 0) {
            // Remove leftover partial temp files from an interrupted run.
            if (name.size() >= 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) fs::remove(entry.path(), ec);
            continue;
        }
        Found f;
        f.key = name.substr(0, name.size() - 4);
        f.path = entry.path().string();
        f.size = (uint64_t) entry.file_size(ec);
        f.mtime = entry.last_write_time(ec);
        if (!ec) existing.push_back(std::move(f));
    }
    std::sort(existing.begin(), existing.end(), [](const Found & a, const Found & b) { return a.mtime < b.mtime; });

    std::lock_guard<std::mutex> lk(cache_mtx_);
    spill_lru_.clear();
    spill_map_.clear();
    spill_used_bytes_ = 0;
    for (auto & f : existing) {
        spill_lru_.push_back({f.key, f.path, f.size});
        spill_map_[f.key] = std::prev(spill_lru_.end());
        spill_used_bytes_ += f.size;
    }
    spill_evict_locked(0);
    return true;
}

void HttpRangeProvider::spill_evict_locked(uint64_t need_bytes) {
    namespace fs = std::filesystem;
    while (!spill_lru_.empty() && spill_used_bytes_ + need_bytes > cache_bytes_) {
        const SpillEntry victim = spill_lru_.front();
        spill_lru_.pop_front();
        spill_map_.erase(victim.key);
        if (spill_used_bytes_ >= victim.size)
            spill_used_bytes_ -= victim.size;
        else
            spill_used_bytes_ = 0;
        std::error_code ec;
        fs::remove(victim.path, ec);
    }
}

bool HttpRangeProvider::spill_read_chunk(const std::string & key, uint64_t inner_off, void * dst, uint64_t len) {
    std::string path;
    {
        std::lock_guard<std::mutex> lk(cache_mtx_);
        auto it = spill_map_.find(key);
        if (it == spill_map_.end()) return false;
        if (inner_off + len > it->second->size) return false;
        // Touch LRU before releasing lock (move to back).
        spill_lru_.splice(spill_lru_.end(), spill_lru_, it->second);
        path = it->second->path;
    }

    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        // File vanished externally; prune stale index entry.
        std::lock_guard<std::mutex> lk(cache_mtx_);
        auto it = spill_map_.find(key);
        if (it != spill_map_.end()) {
            if (spill_used_bytes_ >= it->second->size) spill_used_bytes_ -= it->second->size;
            spill_lru_.erase(it->second);
            spill_map_.erase(it);
        }
        return false;
    }
    bool ok = true;
    if (inner_off > 0) {
#if defined(_WIN32)
        ok = _fseeki64(f, (long long) inner_off, SEEK_SET) == 0;
#else
        ok = fseeko(f, (off_t) inner_off, SEEK_SET) == 0;
#endif
    }
    if (ok && len > 0) {
        ok = std::fread(dst, 1, (size_t) len, f) == (size_t) len;
    }
    std::fclose(f);
    return ok;
}

void HttpRangeProvider::spill_write_chunk(const std::string & key, const void * data, uint64_t len) {
    if (len == 0 || len > cache_bytes_) return;
    namespace fs = std::filesystem;

    // Make room before writing so disk usage never grows past cache_bytes_ at rest.
    {
        std::lock_guard<std::mutex> lk(cache_mtx_);
        if (spill_map_.find(key) != spill_map_.end()) return;
        spill_evict_locked(len);
    }

    const uint64_t seq = spill_tmp_seq_.fetch_add(1, std::memory_order_relaxed);
    const fs::path final_path = fs::path(cache_dir_) / (key + ".bin");
    const fs::path tmp_path = fs::path(cache_dir_) / (key + "." + std::to_string(seq) + ".tmp");

    FILE * f = std::fopen(tmp_path.string().c_str(), "wb");
    if (!f) return;
    const bool wrote = std::fwrite(data, 1, (size_t) len, f) == (size_t) len && std::fflush(f) == 0;
    std::fclose(f);
    if (!wrote) {
        std::error_code ec;
        fs::remove(tmp_path, ec);
        return;
    }

    std::lock_guard<std::mutex> lk(cache_mtx_);
    if (spill_map_.find(key) != spill_map_.end()) {
        std::error_code ec;
        fs::remove(tmp_path, ec);
        return;
    }
    spill_evict_locked(len);
    std::error_code ec;
    fs::rename(tmp_path, final_path, ec);
    if (ec) {
        fs::remove(tmp_path, ec);
        return;
    }
    spill_lru_.push_back({key, final_path.string(), len});
    spill_map_[key] = std::prev(spill_lru_.end());
    spill_used_bytes_ += len;
    spill_evict_locked(0);
}

long long HttpRangeProvider::fetch_http_exact(int lane_id, int shard_idx, void * dst, uint64_t off, uint64_t nbytes) {
    if (nbytes == 0) return 0;
    if (!open_ || !dst || shard_idx < 0 || shard_idx >= (int) shards_.size()) return -1;
    const Shard & sh = shards_[(size_t) shard_idx];
    if (off >= sh.size || nbytes > sh.size - off) return -1;

    const int eff_lane = (lane_id >= 0 && lane_id < lanes_) ? lane_id : 0;
    Lane & lane = *lane_pool_[(size_t) eff_lane];
    std::lock_guard<std::mutex> lk(lane.mtx);

    const uint64_t end_inclusive = off + nbytes - 1;
    char range_buf[96];
    std::snprintf(range_buf, sizeof(range_buf), "bytes=%llu-%llu", (unsigned long long) off,
                  (unsigned long long) end_inclusive);
    const httplib::Headers hdrs = {{"Range", range_buf}};

    std::string cur_host = sh.scheme_host_port;
    std::string cur_path = sh.path_query;

    for (int attempt = 0; attempt <= k_max_retries; ++attempt) {
        if (attempt > 0) {
            const int backoff_ms = 20 * (1 << (attempt - 1)); // 20ms, 40ms, 80ms
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
        }

        bool reset_client = attempt > 0;
        bool retry_needed = false;
        int last_status = 0;
        std::string last_err;

        // Allow up to 4 redirects during a data request in case a signed URL rotates.
        for (int hop = 0; hop <= 4; ++hop) {
            httplib::Client * cli = lane_client(lane, cur_host, reset_client);
            reset_client = false;

            int seen_status = 0;
            std::string seen_location;
            uint64_t written = 0;
            bool overflow = false;

            auto res = cli->Get(
                cur_path, hdrs,
                [&](const httplib::Response & r) -> bool {
                    seen_status = r.status;
                    seen_location = r.get_header_value("Location");
                    return r.status == 206;
                },
                [&](const char * data, size_t len) -> bool {
                    if (written + len > nbytes) {
                        overflow = true;
                        return false;
                    }
                    std::memcpy((char *) dst + written, data, len);
                    written += len;
                    return true;
                });

            if (seen_status == 206 && res && !overflow && written == nbytes) {
                read_bytes_.fetch_add(nbytes, std::memory_order_relaxed);
                return (long long) nbytes;
            }

            if (seen_status != 0 && is_redirect_status(seen_status)) {
                ParsedUrl cur;
                cur.scheme = "http";
                cur.scheme_host_port = cur_host;
                cur.path_query = cur_path;
                const std::string next = resolve_redirect_url(cur, seen_location);
                ParsedUrl np;
                std::string perr;
                if (next.empty() || !parse_url(next, np, perr)) {
                    log_printf(LogLevel::Error, "http_range: invalid redirect for shard %d: %s", shard_idx,
                               perr.c_str());
                    return -1;
                }
                cur_host = np.scheme_host_port;
                cur_path = np.path_query;
                continue;
            }

            last_status = seen_status;
            if (seen_status == 0) {
                last_err = httplib::to_string(res.error());
                retry_needed = true;
            } else if (seen_status == 206 && written < nbytes) {
                // Truncated response body (connection dropped mid-transfer).
                last_err = "short HTTP 206 body (" + std::to_string(written) + "/" + std::to_string(nbytes) + ")";
                retry_needed = true;
            } else if (is_retryable_status(seen_status)) {
                last_err = "HTTP status " + std::to_string(seen_status);
                retry_needed = true;
            } else {
                // Non-retryable HTTP status (e.g. 200 ignoring Range, 404, 416).
                log_printf(LogLevel::Error, "http_range: shard %d [%llu..%llu] failed with HTTP status %d", shard_idx,
                           (unsigned long long) off, (unsigned long long) end_inclusive, seen_status);
                return -1;
            }
            break;
        }

        if (!retry_needed) break;
        if (attempt == k_max_retries) {
            log_printf(LogLevel::Error, "http_range: shard %d [%llu..%llu] failed after %d retries (%s, status=%d)",
                       shard_idx, (unsigned long long) off, (unsigned long long) end_inclusive, k_max_retries,
                       last_err.c_str(), last_status);
        }
    }
    return -1;
}

long long HttpRangeProvider::read_with_spill(int lane_id, int shard_idx, void * dst, uint64_t off, uint64_t nbytes) {
    const Shard & sh = shards_[(size_t) shard_idx];
    const uint64_t first_chunk = off / k_spill_chunk_bytes;
    const uint64_t last_chunk = (off + nbytes - 1) / k_spill_chunk_bytes;
    std::vector<uint8_t> chunk_buf;

    for (uint64_t c = first_chunk; c <= last_chunk; ++c) {
        const uint64_t c_start = c * k_spill_chunk_bytes;
        const uint64_t c_len = std::min<uint64_t>(k_spill_chunk_bytes, sh.size - c_start);
        const uint64_t sub_start = std::max(off, c_start);
        const uint64_t sub_end = std::min(off + nbytes, c_start + c_len);
        const uint64_t inner_off = sub_start - c_start;
        const uint64_t inner_len = sub_end - sub_start;
        char * out_ptr = (char *) dst + (sub_start - off);

        char key_buf[64];
        std::snprintf(key_buf, sizeof(key_buf), "%016llx_s%d_c%llu", (unsigned long long) sh.url_hash, shard_idx,
                      (unsigned long long) c);
        const std::string key(key_buf);

        if (spill_read_chunk(key, inner_off, out_ptr, inner_len)) {
            spill_hits_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        spill_misses_.fetch_add(1, std::memory_order_relaxed);

        // If a single chunk exceeds the whole spill budget, fetch only the requested slice directly.
        if (c_len > cache_bytes_) {
            if (fetch_http_exact(lane_id, shard_idx, out_ptr, sub_start, inner_len) < 0) return -1;
            continue;
        }

        chunk_buf.resize((size_t) c_len);
        if (fetch_http_exact(lane_id, shard_idx, chunk_buf.data(), c_start, c_len) < 0) return -1;
        std::memcpy(out_ptr, chunk_buf.data() + inner_off, (size_t) inner_len);
        spill_write_chunk(key, chunk_buf.data(), c_len);
    }
    return (long long) nbytes;
}

long long HttpRangeProvider::read(int lane, int shard, void * dst, uint64_t off, uint64_t nbytes) {
    if (nbytes == 0) return 0;
    if (!open_ || !dst || shard < 0 || shard >= (int) shards_.size()) return -1;
    const Shard & sh = shards_[(size_t) shard];
    if (off >= sh.size || nbytes > sh.size - off) return -1;

    const auto t0 = clock_t_::now();
    long long got = 0;
    if (!cache_dir_.empty() && cache_bytes_ > 0) {
        got = read_with_spill(lane, shard, dst, off, nbytes);
    } else {
        got = fetch_http_exact(lane, shard, dst, off, nbytes);
    }
    const uint64_t dt = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t_::now() - t0).count();
    io_ns_.fetch_add(dt, std::memory_order_relaxed);
    return got;
}

void HttpRangeProvider::start_workers() {
    worker_stop_ = false;
    batch_gen_ = 0;
    workers_.clear();
    if (lanes_ <= 1) return;
    workers_.reserve((size_t) (lanes_ - 1));
    for (int l = 1; l < lanes_; ++l)
        workers_.emplace_back(&HttpRangeProvider::batch_worker, this, l);
}

void HttpRangeProvider::stop_workers() {
    {
        std::lock_guard<std::mutex> lk(worker_mtx_);
        worker_stop_ = true;
    }
    worker_cv_.notify_all();
    for (std::thread & th : workers_)
        if (th.joinable()) th.join();
    workers_.clear();
}

void HttpRangeProvider::execute_batch_slot(int lane) {
    for (;;) {
        const size_t idx = batch_next_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= cur_count_) break;
        BlockReadReq & r = cur_reqs_[idx];
        const auto t0 = clock_t_::now();
        const long long got = read(lane, r.shard, r.dst, r.off, r.nbytes);
        r.latency_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t_::now() - t0).count();
        r.bytes_moved = got;
        const bool ok = got >= 0;
        if (!ok) batch_all_ok_.store(false, std::memory_order_relaxed);
        if (cur_cb_ && *cur_cb_) (*cur_cb_)(idx, ok);
    }
}

void HttpRangeProvider::batch_worker(int lane) {
    uint64_t seen_gen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(worker_mtx_);
            worker_cv_.wait(lk, [&] { return worker_stop_ || batch_gen_ != seen_gen; });
            if (worker_stop_) return;
            seen_gen = batch_gen_;
        }
        execute_batch_slot(lane);
        {
            std::lock_guard<std::mutex> lk(worker_mtx_);
            if (--active_workers_ == 0) batch_done_cv_.notify_one();
        }
    }
}

bool HttpRangeProvider::read_batch(BlockReadReq * reqs,
                                   size_t count,
                                   const std::function<void(size_t idx, bool ok)> & on_complete,
                                   const std::function<void()> & on_submitted) {
    if (count == 0) {
        if (on_submitted) on_submitted();
        return true;
    }
    if (!open_ || !reqs) return false;

    // Fast path for a single request or single-lane provider: run directly on lane 0.
    if (count == 1 || workers_.empty()) {
        if (on_submitted) on_submitted();
        bool all_ok = true;
        for (size_t i = 0; i < count; ++i) {
            BlockReadReq & r = reqs[i];
            const auto t0 = clock_t_::now();
            const long long got = read(0, r.shard, r.dst, r.off, r.nbytes);
            r.latency_ns =
                (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t_::now() - t0).count();
            r.bytes_moved = got;
            const bool ok = got >= 0;
            if (!ok) all_ok = false;
            if (on_complete) on_complete(i, ok);
        }
        return all_ok;
    }

    std::lock_guard<std::mutex> call_lk(batch_call_mtx_);
    {
        std::lock_guard<std::mutex> lk(worker_mtx_);
        cur_reqs_ = reqs;
        cur_count_ = count;
        cur_cb_ = on_complete ? &on_complete : nullptr;
        batch_next_.store(0, std::memory_order_relaxed);
        batch_all_ok_.store(true, std::memory_order_relaxed);
        active_workers_ = (int) workers_.size();
        ++batch_gen_;
    }
    worker_cv_.notify_all();
    if (on_submitted) on_submitted();

    // Calling thread participates as lane 0 while worker threads execute lanes 1..lanes_-1.
    execute_batch_slot(0);

    {
        std::unique_lock<std::mutex> lk(worker_mtx_);
        batch_done_cv_.wait(lk, [&] { return active_workers_ == 0; });
        cur_reqs_ = nullptr;
        cur_count_ = 0;
        cur_cb_ = nullptr;
    }
    return batch_all_ok_.load(std::memory_order_relaxed);
}

} // namespace bmoe
