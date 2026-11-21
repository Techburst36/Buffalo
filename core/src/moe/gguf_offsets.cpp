#include "gguf_offsets.h"
#include "bmoe/log.h"
#include "../io/block_provider.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace bmoe {

namespace {

// Read a gguf metadata value as an int, accepting any of the integer scalar types the
// writer might have used for a count (llama.cpp writes expert counts as UINT32, but be
// permissive). Returns `dflt` if the key is absent or not an integer scalar.
int meta_int(const gguf_context * ctx, const std::string & key, int dflt) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) return dflt;
    switch (gguf_get_kv_type(ctx, id)) {
    case GGUF_TYPE_UINT8:
        return (int) gguf_get_val_u8(ctx, id);
    case GGUF_TYPE_INT8:
        return (int) gguf_get_val_i8(ctx, id);
    case GGUF_TYPE_UINT16:
        return (int) gguf_get_val_u16(ctx, id);
    case GGUF_TYPE_INT16:
        return (int) gguf_get_val_i16(ctx, id);
    case GGUF_TYPE_UINT32:
        return (int) gguf_get_val_u32(ctx, id);
    case GGUF_TYPE_INT32:
        return (int) gguf_get_val_i32(ctx, id);
    case GGUF_TYPE_UINT64:
        return (int) gguf_get_val_u64(ctx, id);
    case GGUF_TYPE_INT64:
        return (int) gguf_get_val_i64(ctx, id);
    default:
        return dflt;
    }
}

// Single parse; each extraction below reads from the already-parsed context.
gguf_context * open_meta(const char * path) {
    gguf_init_params gp{};
    gp.no_alloc = true; // metadata + offsets only; no tensor bytes touched
    gp.ctx = nullptr;
    return gguf_init_from_file(path, gp);
}

void fill_offsets(const gguf_context * gctx, int file_idx, GgufOffsets & out) {
    const uint64_t data_off = (uint64_t) gguf_get_data_offset(gctx);
    const int64_t n = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        out.off_by_name[name] = data_off + (uint64_t) gguf_get_tensor_offset(gctx, i);
        out.size_by_name[name] = (uint64_t) gguf_get_tensor_size(gctx, i);
        out.type_by_name[name] = (int) gguf_get_tensor_type(gctx, i);
        out.file_by_name[name] = file_idx;
        // Row count, for the one consumer that needs a shape rather than a size: turning a table's
        // total bytes into the bytes a single token gathers from it. Taken here because this parse
        // has already walked the whole KV section, which is exactly what this file's header says a
        // caller must not be made to pay for twice.
        const int64_t * ne = gguf_get_tensor_ne(gctx, i);
        out.rows_by_name[name] = ne ? (uint64_t) (ne[1] > 0 ? ne[1] : 0) : 0;
        // Exactly three dimensions: gguf pads ne to GGML_MAX_DIMS with ones, so a 3-D tensor is one
        // whose fourth extent is 1 and whose third is not.
        out.extents.push_back({name, ne && ne[3] == 1 && ne[2] > 1 ? (uint64_t) ne[2] : 0});
    }
}

void fill_model_info(const gguf_context * gctx, GgufModelInfo & out) {
    const int64_t arch_id = gguf_find_key(gctx, "general.architecture");
    if (arch_id >= 0 && gguf_get_kv_type(gctx, arch_id) == GGUF_TYPE_STRING) {
        out.arch = gguf_get_val_str(gctx, arch_id);
    }
    const int64_t type_id = gguf_find_key(gctx, "general.type");
    if (type_id >= 0 && gguf_get_kv_type(gctx, type_id) == GGUF_TYPE_STRING) out.type = gguf_get_val_str(gctx, type_id);
    const int64_t name_id = gguf_find_key(gctx, "general.name");
    if (name_id >= 0 && gguf_get_kv_type(gctx, name_id) == GGUF_TYPE_STRING) out.name = gguf_get_val_str(gctx, name_id);
    if (!out.arch.empty()) {
        // Arch-prefixed keys, exactly as llama.cpp names them (LLM_KV_EXPERT_COUNT /
        // LLM_KV_EXPERT_USED_COUNT expand "%s" to the architecture).
        out.n_expert = meta_int(gctx, out.arch + ".expert_count", 0);
        out.n_expert_used = meta_int(gctx, out.arch + ".expert_used_count", 0);
        // The hidden state's width. It is what a tensor crossing a host/device boundary actually
        // carries, so the split probe needs it to measure a crossing at the size this model pays.
        out.n_embd = meta_int(gctx, out.arch + ".embedding_length", 0);
        out.n_swa = meta_int(gctx, out.arch + ".attention.sliding_window", 0);
        out.n_block = meta_int(gctx, out.arch + ".block_count", 0);
        out.n_nextn = meta_int(gctx, out.arch + ".nextn_predict_layers", 0);
        out.reads_target = gguf_find_key(gctx, (out.arch + ".target_layers").c_str()) >= 0;
    }
    out.ok = true;
}

// The llama.cpp split filename convention: `<prefix>-%05d-of-%05d.gguf`, 1-based. If
// `path` matches, returns the shard count and sets `prefix` (everything before the
// shard number); the caller rebuilds each sibling with make_shard_path. Returns 0 for
// a plain single-file path. The FIRST shard must be the one asked for — the same rule
// llama.cpp applies — so downstream shard indices line up with llama.cpp's own mmaps.
int split_count_from_path(const std::string & path, std::string & prefix) {
    // <prefix>-DDDDD-of-DDDDD.gguf -> 19 chars after the separating dash
    static const size_t tail_len = 5 + 4 + 5 + 5; // "DDDDD" "-of-" "DDDDD" ".gguf"
    if (path.size() < tail_len + 1) return 0;
    const size_t tail = path.size() - tail_len;
    auto five_digits = [&](size_t at) {
        for (size_t i = at; i < at + 5; ++i)
            if (path[i] < '0' || path[i] > '9') return false;
        return true;
    };
    if (path[tail - 1] != '-' || !five_digits(tail) || path.compare(tail + 5, 4, "-of-") != 0 ||
        !five_digits(tail + 9) || path.compare(tail + 14, 5, ".gguf") != 0)
        return 0;
    const int no = std::stoi(path.substr(tail, 5));
    const int count = std::stoi(path.substr(tail + 9, 5));
    if (count <= 1) return 0; // a -00001-of-00001 file is just a single file
    if (no != 1) {
        log_printf(LogLevel::Error, "%s is shard %d of a split model — pass the first shard (-00001-of-)", path.c_str(),
                   no);
        return -1;
    }
    prefix = path.substr(0, tail - 1);
    return count;
}

std::string make_shard_path(const std::string & prefix, int no, int count) {
    char tail[32];
    std::snprintf(tail, sizeof(tail), "-%05d-of-%05d.gguf", no, count);
    return prefix + tail;
}

// Offsets for `path` and, if it is the first file of a split set, for every sibling
// shard. Any missing or unparsable shard fails the whole read: a partial map would
// surface later as "no gguf offset for tensor X", which points at the wrong culprit.
bool fill_all_offsets(const char * path, GgufOffsets & out, gguf_context * first) {
    std::string prefix;
    const int count = split_count_from_path(path, prefix);
    if (count < 0) return false;

    fill_offsets(first, 0, out);
    out.shard_paths.push_back(path);

    for (int no = 2; no <= count; ++no) {
        const std::string sp = make_shard_path(prefix, no, count);
        gguf_context * gctx = open_meta(sp.c_str());
        if (!gctx) {
            log_printf(LogLevel::Error, "split model is missing shard %s", sp.c_str());
            out = GgufOffsets{};
            return false;
        }
        fill_offsets(gctx, no - 1, out);
        out.shard_paths.push_back(sp);
        gguf_free(gctx);
    }
    out.ok = true;
    return true;
}

// ── Exact zero-overshoot GGUF metadata prefetcher for remote HTTP shards ────────────
//
// `gguf_init_from_callback` reads small scalar fields (4-8 bytes) sequentially. Naive read-ahead
// would overshoot past the end of metadata into tensor data (potentially touching routed expert
// bytes on small split shards). Instead, we walk the GGUF v2/v3 wire framing using a strict
// mathematical lower bound on remaining metadata bytes:
//   - each remaining KV pair >= 8 (key len) + 4 (type) + 1 (min value) = 13 bytes
//   - each remaining string in a string array >= 8 (string len) bytes
//   - each remaining tensor entry >= 8 (name len) + 4 (n_dims) + 4 (ggml_type) + 8 (offset) = 24 bytes
// Every HTTP Range fetch is strictly <= true meta_end <= data_offset, so ZERO bytes of padding or
// tensor payload are ever requested while large vocab arrays still coalesce into <= 1 MiB fetches.

size_t gguf_scalar_byte_size(int32_t t) {
    switch (t) {
    case GGUF_TYPE_UINT8:
    case GGUF_TYPE_INT8:
    case GGUF_TYPE_BOOL:
        return 1;
    case GGUF_TYPE_UINT16:
    case GGUF_TYPE_INT16:
        return 2;
    case GGUF_TYPE_UINT32:
    case GGUF_TYPE_INT32:
    case GGUF_TYPE_FLOAT32:
        return 4;
    case GGUF_TYPE_UINT64:
    case GGUF_TYPE_INT64:
    case GGUF_TYPE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

uint32_t read_le_u32(const uint8_t * p) {
    uint32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

int32_t read_le_i32(const uint8_t * p) {
    int32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

uint64_t read_le_u64(const uint8_t * p) {
    uint64_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

int64_t read_le_i64(const uint8_t * p) {
    int64_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

class ExactGgufMetaReader {
public:
    ExactGgufMetaReader(BlockProvider & prov, int shard_idx)
        : prov_(prov), shard_(shard_idx), fsize_(prov.shard_size(shard_idx)) {}

    bool read_all(std::vector<uint8_t> & out, std::string & err) {
        buf_.clear();
        pos_ = 0;
        if (fsize_ < 24) {
            err = "remote GGUF shard too small (< 24 bytes)";
            return false;
        }
        if (!ensure(24, 24)) {
            err = "failed to read remote GGUF header";
            return false;
        }
        if (std::memcmp(buf_.data(), GGUF_MAGIC, 4) != 0) {
            err = "remote file is not a valid GGUF (bad magic)";
            return false;
        }
        const uint32_t version = read_le_u32(buf_.data() + 4);
        if (version < 2 || version > GGUF_VERSION) {
            err = "unsupported remote GGUF version " + std::to_string(version);
            return false;
        }
        const int64_t n_tensors = read_le_i64(buf_.data() + 8);
        const int64_t n_kv = read_le_i64(buf_.data() + 16);
        if (n_tensors < 0 || n_kv < 0) {
            err = "invalid negative GGUF tensor/KV count";
            return false;
        }
        pos_ = 24;

        const uint64_t nt = (uint64_t) n_tensors;
        const uint64_t nk = (uint64_t) n_kv;

        for (uint64_t k = 0; k < nk; ++k) {
            const uint64_t after_kv = nk - 1 - k;
            const uint64_t tail = min_tail(after_kv, nt);
            // Key string length + key bytes + value type + at least 1 value byte.
            uint64_t klen = 0;
            if (!read_u64(klen, 13 + tail)) {
                err = "truncated GGUF KV key length";
                return false;
            }
            if (!skip_bytes(klen, 5 + tail)) {
                err = "truncated GGUF KV key";
                return false;
            }
            int32_t vtype = 0;
            if (!read_i32(vtype, 5 + tail)) {
                err = "truncated GGUF KV value type";
                return false;
            }
            if (!skip_kv_value(vtype, tail, err)) return false;
        }

        for (uint64_t i = 0; i < nt; ++i) {
            const uint64_t after_t = nt - 1 - i;
            const uint64_t tail = min_tail(0, after_t);
            // Name len + name + n_dims(4) + ggml_type(4) + offset(8) => at least 24 bytes.
            uint64_t nlen = 0;
            if (!read_u64(nlen, 24 + tail)) {
                err = "truncated GGUF tensor name length";
                return false;
            }
            if (!skip_bytes(nlen, 16 + tail)) {
                err = "truncated GGUF tensor name";
                return false;
            }
            uint32_t n_dims = 0;
            if (!read_u32(n_dims, 16 + tail)) {
                err = "truncated GGUF tensor n_dims";
                return false;
            }
            if (n_dims > GGML_MAX_DIMS) {
                err = "invalid GGUF tensor n_dims " + std::to_string(n_dims);
                return false;
            }
            // Exact remaining bytes for this tensor descriptor: n_dims * int64 + ggml_type(4) + offset(8).
            const uint64_t rest = (uint64_t) n_dims * 8ull + 4ull + 8ull;
            if (!skip_bytes(rest, tail)) {
                err = "truncated GGUF tensor descriptor";
                return false;
            }
        }

        out = std::move(buf_);
        return true;
    }

private:
    static constexpr uint64_t k_max_chunk = 1ull << 20; // 1 MiB cap per HTTP Range metadata fetch

    static uint64_t min_tail(uint64_t rem_kv, uint64_t rem_t) {
        if (rem_kv > (1ull << 24) || rem_t > (1ull << 24)) return 0;
        return rem_kv * 13ull + rem_t * 24ull;
    }

    bool ensure(size_t need, uint64_t lower_bound_from_pos) {
        if (pos_ + need <= buf_.size()) return true;
        if (need > fsize_ - pos_) return false;
        const uint64_t want_from_pos = std::max<uint64_t>(need, std::min<uint64_t>(k_max_chunk, lower_bound_from_pos));
        const uint64_t target_end = std::min<uint64_t>(fsize_, pos_ + want_from_pos);
        if (target_end <= buf_.size()) return true;
        const size_t old_sz = buf_.size();
        const uint64_t fetch_len = target_end - old_sz;
        buf_.resize((size_t) target_end);
        const long long got = prov_.read(0, shard_, buf_.data() + old_sz, (uint64_t) old_sz, fetch_len);
        if (got != (long long) fetch_len) {
            buf_.resize(old_sz);
            return false;
        }
        return true;
    }

    bool read_u32(uint32_t & out, uint64_t lb_from_pos) {
        if (!ensure(4, lb_from_pos)) return false;
        out = read_le_u32(buf_.data() + pos_);
        pos_ += 4;
        return true;
    }

    bool read_i32(int32_t & out, uint64_t lb_from_pos) {
        if (!ensure(4, lb_from_pos)) return false;
        out = read_le_i32(buf_.data() + pos_);
        pos_ += 4;
        return true;
    }

    bool read_u64(uint64_t & out, uint64_t lb_from_pos) {
        if (!ensure(8, lb_from_pos)) return false;
        out = read_le_u64(buf_.data() + pos_);
        pos_ += 8;
        return true;
    }

    bool skip_bytes(uint64_t n, uint64_t lb_after_n) {
        if (n > fsize_ - pos_) return false;
        uint64_t rem = n;
        while (rem > 0) {
            const size_t step = (size_t) std::min<uint64_t>(rem, k_max_chunk);
            if (!ensure(step, rem + lb_after_n)) return false;
            pos_ += step;
            rem -= step;
        }
        return true;
    }

    bool skip_kv_value(int32_t vtype, uint64_t tail_after_kv, std::string & err) {
        const size_t scalar_sz = gguf_scalar_byte_size(vtype);
        if (scalar_sz > 0) {
            if (!skip_bytes(scalar_sz, tail_after_kv)) {
                err = "truncated GGUF scalar KV value";
                return false;
            }
            return true;
        }
        if (vtype == GGUF_TYPE_STRING) {
            uint64_t slen = 0;
            if (!read_u64(slen, 8 + tail_after_kv)) {
                err = "truncated GGUF string KV length";
                return false;
            }
            if (!skip_bytes(slen, tail_after_kv)) {
                err = "truncated GGUF string KV payload";
                return false;
            }
            return true;
        }
        if (vtype == GGUF_TYPE_ARRAY) {
            int32_t etype = 0;
            uint64_t arr_n = 0;
            if (!read_i32(etype, 12 + tail_after_kv) || !read_u64(arr_n, 8 + tail_after_kv)) {
                err = "truncated GGUF array header";
                return false;
            }
            if (etype == GGUF_TYPE_STRING) {
                for (uint64_t j = 0; j < arr_n; ++j) {
                    const uint64_t after_str = arr_n - 1 - j;
                    const uint64_t str_tail = (after_str <= (1ull << 24) ? after_str * 8ull : 0ull) + tail_after_kv;
                    uint64_t slen = 0;
                    if (!read_u64(slen, 8 + str_tail)) {
                        err = "truncated GGUF string-array element length";
                        return false;
                    }
                    if (!skip_bytes(slen, str_tail)) {
                        err = "truncated GGUF string-array element payload";
                        return false;
                    }
                }
                return true;
            }
            const size_t elem_sz = gguf_scalar_byte_size(etype);
            if (elem_sz == 0) {
                err = "unsupported GGUF array element type " + std::to_string(etype);
                return false;
            }
            if (arr_n > (fsize_ - pos_) / elem_sz) {
                err = "GGUF numeric array exceeds shard size";
                return false;
            }
            if (!skip_bytes(arr_n * elem_sz, tail_after_kv)) {
                err = "truncated GGUF numeric array payload";
                return false;
            }
            return true;
        }
        err = "unsupported GGUF KV value type " + std::to_string(vtype);
        return false;
    }

    BlockProvider & prov_;
    int shard_ = 0;
    uint64_t fsize_ = 0;
    std::vector<uint8_t> buf_;
    size_t pos_ = 0;
};

struct MetaBufSlice {
    const uint8_t * data = nullptr;
    size_t size = 0;
};

size_t meta_buf_read_cb(void * ud, void * out, uint64_t off, size_t len) {
    const auto * s = static_cast<const MetaBufSlice *>(ud);
    if (!s || off >= s->size) return 0;
    const size_t n = std::min<size_t>(len, s->size - (size_t) off);
    std::memcpy(out, s->data + off, n);
    return n;
}

gguf_context * parse_remote_shard_gguf(BlockProvider & prov,
                                       int shard_idx,
                                       ggml_context ** out_ggml_ctx,
                                       std::string & err) {
    std::vector<uint8_t> meta_bytes;
    ExactGgufMetaReader reader(prov, shard_idx);
    if (!reader.read_all(meta_bytes, err)) return nullptr;

    MetaBufSlice slice{meta_bytes.data(), meta_bytes.size()};
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx = out_ggml_ctx;
    gguf_context * gctx = gguf_init_from_callback(&meta_buf_read_cb, &slice, 0, prov.shard_size(shard_idx), gp);
    if (!gctx) {
        err = "gguf_init_from_callback failed for remote shard " + std::to_string(shard_idx);
        return nullptr;
    }
    return gctx;
}

} // namespace

RemoteGgufMeta::RemoteGgufMeta() : gguf_ctx(nullptr, gguf_free) {}
RemoteGgufMeta::~RemoteGgufMeta() = default;
RemoteGgufMeta::RemoteGgufMeta(RemoteGgufMeta &&) noexcept = default;
RemoteGgufMeta & RemoteGgufMeta::operator=(RemoteGgufMeta &&) noexcept = default;

RemoteGgufMeta read_remote_gguf_meta(const char * url, int lanes, const HttpProviderOptions & http_opt) {
    RemoteGgufMeta out;
    if (!url || !*url) {
        out.error = "empty remote model URL";
        return out;
    }

    const std::vector<std::string> shard_urls = expand_split_urls(url);
    if (shard_urls.empty()) {
        out.error = std::string(url) + " is not the first shard (-00001-of-) of a split model";
        return out;
    }

    const int eff_lanes = lanes > 0 ? lanes : (http_opt.lanes > 0 ? http_opt.lanes : MoeStreamConfig::http_threads_default);
    std::string perr;
    out.provider = open_block_provider(shard_urls, eff_lanes, /*direct=*/false, /*max_read=*/0, IoBackend::Auto,
                                       http_opt, &perr);
    if (!out.provider) {
        out.error = perr.empty() ? (std::string("failed to open remote GGUF: ") + url) : perr;
        return out;
    }

    // Parse shard 0 metadata (0 bytes overshoot past meta_end).
    std::string serr;
    gguf_context * merged = parse_remote_shard_gguf(*out.provider, 0, nullptr, serr);
    if (!merged) {
        out.error = serr;
        out.provider.reset();
        return out;
    }
    out.gguf_ctx.reset(merged);

    fill_model_info(merged, out.meta.info);
    fill_offsets(merged, 0, out.meta.offsets);
    out.meta.offsets.shard_paths = shard_urls;

    const int declared_splits = meta_int(merged, "split.count", 1);
    if (declared_splits > 1 && shard_urls.size() == 1) {
        out.error = std::string("remote GGUF declares split.count=") + std::to_string(declared_splits) +
                    ", but URL does not follow -00001-of-NNNNN.gguf naming: " + url;
        out.gguf_ctx.reset();
        out.provider.reset();
        return out;
    }

    // For split remote GGUFs, parse sibling shards 1..N-1 (metadata only, 0 bytes overshoot)
    // and merge their tensor descriptors into `merged` so llama_model_init_from_user() sees
    // the unified tensor manifest while `out.meta.offsets` retains true per-shard byte offsets.
    for (int s = 1; s < (int) shard_urls.size(); ++s) {
        ggml_context * tmp_ctx = nullptr;
        gguf_context * shard_gguf = parse_remote_shard_gguf(*out.provider, s, &tmp_ctx, serr);
        if (!shard_gguf) {
            if (tmp_ctx) ggml_free(tmp_ctx);
            out.error = "remote split shard " + shard_urls[(size_t) s] + ": " + serr;
            out.gguf_ctx.reset();
            out.provider.reset();
            return out;
        }
        fill_offsets(shard_gguf, s, out.meta.offsets);
        if (tmp_ctx) {
            for (ggml_tensor * t = ggml_get_first_tensor(tmp_ctx); t; t = ggml_get_next_tensor(tmp_ctx, t)) {
                if (gguf_find_tensor(merged, t->name) == -1) gguf_add_tensor(merged, t);
            }
            ggml_free(tmp_ctx);
        }
        gguf_free(shard_gguf);
    }

    if (shard_urls.size() > 1) {
        gguf_set_val_u16(merged, "split.count", 1);
        gguf_set_val_u16(merged, "split.no", 0);
        gguf_set_val_i32(merged, "split.tensors.count", (int32_t) gguf_get_n_tensors(merged));
    }

    out.meta.offsets.ok = true;
    out.meta.ok = true;
    out.ok = true;
    return out;
}

RemoteGgufMeta read_local_gguf_for_user(const char * path,
                                        int lanes,
                                        bool direct,
                                        size_t max_read,
                                        IoBackend io_backend) {
    RemoteGgufMeta out;
    if (!path || !*path) {
        out.error = "empty local model path";
        return out;
    }

    std::string prefix;
    const int split_count = split_count_from_path(path, prefix);
    if (split_count < 0) {
        out.error = std::string(path) + " is not the first shard (-00001-of-) of a split model";
        return out;
    }

    std::vector<std::string> shard_paths;
    shard_paths.push_back(path);
    for (int no = 2; no <= split_count; ++no)
        shard_paths.push_back(make_shard_path(prefix, no, split_count));

    // Parse shard 0 metadata only (no_alloc = true; never mmaps or reads tensor payloads).
    gguf_context * merged = open_meta(shard_paths[0].c_str());
    if (!merged) {
        out.error = std::string("failed to parse GGUF metadata: ") + shard_paths[0];
        return out;
    }
    out.gguf_ctx.reset(merged);

    fill_model_info(merged, out.meta.info);
    fill_offsets(merged, 0, out.meta.offsets);
    out.meta.offsets.shard_paths = shard_paths;

    const int declared_splits = meta_int(merged, "split.count", 1);
    if (declared_splits > 1 && shard_paths.size() == 1) {
        out.error = std::string("GGUF declares split.count=") + std::to_string(declared_splits) +
                    ", but path does not follow -00001-of-NNNNN.gguf naming: " + path;
        out.gguf_ctx.reset();
        return out;
    }

    // Merge sibling split shards into the single gguf_context for llama_model_init_from_user(),
    // while retaining real per-shard file indices and byte offsets in out.meta.offsets.
    for (int s = 1; s < (int) shard_paths.size(); ++s) {
        ggml_context * tmp_ctx = nullptr;
        gguf_init_params gp{};
        gp.no_alloc = true;
        gp.ctx = &tmp_ctx;
        gguf_context * shard_gguf = gguf_init_from_file(shard_paths[(size_t) s].c_str(), gp);
        if (!shard_gguf) {
            if (tmp_ctx) ggml_free(tmp_ctx);
            out.error = "split model is missing or unreadable shard " + shard_paths[(size_t) s];
            out.gguf_ctx.reset();
            return out;
        }
        fill_offsets(shard_gguf, s, out.meta.offsets);
        if (tmp_ctx) {
            for (ggml_tensor * t = ggml_get_first_tensor(tmp_ctx); t; t = ggml_get_next_tensor(tmp_ctx, t)) {
                if (gguf_find_tensor(merged, t->name) == -1) gguf_add_tensor(merged, t);
            }
            ggml_free(tmp_ctx);
        }
        gguf_free(shard_gguf);
    }

    if (shard_paths.size() > 1) {
        gguf_set_val_u16(merged, "split.count", 1);
        gguf_set_val_u16(merged, "split.no", 0);
        gguf_set_val_i32(merged, "split.tensors.count", (int32_t) gguf_get_n_tensors(merged));
    }

    const int eff_lanes = std::max(1, lanes);
    out.provider = open_local_block_provider(shard_paths, eff_lanes, direct, max_read, io_backend);
    if (!out.provider) {
        out.error = std::string("failed to open local block provider for ") + path;
        out.gguf_ctx.reset();
        return out;
    }

    out.meta.offsets.ok = true;
    out.meta.ok = true;
    out.ok = true;
    return out;
}

GgufOffsets read_gguf_offsets(const char * path) {
    if (path && is_remote_url(path)) return read_remote_gguf_meta(path).meta.offsets;
    GgufOffsets out;
    if (gguf_context * gctx = open_meta(path)) {
        fill_all_offsets(path, out, gctx);
        gguf_free(gctx);
    }
    return out;
}

GgufModelInfo read_gguf_model_info(const char * path) {
    if (path && is_remote_url(path)) return read_remote_gguf_meta(path).meta.info;
    GgufModelInfo out;
    if (gguf_context * gctx = open_meta(path)) {
        fill_model_info(gctx, out);
        gguf_free(gctx);
    }
    return out;
}

GgufMeta read_gguf_meta(const char * path) {
    if (path && is_remote_url(path)) return read_remote_gguf_meta(path).meta;
    GgufMeta out;
    if (gguf_context * gctx = open_meta(path)) {
        const bool offs_ok = fill_all_offsets(path, out.offsets, gctx);
        fill_model_info(gctx, out.info);
        gguf_free(gctx);
        out.ok = offs_ok;
    }
    return out;
}

} // namespace bmoe
