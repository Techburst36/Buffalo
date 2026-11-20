// Unit tests for BlockProvider, PosixBlockProvider, and raw Linux IoUringBlockProvider.
//
// Exercises multi-shard positioned reads, batched reads with completion callbacks, aligned zero-copy
// paths, unaligned offsets, sub-alignment EOF tails, out-of-bounds rejection, reopen(), and
// open_local_block_provider() automatic fallback when io_uring_setup is blocked by kernel/sandbox
// policy (ENOSYS / EPERM / EACCES).
//
// Checks are explicit (not <cassert>): the Release build defines NDEBUG, which compiles assert out.
#include "block_provider.h"
#include "platform_io.h"
#ifdef BMOE_HAS_IO_URING
#include "iouring_provider.h"
#include <cerrno>
#include <cstring>
#endif

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace bmoe;

static int failures = 0;

static void check(bool ok, const std::string & what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

static unsigned char pattern(int shard, uint64_t i) {
    return (unsigned char) (((i * 131u) + (uint64_t) shard * 53u + 7u) & 0xffu);
}

static bool matches(int shard, const unsigned char * buf, size_t n, uint64_t off) {
    for (size_t i = 0; i < n; ++i)
        if (buf[i] != pattern(shard, off + i)) return false;
    return true;
}

static void exercise_provider(BlockProvider & p,
                              const std::string & label,
                              const std::vector<uint64_t> & sizes) {
    check(p.is_open(), label + ": is_open");
    check(p.shard_count() == (int) sizes.size(), label + ": shard_count matches");
    for (size_t s = 0; s < sizes.size(); ++s)
        check(p.shard_size((int) s) == sizes[s], label + ": shard_size(" + std::to_string(s) + ") matches");

    // 1. Single-item unaligned read across shards
    {
        std::vector<unsigned char> buf(5000, 0xEE);
        const long long got = p.read(0, 0, buf.data(), 1000, buf.size());
        check(got >= (long long) buf.size() && matches(0, buf.data(), buf.size(), 1000),
              label + ": single unaligned read shard 0");
    }
    // 2. Single-item fully aligned read (exercises zero-copy fast path under O_DIRECT + io_uring)
    {
        const size_t al = p.buffer_align();
        const size_t len = al * 2;
        void * aligned_buf = pio::alloc_aligned(al, len);
        check(aligned_buf != nullptr, label + ": alloc_aligned");
        if (aligned_buf) {
            std::memset(aligned_buf, 0xDD, len);
            const long long got = p.read(0, 1, aligned_buf, al, len);
            check(got >= (long long) len && matches(1, (const unsigned char *) aligned_buf, len, al),
                  label + ": single aligned zero-copy read shard 1");
            pio::aligned_free(aligned_buf);
        }
    }
    // 3. Sub-alignment EOF tail read
    {
        const uint64_t tail = 3000;
        std::vector<unsigned char> buf(tail, 0xCC);
        const long long got = p.read(0, 1, buf.data(), sizes[1] - tail, tail);
        check(got >= (long long) tail && matches(1, buf.data(), buf.size(), sizes[1] - tail),
              label + ": sub-alignment EOF-tail read");
    }
    // 4. Out-of-bounds / overflow rejection
    {
        std::vector<unsigned char> buf(64, 0xBB);
        check(p.read(0, 0, buf.data(), sizes[0] - 32, 64) == -1, label + ": rejects read past EOF");
        check(p.read(0, 0, buf.data(), UINT64_MAX - 8, 64) == -1, label + ": rejects offset overflow");
        check(p.read(0, 99, buf.data(), 0, 64) == -1, label + ": rejects invalid shard");
        check(p.read(0, 0, buf.data(), 0, 0) == 0, label + ": zero-length read returns 0");
    }

    // 5. Multi-shard batched read with >qd requests (tests chunking/sliding window, aligned + unaligned + EOF tail,
    //    and on_complete callback completeness)
    {
        const size_t n_reqs = 40;
        const size_t al = p.buffer_align();
        std::vector<void *> bufs(n_reqs, nullptr);
        std::vector<size_t> lens(n_reqs, 0);
        std::vector<BlockReadReq> reqs(n_reqs);

        for (size_t i = 0; i < n_reqs; ++i) {
            const int shard = (int) (i % sizes.size());
            const uint64_t fsz = sizes[(size_t) shard];
            uint64_t off = 0;
            size_t len = 0;
            if (i % 5 == 0) {
                // Aligned zero-copy candidate
                off = al * (1 + (i % 4));
                len = al;
            } else if (i % 5 == 1) {
                // Sub-alignment EOF tail
                len = 2500 + (i * 17) % 1000;
                off = fsz - len;
            } else if (i % 5 == 2) {
                // Zero-length item in the middle of a batch
                off = 123;
                len = 0;
            } else {
                // Arbitrary unaligned slice
                len = 3333 + (i * 257) % 8192;
                off = ((i * 7919) % (fsz - len - 16)) + 3;
            }
            lens[i] = len;
            bufs[i] = pio::alloc_aligned(al, std::max<size_t>(len, al));
            if (bufs[i] && len > 0) std::memset(bufs[i], 0xAA, len);
            reqs[i] = BlockReadReq{shard, off, len, bufs[i], 0, 0};
        }

        std::vector<int> cb_counts(n_reqs, 0);
        bool all_cb_ok = true;
        const bool ok = p.read_batch(reqs.data(), reqs.size(), [&](size_t idx, bool item_ok) {
            if (idx < n_reqs) cb_counts[idx]++;
            if (!item_ok) all_cb_ok = false;
        });
        check(ok && all_cb_ok, label + ": read_batch(40 reqs across shards) succeeded");

        bool bytes_ok = true;
        bool cbs_ok = true;
        for (size_t i = 0; i < n_reqs; ++i) {
            if (cb_counts[i] != 1) cbs_ok = false;
            if (lens[i] == 0) {
                if (reqs[i].bytes_moved != 0) bytes_ok = false;
            } else {
                if (reqs[i].bytes_moved < (long long) lens[i] ||
                    !matches(reqs[i].shard, (const unsigned char *) bufs[i], lens[i], reqs[i].off))
                    bytes_ok = false;
            }
            if (bufs[i]) pio::aligned_free(bufs[i]);
        }
        check(cbs_ok, label + ": on_complete fired exactly once per request");
        check(bytes_ok, label + ": all 40 batched slices matched exact byte pattern");
    }

    // 6. Batch with an out-of-bounds item fails cleanly and still fires on_complete for every item
    {
        std::vector<unsigned char> good(4096, 0x11), bad(64, 0x22);
        BlockReadReq reqs[2] = {
            BlockReadReq{0, 4096, good.size(), good.data(), 0, 0},
            BlockReadReq{0, sizes[0] - 10, bad.size(), bad.data(), 0, 0},
        };
        int seen[2] = {0, 0};
        bool item_ok[2] = {false, true};
        const bool ok = p.read_batch(reqs, 2, [&](size_t idx, bool ok_i) {
            seen[idx]++;
            item_ok[idx] = ok_i;
        });
        check(!ok && seen[0] == 1 && seen[1] == 1 && item_ok[0] && !item_ok[1] &&
                  matches(0, good.data(), good.size(), 4096),
              label + ": batch isolates OOB error without corrupting valid sibling");
    }

    // 7. Reopen preserves readability and accounting
    const long long bytes_before = p.read_bytes();
    check(bytes_before > 0, label + ": read_bytes accounting > 0");
    check(p.reopen(), label + ": reopen() succeeds");
    {
        std::vector<unsigned char> buf(4096, 0xEE);
        const long long got = p.read(0, 0, buf.data(), 8192, buf.size());
        check(got >= (long long) buf.size() && matches(0, buf.data(), buf.size(), 8192),
              label + ": reads cleanly after reopen()");
    }
}

int main() {
    const std::vector<std::string> paths = {"bmoe_bp_shard0.bin", "bmoe_bp_shard1.bin"};
    const std::vector<uint64_t> sizes = {3 * 65536 + 777, 2 * 65536 + 1337}; // non-multiples of any granule
    for (size_t s = 0; s < paths.size(); ++s) {
        FILE * f = std::fopen(paths[s].c_str(), "wb");
        if (!f) {
            std::printf("[FAIL] cannot create scratch shard %zu\n", s);
            return 1;
        }
        for (uint64_t i = 0; i < sizes[s]; ++i)
            std::fputc(pattern((int) s, i), f);
        std::fclose(f);
    }

    // PosixBlockProvider: buffered & direct
    for (bool direct : {false, true}) {
        PosixBlockProvider p;
        const std::string tag = std::string("posix(") + (direct ? "direct" : "buffered") + ")";
        check(p.open(paths, 2, direct, 1 << 16), tag + ": open");
        check(!p.prefers_batch() && std::string(p.backend_name()) == "posix", tag + ": metadata");
        if (p.is_open()) exercise_provider(p, tag, sizes);
    }

    // Factory: explicit posix
    {
        auto p = open_local_block_provider(paths, 2, true, 1 << 16, IoBackend::Posix);
        check(p != nullptr && std::string(p->backend_name()) == "posix", "factory(posix): returns PosixBlockProvider");
    }

    // Raw Linux IoUringBlockProvider + factory auto/io_uring fallback check
#ifdef BMOE_HAS_IO_URING
    {
        IoUringBlockProvider ring;
        int ring_errno = 0;
        const bool opened = ring.open(paths, 4, true, 1 << 16, &ring_errno);
        if (opened) {
            std::printf("[INFO] io_uring: running NATIVE kernel io_uring tests\n");
            check(ring.prefers_batch() && std::string(ring.backend_name()) == "io_uring", "io_uring: metadata");
            exercise_provider(ring, "io_uring(direct)", sizes);
            ring.close();

            IoUringBlockProvider ring_buf;
            check(ring_buf.open(paths, 4, false, 1 << 16, &ring_errno), "io_uring(buffered): open");
            if (ring_buf.is_open()) exercise_provider(ring_buf, "io_uring(buffered)", sizes);

            auto p_auto = open_local_block_provider(paths, 4, true, 1 << 16, IoBackend::Auto);
            check(p_auto != nullptr && std::string(p_auto->backend_name()) == "io_uring",
                  "factory(auto): selected io_uring natively");
        } else {
            const bool expected_block =
                ring_errno == ENOSYS || ring_errno == EPERM || ring_errno == EACCES || ring_errno == ENOMEM;
            std::printf("[INFO] io_uring: io_uring_setup refused (%s, errno=%d) — verifying graceful fallback to posix\n",
                        std::strerror(ring_errno), ring_errno);
            check(expected_block, "io_uring: refused only with expected policy errno (ENOSYS/EPERM/EACCES/ENOMEM)");
            auto p_auto = open_local_block_provider(paths, 4, true, 1 << 16, IoBackend::Auto);
            auto p_req = open_local_block_provider(paths, 4, true, 1 << 16, IoBackend::IoUring);
            check(p_auto != nullptr && std::string(p_auto->backend_name()) == "posix",
                  "factory(auto): cleanly fell back to posix");
            check(p_req != nullptr && std::string(p_req->backend_name()) == "posix",
                  "factory(io_uring): cleanly fell back to posix");
            if (p_auto) exercise_provider(*p_auto, "factory(auto->posix fallback)", sizes);
        }
    }
#else
    std::printf("[INFO] io_uring: not compiled on this platform (BMOE_HAS_IO_URING unset)\n");
    {
        auto p = open_local_block_provider(paths, 4, true, 1 << 16, IoBackend::Auto);
        check(p != nullptr && std::string(p->backend_name()) == "posix", "factory(auto): returns posix");
    }
#endif

    // Missing file must NOT be masked by fallback
    {
        auto missing = open_local_block_provider({"bmoe_nonexistent_shard_999.bin"}, 1, true, 4096, IoBackend::Auto);
        check(missing == nullptr, "factory(auto): missing file returns nullptr");
    }

    for (const std::string & p : paths)
        std::remove(p.c_str());

    if (failures) {
        std::printf("\n%d check(s) FAILED\n", failures);
        return 1;
    }
    std::printf("\nall block provider checks passed\n");
    return 0;
}
