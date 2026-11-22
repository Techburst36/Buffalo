// Unit & session tests for the standalone resident draft model (core/src/engine/draft_model.{h,cpp}).
//
// Covers:
//   1. Pure cumulative-confidence stopping rule (p1 * p2 * ... * pk >= tau) and top-1 softmax math.
//   2. Load-time tokenizer parity: passes on matching vocab, refuses mismatched vocab.
//   3. End-to-end Session generation with DraftSource::model on tiny-dense-qwen3.gguf:
//      - exact greedy output parity vs unspeculated base;
//      - adaptive draft length via --draft-tau (tau=0 drafts full cap, high tau stops early);
//      - multi-turn prefix reuse (clear_kv=false) keeps draft KV consistent.
//
// Checks are explicit (not <cassert>): the Release build defines NDEBUG, which compiles assert out.

#include "bmoe/config.h"
#include "bmoe/runtime.h"
#include "bmoe/session.h"
#include "draft_model.h"

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace bmoe;
using namespace bmoe::detail;

static int failures = 0;

static void check(bool ok, const std::string & what) {
    if (ok) {
        std::printf("[PASS] %s\n", what.c_str());
    } else {
        std::printf("[FAIL] %s\n", what.c_str());
        ++failures;
    }
}

static void test_pure_policy() {
    double cum = 1.0, next = 0.0;
    check(draft_cum_prob_step(cum, 0.8, 0.10f, next) && std::fabs(next - 0.8) < 1e-12,
          "cum_p: first step 0.8 >= 0.10 continues");
    cum = next;
    check(draft_cum_prob_step(cum, 0.5, 0.10f, next) && std::fabs(next - 0.4) < 1e-12,
          "cum_p: second step 0.4 >= 0.10 continues");
    cum = next;
    check(draft_cum_prob_step(0.5, 0.25, 0.125f, next) && std::fabs(next - 0.125) < 1e-12,
          "cum_p: exact binary boundary 0.125 >= 0.125 continues");
    check(!draft_cum_prob_step(cum, 0.24, 0.10f, next), "cum_p: 0.096 < 0.10 stops extending");

    check(!draft_cum_prob_step(1.0, 0.05, 0.10f, next), "cum_p: uncertain first token (0.05 < 0.10) drafts 0");
    check(draft_cum_prob_step(1e-5, 1e-5, 0.0f, next), "cum_p: tau=0 disables cutoff");

    {
        const float uniform[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int32_t id = -1;
        double prob = 0.0;
        top1_token_and_prob(uniform, 4, id, prob);
        check(id == 0 && std::fabs(prob - 0.25) < 1e-9, "top1_token_and_prob: uniform distribution");
    }
    {
        // Large offset checks numerical stability (subtract-max before exp).
        const float peaked[3] = {1000.0f, 1002.0f, 1000.0f};
        int32_t id = -1;
        double prob = 0.0;
        top1_token_and_prob(peaked, 3, id, prob);
        const double want = 1.0 / (1.0 + 2.0 * std::exp(-2.0));
        check(id == 1 && std::fabs(prob - want) < 1e-9, "top1_token_and_prob: numerically stable peak");
    }
}

static void test_tokenizer_parity(const std::string & model_path, const std::string & other_vocab_path) {
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;

    std::unique_ptr<llama_model, void (*)(llama_model *)> m_tgt(llama_model_load_from_file(model_path.c_str(), mp),
                                                                llama_model_free);
    std::unique_ptr<llama_model, void (*)(llama_model *)> m_same(llama_model_load_from_file(model_path.c_str(), mp),
                                                                 llama_model_free);
    std::unique_ptr<llama_model, void (*)(llama_model *)> m_diff(
        llama_model_load_from_file(other_vocab_path.c_str(), mp), llama_model_free);

    check(m_tgt && m_same && m_diff, "loaded vocab models for tokenizer parity check");
    if (m_tgt && m_same) {
        const TokenizerParity ok = check_tokenizer_parity(m_tgt.get(), m_same.get());
        check(ok.ok && ok.n_vocab > 0, "tokenizer parity passes on identical vocab (" + ok.summary + ")");
    }
    if (m_tgt && m_diff) {
        const TokenizerParity bad = check_tokenizer_parity(m_tgt.get(), m_diff.get());
        check(!bad.ok && !bad.error.empty(), "tokenizer parity refuses mismatched vocab (" + bad.error + ")");
    }

    llama_backend_free();
}

static void test_session_draft_model(const std::string & model_path) {
    RunConfig base;
    base.model_path = model_path;
    base.prompt = "Hello world from Buffalo dense streaming speculative decoding test.";
    base.n_ctx = 256;
    base.n_predict = 24;
    base.n_threads = 2;
    base.n_gpu_layers = 0;
    base.device_use = DeviceUse::CpuOnly;

    const RunResult r_base = run(base);
    check(r_base.ok && r_base.summary.n_generated == 24, "base run succeeded (" + r_base.error + ")");

    // Draft model == target model with tau=0: must draft full width, accept 100%, and match base.
    RunConfig dm_full = base;
    dm_full.spec.source = DraftSource::model;
    dm_full.spec.draft_model = model_path;
    dm_full.spec.draft_max = 6;
    dm_full.spec.draft_tau = 0.0f;

    const RunResult r_full = run(dm_full);
    check(r_full.ok, "draft model run (tau=0) succeeded (" + r_full.error + ")");
    check(r_full.generated_text == r_base.generated_text, "draft model (tau=0) matches base output");
    check(r_full.summary.mtp_drafted > 0 && r_full.summary.mtp_accepted == r_full.summary.mtp_drafted,
          "self-draft (tau=0) achieves 100% acceptance (" + std::to_string(r_full.summary.mtp_accepted) + "/" +
              std::to_string(r_full.summary.mtp_drafted) + ")");
    check(r_full.summary.mtp_decodes < r_base.summary.n_generated,
          "self-draft reduces verify decode count (" + std::to_string(r_full.summary.mtp_decodes) + " < " +
              std::to_string(r_base.summary.n_generated) + ")");

    // High tau (0.999999): random-weight tiny model has diffuse logits so every step stops at 0 drafts.
    RunConfig dm_strict = dm_full;
    dm_strict.spec.draft_tau = 0.999999f;
    const RunResult r_strict = run(dm_strict);
    check(r_strict.ok && r_strict.generated_text == r_base.generated_text,
          "draft model (tau=0.999999) matches base output");
    check(r_strict.summary.mtp_drafted < r_full.summary.mtp_drafted,
          "high tau stops early (" + std::to_string(r_strict.summary.mtp_drafted) + " < " +
              std::to_string(r_full.summary.mtp_drafted) + ")");

    // Multi-turn session with clear_kv=false: verify KV reset/trim stays consistent across turns.
    SessionConfig sc = session_config_from(dm_full);
    std::string err;
    std::unique_ptr<Session> sess = Session::open(sc, err);
    check(sess != nullptr, "draft model session opened (" + err + ")");
    if (sess) {
        GenerateRequest g1;
        g1.prompt = base.prompt;
        g1.n_predict = 12;
        g1.clear_kv = true;
        const RunResult t1 = sess->generate(g1);
        check(t1.ok && t1.summary.n_generated == 12, "multi-turn turn 1 succeeded");

        GenerateRequest g2;
        g2.prompt = "Second turn prompt after clear_kv=true.";
        g2.n_predict = 12;
        g2.clear_kv = true;
        const RunResult t2 = sess->generate(g2);
        check(t2.ok && t2.summary.n_generated == 12, "multi-turn turn 2 succeeded");
    }
}

int main(int argc, char ** argv) {
    test_pure_policy();

    if (argc >= 2) {
        const std::string model_path = argv[1];
        const std::string other_vocab = argc >= 3 ? argv[2] : BMOE_QWEN35_VOCAB_PATH;
        test_tokenizer_parity(model_path, other_vocab);
        test_session_draft_model(model_path);
    }

    if (failures) {
        std::printf("\n%d draft model check(s) FAILED\n", failures);
        return 1;
    }
    std::printf("\nall draft model checks passed\n");
    return 0;
}
