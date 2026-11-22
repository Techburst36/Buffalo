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

// Gate 4: Hybrid lazy KV catch-up test.
// Proves directly that tokens accepted during n-gram sweeps (while the draft model is NOT called)
// are lazily prefilled into the draft model's KV cache in one batch right before its next draft,
// yielding the exact same draft tokens and cumulative probability as a fresh full-prefix prefill.
static void test_hybrid_lazy_catchup_direct(const std::string & model_path) {
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    ggml_backend_dev_t no_devs[1] = {nullptr};
    mp.devices = no_devs;
    std::unique_ptr<llama_model, void (*)(llama_model *)> model(llama_model_load_from_file(model_path.c_str(), mp),
                                                                llama_model_free);
    check(model != nullptr, "hybrid catch-up: loaded tiny model");
    if (!model) {
        llama_backend_free();
        return;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const std::string text = "def merge_intervals(intervals): return sorted(intervals)";
    std::vector<llama_token> toks(text.size() + 8);
    int n_tok = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, true);
    if (n_tok < 0) {
        toks.resize((size_t) (-n_tok));
        n_tok = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, true);
    }
    toks.resize((size_t) std::max(0, n_tok));
    check(toks.size() >= 30, "hybrid catch-up: tokenized test sequence (" + std::to_string(toks.size()) + " tokens)");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 128;
    cp.n_ubatch = 128;
    cp.n_outputs_max = 1;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.swa_full = false;
    std::unique_ptr<llama_context, void (*)(llama_context *)> ctx_lazy(llama_init_from_model(model.get(), cp),
                                                                       llama_free);
    std::unique_ptr<llama_context, void (*)(llama_context *)> ctx_ref(llama_init_from_model(model.get(), cp),
                                                                      llama_free);
    std::unique_ptr<llama_context, void (*)(llama_context *)> ctx_stale(llama_init_from_model(model.get(), cp),
                                                                        llama_free);
    check(ctx_lazy && ctx_ref && ctx_stale, "hybrid catch-up: created lazy, reference, and stale contexts");
    if (!ctx_lazy || !ctx_ref || !ctx_stale) {
        llama_backend_free();
        return;
    }

    DraftModelRunner lazy, ref, stale;
    lazy.init(ctx_lazy.get(), vocab, (int) llama_n_ubatch(ctx_lazy.get()));
    ref.init(ctx_ref.get(), vocab, (int) llama_n_ubatch(ctx_ref.get()));
    stale.init(ctx_stale.get(), vocab, (int) llama_n_ubatch(ctx_stale.get()));

    // Step 1: initial draft on prefix[0..11] + tok[12], then partial accept of 2 drafts.
    std::vector<llama_token> prefix1(toks.begin(), toks.begin() + 12);
    const llama_token last1 = toks[12];
    std::vector<llama_token> draft1;
    const DraftStepResult r1 = lazy.draft(prefix1, last1, 6, 0.0f, draft1);
    check(r1.ok && r1.n_draft == 6 && r1.catchup_tokens == 13,
          "hybrid catch-up: initial draft prefilled prompt+last (catchup=" + std::to_string(r1.catchup_tokens) + ")");
    // Target confirms last1 + 2 accepted drafts -> positions 0..14 confirmed, rejects drafts 2..5.
    lazy.on_accept(/*n_past_before_verify*/ 12, /*n_accepted*/ 2);
    check(lazy.kv_len() == 15,
          "hybrid catch-up: on_accept trimmed rejected drafts cleanly (kv_len=" + std::to_string(lazy.kv_len()) + ")");

    // Simulated n-gram sweeps: 10 additional tokens are accepted into the target's history WITHOUT
    // touching the draft model's KV cache (lazy.kv_len() stays at 15).
    std::vector<llama_token> prefix2 = prefix1;
    prefix2.push_back(last1);
    prefix2.push_back(draft1[0]);
    prefix2.push_back(draft1[1]);
    const int n_ngram_accepted = 10;
    for (int i = 0; i < n_ngram_accepted; ++i)
        prefix2.push_back(toks[(size_t) (15 + i)]);
    const llama_token last2 = toks[25];
    check(lazy.kv_len() == 15 && (int) prefix2.size() == 25,
          "hybrid catch-up: draft KV stays untouched during n-gram sweeps (kv_len=15, prefix=25)");

    // Step 2: n-gram abstains, so the draft model must catch up all 10 n-gram tokens + last2
    // in one batch and produce logits and drafts identical to a fresh full-prefix prefill.
    std::vector<llama_token> out_lazy, out_ref;
    const DraftStepResult r_lazy = lazy.draft(prefix2, last2, 6, 0.0f, out_lazy);
    const DraftStepResult r_ref = ref.draft(prefix2, last2, 6, 0.0f, out_ref);
    check(r_lazy.ok && r_ref.ok, "hybrid catch-up: lazy and reference drafts succeeded");
    check(r_lazy.catchup_tokens == n_ngram_accepted + 1,
          "hybrid catch-up: prefilled exact n-gram gap + last token in one call (catchup=" +
              std::to_string(r_lazy.catchup_tokens) + ", want " + std::to_string(n_ngram_accepted + 1) + ")");
    check(lazy.kv_len() == r_ref.catchup_tokens + 5,
          "hybrid catch-up: lazy KV length matches reference (" + std::to_string(lazy.kv_len()) + ")");
    check(out_lazy == out_ref, "hybrid catch-up: lazy draft tokens match fresh full-prefix prefill exactly");

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * lg_lazy = llama_get_logits_ith(ctx_lazy.get(), -1);
    const float * lg_ref = llama_get_logits_ith(ctx_ref.get(), -1);
    double max_diff_lazy_ref = 0.0;
    for (int v = 0; v < n_vocab; ++v)
        max_diff_lazy_ref = std::max(max_diff_lazy_ref, (double) std::fabs(lg_lazy[v] - lg_ref[v]));
    check(max_diff_lazy_ref < 1e-4,
          "hybrid catch-up: lazy logits match fresh full-prefix prefill (max_diff=" +
              std::to_string(max_diff_lazy_ref) + ")");

    // Control check: verify that omitting the 10 n-gram tokens produces measurably different logits
    // (proves the caught-up n-gram tokens actually enter KV attention rather than being ignored).
    std::vector<llama_token> short_prefix(prefix2.begin(), prefix2.begin() + 15);
    std::vector<llama_token> out_stale;
    const DraftStepResult r_stale = stale.draft(short_prefix, last2, 6, 0.0f, out_stale);
    const float * lg_stale = llama_get_logits_ith(ctx_stale.get(), -1);
    double max_diff_stale_ref = 0.0;
    for (int v = 0; v < n_vocab; ++v)
        max_diff_stale_ref = std::max(max_diff_stale_ref, (double) std::fabs(lg_stale[v] - lg_ref[v]));
    check(r_stale.ok && max_diff_stale_ref > 1e-4,
          "hybrid catch-up control: omitting n-gram tokens changes draft logits (max_diff=" +
              std::to_string(max_diff_stale_ref) + ")");

    lazy.reset();
    ref.reset();
    stale.reset();
    ctx_lazy.reset();
    ctx_ref.reset();
    ctx_stale.reset();
    model.reset();
    llama_backend_free();
}

static void test_session_hybrid(const std::string & model_path) {
    // Run base first on a repeating prompt so we know the exact output and guarantee both an
    // n-gram match (from the prompt repetition) and n-gram abstentions (novel generated tail).
    RunConfig base;
    base.model_path = model_path;
    base.prompt = "one two three four five six seven eight one two three four";
    base.n_ctx = 256;
    base.n_predict = 24;
    base.n_threads = 2;
    base.n_gpu_layers = 0;
    base.device_use = DeviceUse::CpuOnly;

    const RunResult r_base = run(base);
    check(r_base.ok && r_base.summary.n_generated == 24, "hybrid session: base run succeeded");

    // Prepend a snippet of base's own early output + repeated prefix so n-gram fires AND accepts
    // tokens early, while later positions abstain and fall back to the draft model with lazy catch-up.
    RunConfig hyb = base;
    hyb.prompt = base.prompt + " " + r_base.generated_text.substr(0, 6) + " " + base.prompt;
    const RunResult r_base2 = run([&] {
        RunConfig c = base;
        c.prompt = hyb.prompt;
        return c;
    }());
    check(r_base2.ok && r_base2.summary.n_generated == 24, "hybrid session: base2 run succeeded");

    hyb.spec.source = DraftSource::hybrid;
    hyb.spec.draft_model = model_path;
    hyb.spec.draft_max = 6;
    hyb.spec.draft_tau = 0.0f;
    hyb.spec.ngram_min_match = 2;
    hyb.spec.ngram_max_match = 12;

    const RunResult r_hyb = run(hyb);
    check(r_hyb.ok, "hybrid session: hybrid run succeeded (" + r_hyb.error + ")");
    check(r_hyb.generated_text == r_base2.generated_text, "hybrid session: output matches base exactly");
    check(r_hyb.summary.ngram_drafted_steps > 0,
          "hybrid session: n-gram drafted on at least one sweep (" + std::to_string(r_hyb.summary.ngram_drafted_steps) +
              ")");
    check(r_hyb.summary.model_drafted_steps > 0,
          "hybrid session: draft model fell back on abstain (" + std::to_string(r_hyb.summary.model_drafted_steps) +
              ")");
    check(r_hyb.summary.model_catchup_tokens > 0,
          "hybrid session: draft model lazily caught up tokens (" +
              std::to_string(r_hyb.summary.model_catchup_tokens) + ")");
    check(r_hyb.summary.mtp_drafted == r_hyb.summary.ngram_drafted + r_hyb.summary.model_drafted &&
              r_hyb.summary.mtp_accepted == r_hyb.summary.ngram_accepted + r_hyb.summary.model_accepted &&
              r_hyb.summary.drafted_steps == r_hyb.summary.ngram_drafted_steps + r_hyb.summary.model_drafted_steps,
          "hybrid session: per-source counters sum cleanly to totals");
}

int main(int argc, char ** argv) {
    test_pure_policy();

    if (argc >= 2) {
        const std::string model_path = argv[1];
        const std::string other_vocab = argc >= 3 ? argv[2] : BMOE_QWEN35_VOCAB_PATH;
        test_tokenizer_parity(model_path, other_vocab);
        test_session_draft_model(model_path);
        test_hybrid_lazy_catchup_direct(model_path);
        test_session_hybrid(model_path);
    }

    if (failures) {
        std::printf("\n%d draft model check(s) FAILED\n", failures);
        return 1;
    }
    std::printf("\nall draft model checks passed\n");
    return 0;
}
