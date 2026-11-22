#pragma once

// Standalone resident draft model for speculative decoding (DraftSource::model / hybrid).
//
// Unlike MTP (which shares the target's hidden states via common/speculative.h), a standalone
// draft model is an independent causal LM with its own KV cache (e.g. Llama-3.2-1B-Instruct
// drafting for Llama-3.3-70B-Instruct). It uses only llama.cpp's public C API:
//   - check_tokenizer_parity() refuses to start if vocab size, special IDs (BOS/EOS/EOT),
//     control/EOG flags, token texts, or chat-template special tokens diverge from the target;
//   - DraftModelRunner maintains a lazily caught-up KV cache (`kv_len_`) and drafts greedily
//     while the cumulative top-1 probability product p1 * p2 * ... * pk stays >= tau.

#include "logits.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#ifdef BMOE_HAVE_LLAMA
#include "llama.h"
#endif

namespace bmoe::detail {

// Pure cumulative-confidence step: updates `next_cum_p = cum_p * p_top1` and returns whether
// extending with this token stays at or above the floor `tau`. `tau <= 0` disables the cutoff.
inline bool draft_cum_prob_step(double cum_p, double p_top1, float tau, double & next_cum_p) {
    next_cum_p = cum_p * p_top1;
    return tau <= 0.0f || next_cum_p >= (double) tau;
}

// Greedy argmax and exact softmax probability of the winning token from one logit row.
// Requires n_vocab >= 1.
inline void top1_token_and_prob(const float * logits, int n_vocab, int32_t & out_id, double & out_prob) {
    int32_t best = 0;
    float best_v = logits[0];
    for (int v = 1; v < n_vocab; ++v) {
        if (logits[v] > best_v) {
            best_v = logits[v];
            best = v;
        }
    }
    const LogNorm norm = log_norm(logits, n_vocab);
    out_id = best;
    out_prob = std::exp(norm.logp(best_v));
}

#ifdef BMOE_HAVE_LLAMA

struct TokenizerParity {
    bool ok = false;
    std::string error;
    std::string summary;
    int n_vocab = 0;
    llama_token bos = -1;
    llama_token eos = -1;
    llama_token eot = -1;
    int n_eog = 0;
    int n_control = 0;
    int n_template_specials = 0;
};

// Verify target and draft models share the exact same vocabulary: size, type, BOS/EOS/EOT IDs,
// token texts, control/EOG attributes, and tokenization of every special token referenced in
// the target's chat template. Refuses any mismatch before decoding begins.
TokenizerParity check_tokenizer_parity(const llama_model * tgt, const llama_model * dft);

struct DraftStepResult {
    bool ok = true;
    int n_draft = 0;
    int catchup_tokens = 0; // tokens prefilled into the draft KV before proposing drafts
    double final_cum_p = 1.0;
    std::string error;
};

// Manages KV state and synchronous greedy drafting for a standalone draft llama_context.
//
// Invariant: at the start of draft(), positions [0, kv_len_) in `ctx_`'s sequence-0 KV cache
// hold `prefix[0 .. kv_len_ - 1]`, and no cells exist at positions >= kv_len_.
class DraftModelRunner {
public:
    DraftModelRunner() = default;
    ~DraftModelRunner() { reset(); }

    DraftModelRunner(const DraftModelRunner &) = delete;
    DraftModelRunner & operator=(const DraftModelRunner &) = delete;

    // Bind to an already-created draft context (non-owning; call reset() before destroying ctx).
    void init(llama_context * ctx, const llama_vocab * vocab, int n_ubatch);
    void reset();

    bool active() const { return ctx_ != nullptr; }
    int kv_len() const { return kv_len_; }

    // Clear the draft KV cache completely and reset `kv_len_` to 0.
    void reset_kv();

    // Trim the draft KV cache so positions >= keep_len are removed. If the draft memory refuses
    // partial removal, clears the cache and sets `kv_len_` to 0 so the next draft re-prefills.
    void trim_kv(int keep_len);

    // Lazily prefill `prefix[kv_len_ .. prefix.size()-1]` + `last_tok` into the draft KV cache,
    // then draft up to `n_max` continuation tokens greedily while cumulative top-1 probability
    // stays >= `tau` (stopping after an EOG token or at `n_max`).
    DraftStepResult draft(const std::vector<llama_token> & prefix,
                          llama_token last_tok,
                          int n_max,
                          float tau,
                          std::vector<llama_token> & out);

    // After target verification of `[last_tok, out[0..n_draft-1]]` at `n_past_before_verify`
    // accepts `n_accepted` drafts, discard any rejected draft positions from the draft KV cache
    // while retaining `last_tok` and accepted drafts already decoded into it.
    void on_accept(int n_past_before_verify, int n_accepted);

private:
    llama_context * ctx_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    int n_vocab_ = 0;
    int n_ubatch_ = 0;
    llama_batch batch_{};
    bool batch_owned_ = false;
    int kv_len_ = 0;
};

#endif // BMOE_HAVE_LLAMA

} // namespace bmoe::detail
