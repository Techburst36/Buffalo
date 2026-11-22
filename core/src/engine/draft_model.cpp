#include "draft_model.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>

namespace bmoe::detail {

namespace {

const char * safe_text(const llama_vocab * v, llama_token id) {
    const char * s = llama_vocab_get_text(v, id);
    return s ? s : "";
}

std::vector<llama_token> tokenize_literal(const llama_vocab * vocab, const std::string & s) {
    if (s.empty()) return {};
    std::vector<llama_token> out(s.size() + 8);
    int n = llama_tokenize(vocab, s.c_str(), (int) s.size(), out.data(), (int) out.size(),
                           /*add_special*/ false, /*parse_special*/ true);
    if (n < 0) {
        out.resize((size_t) (-n));
        n = llama_tokenize(vocab, s.c_str(), (int) s.size(), out.data(), (int) out.size(), false, true);
    }
    if (n <= 0) return {};
    out.resize((size_t) n);
    return out;
}

std::string format_tokens(const std::vector<llama_token> & toks) {
    std::string s = "[";
    for (size_t i = 0; i < toks.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(toks[i]);
    }
    return s + "]";
}

// Collect special-token literals referenced by `tmpl`: every `<|...|>` marker plus every
// control / user-defined token text from `vocab` (length >= 2) that appears in the template.
std::set<std::string> template_special_literals(const llama_vocab * vocab, int n_vocab, const char * tmpl) {
    std::set<std::string> out;
    if (!tmpl || !*tmpl) return out;
    const std::string t(tmpl);

    // 1. Angle-pipe markers `<|...|>` (Llama-3, Qwen, GPT-OSS, etc.)
    size_t pos = 0;
    while ((pos = t.find("<|", pos)) != std::string::npos) {
        const size_t end = t.find("|>", pos + 2);
        if (end == std::string::npos) break;
        const size_t len = end + 2 - pos;
        if (len <= 64) out.insert(t.substr(pos, len));
        pos = end + 2;
    }

    // 2. Any control / user-defined token string present in the template (e.g. <s>, </s>, [INST]).
    for (llama_token id = 0; id < n_vocab; ++id) {
        const auto attr = llama_vocab_get_attr(vocab, id);
        const bool special = llama_vocab_is_control(vocab, id) || llama_vocab_is_eog(vocab, id) ||
                             (attr & LLAMA_TOKEN_ATTR_USER_DEFINED) != 0;
        if (!special) continue;
        const char * txt = safe_text(vocab, id);
        const size_t len = std::strlen(txt);
        if (len < 2) continue;
        if (t.find(txt) != std::string::npos) out.emplace(txt, len);
    }
    return out;
}

} // namespace

TokenizerParity check_tokenizer_parity(const llama_model * tgt, const llama_model * dft) {
    TokenizerParity r;
    auto fail = [&](std::string msg) {
        r.ok = false;
        r.error = std::move(msg);
        return r;
    };
    if (!tgt || !dft) return fail("tokenizer parity: missing target or draft model");

    const llama_vocab * vt = llama_model_get_vocab(tgt);
    const llama_vocab * vd = llama_model_get_vocab(dft);
    if (!vt || !vd) return fail("tokenizer parity: missing target or draft vocabulary");

    const int nt = llama_vocab_n_tokens(vt);
    const int nd = llama_vocab_n_tokens(vd);
    if (nt != nd) {
        return fail("draft model tokenizer mismatch: vocab size differs (target=" + std::to_string(nt) +
                    ", draft=" + std::to_string(nd) + ")");
    }
    const enum llama_vocab_type tt = llama_vocab_type(vt);
    const enum llama_vocab_type td = llama_vocab_type(vd);
    if (tt != td) {
        return fail("draft model tokenizer mismatch: vocab type differs (target=" + std::to_string((int) tt) +
                    ", draft=" + std::to_string((int) td) + ")");
    }

    const llama_token bos_t = llama_vocab_bos(vt), bos_d = llama_vocab_bos(vd);
    if (bos_t != bos_d) {
        return fail("draft model tokenizer mismatch: BOS token ID differs (target=" + std::to_string(bos_t) +
                    ", draft=" + std::to_string(bos_d) + ")");
    }
    const llama_token eos_t = llama_vocab_eos(vt), eos_d = llama_vocab_eos(vd);
    if (eos_t != eos_d) {
        return fail("draft model tokenizer mismatch: EOS token ID differs (target=" + std::to_string(eos_t) +
                    ", draft=" + std::to_string(eos_d) + ")");
    }
    const llama_token eot_t = llama_vocab_eot(vt), eot_d = llama_vocab_eot(vd);
    if (eot_t != eot_d) {
        return fail("draft model tokenizer mismatch: EOT token ID differs (target=" + std::to_string(eot_t) +
                    ", draft=" + std::to_string(eot_d) + ")");
    }

    int n_eog = 0;
    int n_control = 0;
    for (llama_token id = 0; id < nt; ++id) {
        const char * st = safe_text(vt, id);
        const char * sd = safe_text(vd, id);
        if (std::strcmp(st, sd) != 0) {
            return fail("draft model tokenizer mismatch: token ID " + std::to_string(id) + " text differs (target='" +
                        st + "', draft='" + sd + "')");
        }
        const bool eog_t = llama_vocab_is_eog(vt, id);
        const bool eog_d = llama_vocab_is_eog(vd, id);
        if (eog_t != eog_d) {
            return fail("draft model tokenizer mismatch: token ID " + std::to_string(id) + " ('" + st +
                        "') EOG flag differs (target=" + std::to_string((int) eog_t) +
                        ", draft=" + std::to_string((int) eog_d) + ")");
        }
        const bool ctl_t = llama_vocab_is_control(vt, id);
        const bool ctl_d = llama_vocab_is_control(vd, id);
        if (ctl_t != ctl_d) {
            return fail("draft model tokenizer mismatch: token ID " + std::to_string(id) + " ('" + st +
                        "') control flag differs (target=" + std::to_string((int) ctl_t) +
                        ", draft=" + std::to_string((int) ctl_d) + ")");
        }
        if (eog_t) ++n_eog;
        if (ctl_t) ++n_control;
    }

    // Verify every special token referenced in the target's chat template tokenizes identically
    // on both models (including <|eot_id|>, <|start_header_id|>, <|end_header_id|>, etc.).
    const char * tmpl = llama_model_chat_template(tgt, nullptr);
    const std::set<std::string> tmpl_specials = template_special_literals(vt, nt, tmpl);
    for (const std::string & lit : tmpl_specials) {
        const std::vector<llama_token> tok_t = tokenize_literal(vt, lit);
        const std::vector<llama_token> tok_d = tokenize_literal(vd, lit);
        if (tok_t != tok_d) {
            return fail("draft model tokenizer mismatch: chat-template special '" + lit +
                        "' tokenizes differently (target=" + format_tokens(tok_t) +
                        ", draft=" + format_tokens(tok_d) + ")");
        }
    }

    r.ok = true;
    r.n_vocab = nt;
    r.bos = bos_t;
    r.eos = eos_t;
    r.eot = eot_t;
    r.n_eog = n_eog;
    r.n_control = n_control;
    r.n_template_specials = (int) tmpl_specials.size();

    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "vocab=%d bos=%d eos=%d eot=%d eog=%d control=%d template_specials=%d",
                  r.n_vocab, (int) r.bos, (int) r.eos, (int) r.eot, r.n_eog, r.n_control, r.n_template_specials);
    r.summary = buf;
    return r;
}

void DraftModelRunner::init(llama_context * ctx, const llama_vocab * vocab, int n_ubatch) {
    reset();
    if (!ctx || !vocab) return;
    ctx_ = ctx;
    vocab_ = vocab;
    n_vocab_ = llama_vocab_n_tokens(vocab);
    n_ubatch_ = std::max(1, n_ubatch);
    batch_ = llama_batch_init(n_ubatch_, /*embd*/ 0, /*n_seq_max*/ 1);
    batch_owned_ = true;
    kv_len_ = 0;
}

void DraftModelRunner::reset() {
    if (batch_owned_) {
        llama_batch_free(batch_);
        batch_ = {};
        batch_owned_ = false;
    }
    ctx_ = nullptr;
    vocab_ = nullptr;
    n_vocab_ = 0;
    n_ubatch_ = 0;
    kv_len_ = 0;
}

void DraftModelRunner::reset_kv() {
    if (ctx_) llama_memory_clear(llama_get_memory(ctx_), true);
    kv_len_ = 0;
}

void DraftModelRunner::trim_kv(int keep_len) {
    if (!ctx_) {
        kv_len_ = 0;
        return;
    }
    if (keep_len <= 0) {
        reset_kv();
        return;
    }
    if (keep_len >= kv_len_) return;
    if (!llama_memory_seq_rm(llama_get_memory(ctx_), /*seq*/ 0, (llama_pos) keep_len, -1)) {
        reset_kv();
        return;
    }
    kv_len_ = keep_len;
}

DraftStepResult DraftModelRunner::draft(const std::vector<llama_token> & prefix,
                                        llama_token last_tok,
                                        int n_max,
                                        float tau,
                                        std::vector<llama_token> & out) {
    DraftStepResult r;
    out.clear();
    if (!ctx_ || !vocab_ || n_vocab_ <= 0) {
        r.ok = false;
        r.error = "draft model runner is not initialised";
        return r;
    }
    if (n_max <= 0) return r;

    const int p_len = (int) prefix.size();
    if (kv_len_ > p_len) trim_kv(p_len);

    // Lazy KV catch-up: feed every confirmed token not yet in the draft context plus `last_tok`
    // in one batched sequence (chunked only by n_ubatch_), requesting logits on the final token.
    const int n_catchup = (p_len - kv_len_) + 1;
    int offset = 0;
    while (offset < n_catchup) {
        const int chunk = std::min(n_ubatch_, n_catchup - offset);
        batch_.n_tokens = chunk;
        for (int j = 0; j < chunk; ++j) {
            const int idx = kv_len_ + j;
            batch_.token[j] = idx < p_len ? prefix[(size_t) idx] : last_tok;
            batch_.pos[j] = idx;
            batch_.n_seq_id[j] = 1;
            batch_.seq_id[j][0] = 0;
            batch_.logits[j] = (j == chunk - 1) ? 1 : 0;
        }
        if (llama_decode(ctx_, batch_) != 0) {
            r.ok = false;
            r.error = "draft model decode failed during KV catch-up";
            return r;
        }
        kv_len_ += chunk;
        offset += chunk;
    }
    r.catchup_tokens = n_catchup;

    // Greedy autoregressive drafting: keep extending while cumulative top-1 probability
    // p1 * p2 * ... * pk stays >= tau, stopping at n_max or right after proposing an EOG token.
    double cum_p = 1.0;
    for (int k = 0; k < n_max; ++k) {
        const float * lg = llama_get_logits_ith(ctx_, -1);
        if (!lg) {
            r.ok = false;
            r.error = "draft model produced null logits";
            return r;
        }
        int32_t best_id = 0;
        double p_top1 = 0.0;
        top1_token_and_prob(lg, n_vocab_, best_id, p_top1);

        double next_cum_p = cum_p;
        if (!draft_cum_prob_step(cum_p, p_top1, tau, next_cum_p)) break;

        cum_p = next_cum_p;
        out.push_back((llama_token) best_id);
        r.final_cum_p = cum_p;

        if (llama_vocab_is_eog(vocab_, (llama_token) best_id)) break;
        if (k + 1 >= n_max) break;

        batch_.n_tokens = 1;
        batch_.token[0] = (llama_token) best_id;
        batch_.pos[0] = kv_len_;
        batch_.n_seq_id[0] = 1;
        batch_.seq_id[0][0] = 0;
        batch_.logits[0] = 1;
        if (llama_decode(ctx_, batch_) != 0) {
            r.ok = false;
            r.error = "draft model decode failed during drafting";
            return r;
        }
        ++kv_len_;
    }

    r.n_draft = (int) out.size();
    return r;
}

void DraftModelRunner::on_accept(int n_past_before_verify, int n_accepted) {
    // Positions [0, n_past_before_verify + 1 + n_accepted) are confirmed (`prefix` + `last_tok` +
    // accepted drafts). Discard any further positions the drafter decoded past the first mismatch.
    trim_kv(n_past_before_verify + 1 + n_accepted);
}

} // namespace bmoe::detail
