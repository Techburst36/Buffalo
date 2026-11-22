# Build brief for Neon: resident 1B drafter for dense streaming (P1)

Repo: `Techburst36/Buffalo`, branch `m4-speed-wall`. Pre-registered line: `Decisions/LOG.md`,
section "speed P1-dense", including amendment 1. Speed numbers count only after the gates below pass.

## Why this build

On Llama-3.3-70B Q4_K_M with `--dense-stream`, one sweep costs the same for 1 position or ~17,
and only 1.2x at 33 (0A). So checking a wider draft is nearly free: extra positions cost
~2% of a sweep. Store drafting (`--ngram --draft 32 --ngram-min-match 2`) already gives
3.29x lossless on a code edit (10.19 → 3.10 s/token). Still, 14 of 18 sweeps were plain:
those steps write new text that the store can't copy. A resident small drafter is for exactly
those steps.

## Step 0: clean tree (before any P1 code)

1. Commit the local `draft_max_limit` 8 → 64 change in `core/include/bmoe/config.h` on
   `m4-speed-wall`, with a one-line rationale (0A).
2. Review the guards that assume the old limit: the `prefill.min_tokens` check
   (`core/src/config.cpp` ~line 288, `planner.cpp` ~line 2722) and the `n_ubatch >= draft + 1`
   rule. Fix or document them.
3. All `ctest` tests pass. Zero diffs to `third_party/llama.cpp` unless explicitly agreed.

## What to build

**Draft model.** Llama-3.2-1B-Instruct GGUF (Q8_0 preferred; Q4_K_M if VRAM is tight),
fully resident on the 3080, with its own KV cache. A new `DraftSource` next to `mtp` and `ngram`,
reusing the existing verify loop and catch-up plumbing in `core/src/engine/session.cpp`.

**Draft length: stop on cumulative confidence, not a fixed K.** Draft greedily from the 1B
model and keep extending while the product of its top-1 probabilities, p1 × p2 × … × pk,
stays ≥ τ. Defaults: τ = 0.10 and a hard cap of 32 (≤ `draft_max_limit`).
- New flags: `--draft-model PATH`, `--draft-tau F`, plus the existing `--draft N` as the cap.
- Log per sweep: draft length, accepted length.
- Easy stretches (syntax, boilerplate) should run long; uncertain ones stop early.

**Hybrid source.** Try the n-gram store first. Only when it abstains (no match ≥ `--ngram-min-match`)
does the 1B model draft. The 1B model's KV cache catches up **lazily**: right before its next
draft call, it prefills the tokens accepted since its last call, in one batch. It does not sync
after every n-gram sweep.

**Keep drafting synchronous in P1.** Sixteen 1B tokens take well under 100 ms against a ~10 s sweep,
so async overlap isn't worth its bugs yet.

## Gates (all must pass before any speed number counts)

1. **Tokenizer parity at load time.** Same vocabulary size and the same IDs for BOS, EOS,
   `<|eot_id|>` and every special token used by the chat template. Refuse to start otherwise.
2. **VRAM, no spill.** Measure peak device memory with `--gpu-layers 12` plus the drafter.
   It must fit in dedicated VRAM with ≥ 300 MiB headroom. On WSL2, spilling into Windows
   "shared GPU memory" silently slows CUDA, so check Task Manager's *Shared GPU memory* stays
   at baseline during a run. If it doesn't fit, use Q4_K_M or lower `--gpu-layers` and report it.
3. **Output identity at `--temp 0`, against `base`, on every P1 prompt.** The rule:
   - The output is identical to `base`, **or**
   - The first differing token sits on a near-tie: the target's top-2 logit gap at that
     position is < 0.05 in `base`. Wider verify batches change floating-point summation order,
     and Buffalo's own `mtp.md` shows batch width alone can flip near-ties.

   Any divergence that isn't a near-tie is a bug, and that arm's speed numbers don't count.
   Report the near-tie count per arm.
4. **Tests.** All `ctest` tests pass, plus a new test for the hybrid catch-up (tokens accepted
   via n-gram are in the 1B model's KV cache before its next draft).

## The P1 matrix

Model: Llama-3.3-70B-Instruct Q4_K_M, `--dense-stream --dense-cache-mb 5120 --gpu-layers 12 --temp 0 -n 128`.

| Arm | Flags |
| --- | --- |
| base | none |
| ngram | `--ngram --draft 32 --ngram-min-match 2` |
| draft-1b | `--draft-model <1B> --draft 32 --draft-tau 0.10` |
| hybrid | n-gram first, 1B fallback, same caps |

Prompts, fixed before running (22 total):
- **8 code generation:** the first 8 HumanEval problems.
- **8 edit/repair:**
  - 4 edits like the `merge_intervals` type-hint prompt;
  - 4 repairs, each giving a buggy function plus its failing test output and asking for the fixed function.
- **6 prose:** 3 explanations, 3 summaries of a pasted paragraph.

Time budget: ~15–20 h of machine time, since the base arm alone is ~22 min per prompt. Run in two overnight batches,
code/edit first. Record warm/cold page cache. One run per prompt per arm.

Report per arm and per slice:
- s/token and the speedup vs base;
- tokens per sweep;
- the share of sweeps drafted, and by which source (hybrid);
- draft acceptance;
- peak VRAM;
- the gate results.

**Line (from the LOG):** PASS if the best arm reaches ≥ 2.0x base wall-clock on the code slices
(generation and edit/repair pooled). Also reported: draft-1b vs hybrid. If hybrid ≤ draft-1b
on edit/repair, the store adds nothing over the 1B model there.

## Explicitly out of scope for P1

- MoE drafting. Measured slower at every width on gpt-oss (0.58–0.90x); it stays off.
- Speculative-cache-bypass for MoE: noted for later, only matters if MoE drafting returns.
- Async drafting, relaxed or judge acceptance, and any lossy lever. Those belong to the LADDER.
