# mtp-lend rebuild A/B — 2026-10-12

Same model/config throughout: abliterated IQ3_XXS, 262K ctx, `--kv-resident 16384`,
MTP spec=4, RTX 2080 Ti. Decode = 512-token-gen prompt (3 runs). Prefill = ~12,940-token
prompt (2 runs). Raw numbers in `baseline-pre-rebuild.json` / `post-rebuild-mtplend.json`.

## What changed between the two

| | baseline (pre-rebuild) | post-rebuild |
|---|---|---|
| engine binary | pre-merge (sm75-patches tip) | merged `mtp-lend` (b4b5c838) |
| `--mtp-lend` | off | **on** (swapped `--vram-elastic` → `--mtp-lend`) |
| sm75 cherry-picks | on (`STRATA_SELECT_SIMT=2`, `STRATA_PA_V2T=1`, `STRATA_REFILL_PARTS=1`) | on (same) |
| draft layer VRAM | 835 MiB (experts 675 resident) | **160 MiB** (experts 675 in pinned RAM) |
| prompt-path borrow | 3.50 GiB | **4.16 GiB** (+675 MiB = the MTP tail) |

## Results

| metric | baseline | post-rebuild | change |
|---|---|---|---|
| decode run1 | 43.3 tok/s | 40.9 tok/s | ~same (run-to-run) |
| decode run2 | 53.7 tok/s | 49.5 tok/s | ~same |
| decode run3 | 60.0 tok/s | 59.4 tok/s | ~same |
| prefill run1 | 414.1 tok/s | **815.6 tok/s** | **+97%** |
| prefill run2 | 414.1 tok/s | **817.3 tok/s** | **+98%** |

Decode is flat (as designed — mtp-lend only touches the prompt path). Prefill is ~2×.

## Interpretation

The 675 MiB the MTP experts no longer hold in VRAM lets the prompt path borrow a larger
chunk of the expert cache (4.16 vs 3.50 GiB). On this 11 GB card that roughly doubles
prompt throughput at ~13K-token prompts. Decode is unchanged, MTP acceptance is within
normal run-to-run spread.

Note: `--vram-elastic` had to be dropped to enable mtp-lend (it's an auto-off — elastic
shrink would unmap the cache tail). The model still fits 262K context without it
(verified at 19:44 in the log before this rebuild).

## vs. the 2080 Ti / IQ3_S community report

`bench/results/2026-10-02-community-rtx2080ti-11gb/` (homeofe, 2026-10-02). Same GPU
(2080 Ti 11 GB), same 262K ctx, same MTP spec=4 — but **not** a like-for-like config:

| | this A/B | community report |
|---|---|---|
| quant | **IQ3_XXS (abliterated)** | IQ3_S |
| mtp-lend | **on** | off |
| `--kv-resident` | 16,384 | 32,768 |
| CPU | i7-9700KF (8c, no HT) | Threadripper 3960X (24c/48t) |
| vram-reserve | 512 MiB | 700 MiB |

| metric | this A/B (IQ3_XXS, mtp-lend on) | community (IQ3_S) |
|---|---|---|
| prefill @ ~13K | **816 tok/s** | ~560 (interp. between 497@4K and 637@32K) |
| prefill @ 4K / 32K / 128K / 250K | — | 497 / 637 / 566 / 466 |
| decode | **~50–59 tok/s** (run2/3, @13K ctx) | 39.6 @4K → 34.1 @128K |

Read: my prefill at ~13K (~816) is ~40–45% above the community report's interpolated
13K figure and ~64% above its 4K number; decode is ~30–50% above their 4K figure.
**Not apples-to-apples** — three things favor me here: (1) IQ3_XXS is a smaller model
than IQ3_S (fewer bytes per expert → faster prompt compute), (2) mtp-lend is on,
(3) the community CPU is *stronger* (24c Zen 2 vs my 8c Coffee Lake), which should
have *helped* their CPU-bound expert refill — so the gap is mostly model size, not CPU.
Their `--kv-resident 32768` (vs my 16,384) would favor them at long context, but at
13K it doesn't dominate.

Caveat: the community report is methodologically heavier (3-run medians, foreign-traffic
guarding, 1 Hz telemetry, needle recall, image checks). My A/B is lighter (2–3 runs, no
telemetry). Treat the comparison as directional, not a benchmark.
