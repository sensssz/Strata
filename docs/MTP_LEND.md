# `--mtp-lend`: the draft layer's experts lent to the prompt path (opt-in)

The MTP draft layer keeps its 512 routed experts (512 x 1,382,400 bytes = 675 MiB) in VRAM for the whole session,
but only a draft round reads them. Its prompt pass (the K/V of the prompt's cells: `Prefill::draft_kv`,
`MtpDrafter::prefill`) uses the dense weights alone. With `--mtp-lend` (or `STRATA_MTP_LEND=1`):

1. the drafter reads its experts into pinned RAM at load, and a placeholder holds their VRAM until the expert cache
   opens, so the cache gets the same number of slots as without the switch;
2. the cache opens with the experts' block as its **tail**, right after the last slot (`ExpertCache::set_tail_bytes`),
   and the drafter points at it (`MtpDrafter::attach_experts`, uploaded and read back once);
3. the prompt path's loan - the cache's last slots, lent for a prompt and refilled before any verify window - ends at
   the tail's end, so a prompt chunk needs ~675 MiB fewer cache slots, or `--prefill auto` affords a larger chunk under
   the same lend cap (85-90% of the slots);
4. after each prompt the experts are copied back from pinned RAM (blocking, before the lent slots are refilled), and
   decoding is unchanged: same slots, same kernels, the experts at the same address.

**Where it helps.** Prompt reading on cards whose auto chunk is held back by the lend cap: 8-12 GB cards (an RTX
2080 Ti's 3.25 GiB cache gains 20% more lendable bytes). A card whose chunk is already at its ceiling saves only the
~675 MiB per chunk it no longer streams. Decode speed does not change. Not measured yet - compare with and without
the switch on your own card (`STRATA_TRACE=1` prints the chunk, the loan and the copy-back time).

**Costs.** 675 MiB of pinned RAM, and a 675 MiB copy after each prompt that borrowed the cache (~30 ms on PCIe 4.0
x16, ~55 ms on PCIe 3.0 x16) in place of refilling about as many bytes of lent cache slots.

**Off, with a line saying why, on:** a layer split, `--batch-mtp`, `--pipeline-windows`, `--vram-elastic`, the elastic
K/V (`--kv-grow`), helper expert caches on other GPUs, `--no-prefill-borrow`, a GPU that shares the system's memory,
or when the pinned host copy cannot be allocated. Without an expert cache the experts get a block of their own, as
without the switch.

**Checks.** The experts are read back and compared once at start; `STRATA_MTP_LEND_VERIFY=1` compares them after every
copy-back. A draft round that would read them while they are lent fails with "the draft layer's experts are lent to the
prompt path" instead of drafting from the prompt path's buffers. `tests/program/mtp_lend_rules_test.cpp` checks the
decision and the loan arithmetic.

**Code.** `include/strata/program/mtp_lend.hpp` (the hooks), `include/strata/program/mtp_lend_rules.hpp` (the pure
rules), `src/core/mtp_lend.inl` (the drafter's side), `ExpertCache::set_tail_bytes` / `lendable_bytes_from`, and
one-line calls in `src/program/generate.cpp`.
