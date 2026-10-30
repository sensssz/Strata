# confs archive

- `iq3_xxs.pre-switch-20261017.json` = copy of `~/Strata/strata-abliterated-iq3_xxs.json.bak-20261017`
  (breadcrumb written when the pack switch to iq3_s happened on 2026-10-17 13:46).
  This is the XXS config as it ran in the engine 0.1.44.1 era (2026-10-13/14 "ladder 8192,
  prefill 950-1000" window): 512 MiB reserve, kv-resident 16384, mtp-lend,
  conversation-cache 6144 + min-tokens 4096, 262144 ctx.
- Live configs (`../iq3_s.json`, `../iq3_xxs.json`) are the ACTIVE files read through
  `~/Strata/strata-current.json` (symlink flipped by `~/strata-switch.sh`).
- Rule: every conf edit gets committed on `mtp-lend` with WHY in the message
  (measured deltas -> reference `bench/results/` file names).
