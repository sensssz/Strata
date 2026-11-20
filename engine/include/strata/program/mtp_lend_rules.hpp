// include/strata/program/mtp_lend_rules.hpp - --mtp-lend's decision and loan arithmetic (pure: no GPU, no
// environment read).  The hooks that use them, and what the switch does, are in strata/program/mtp_lend.hpp;
// tests/program/mtp_lend_rules_test.cpp checks them.
#pragma once

#include "strata/program/split_ring.hpp"

#include <algorithm>
#include <cstdint>

namespace strata::program::mtp_lend {

// ---- pure (no GPU): the decision and the loan arithmetic

/// What generate.cpp knows when the drafter loads.
struct Config {
    bool asked = false;              ///< --mtp-lend (STRATA_MTP_LEND=0/1 overrides)
    bool drafter = false;            ///< a draft layer is loaded (--mtp with --spec >= 2)
    bool multi_gpu = false;          ///< a layer split
    bool batch_mtp = false;          ///< --batch-mtp: slot drafters share the experts and draft between prompt chunks
    bool pipeline = false;           ///< --pipeline-windows >= 2: chains in flight across windows
    bool vram_elastic = false;       ///< --vram-elastic: the cache's tail segments are given back
    bool cache_vmm = false;          ///< the elastic K/V: the cache in a VMM range
    bool helper_caches = false;      ///< --expert-cache-device1..3
    bool no_borrow = false;          ///< --no-prefill-borrow: nothing is ever lent
    bool integrated = false;         ///< unified memory: no PCIe copy to save, the host copy would hold it twice
};

/// The switch's value: STRATA_MTP_LEND (when set and not empty) wins over the flag.
inline bool asked(bool flag, const char* env) { return env != nullptr && env[0] != '\0' ? env[0] != '0' : flag; }

/// Null: on.  Otherwise why it is off (only said when it was asked for).
inline const char* why_off(const Config& c) {
    if (!c.asked) return "not asked for";
    if (!c.drafter) return "there is no draft layer (--mtp with --spec 2 or more)";
    if (c.multi_gpu) return "a layer split (one GPU only for now)";
    if (c.batch_mtp) return "--batch-mtp (its slot drafters draft between a prompt's chunks)";
    if (c.pipeline) return "--pipeline-windows";
    if (c.vram_elastic) return "--vram-elastic (it gives the cache's tail back)";
    if (c.cache_vmm) return "the elastic K/V (--kv-grow)";
    if (c.helper_caches) return "helper expert caches on other GPUs";
    if (c.no_borrow) return "--no-prefill-borrow (nothing is lent)";
    if (c.integrated) return "the GPU shares the system's memory (nothing crosses PCIe)";
    return nullptr;
}

/// The cache's last slots a loan of `need` bytes takes when `extra` bytes (the tail and its alignment pad) follow
/// them - split_lend_slots' arithmetic on what the tail leaves.  Never 0: a loan always takes at least one slot, so a
/// count > 0 keeps meaning "a loan" to every caller (and the tail alone never has to be laid out from slot `slots`).
inline int64_t slots_for_need(uint64_t need, uint64_t extra, int64_t slots, int64_t slot_bytes, const uint64_t* offsets,
                              uint64_t max_blob) {
    const uint64_t rest = need > extra ? need - extra : 0;
    return std::max<int64_t>(1, split_lend_slots(slots, slot_bytes, offsets, max_blob, rest));
}

}  // namespace strata::program::mtp_lend
