// include/strata/program/mtp_lend.hpp - --mtp-lend (opt-in): the MTP draft layer's experts lent to the prompt path.
//
// The draft layer keeps its 512 routed experts (675 MiB) in VRAM for the whole session, but only a draft round reads
// them: its prompt pass (the K/V of the prompt's cells) uses the dense weights alone.  With --mtp-lend (or
// STRATA_MTP_LEND=1):
//
//   1. the drafter loads its experts into pinned RAM (MtpDrafter::set_defer_experts) and their VRAM is held by a
//      placeholder until the expert cache opens, so every cache-sizing rule sees the free VRAM it saw before;
//   2. the cache opens with the experts' block as its TAIL (ExpertCache::set_tail_bytes), right after the last slot,
//      and the drafter points at it (MtpDrafter::attach_experts): the cache's slot count is what it was;
//   3. the prompt path's loan - the cache's last slots, refilled before any verify window - now ends at the tail's
//      end, so a chunk needs ~675 MiB fewer cache slots (or a larger chunk fits the same lend cap);
//   4. the experts are copied back from pinned RAM wherever the lent slots are (MtpDrafter::restore_experts).
//
// Decode is unchanged: the experts are back before any draft round, at the same address (the captured graphs keep
// it).  v1: one GPU, no --batch-mtp, no --vram-elastic, no elastic K/V, no helper caches; each turns it off with a
// line saying so.  The default (switch off) runs none of this: every hook below returns at once.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/mtp.hpp"
#include "strata/program/mtp_lend_rules.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::program::mtp_lend {

// ---- the hooks generate.cpp calls (each returns at once when the switch is off)

struct State {
    bool on = false;            ///< the drafter's experts are deferred to the cache's tail
    void* hold = nullptr;       ///< the tail's VRAM, held from the drafter's load until the cache opens
    State() = default;
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    ~State() { if (hold != nullptr) cudaFree(hold); }
};

inline bool device_integrated() {
#if defined(STRATA_HIP_GFX906)   // (the gfx906 compat layer has no cudaDevAttrIntegrated; that GPU is discrete)
    return false;
#else
    int dev = 0, v = 0;
    return cudaGetDevice(&dev) == cudaSuccess && cudaDeviceGetAttribute(&v, cudaDevAttrIntegrated, dev) == cudaSuccess && v != 0;
#endif
}

/// Before MtpDrafter::load: decide, and defer the experts when on.
inline void before_load(State& s, core::MtpDrafter& mtp, const Config& c) {
    const char* why = why_off(c);
    s.on = why == nullptr;
    if (!s.on && c.asked) std::fprintf(stderr, "strata generate: --mtp-lend is off: %s\n", why);
    mtp.set_defer_experts(s.on);
}

/// After MtpDrafter::load: hold the experts' VRAM until the cache opens (the cache is sized from the free VRAM, which
/// must be what it was with the experts in VRAM).
inline bool after_load(State& s, core::MtpDrafter& mtp, std::string& err) {
    if (!s.on) return true;
    if (!mtp.defer_experts()) { s.on = false; return true; }   // no pinned RAM: load() kept them in VRAM (said)
    if (cudaMalloc(&s.hold, (size_t) mtp.expert_bytes()) != cudaSuccess) {
        (void) cudaGetLastError();
        s.hold = nullptr;
        err = "mtp: the 512 experts do not fit in VRAM";
        return false;
    }
    return true;
}

/// Before each ExpertCache::open/open_sized attempt: give the held VRAM to the cache's tail.
inline void before_cache_open(State& s, core::ExpertCache& xc, const core::MtpDrafter& mtp) {
    if (!s.on) return;
    if (s.hold != nullptr) { cudaFree(s.hold); s.hold = nullptr; }
    xc.set_tail_bytes(mtp.expert_bytes());
}

/// After the cache is open (or was not made), before the prompt path, bind() and any capture: the drafter's experts
/// into the tail, or - no cache, no tail - into a block of their own (the switch is then off).
inline bool attach(State& s, core::MtpDrafter& mtp, core::ExpertCache& xc, std::string& err) {
    if (!mtp.defer_experts()) return true;
    if (s.hold != nullptr) { cudaFree(s.hold); s.hold = nullptr; }
    uint8_t* dev = xc.valid() && xc.tail_bytes() >= mtp.expert_bytes() ? xc.tail() : nullptr;
    if (dev == nullptr) {
        s.on = false;
        std::fprintf(stderr, "strata generate: --mtp-lend is off: no expert cache to hold the draft layer's experts\n");
    }
    if (!mtp.attach_experts(dev, err)) return false;
    if (dev != nullptr)
        std::fprintf(stderr, "strata generate: --mtp-lend: the draft layer's experts (%.0f MiB) are the expert cache's "
                             "tail: the prompt path borrows them with its last slots and they are copied back after "
                             "each prompt (verified)\n", (double) mtp.expert_bytes() / 1048576.0);
    return true;
}

/// The generate path's slots_from_bytes / the serve path's cache_slots_for, when the cache has a tail.
inline int64_t slots_for_need(const core::ExpertCache& xc, uint64_t need, uint64_t max_blob) {
    return slots_for_need(need, xc.lendable_bytes_from(xc.slots()), xc.slots(), xc.bytes(), xc.slot_offsets(), max_blob);
}

/// The generate path's bytes_from_slots, when the cache has a tail: the last k slots and the tail.
inline uint64_t bytes_of_last(const core::ExpertCache& xc, int64_t k) {
    return xc.lendable_bytes_from(k <= 0 ? xc.slots() : xc.slots() - k);
}

/// The prompt path is about to lay its buffers out over the loan (and so over the tail).
inline bool lend(State& s, core::MtpDrafter& mtp, std::string& err) {
    if (!s.on || !mtp.experts_lendable() || !mtp.experts_resident()) return true;
    return mtp.lend_experts(err);
}

/// Before any verify window reads (wherever the lent slots are refilled): the experts back from pinned RAM.
inline bool refill(State& s, core::MtpDrafter& mtp, std::string& err) {
    if (!s.on || mtp.experts_resident()) return true;
    static const bool trace = [] { const char* v = std::getenv("STRATA_TRACE"); return v != nullptr && v[0] != '0'; }();
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = mtp.restore_experts(err);   // blocking: done when it returns
    if (ok && trace)
        std::fprintf(stderr, "strata trace: --mtp-lend: the draft layer's experts copied back in %.1f ms\n",
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return ok;
}

/// The one-shot generate path: the loan it lays out from slot `first` covers the tail too.
inline bool lend_oneshot(State& s, core::MtpDrafter& mtp, const core::ExpertCache& xc, int64_t first,
                         uint64_t& borrow_bytes, std::string& err) {
    if (!s.on || xc.tail_bytes() == 0) return true;
    borrow_bytes = xc.lendable_bytes_from(first);
    return lend(s, mtp, err);
}

}  // namespace strata::program::mtp_lend
