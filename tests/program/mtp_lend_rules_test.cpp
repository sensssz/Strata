// --mtp-lend's decision and loan arithmetic (strata/program/mtp_lend_rules.hpp): no GPU, no model.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "strata/program/mtp_lend_rules.hpp"

using namespace strata::program::mtp_lend;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // ---- the switch: STRATA_MTP_LEND (set, not empty) wins over the flag
    CHECK(!asked(false, nullptr));
    CHECK(asked(true, nullptr));
    CHECK(asked(true, ""));
    CHECK(!asked(true, "0"));
    CHECK(asked(false, "1"));

    // ---- the decision: on only when asked, with a drafter, and none of the v1 exclusions
    Config c;
    CHECK(why_off(c) != nullptr);                       // not asked: off (and nothing is said)
    c.asked = true;
    CHECK(why_off(c) != nullptr);                       // no draft layer
    c.drafter = true;
    CHECK(why_off(c) == nullptr);                       // one GPU, nothing else: on
    for (bool Config::*f : {&Config::multi_gpu, &Config::batch_mtp, &Config::pipeline, &Config::vram_elastic,
                            &Config::cache_vmm, &Config::helper_caches, &Config::no_borrow, &Config::integrated}) {
        Config d = c;
        d.*f = true;
        CHECK(why_off(d) != nullptr);
    }

    // ---- the loan: uniform slots of 1000 bytes, a 600-byte tail (with its pad) after the last of 10 slots
    const uint64_t blob = 1000;
    // without a tail it is split_lend_slots exactly (but never 0)
    CHECK(slots_for_need(2500, 0, 10, 10000, nullptr, blob) == 3);
    CHECK(slots_for_need(3000, 0, 10, 10000, nullptr, blob) == 3);
    CHECK(slots_for_need(0, 0, 10, 10000, nullptr, blob) == 1);
    // the tail covers the first 600 bytes of any loan
    CHECK(slots_for_need(2500, 600, 10, 10000, nullptr, blob) == 2);   // 1900 left -> 2 slots
    CHECK(slots_for_need(2600, 600, 10, 10000, nullptr, blob) == 2);   // 2000 left -> 2 slots
    CHECK(slots_for_need(2601, 600, 10, 10000, nullptr, blob) == 3);
    // a loan the tail alone holds still takes one slot: a count > 0 keeps meaning "a loan" to the callers
    CHECK(slots_for_need(500, 600, 10, 10000, nullptr, blob) == 1);
    // more than the cache holds: the callers' "k + 128 > slots" test refuses it, as before
    CHECK(slots_for_need(50000, 600, 10, 10000, nullptr, blob) > 10);

    // ---- sized slots (a native pack): per-slot offsets, slots taken from the end until they hold what is left
    const std::vector<uint64_t> off = {0, 1024, 2048, 3328, 4608, 5888};   // 5 slots: 1024, 1024, 1280, 1280, 1280
    const int64_t bytes = (int64_t) off.back();
    CHECK(slots_for_need(1280, 0, 5, bytes, off.data(), 1280) == 1);
    CHECK(slots_for_need(1281, 0, 5, bytes, off.data(), 1280) == 2);
    CHECK(slots_for_need(1281, 1, 5, bytes, off.data(), 1280) == 1);
    CHECK(slots_for_need(3840 + 700, 700, 5, bytes, off.data(), 1280) == 3);
    CHECK(slots_for_need(3841 + 700, 700, 5, bytes, off.data(), 1280) == 4);
    CHECK(slots_for_need(100000, 700, 5, bytes, off.data(), 1280) == 5);   // all of them (the caller refuses it)

    if (failed == 0) std::printf("mtp_lend_rules_test: ok\n");
    return failed == 0 ? 0 : 1;
}
