// src/core/mtp_lend.inl - --mtp-lend (opt-in): the MTP draft layer's routed experts lent to the prompt path.
//
// Included at the end of mtp.cpp (it defines MtpDrafter members); the API is in include/strata/core/mtp.hpp.
//
// The draft layer reads its 512 routed experts (512 x BLOB = 675 MiB) only in a draft round (record_rest); its prompt
// pass - the K/V of the prompt's cells, MtpDrafter::prefill and Prefill::draft_kv - uses the dense weights alone.  So
// with --mtp-lend the experts live in the expert cache's tail (ExpertCache::set_tail_bytes), the prompt path's loan
// covers that tail together with the cache's last slots, and the experts are copied back from a pinned host copy
// before the next draft round (strata/program/mtp_lend.hpp drives it).  Without the switch none of this runs.

namespace strata::core {

namespace {
bool mtp_lend_verify_env() {   // STRATA_MTP_LEND_VERIFY=1 (diagnostic): read the block back after every copy
    static const bool on = [] { const char* v = std::getenv("STRATA_MTP_LEND_VERIFY"); return v != nullptr && v[0] == '1'; }();
    return on;
}

// The block on the device against the host copy, byte for byte, 64 MiB at a time.
bool mtp_lend_compare(const uint8_t* dev, const uint8_t* host, uint64_t bytes, std::string& err) {
    std::vector<uint8_t> back((size_t) std::min<uint64_t>(bytes, 64ull << 20));
    for (uint64_t off = 0; off < bytes;) {
        const uint64_t n = std::min<uint64_t>(back.size(), bytes - off);
        if (cudaMemcpy(back.data(), dev + off, (size_t) n, cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = std::string("mtp: --mtp-lend: reading the experts back failed: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        if (std::memcmp(back.data(), host + off, (size_t) n) != 0) {
            err = "mtp: --mtp-lend: the experts on the device differ from the host copy near byte " + std::to_string(off);
            return false;
        }
        off += n;
    }
    return true;
}
}  // namespace

uint64_t MtpDrafter::expert_bytes() const {
    return g_ != nullptr ? (uint64_t) g_->n_expert * (uint64_t) strata::kernels::cpu::BLOB : 0;
}

bool MtpDrafter::load_experts_host(const std::string& rt_dir, const ModelGeometry& g, std::string& err) {
    const uint64_t bytes = (uint64_t) g.n_expert * (uint64_t) strata::kernels::cpu::BLOB;
    if (cudaHostAlloc((void**) &h_experts_, (size_t) bytes, cudaHostAllocPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        h_experts_ = nullptr;
        defer_experts_ = false;   // load() puts them in VRAM, as without --mtp-lend
        std::fprintf(stderr, "strata mtp: --mtp-lend is off: no pinned RAM for the experts' host copy (%.0f MiB); "
                             "they stay in VRAM\n", (double) bytes / 1048576.0);
        return true;
    }
    FILE* f = std::fopen((rt_dir + "/experts.bin").c_str(), "rb");
    if (f == nullptr) { err = "mtp: cannot open experts.bin"; return false; }
    struct Closer {
        FILE* f;
        ~Closer() { if (f != nullptr) std::fclose(f); }
    } closer{f};
#if !defined(_WIN32)
    strata::platform::advise_willneed(fileno(f), 0, bytes);
#endif
    for (uint64_t off = 0; off < bytes;) {   // one fread per 64 MiB (MSVC's stream splits smaller ones: loaderfix.2)
        const uint64_t n = std::min<uint64_t>(64ull << 20, bytes - off);
        if (std::fread(h_experts_ + off, 1, (size_t) n, f) != (size_t) n) { err = "mtp: experts.bin is truncated"; return false; }
        off += n;
    }
    experts_ = nullptr;
    experts_resident_ = false;   // until attach_experts
    std::fprintf(stderr, "strata mtp: --mtp-lend: the %lld experts (%.0f MiB) wait in pinned RAM for the expert cache's "
                         "tail\n", (long long) g.n_expert, (double) bytes / 1048576.0);
    return true;
}

bool MtpDrafter::attach_experts(uint8_t* dev, std::string& err) {
    if (!defer_experts_ || h_experts_ == nullptr) return true;   // nothing was deferred
    const OnDevice on_device(device_);
    const uint64_t bytes = expert_bytes();
    if (dev == nullptr) {   // no tail was made: a block of its own, as without --mtp-lend
        if (cudaMalloc((void**) &experts_, (size_t) bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            experts_ = nullptr;
            err = "mtp: the 512 experts do not fit in VRAM";
            return false;
        }
        experts_borrowed_ = false;
        vram_ += bytes;
    } else {
        experts_ = dev;
        experts_borrowed_ = true;
    }
    if (cudaMemcpy(experts_, h_experts_, (size_t) bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        err = std::string("mtp: --mtp-lend: uploading the experts failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    // read back once: a block that is right about its address and wrong about its bytes drafts plausible tokens that
    // the verify window then rejects - slower, never wrong, and so easy to miss (expert_cache.hpp: verify_slot)
    if (!mtp_lend_compare(experts_, h_experts_, bytes, err)) return false;
    experts_resident_ = true;
    if (!experts_borrowed_) {   // nothing to lend: the host copy has no use
        cudaFreeHost(h_experts_);
        h_experts_ = nullptr;
        defer_experts_ = false;
    }
    return true;
}

bool MtpDrafter::lend_experts(std::string& err) {
    if (!experts_borrowed_ || !experts_resident_) return true;
    if (chain_live_) { err = "mtp: --mtp-lend: a pipelined chain is still in flight"; return false; }
    const OnDevice on_device(device_);
    if ((cs_ && cudaStreamSynchronize(cs_) != cudaSuccess) || (side_ && cudaStreamSynchronize(side_) != cudaSuccess)) {
        err = "mtp: its stream failed";
        return false;
    }
    experts_resident_ = false;
    return true;
}

bool MtpDrafter::restore_experts(std::string& err) {
    if (!experts_borrowed_ || experts_resident_) return true;
    const OnDevice on_device(device_);
    const uint64_t bytes = expert_bytes();
    // blocking, from pinned RAM: the prompt path's stream has finished with the region (its part ended with a sync)
    if (cudaMemcpy(experts_, h_experts_, (size_t) bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        err = std::string("mtp: --mtp-lend: copying the experts back failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    if (mtp_lend_verify_env() && !mtp_lend_compare(experts_, h_experts_, bytes, err)) return false;
    experts_resident_ = true;
    return true;
}

}  // namespace strata::core
