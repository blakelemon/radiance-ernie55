/* tests/r4d_emu/verdict_replay.h -- what the card backend does to a launcher's launches, modelled on the SIMT emulator (review of fix series 4, B1/B2).
 *
 * The engine's device backend (core/device/aql.cpp) intercepts every hipLaunchKernel of a plugin, records a decode pass as a tape (the kernel, the grid and the kernel-argument BYTES
 * of every launch), plays the tape again for every later step, and every kAuditPlays plays issues the pass once more live and compares its tape byte for byte against the
 * recording (core/runtime/ctx.cpp, Ctx::run_pass; a difference is RAD_E_STATE and stops the engine). The emulator's launch tap (emu_launch_tap) gives a test the same view:
 * it records the launches of one issue and plays them again from the recorded bytes, with the device memory (operands, outputs, the library's verdict state) as it is at
 * the time of the play. What this cannot model: the card's hazard ordering between ops and the second copy of `__device__` globals in a separately loaded code object (the
 * library no longer has any such global; tests/CMakeLists.txt checks that on the sources). */
#pragma once
#include "hip/hip_runtime.h"
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

extern bool (*emu_launch_tap)(EmuLaunchRec&);

namespace verdict_replay {

constexpr int kAuditPlays = 1024;   // core/runtime/ctx.h Ctx::kAuditPlays

using Tape = std::vector<EmuLaunchRec>;
inline Tape* g_rec = nullptr;
inline bool tap(EmuLaunchRec& r) { if (g_rec) g_rec->push_back(r); return true; }

/* The launches one live issue makes, as the backend would record them (the issue also runs). */
inline Tape record(const std::function<int()>& issue, int* rc = nullptr) {
    Tape t;
    g_rec = &t;
    emu_launch_tap = tap;
    const int r = issue();
    emu_launch_tap = nullptr;
    g_rec = nullptr;
    if (rc) *rc = r;
    return t;
}

/* Ctx::tapes_agree: the same kernels, grids and kernel-argument bytes, in the same order. */
inline bool agree(const Tape& a, const Tape& b, std::string* why) {
    if (a.size() != b.size()) { if (why) *why = "launch count " + std::to_string(a.size()) + " vs " + std::to_string(b.size()); return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].kfn != b[i].kfn || a[i].g.x != b[i].g.x || a[i].g.y != b[i].g.y || a[i].g.z != b[i].g.z || a[i].b.x != b[i].b.x || a[i].b.y != b[i].b.y ||
            a[i].b.z != b[i].b.z) { if (why) *why = "launch " + std::to_string(i) + " is another kernel or grid"; return false; }
        if (a[i].args.size() != b[i].args.size()) { if (why) *why = "launch " + std::to_string(i) + " argument size"; return false; }
        for (size_t k = 0; k < a[i].args.size(); ++k)
            if (a[i].args[k] != b[i].args[k]) { if (why) *why = "launch " + std::to_string(i) + ": its arguments differ at byte " + std::to_string(k); return false; }
    }
    return true;
}

inline void play(const Tape& t) { for (const EmuLaunchRec& r : t) r.replay(); }

/* One op: how to issue it, break and repair its metadata, poison and read its output, and take the verdict the engine polls. */
struct Case {
    std::string name;
    std::function<int()> issue;
    std::function<void()> corrupt, restore, poison;
    std::function<std::vector<uint16_t>()> output;
    std::function<bool()> untouched;      // the output is still what poison() left
    std::function<int()> take;            // the events since the last take, OR-ed (0: none)
};

/* Records a pass, then plays it plays times (default past kAuditPlays) with validation on; at chosen plays the metadata is invalid. Checks: every audit (the live issue every
 * kAuditPlays plays, and after a refusal) has the recorded bytes; a refusal on play k is taken exactly once and stores nothing; the next play (metadata repaired) is a pass,
 * with the output of the first play and no event. */
inline bool replay_audit(const Case& c, int plays, std::string* detail) {
    bool ok = true;
    auto fail = [&](const std::string& m) { if (ok && detail) *detail = m; ok = false; };
    (void)c.take();
    c.poison();
    int rc = -1;
    Tape tape = record(c.issue, &rc);
    if (rc != 0 || tape.empty()) { fail("the first issue failed or launched nothing"); return false; }
    const std::vector<uint16_t> good = c.output();
    if (c.take() != 0) fail("the first (valid) issue raised an event");
    {   /* an issue after the recording: the backend's audit */
        std::string why;
        Tape fresh = record(c.issue);
        if (!agree(tape, fresh, &why)) fail("a second issue's kernel arguments differ from the first's: " + why);
    }
    (void)c.take();
    const int refuse[] = {5, kAuditPlays + 1, plays - 1};
    for (int k = 0; k < plays && ok; ++k) {
        bool refused_play = false;
        for (int r : refuse) refused_play = refused_play || k == r;
        if (refused_play) c.corrupt();
        c.poison();
        play(tape);
        const int ev = c.take();
        if (refused_play) {
            if (ev == 0) fail("play " + std::to_string(k) + ": a refusal was not seen");
            if (!c.untouched()) fail("play " + std::to_string(k) + ": a refused play stored output");
            if (c.take() != 0) fail("play " + std::to_string(k) + ": the refusal was reported twice");
            c.restore();
        } else {
            if (ev != 0) fail("play " + std::to_string(k) + ": a valid play reported an event (0x" + [&] { char b[16]; std::snprintf(b, sizeof b, "%x", (unsigned)ev); return std::string(b); }() + ")");
            if (c.output() != good) fail("play " + std::to_string(k) + ": the output differs from the first play's");
        }
        if (k % kAuditPlays == 0 && ok) {   /* the audit: a live issue is compared with the tape */
            std::string why;
            Tape fresh = record(c.issue);
            if (!agree(tape, fresh, &why)) fail("audit at play " + std::to_string(k) + ": " + why);
            (void)c.take();
        }
        if (refused_play && ok) {   /* an issue of a refused launch has the same bytes as the recording too */
            c.corrupt();
            std::string why;
            Tape fresh = record(c.issue);
            if (!agree(tape, fresh, &why)) fail("an issue of a refused launch differs from the recording: " + why);
            (void)c.take();
            c.restore();
        }
    }
    return ok;
}

}  // namespace verdict_replay
