/* r4d_ernie_verdict.h -- the verdict of an ERNIE device op that libref would have refused.
 *
 * libref refuses an invalid launch with an error code before its first store. A card launcher cannot: its metadata operands are device memory, the launch is
 * asynchronous, and a host read would be a stream synchronisation inside an issued-ahead step. So a card op detects the violation on the device and the
 * ENGINE polls the verdict at the point where it already waits for the step (core/engine.cpp, after collect_step, before the scheduler commits the step's
 * tokens) and fails the step with the code libref returns (RAD_E_SHAPE).
 *
 * WHERE THE STATE LIVES. All verdict state is one ordinary device allocation per device (hipMalloc, State below), made by this library on first use and passed to
 * every kernel as a POINTER in its kernel arguments (Vc). Nothing is a `__device__` global: the engine's device backend (core/device/aql.cpp) loads the code object
 * into its own hsa_executable, a separate copy of every global from the one HIP's own runtime holds, so a host read through hipGetSymbolAddress would never see what
 * the launched kernels wrote. A pointer to a hipMalloc'd buffer names the same bytes on both paths. The host reads it back by that pointer.
 *
 * REPLAY. The backend records a decode pass and replays it, and audits a replay byte for byte against a fresh issue (core/runtime/ctx.cpp, Ctx::run_pass). So the
 * kernel arguments of a call site must not change from one issue to the next: no generation counter, no per-launch slot. Every value in Vc is a function of the
 * call site (its operand pointers), and every bit of state that changes lives on the device (r4d_ar_seq.h does the same).
 *
 *   1. The EVENT LOG (Dev, one per family): a monotonic counter and a ring of the bit sets of the last kLog events. A violating lane or block raises an event
 *      (atomicAdd on the counter, a store of its bits into its own ring entry). Nothing ever clears it and nothing ever overwrites a count, so one launch
 *      cannot hide another's refusal. The host remembers how many events it has taken per (family, device); take() returns the bits of the events since the
 *      last take (bit 0x40000000 if events happened whose bits were not yet stored or overflowed the ring).
 *   2. The GATE (union attention, DSpark attention): one 32-bit word per CALL SITE, found by the launcher from the call site's operand pointers (the same operands
 *      give the same word on every issue and every replay; two call sites that can run at the same time have different words). Per launch, on the launch's stream:
 *      union attention runs gate_open (one thread: word = kGateOpen), then the validator (a violating block ORs its bits into the word), then the attention
 *      kernel, whose blocks read the word first; the DSpark validator is one block and writes the whole word. Every launch therefore starts from a clean word, so
 *      a refusal on one replay never sticks to the next. A launch of a call site cannot overlap the previous launch of it: they write the same output, which the
 *      backend's hazard check orders (core/runtime/ctx.cpp, Ctx::haz_mark); on a HIP stream the order is the stream's. Slots are never reused (a table of kSlots
 *      call sites per device); a device that runs out gets RAD_E_DEVICE from the launcher rather than a shared word.
 *
 * Validation is OFF by default on a card (fix series 5): R4D_ERNIE_VALIDATE=1 turns it on. Off costs nothing: no State is allocated, no validator, gate or poll copy is
 * launched, every Vc in a kernel's arguments is null (so the arguments are constant and the kernels' refusal branches are not taken: a row libref would refuse is computed
 * as it was before this series). The emulator keeps validation on by default (R4D_ERNIE_VALIDATE=0 turns it off; R4D_ERNIE_CARD_DEFAULTS=1 makes an unset variable mean off,
 * as on a card, so a test can hold the card default); the older per-family variables (R4D_DSPARK_VALIDATE, R4D_HCA_VALIDATE) are honoured on top of it.
 * SPDX-License-Identifier: Apache-2.0 */
#ifndef R4D_ERNIE_VERDICT_H
#define R4D_ERNIE_VERDICT_H

#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace r4d_verdict {

constexpr int kLog = 64;
constexpr int kSlots = 8192;                     /* gate words (call sites) a device can hold */
constexpr int kMaxDev = 64;
constexpr unsigned kUnknownBits = 0x40000000u;   /* an event was counted whose bits are not readable (still being stored, or the ring wrapped) */

struct Dev { unsigned cnt; unsigned log[kLog]; unsigned pad_; unsigned long long ent[kLog]; };   /* ent[i % kLog] = (event index + 1) << 32 | bits: an entry that does not carry the index it is read for is not that event's */

/* The families that keep a log, in State order. */
enum Fam { kFamDspark = 0, kFamHca = 1, kFamMla = 2, kFamOps = 3, kFamDsa = 4, kFamilies = 5 };   /* kFamDsa: the DSA selection ops (untiled select, tiled op), integration round 4 */

/* The device allocation: the four logs first (the poll copies exactly that prefix), then the gate words. */
struct State { Dev log[kFamilies]; unsigned gate[kSlots]; unsigned inject; unsigned inject_site1; };   /* inject, inject_site1: the self-test (see inject_now); inject must follow gate[kSlots] */
constexpr size_t kLogBytes = sizeof(Dev) * kFamilies;

/* What a kernel gets in its arguments. `log` is its family's event log (null: validation is off, nothing is raised); `gate` is the word of its call site (null for
 * a kernel that is not gated). Constant across every issue and replay of a call site. */
struct Vc { Dev* log; unsigned* gate; unsigned* inject; };

constexpr unsigned kGateOpen = 0x80000000u;   /* the word of a launch whose open step ran: bit 31 set, the verdict bits below */
constexpr unsigned kInjected = 0x10000000u;   /* the self-test refusal (R4D_ERNIE_SELFTEST_REFUSE): a validator reports it as a violation of the Nth gated launch */
constexpr unsigned kGateLost = 0x20000000u;   /* what a store block reports when the word is not open: the launch's validator did not run (or the word was cleared) */

/* Device side: raise one event with `bits` (nonzero). */
__device__ __forceinline__ void raise(Dev* v, unsigned bits) {
    if (!v) return;
    const unsigned i = atomicAdd(&v->cnt, 1u);
    v->ent[i % (unsigned)kLog] = ((unsigned long long)(i + 1u) << 32) | bits;
}

/* Device side of the SELF-TEST. `State::inject` is 0 (off) or N: the validator of the Nth gated launch from now reports kInjected as if it had found a violation (the launch stores
 * nothing, the event is raised and polled like any other). It lets a card run prove, with a real model and no bad input, that an event raised on the engine's own queue reaches the
 * engine's poll (CARD-CHECKS.md). `State::inject_site1` aims it: 0 counts every gated launch of the device; S + 1 counts only the launches of the gate word (call site) number S
 * of that device, so N is then the Nth LAUNCH OF THAT SITE, i.e. the Nth replay of a recorded decode pass at that layer. Sites are numbered in the order they first launch (the
 * first decode pass visits them in layer order, so the last site of that pass is its last gated layer). One validator thread calls it per launch. `gate` is the launch's gate word. */
__device__ __forceinline__ bool inject_now(unsigned* p, const unsigned* gate) {
    if (!p) return false;
    const unsigned site1 = *(volatile unsigned*)(p + 1);
    if (site1 != 0u && (unsigned)(gate - (p - kSlots)) + 1u != site1) return false;
    unsigned n = *(volatile unsigned*)p;
    while (n != 0u) {   /* n -> n - 1 only while n > 0: two validators on different queues cannot take the counter below zero (it would wrap to 4G and fire again) */
        const unsigned old = atomicCAS(p, n, n - 1u);
        if (old == n) return n == 1u;
        n = old;
    }
    return false;
}

/* Device side of the GATE. gate_open: one thread, the start of a launch. gate_add: a validator block with a violation ORs its bits in. gate_read: the word a store
 * block decides on: 0 = pass, else the refusal bits (kGateLost if the word is not open). */
__device__ __forceinline__ void gate_open(unsigned* g) { atomicExch(g, kGateOpen); }
__device__ __forceinline__ void gate_add(unsigned* g, unsigned bits) { if (bits) atomicOr(g, bits); }
__device__ __forceinline__ unsigned gate_read(const unsigned* g) {
    const unsigned v = *(const volatile unsigned*)g;
    return (v & kGateOpen) ? (v & ~kGateOpen) : kGateLost;
}
/* The verdict as ONE value for the whole block: thread 0 reads the word once and the block decides on that read (a word that changed between two threads' loads
 * cannot split a block into storing and not storing). Every thread of the block must reach it. A lost word is raised once (thread 0). */
__device__ __forceinline__ bool gate_refused_block(const Vc& vc) {
    __shared__ unsigned s_gate_v;
    if (threadIdx.x == 0) s_gate_v = gate_read(vc.gate);
    __syncthreads();
    const unsigned v = s_gate_v;
    if (v == kGateLost && threadIdx.x == 0) raise(vc.log, kGateLost);
    return v != 0u;
}
/* The same for a kernel whose waves never synchronise (no barrier, no LDS): the decision is one value per wave (lane 0's read, broadcast), raised once per wave. */
__device__ __forceinline__ bool gate_refused_wave(const Vc& vc) {
    unsigned v = gate_read(vc.gate);
    v = (unsigned)__builtin_amdgcn_readfirstlane((int)v);
    if (v == kGateLost && (threadIdx.x & 31u) == 0) raise(vc.log, kGateLost);
    return v != 0u;
}

/* R4D_ERNIE_CARD_FORM=1 (the emulator only; a HIP build is always card form): run a launch the way a card build does -- no host scan of the metadata operands,
 * the verdict polled. Read at every call (a test toggles it). */
inline bool card_form() {
#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
    return true;
#else
    const char* e = std::getenv("R4D_ERNIE_CARD_FORM");
    return e && *e == '1';
#endif
}

/* R4D_ERNIE_VALIDATE: "1" on, "0" off, anything else (or unset) the build's default: OFF on a card build, ON in the emulator. R4D_ERNIE_CARD_DEFAULTS=1 (emulator only) gives the emulator
 * the card's default. Read at every call where a launch completes before it returns (the emulator, R4D_SYNC_LAUNCH, so a test can toggle it); a card build reads it once. */
inline bool defaults_off() {
#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
    return true;
#else
    const char* e = std::getenv("R4D_ERNIE_CARD_DEFAULTS");
    return e && *e == '1';
#endif
}
inline bool master_on_text(const char* e) {
    if (e && *e == '0') return false;
    if (e && *e == '1') return true;
    return !defaults_off();
}
inline bool master_on() {
#ifdef R4D_SYNC_LAUNCH
    return master_on_text(std::getenv("R4D_ERNIE_VALIDATE"));
#else
    static const bool on = master_on_text(std::getenv("R4D_ERNIE_VALIDATE"));
    return on;
#endif
}

/* The cost model, readable by tests and by a GPU A/B: what the validators and the poll have done since process start. */
struct Stats {
    std::atomic<long long> validator_launches{0};   /* extra kernel launches ahead of an op that exist only for validation (the gate open step is counted too) */
    std::atomic<long long> dedupe_launches{0};      /* extra launches of the last-writer pass (clear + mark) a state_write / compress past 256 rows makes; they run whether or not validation is on */
    std::atomic<long long> validator_bytes_bound{0};/* upper bound of the metadata bytes those launches read */
    std::atomic<long long> polls{0};                /* take() / take_all() calls */
    std::atomic<long long> poll_copies{0};          /* device-to-host copies the polls issued (one per used device per poll) */
    std::atomic<long long> poll_bytes{0};           /* device-to-host bytes the polls copied */
};
inline Stats& stats() { static Stats s; return s; }

/* Host side: the events since the last take of this (family, device) (see the header). `h` is a device-to-host copy of the Dev. */
class Taken {
public:
    int fold(int device, const Dev& h) {
        std::lock_guard<std::mutex> lk(mu_);
        if (device < 0 || device >= kMaxDev) device = 0;
        const unsigned n = h.cnt - last_[device];
        last_[device] = h.cnt;
        if (n == 0) return 0;
        unsigned bits = 0;
        const unsigned m = n < (unsigned)kLog ? n : (unsigned)kLog;
        bool unknown = n > (unsigned)kLog;
        for (unsigned k = 0; k < m; ++k) {
            const unsigned idx = h.cnt - 1u - k;
            const unsigned long long e = h.ent[idx % (unsigned)kLog];
            if ((unsigned)(e >> 32) != idx + 1u || (unsigned)e == 0u) unknown = true;   /* not published yet, or already overwritten by a later event */
            else bits |= (unsigned)e;
        }
        if (unknown || bits == 0) bits |= kUnknownBits;
        return (int)bits;
    }
private:
    std::mutex mu_;
    unsigned last_[kMaxDev] = {};
};

/* Host side: the per-device State, the call-site table of gate words, and the poll. One instance per process (an inline function's static: one copy for every
 * translation unit of the library). */
class Registry {
public:
    struct Dv {
        State* st = nullptr;
        unsigned next = 0;                                  /* gate words handed out */
        std::unordered_map<std::string, unsigned> sites;    /* call-site key -> word index */
        void* stream = nullptr;                             /* the poll's private stream (card builds) */
    };

    /* The State of the current device, allocated (and zeroed) on first use; null if the allocation failed. */
    State* state() {
        int d = 0;
        if (hipGetDevice(&d) != hipSuccess || d < 0 || d >= kMaxDev) return nullptr;
        State* s = fast_[d].load(std::memory_order_acquire);   /* every launch of every rank thread asks: no lock once the device has its state */
        return s ? s : state_of(d);
    }
    State* state_of(int d) {
        std::lock_guard<std::mutex> lk(mu_);
        return state_locked(d);
    }
    /* The gate word of a call site on the current device (null if the state could not be allocated or the table is full). `key` is the raw bytes of the call
     * site's operand pointers: equal operands, equal word, on every issue. */
    unsigned* gate_for(const std::string& key, State** stp) {
        int d = 0;
        if (hipGetDevice(&d) != hipSuccess || d < 0 || d >= kMaxDev) return nullptr;
        std::lock_guard<std::mutex> lk(mu_);
        State* st = state_locked(d);
        if (stp) *stp = st;
        if (!st) return nullptr;
        Dv& v = dv_[d];
        auto it = v.sites.find(key);
        if (it != v.sites.end()) return &st->gate[it->second];
        if (v.next >= (unsigned)kSlots) {
            if (!warned_full_) { warned_full_ = true; std::fprintf(stderr, "r4d: ERNIE verdict: device %d has %d gated call sites; the next one is refused (RAD_E_DEVICE)\n", d, kSlots); }
            return nullptr;
        }
        const unsigned i = v.next++;
        v.sites.emplace(key, i);
        return &st->gate[i];
    }
    /* How many devices have a State (0 while validation has never been on). */
    int states() { std::lock_guard<std::mutex> lk(mu_); int n = 0; for (int d = 0; d < kMaxDev; ++d) n += dv_[d].st != nullptr; return n; }
    int gate_sites(int d) { std::lock_guard<std::mutex> lk(mu_); return d >= 0 && d < kMaxDev ? (int)dv_[d].next : 0; }

    /* The engine's own readback (core/plugin/registry.h, Plugin::kernel_poll_source / _fold): where the log of `device` lives, so the ENGINE copies it through that rank's own
     * queue (rad_memcpy_async behind the rank's step work, then an event it waits for), and the fold of that copy. 0 from log_source: validation is off and no always-on
     * family (acquire_always) has launched on the device, or no launch has run there (no State). */
    int log_source(int d, void** dev, int64_t* bytes) {
        if (d < 0 || d >= kMaxDev || !(master_on() || always_[d].load(std::memory_order_acquire))) return 0;
        State* s = fast_[d].load(std::memory_order_acquire);
        if (!s) return 0;
        *dev = (void*)s->log;   /* Dev log[kFamilies] is the first member of State: the poll copies exactly that prefix */
        *bytes = (int64_t)kLogBytes;
        return 1;
    }
    int fold_copy(int d, const void* copy, int64_t bytes) {
        Stats& stt = stats();
        stt.polls.fetch_add(1, std::memory_order_relaxed);
        stt.poll_copies.fetch_add(1, std::memory_order_relaxed);
        stt.poll_bytes.fetch_add((long long)bytes, std::memory_order_relaxed);
        if (!copy || bytes != (int64_t)kLogBytes) return -1;
        const Dev* h = (const Dev*)copy;
        int bits = 0;
        for (int f = 0; f < kFamilies; ++f) bits |= taken_[f].fold(d, h[f]);
        return bits;
    }

    /* The poll. take_family folds one family; take_all folds every family with one device copy per used device. Returns the OR of the bits, -1 if a copy failed.
     * Only devices that have a State (a launch ran there) are visited. */
    int take(int fam) { return poll(fam); }
    int take_all() { return poll(-1); }

    /* The State of the current device for an always-on family (acquire_always), allocated on first use; marks the device so that log_source and the poll answer for it with
     * validation off as well. Null if the allocation failed. */
    State* state_always() {
        int d = 0;
        if (hipGetDevice(&d) != hipSuccess || d < 0 || d >= kMaxDev) return nullptr;
        State* s = fast_[d].load(std::memory_order_acquire);
        if (!s) s = state_of(d);
        if (s && !always_[d].load(std::memory_order_relaxed)) always_[d].store(true, std::memory_order_release);
        return s;
    }

private:
    State* state_locked(int d) {
        Dv& v = dv_[d];
        if (v.st) return v.st;
        void* p = nullptr;
        if (hipMalloc(&p, sizeof(State)) != hipSuccess || !p) return nullptr;
        if (hipMemset(p, 0, sizeof(State)) != hipSuccess) { (void)hipFree(p); return nullptr; }
        v.st = (State*)p;
        arm_selftest(d, v.st);
        fast_[d].store(v.st, std::memory_order_release);
        return v.st;
    }
    /* R4D_ERNIE_SELFTEST_REFUSE=N arms the self-test on a device's State: the Nth gated launch is reported as a violation (see inject_now). Aiming (all optional):
     *   R4D_ERNIE_SELFTEST_DEVICE=d   only device d is armed (the others never inject), so a card check can raise on one non-zero rank
     *   R4D_ERNIE_SELFTEST_SITE=S     N counts only the launches of the device's call site S (gate word S, numbered in order of first launch), so N is a replay count and the
     *                                  refusal falls at one layer of a replayed decode pass, not at a position in the launch stream
     * Public so a test can arm a State of its own. */
public:
    static void arm_selftest(int d, State* s) {
        const char* e = std::getenv("R4D_ERNIE_SELFTEST_REFUSE");
        if (!e || !*e) return;
        const unsigned n = (unsigned)std::strtoul(e, nullptr, 10);
        if (!n) return;
        if (const char* dv = std::getenv("R4D_ERNIE_SELFTEST_DEVICE"); dv && *dv && (int)std::strtol(dv, nullptr, 10) != d) return;
        unsigned site1 = 0;
        if (const char* sv = std::getenv("R4D_ERNIE_SELFTEST_SITE"); sv && *sv) site1 = (unsigned)std::strtoul(sv, nullptr, 10) + 1u;
#ifdef R4D_SYNC_LAUNCH
        s->inject_site1 = site1;
        s->inject = n;
#else
        (void)hipMemcpy(&s->inject_site1, &site1, sizeof site1, hipMemcpyHostToDevice);
        (void)hipMemcpy(&s->inject, &n, sizeof n, hipMemcpyHostToDevice);
#endif
    }
private:
    int poll(int fam) {
        Stats& stt = stats();
        stt.polls.fetch_add(1, std::memory_order_relaxed);
        const bool on = master_on();
        State* sts[kMaxDev] = {};
        {
            std::lock_guard<std::mutex> lk(mu_);
            /* validation off: only the devices an always-on family launched on (none: no copy, 0) */
            for (int d = 0; d < kMaxDev; ++d) sts[d] = (on || always_[d].load(std::memory_order_acquire)) ? dv_[d].st : nullptr;
        }
        std::vector<Dev> hv;
        int used[kMaxDev], nused = 0;
        for (int d = 0; d < kMaxDev; ++d) if (sts[d]) used[nused++] = d;
        hv.resize((size_t)nused * kFamilies);
        Dev (*h)[kFamilies] = nused ? (Dev (*)[kFamilies])hv.data() : nullptr;   /* h[i] = the logs of used[i] */
        int rc = 0, bits = 0;
#ifdef R4D_SYNC_LAUNCH
        for (int i = 0; i < nused; ++i) {
            stt.poll_copies.fetch_add(1, std::memory_order_relaxed);
            stt.poll_bytes.fetch_add((long long)kLogBytes, std::memory_order_relaxed);
            std::memcpy(h[i], sts[used[i]]->log, kLogBytes);
        }
#else
        /* On a card the copy goes through a private non-blocking stream per device and waits for that stream alone: the engine polls where it has already waited for
         * the step (collect_step), a step issued ahead sits in the engine's own streams, and a synchronous copy is ordered against the null stream. All copies are
         * issued before the first wait. */
        int cur = 0;
        if (hipGetDevice(&cur) != hipSuccess) return -1;
        bool issued[kMaxDev] = {};   /* by position in used[] */
        for (int i = 0; i < nused; ++i) {
            const int d = used[i];
            hipStream_t s = nullptr;
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (hipSetDevice(d) != hipSuccess) { rc = -1; continue; }
                if (!dv_[d].stream) {
                    hipStream_t ns = nullptr;
                    if (hipStreamCreateWithFlags(&ns, hipStreamNonBlocking) != hipSuccess) ns = nullptr;
                    dv_[d].stream = (void*)ns;
                }
                s = (hipStream_t)dv_[d].stream;
            }
            stt.poll_copies.fetch_add(1, std::memory_order_relaxed);
            stt.poll_bytes.fetch_add((long long)kLogBytes, std::memory_order_relaxed);
            if (!s || hipMemcpyAsync(h[i], sts[d]->log, kLogBytes, hipMemcpyDeviceToHost, s) != hipSuccess) { rc = -1; continue; }
            issued[i] = true;
        }
        for (int i = 0; i < nused; ++i) {
            if (!issued[i]) continue;
            if (hipStreamSynchronize((hipStream_t)dv_[used[i]].stream) != hipSuccess) { rc = -1; issued[i] = false; }
        }
        (void)hipSetDevice(cur);
#endif
        for (int i = 0; i < nused; ++i) {
#ifndef R4D_SYNC_LAUNCH
            if (!issued[i]) continue;
#endif
            for (int f = 0; f < kFamilies; ++f)
                if (fam < 0 || fam == f) bits |= taken_[f].fold(used[i], h[i][f]);
        }
        return rc < 0 ? -1 : bits;
    }

    std::mutex mu_;
    std::atomic<State*> fast_[kMaxDev] = {};
    std::atomic<bool> always_[kMaxDev] = {};   /* an always-on family has launched on the device (state_always) */
    Dv dv_[kMaxDev];
    Taken taken_[kFamilies];
    bool warned_full_ = false;
};
inline Registry& registry() { static Registry r; return r; }

/* What a launcher gets: the Vc for its kernels, or ok = false (the device state could not be allocated, or the device has no gate word left: the launcher returns
 * RAD_E_DEVICE). With validation off (R4D_ERNIE_VALIDATE=0) the log is null and nothing is allocated. */
struct Launch { Vc vc; bool ok; };

inline Launch acquire(Fam f) {
    Launch l{{nullptr, nullptr, nullptr}, true};
    if (!master_on()) return l;
    State* st = registry().state();
    if (!st) { l.ok = false; return l; }
    l.vc.log = &st->log[f];
    l.vc.inject = &st->inject;
    return l;
}

/* A family whose card refusal leaves stale state behind (the quantised latent store: a refused row keeps the slot's previous bytes, and later attention reads them) keeps its
 * event log with validation OFF too, so the engine's poll still fails the step before its commit. Only the log: no validator launch, no gate word, no self-test injection, no
 * generation-varying argument (the pointer is constant once the State exists). Cost with validation off: one State per device (sizeof(State), once) and, once such a launch
 * has run on a device, the engine's poll copy of kLogBytes through each rank's queue at every polled issue; no extra kernel launch. */
inline Launch acquire_always(Fam f) {
    if (master_on()) return acquire(f);
    Launch l{{nullptr, nullptr, nullptr}, true};
    State* st = registry().state_always();
    if (!st) { l.ok = false; return l; }
    l.vc.log = &st->log[f];
    return l;
}

/* A gated launch: the key is the raw bytes of the call site's operand pointers (and what else tells two call sites apart). */
template <class... P>
inline Launch acquire_gated(Fam f, P... ptrs) {
    Launch l = acquire(f);
    if (!l.ok || !l.vc.log) return l;
    const void* const k[] = {(const void*)ptrs...};
    State* st = nullptr;
    unsigned* g = registry().gate_for(std::string((const char*)k, sizeof k) + (char)f, &st);
    if (!g) { l.ok = false; return l; }
    l.vc.gate = g;
    return l;
}

}  /* namespace r4d_verdict */

/* In a launcher: take the family's Vc into the kernel arguments `S` (a variable of a struct with a `vc` member), or return RAD_E_DEVICE. */
#define R4D_VC(S, FAM) const r4d_verdict::Launch S##_vl = r4d_verdict::acquire(FAM); if (!S##_vl.ok) return RAD_E_DEVICE; (S).vc = S##_vl.vc
/* The same for an always-on family (acquire_always): the log is passed whether or not validation is on. */
#define R4D_VC_ALWAYS(S, FAM) const r4d_verdict::Launch S##_vl = r4d_verdict::acquire_always(FAM); if (!S##_vl.ok) return RAD_E_DEVICE; (S).vc = S##_vl.vc

#endif
