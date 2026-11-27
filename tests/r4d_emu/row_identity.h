/* tests/r4d_emu/row_identity.h -- the row-identity check of the row-identity checks: a row's output must be the same bits whatever
 * else shares its launch. A family supplies run(rows, out): it gathers the listed rows of its input pool into one launch of rows.size()
 * rows and writes rows.size() * out_bytes bytes. The probe rows are computed alone (one-row launches), then placed at spread positions
 * (first, last, middle and the rest evenly) of launches of each size, in three neighbour modes, and every probe's output must equal its one-row
 * output byte for byte:
 *   plain    neighbours drawn at random from the pool;
 *   mixed    the same with other neighbours and other probe positions;
 *   hostile  (when the family lists hostile rows) neighbours drawn only from the hostile rows: extreme magnitudes, all-zero rows, fully masked rows.
 *            A dependence that exists only beside an outlier (a batch-wide scale or max, a clamp, a shared accumulator) shows here.
 * The probes are six normal rows (three fixed, three drawn from the seed) and, when there are hostile rows, the first hostile row, so a hostile row is
 * itself checked alone against in-batch. */
#ifndef R4D_EMU_ROW_IDENTITY_H
#define R4D_EMU_ROW_IDENTITY_H
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace rowid {
struct Family {
    const char* name;
    int pool;          /* rows in the input pool (>= the largest launch) */
    size_t out_bytes;  /* output bytes per row */
    std::function<int(const std::vector<int>& rows, std::vector<uint8_t>& out)> run;
    std::vector<int> hostile = {};   /* pool rows that are extreme, zero or masked: the third neighbour mode draws from these only */
};

/* Launches the checks below saw refused (one-row or batched), since process start. A control arm that expects differences reads it before and after: a refused launch
 * counts as a failure in check()'s return value but is no evidence of a numerical difference. */
inline int& refused() { static int n = 0; return n; }

/* Returns the number of failures; prints one line per launch. */
inline int check(const Family& f, const std::vector<int>& sizes, unsigned seed = 11) {
    std::mt19937 g(seed);
    auto is_hostile = [&](int r) { return std::find(f.hostile.begin(), f.hostile.end(), r) != f.hostile.end(); };
    std::vector<int> probes;
    auto add = [&](int r) { r = ((r % f.pool) + f.pool) % f.pool; if (std::find(probes.begin(), probes.end(), r) == probes.end()) probes.push_back(r); };
    add(3); add(f.pool / 2 + 1); add(f.pool - 2);
    for (int tries = 0; probes.size() < 6 && tries < 100; ++tries) { const int r = (int)(g() % (unsigned)f.pool); if (!is_hostile(r)) add(r); }
    const size_t n_normal = probes.size();
    if (!f.hostile.empty()) add(f.hostile[0]);
    const int np = (int)probes.size();
    std::vector<std::vector<uint8_t>> alone(np);
    int bad = 0;
    for (int i = 0; i < np; ++i) {
        if (int rc = f.run({probes[i]}, alone[i])) { ++refused(); std::printf("FAIL %s one-row launch refused rc=%d\n", f.name, rc); return 1; }
    }
    const int modes = f.hostile.empty() ? 2 : 3;
    for (int T : sizes) {
        for (int mode = 0; mode < modes; ++mode) {
            std::vector<int> rows(T);
            for (auto& r : rows) {
                if (mode == 2) r = f.hostile[g() % f.hostile.size()];
                else do r = (int)(g() % (unsigned)f.pool); while (std::find(probes.begin(), probes.end(), r) != probes.end());
            }
            /* probe positions: first, last, middle, then evenly; mode shifts them so every mode meets other neighbours. */
            std::vector<int> pos(np, -1);
            std::vector<char> taken(T, 0);
            const int shift = mode == 0 ? 0 : (T > 1 ? 1 + mode : 0);
            for (int i = 0; i < np; ++i) {
                int p = i == 0 ? 0 : i == 1 ? T - 1 : i == 2 ? T / 2 : (int)((long long)T * (i - 2) / (np - 1));
                if (T > 3 && i >= 2) p = (p + shift) % T;
                if (!taken[p]) { taken[p] = 1; pos[i] = p; rows[p] = probes[i]; }
            }
            std::vector<uint8_t> out;
            const char* mn = mode == 0 ? "plain" : mode == 1 ? "mixed" : "hostile";
            if (int rc = f.run(rows, out)) { ++refused(); std::printf("FAIL %s T=%d %s refused rc=%d\n", f.name, T, mn, rc); ++bad; continue; }
            int differ = 0, checked = 0;
            for (int i = 0; i < np; ++i) {
                if (pos[i] < 0) continue;
                ++checked;
                differ += std::memcmp(&out[(size_t)pos[i] * f.out_bytes], alone[i].data(), f.out_bytes) != 0;
            }
            std::printf("%s %s T=%4d %-7s: %d of %d probe rows differ from the same row alone (%d normal + %d hostile probes)\n", differ ? "FAIL" : "PASS", f.name, T, mn,
                        differ, checked, (int)n_normal, np - (int)n_normal);
            bad += differ != 0;
        }
    }
    return bad;
}
}  // namespace rowid
#endif
