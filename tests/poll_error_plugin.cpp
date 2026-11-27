/* poll_error_plugin.cpp -- a kernel library with no kernels that reports a refused launch, for engine_e2e_test.
 *
 * A device kernel library answers `rad_kernel_poll_error` (core/plugin/registry.h, Plugin::kernel_poll_error) with the verdict bits of the launches libref would have
 * refused. This library stands in for one: its poll returns RAD_POLL_PLUGIN_BITS (default 0x10) at the RAD_POLL_PLUGIN_AFTER-th poll (default 0, the first) and 0 at every
 * other, so the engine's handling of a nonzero verdict is tested without a card. RAD_POLL_PLUGIN_BITS=0 never reports. */
#include "rad_abi.h"
#include "rad_abi_layout.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

extern "C" {

uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT

const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = { RAD_PLUGIN_KERNEL, "pollerr", "0.1.0", "reports a refused launch, for engine_e2e_test", "host" };
    return &info;
}
int rad_kernel_schema_count(void) { return 0; }
const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }
int rad_kernel_count(void) { return 0; }
const RadKernelInfo* rad_kernel_at(int) { return nullptr; }

int rad_kernel_poll_error(void) {
    static std::atomic<long> polls{0};
    const char* b = std::getenv("RAD_POLL_PLUGIN_BITS");
    const char* a = std::getenv("RAD_POLL_PLUGIN_AFTER");
    const long bits = b && *b ? std::strtol(b, nullptr, 0) : 0x10;
    const long after = a && *a ? std::strtol(a, nullptr, 10) : 0;
    return polls.fetch_add(1) == after ? (int)bits : 0;
}

/* The engine's own readback form (Plugin::kernel_poll_source / _fold), RAD_POLL_PLUGIN_MODE=source: each "device" has a 64-byte log in plain memory (the host backend's device memory
 * is host memory). The engine asks where the log of a device is once per rank per polled issue, behind that rank's step work, and copies it through the rank's own stream. The
 * stand-in raises RAD_POLL_PLUGIN_BITS into the log of device RAD_POLL_PLUGIN_DEVICE when that device is asked for the RAD_POLL_PLUGIN_AFTER-th time (0 = the first), i.e. at the
 * step the engine is issuing, after the work of that step: the copy the engine makes next sees it, and nothing before did. Another device is never touched. fold returns the log
 * and clears it. */
static bool source_mode() { const char* m = std::getenv("RAD_POLL_PLUGIN_MODE"); return m && !std::strcmp(m, "source"); }
static int g_log[64][16];
static std::atomic<long> g_calls[64];
int rad_kernel_poll_source(int device, void** dev_ptr, int64_t* bytes) {
    if (!source_mode() || device < 0 || device >= 64) return 0;
    const char* d = std::getenv("RAD_POLL_PLUGIN_DEVICE");
    const char* b = std::getenv("RAD_POLL_PLUGIN_BITS");
    const char* a = std::getenv("RAD_POLL_PLUGIN_AFTER");
    const long want_dev = d && *d ? std::strtol(d, nullptr, 10) : 0;
    const long bits = b && *b ? std::strtol(b, nullptr, 0) : 0x10;
    const long after = a && *a ? std::strtol(a, nullptr, 10) : 0;
    const long n = g_calls[device].fetch_add(1);
    if (device == want_dev && n == after && bits) g_log[device][0] = (int)bits;
    *dev_ptr = g_log[device];
    *bytes = sizeof g_log[device];
    return 1;
}
int rad_kernel_poll_fold(int device, const void* copy, int64_t bytes) {
    if (device < 0 || device >= 64 || bytes != (int64_t)sizeof g_log[0]) return -1;
    const int v = *(const int*)copy;
    g_log[device][0] = 0;
    return v;
}

}  // extern "C"
