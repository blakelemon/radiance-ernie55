// fixture for ernie_no_device_globals.cmake (EXPECT_FAIL): every line below is a program-scope device variable the lint must find
__device__ int g_plain;
__device__ int g_init = compute(3);
__constant__ float g_table[4];
static __device__ unsigned g_static = make(1, 2);
__device__ __managed__ int g_managed;
// and these are functions, which it must not report
__device__ __forceinline__ int fn_a(int x) { return x; }
__device__ int fn_b(int x);
