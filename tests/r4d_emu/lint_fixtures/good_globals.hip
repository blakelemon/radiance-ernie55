// fixture for ernie_no_device_globals.cmake: functions only
__device__ __forceinline__ int fn_a(int x) { return x; }
__device__ int fn_b(int x);
__global__ void k(int x) { __shared__ int s; s = x; }
