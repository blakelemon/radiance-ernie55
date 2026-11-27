// fixture for the argument-declaration check: the macro, a kernel parameter and a pointer are fine
__global__ void k(FooArgs A) { }
int f(const FooArgs* a) {
    R4D_ARGS(FooArgs, P);
    P.x = 1;
    return a->x;
}
