// fixture for the argument-declaration check (EXPECT_FAIL): every spelling below leaves the padding of a kernel-argument struct undefined
int f() {
    FooArgs P{};
    FooArgs Q;
    FooArgs R = {};
    launch(kernel, grid, block, stream, (FooArgs{a, b}));
    FooArgs S(1, 2);
    return 0;
}
