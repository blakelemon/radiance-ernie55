# Two source lints for the ERNIE kernel sources (libr4d/r4d_ernie_*). Run:
#   cmake -DSRC=<repo root> [-DCHECK=globals|args] [-DEXPECT_FAIL=1 -DFILES="a;b"] -P ernie_no_device_globals.cmake
#
# CHECK=globals (default), review B2 + m-4: no program-scope device variable (`__device__`, `__constant__`, `__managed__`, with or without an initialiser, with or without
# parentheses in it). The engine's device backend loads a kernel library's code object into its own hsa_executable, a separate copy of every program-scope variable from
# the one HIP's runtime holds; a host read of such a variable would see a different object from the one the launched kernels write. Verdict state is a hipMalloc'd buffer
# passed by pointer (libr4d/r4d_ernie_verdict.h). A device FUNCTION has a `(` before its first `;`, `=` or `{`; a variable declaration has `=` or `;` first.
#
# CHECK=args (fix series 5, m-1): a kernel-argument struct (a type named *Args) is declared with R4D_ARGS(T, name) and nowhere else in a launcher: `T x{};`, `T x;`, `T x = {};`,
# `T x(...)` and a braced temporary `T{...}` all leave the struct's padding undefined, and the backend copies and audits those bytes.
if(NOT CHECK)
  set(CHECK globals)
endif()
if(NOT FILES)
  if(CHECK STREQUAL "args")
    set(FILES ${SRC}/libr4d/r4d_ernie_hca.hip ${SRC}/libr4d/r4d_ernie_dspark.hip ${SRC}/libr4d/r4d_ernie_mla.hip ${SRC}/libr4d/r4d_ernie_ops.hip ${SRC}/libr4d/r4d_ernie_mhc.hip ${SRC}/libr4d/r4d_ernie_small.hip)
  else()
    file(GLOB FILES ${SRC}/libr4d/r4d_ernie_*.hip ${SRC}/libr4d/r4d_ernie_*.h)
  endif()
endif()
set(bad "")
foreach(f ${FILES})
  file(READ ${f} text)
  if(CHECK STREQUAL "args")
    string(REGEX MATCHALL "\n[ \t]*[A-Za-z0-9_]+Args[ \t]+[A-Za-z0-9_]+[ \t]*(\\{[^\n]*\\}|;|=[^\n]*|\\([^\n]*)" hits "${text}")
    string(REGEX MATCHALL "[^A-Za-z0-9_][A-Za-z0-9_]+Args\\{" temps "${text}")
    list(APPEND hits ${temps})
  else()
    string(REGEX MATCHALL "(^|\n)[ \t]*((static|extern|inline)[ \t]+)*(__device__|__constant__|__managed__)([ \t]+(__device__|__constant__|__managed__))*[^(){}\n;=]*(=|;)" hits "${text}")
  endif()
  foreach(h ${hits})
    string(STRIP "${h}" h1)
    list(APPEND bad "${f}: ${h1}")
  endforeach()
endforeach()
if(bad)
  foreach(b ${bad})
    message(STATUS "${CHECK}: ${b}")
  endforeach()
  if(NOT EXPECT_FAIL)
    if(CHECK STREQUAL "args")
      message(FATAL_ERROR "a kernel-argument struct is declared without R4D_ARGS (its padding would be undefined)")
    else()
      message(FATAL_ERROR "ERNIE kernel sources declare program-scope device variables; the host cannot read them on the device backend")
    endif()
  endif()
else()
  if(EXPECT_FAIL)
    message(FATAL_ERROR "expected hits, found none")
  endif()
endif()
