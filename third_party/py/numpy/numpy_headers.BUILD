# numpy ships its C headers inside the wheel but exposes no cc_library for
# them, and the include root moved from `core` to `_core` in numpy 2, so both
# are declared and the wrapper depends on whichever glob matched.
cc_library(
    name = "numpy_headers_2",
    hdrs = glob(["site-packages/numpy/_core/include/**/*.h"]),
    strip_include_prefix = "site-packages/numpy/_core/include/",
)

cc_library(
    name = "numpy_headers_1",
    hdrs = glob(["site-packages/numpy/core/include/**/*.h"]),
    strip_include_prefix = "site-packages/numpy/core/include/",
)

cc_library(
    name = "numpy_headers",
    deps = [
        ":numpy_headers_1",
        ":numpy_headers_2",
    ],
)
