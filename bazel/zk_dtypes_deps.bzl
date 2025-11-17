load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def zk_dtypes_deps():
    http_archive(
        name = "rules_cc",
        urls = ["https://github.com/bazelbuild/rules_cc/archive/refs/tags/0.1.0.tar.gz"],
        strip_prefix = "rules_cc-0.1.0",
        sha256 = "4b12149a041ddfb8306a8fd0e904e39d673552ce82e4296e96fac9cbf0780e59",
    )

    http_archive(
        name = "com_google_absl",
        sha256 = "9b2b72d4e8367c0b843fa2bcfa2b08debbe3cee34f7aaa27de55a6cbb3e843db",
        strip_prefix = "abseil-cpp-20250814.0",
        urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20250814.0.tar.gz"],
        patches = [
            "//third_party/absl:endian.patch",
            "//third_party/absl:googletest.patch",
        ],
        patch_args = ["-p1"],
    )

    http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "ef516fb84824a597c4d5d0d6d330daedb18363b5a99eda87d027e6bdd9cba299",
        strip_prefix = "re2-03da4fc0857c285e3a26782f6bc8931c4c950df4",
        urls = ["https://github.com/google/re2/archive/03da4fc0857c285e3a26782f6bc8931c4c950df4.tar.gz"],
    )

    http_archive(
        name = "com_google_googletest",
        sha256 = "81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2",
        strip_prefix = "googletest-release-1.12.1",
        urls = ["https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz"],
    )

    EIGEN_COMMIT = "4c38131a16803130b66266a912029504f2cf23cd"
    http_archive(
        name = "eigen_archive",
        build_file = "//third_party/eigen3:eigen_archive.BUILD",
        sha256 = "1a432ccbd597ea7b9faa1557b1752328d6adc1a3db8969f6fe793ff704be3bf0",
        strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
        urls = ["https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT)],
    )
