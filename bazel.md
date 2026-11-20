# Bazel 使用指南：在 mlir-tutorial 中使用 Bazel

> 本课程的主线使用 CMake；上游教程文章使用 Bazel。两种构建系统生成的是
> 同一套 `tutorial-opt` 和同一批 pass，但构建树、外部依赖和命令完全不同。
>
> 本文不是要替代前面的课程，而是补上“如果按上游文章使用 Bazel，应该怎么
> 做”。建议先完成第 0 课，知道 `tutorial-opt`、`mlir-opt`、lit 测试分别
> 是什么，再阅读本文。

## 1. 先给一条最短路径

当前环境已经安装了 Bazelisk，并且仓库把版本固定在 `.bazelversion`：

```
cd /home/zhouzhou/npu/mlir-tutorial

bazel version
# bazel 8.3.1

bazel build -c fastbuild //tools:tutorial-opt
bazel-bin/tools/tutorial-opt tests/mul_to_add_pdll.mlir --mul-to-add-pdll
```

运行一个 lit 测试：

```
bazel test -c fastbuild //tests:mul_to_add_pdll.mlir.test \
  --test_output=all
```

第一次运行 Bazel 会下载并构建自己的 LLVM/MLIR 和第三方依赖，可能比
CMake 首次构建还慢；后续增量构建会快很多。

### 1.1 安装/确认 Bazelisk

推荐使用 Bazelisk，而不是手动安装某个固定版本的 Bazel。Bazelisk 会读取
仓库根目录的 `.bazelversion`，自动下载并运行项目要求的 Bazel 版本。

安装方式见 Bazelisk 官方仓库：<https://github.com/bazelbuild/bazelisk>。

安装后确认：

```
which bazel
bazel version
```

本环境已经装好，输出为：

```
bazel 8.3.1
```

Bazel 默认使用系统 C++ 编译器（本环境是 GCC 13）；本仓库没有自定义 C++
toolchain。先用默认编译器把 Bazel 路线跑通，再考虑切换到 clang。

首次构建还需要：

- 足够的磁盘空间（LLVM、or-tools 和缓存会占用数 GB）
- 可访问 GitHub、BCR 和 Bazel 的依赖下载地址
- 足够的并行资源；可以用 `--jobs=N` 限制并发

---

## 2. Bazel 是什么

Bazel 是一个“声明式、可缓存、可并行、沙箱化”的构建系统。它最初由 Google
开源，LLVM、MLIR、TensorFlow、Protobuf 等项目都使用它。

严格来说，Bazel 是构建系统，Bzlmod 是 Bazel 8 默认使用的模块依赖管理机制；
两者经常一起出现，但不是同一个概念。本文后面会分别解释 `BUILD`（构建规则）
和 `MODULE.bazel`（模块依赖）。

和 CMake/Ninja 的核心区别：

| 维度 | CMake + Ninja | Bazel |
| --- | --- | --- |
| 构建描述 | `CMakeLists.txt` + 生成构建文件 | 每个目录的 `BUILD` 文件 |
| 依赖声明 | target link/include 关系 | 每个 target 的 `deps` 精确声明 |
| 增量依据 | 时间戳 + depfile | action 的内容哈希 |
| 缓存 | 本地构建目录 | 内容寻址缓存，可共享 |
| 沙箱 | 通常不强制 | 默认沙箱执行 action |
| 外部依赖 | `find_package` / FetchContent / submodule | Bzlmod / 外部仓库 |
| 构建产物 | `build/` | `bazel-bin/`、`bazel-out/`、`bazel-testlogs/` |
| 测试 | 外部 lit/ctest | 原生 `bazel test`，也可包装 lit |

Bazel 关注的不是“执行这条编译命令”，而是：

```
目标：//tools:tutorial-opt
  -> 需要哪些源文件
  -> 需要哪些库
  -> 需要先运行哪些代码生成器（TableGen、PDLL）
  -> 每个 action 的输入哈希是否变化
  -> 只执行需要重新执行的部分
```

### 2.1 几个核心概念

#### Package（包）

一个包含 `BUILD`（或 `BUILD.bazel`）文件的目录就是一个 package。

例如：

```
tools/BUILD                 -> package //tools
lib/Dialect/Poly/BUILD      -> package //lib/Dialect/Poly
tests/BUILD                 -> package //tests
```

`lib/Dialect` 本身没有 `BUILD`，这没有关系；Bazel 只把真正有 `BUILD` 的
目录当作 package。

#### Target（目标）

`BUILD` 里的每一条规则就是一个 target，例如：

```
cc_library(
    name = "Poly",
    ...
)
```

对应标签：

```
//lib/Dialect/Poly:Poly
```

#### Label（标签）

常用的三种形式：

```
//tools:tutorial-opt                 本仓库的 target
//lib/Transform/Arith:Passes         本仓库另一个 package 的 target
@llvm-project//mlir:IR              外部仓库 llvm-project 的 target
@or-tools//ortools/linear_solver    外部仓库 or-tools 的 target
```

可以读成：

```
@仓库//包:目标
```

省略 `:目标` 时，Bazel 会尝试使用与包最后一段同名的 target。例如
`//tools` 等价于 `//tools:tools`，但本项目没有这个 target，所以写完整标签更安全。

#### Rule（规则）

`BUILD` 文件使用 Starlark 语法：看起来很像 Python，但它不是 Python，不能
在 `BUILD` 里写任意 Python 代码。需要复用逻辑时，应该写成 `.bzl` 里的宏
（macro）或规则（rule），例如本项目的 `bazel/lit.bzl`。

Bazel 内置的规则包括：

| 规则 | 作用 | 本项目例子 |
| --- | --- | --- |
| `cc_library` | C++ 库 | `//lib/Dialect/Poly:Poly` |
| `cc_binary` | C++ 可执行文件 | `//tools:tutorial-opt` |
| `py_test` | Python 测试 | lit 包装出来的测试 |
| `filegroup` | 一组文件 | lit 测试的数据 |
| `gentbl_cc_library` | 调用 TableGen/PDLL 生成文件 | `Poly` 的 `.inc` 生成 |
| `td_library` | 声明一组 `.td` 源文件 | `//lib/Dialect/Poly:td_files` |

`gentbl_cc_library` 不是 Bazel 内置规则，是 MLIR 在
`@llvm-project//mlir:tblgen.bzl` 里提供的宏；`glob_lit_tests` 则是本项目
`bazel/lit.bzl` 定义的宏。

#### 工作区 / 模块

Bazel 8 使用 **Bzlmod** 管理模块。项目根目录的 `MODULE.bazel` 是模块声明，
不需要旧式的 `WORKSPACE` 文件。

#### 执行阶段

Bazel 一次命令通常经过：

```
Loading    读取 BUILD / MODULE / .bzl
Analysis   建立 action 依赖图，解析 label
Execution  在沙箱里执行编译、链接、代码生成
Testing    运行测试 target
```

出错时知道自己在哪个阶段，能更快定位问题：

- `no such target`：Loading/Analysis 阶段，label 或 BUILD 写错
- `undeclared inclusion`：编译阶段，头文件依赖没有写进 `deps`
- test timeout：Testing 阶段，测试本身太慢或卡住

---

## 3. 本项目中的 Bazel 文件

### 3.1 文件总览

| 文件 | 作用 |
| --- | --- |
| `.bazelversion` | 固定 Bazel 版本：`8.3.1` |
| `.bazelrc` | 全局默认参数：Bzlmod、C++17、Debug、测试输出等 |
| `.bazelignore` | 告诉 Bazel 忽略 `externals/` |
| `MODULE.bazel` | Bzlmod 模块、BCR 依赖、LLVM 配置、Python 工具链 |
| `extensions.bzl` | 自定义 module extension，下载 pinned LLVM 源码 |
| `BUILD` | 根 package，设置默认可见性 |
| `bazel/BUILD` | `bazel` package 的占位 BUILD |
| `bazel/lit.bzl` | 把 `.mlir` 文件包装成 `py_test` 的 lit 宏 |
| `tests/BUILD` | 测试工具集合和 `glob_lit_tests()` |
| `tools/BUILD` | `tutorial-opt` 可执行文件 |
| `lib/**/BUILD` | 各库的 C++ / TableGen target |

### 3.2 `.bazelversion`

```
8.3.1
```

Bazelisk 会根据这个文件自动选择对应版本的 Bazel。因此建议用 **Bazelisk**
（命令名通常也是 `bazel`），而不是系统里任意一个旧版 Bazel。

查看当前版本：

```
bazel version
```

如果版本不是 8.3.1，先检查 `bazel` 是否是 Bazelisk、是否在仓库根目录运行。

### 3.3 `.bazelrc`

本项目的 `.bazelrc`：

```
common --enable_bzlmod
common --action_env=BAZEL_CXXOPTS=-std=c++17
common --cxxopt='-std=c++17'
common --deleted_packages=externals
build:macos --apple_platform_type=macos
build:macos --macos_minimum_os=10.13
build:macos --macos_sdk_version=10.13
build:macos_arm64 --cpu=darwin_arm64
common --copt=-fdiagnostics-color=always
common --test_output=errors
common -c dbg
```

逐项含义：

- `--enable_bzlmod`：启用 Bzlmod
- `--cxxopt=-std=c++17`：所有 C++ 编译使用 C++17
- `--action_env=BAZEL_CXXOPTS=-std=c++17`：把编译选项传进 action 环境
- `--deleted_packages=externals`：不把 `externals` 当作当前 workspace 的 package
- macOS 配置：设置 SDK 和 Apple Silicon CPU
- `--test_output=errors`：测试失败时才打印输出
- `-c dbg`：默认构建配置是 Debug

因此直接运行 `bazel build //tools:tutorial-opt` 是 Debug 构建。想加快普通
构建可以显式覆盖：

```
bazel build -c fastbuild //tools:tutorial-opt
bazel build -c opt      //tools:tutorial-opt
```

Debug 适合 gdb 和排查断言；`fastbuild` 接近 CI 的默认选择；`opt` 运行最快。

### 3.4 `.bazelignore`

```
externals
build
```

本仓库同时保留 CMake 用的 `externals/llvm-project` 子模块。Bazel **不使用**
这个子模块，而是从 `extensions.bzl` 拉取自己的 LLVM 副本。因此把
`externals` 整个忽略，避免 Bazel 把它当作源码包扫描。

`build` 是 CMake 的构建树。它里面会有 `build/_deps/...` 等第三方源码，
其中一些目录带 `BUILD` 文件；如果不忽略，`bazel query //...` 可能误入这些
目录并报“找不到 pybind11_bazel”之类的错误。本课程在 CMake 和 Bazel
共存的环境里运行，所以 `.bazelignore` 额外加入了 `build`。

同理，`.bazelrc` 里的 `--deleted_packages=externals` 也是这个目的。

> 重要：修改 `externals/llvm-project` 里的代码不会影响 Bazel 构建。
> Bazel 的 LLVM 源码在 Bazel 的 output base 里。要改 LLVM 版本，应该修改
> `extensions.bzl` 中的 commit，并让 CMake 子模块与之对齐。

### 3.5 `MODULE.bazel`

核心内容可以分成四块。

#### 模块声明

```
module(
    name = "mlir_tutorial",
    version = "1.0.0",
    repo_name = "mlir_tutorial",
)
```

- `name`：Bzlmod 模块名
- `repo_name`：其他 `.bzl` 文件里使用的仓库名
- `bazel/lit.bzl` 里的 `@mlir_tutorial//tests:test_utilities` 就依赖这个名称

#### BCR 依赖

```
bazel_dep(name = "bazel_skylib", version = "1.7.1")
bazel_dep(name = "rules_python", version = "1.2.0")
bazel_dep(name = "rules_cc", version = "0.1.1")
bazel_dep(name = "protobuf", version = "30.1")
bazel_dep(name = "or-tools", version = "9.12")
bazel_dep(name = "scip", version = "9.2.0.bcr.3")
bazel_dep(name = "zlib-ng", version = "2.0.7")
...
```

BCR 是 Bazel Central Registry，可以理解为 Bazel 的模块仓库。Bzlmod 会根据
版本解析依赖图，并把下载的模块放到 Bazel 的缓存中。

`or-tools` 在这里是 9.12；CMake 的 `FetchContent` 使用的是 v9.11。两者
版本略有差异，但本课程相关的 ILP API 行为一致。

#### 自定义 LLVM 模块扩展

```
mlir_tutorial_deps = use_extension("//:extensions.bzl", "mlir_tutorial_deps")
use_repo(mlir_tutorial_deps,
    "llvm-raw",
    "llvm_zstd",
    "llvm_zlib"
)
```

`extensions.bzl` 做三件事：

1. 用 `new_git_repository` 下载 LLVM pinned commit
2. 用 `http_archive` 提供 zstd
3. 用 `http_archive` 提供 zlib-ng

LLVM commit 固定在：

```
commit = "d9190f8141661bd6120dea61d28ae8940fd775d0"
```

这和 CMake 使用的 LLVM 子模块 commit 应该保持一致。

#### 配置 LLVM target

```
_LLVM_TARGETS = ["X86", "NVPTX", "AArch64"]

llvm_configure = use_repo_rule(
    "@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
llvm_configure(
    name = "llvm-project",
    targets = _LLVM_TARGETS,
)
```

`llvm_configure` 读取 `@llvm-raw` 的源码，在 Bazel 里生成一个配置好的
`@llvm-project` 仓库。之后所有的：

```
@llvm-project//mlir:IR
@llvm-project//mlir:ArithDialect
@llvm-project//llvm:Support
```

都来自这里。

`_LLVM_TARGETS` 中的 NVPTX/AArch64 会让首次构建更慢，但这是上游为了
LLVM/MLIR 自身的 target 图能够完整加载而保留的；不要随手删掉，除非你明确
知道哪些测试或 target 不再需要。

#### Python 和 pip

```
python.toolchain(python_version = "3.13")
use_repo(python, "python_3_13")

pip.parse(
    hub_name = "mlir_tutorial_pip_deps",
    python_version = "3.13",
    requirements_lock = "//:requirements.txt",
)
use_repo(pip, "mlir_tutorial_pip_deps")
```

`requirements.txt` 里是 lit。`tests/BUILD` 的 lit 测试通过
`@mlir_tutorial_pip_deps//lit` 使用它。因此 Bazel 测试不依赖你系统里是否
预先安装了 lit。

### 3.6 依赖图

把上面的关系画出来：

```
MODULE.bazel
│
├── bazel_dep(...) ────────────────► BCR 中的 rules_python、or-tools、scip...
│
├── mlir_tutorial_deps ───────────► extensions.bzl
│     ├── @llvm-raw（pinned git commit）
│     ├── @llvm_zstd
│     └── @llvm_zlib
│
├── llvm_configure ───────────────► @llvm-project（Bazel 可用的 LLVM/MLIR）
│
└── python / pip ─────────────────► @mlir_tutorial_pip_deps
                                      └── lit
```

本仓库自己的 target 再依赖这些外部 target。

---

## 4. BUILD 文件怎么读

### 4.1 `tools/BUILD`：最终可执行文件

```
cc_binary(
    name = "tutorial-opt",
    srcs = ["tutorial-opt.cpp"],
    includes = ["include"],
    deps = [
        "//lib/Conversion/PolyToStandard",
        "//lib/Dialect/Box",
        "//lib/Dialect/Noisy",
        "//lib/Dialect/Poly",
        "//lib/Transform/Affine:Passes",
        "//lib/Transform/Arith:Passes",
        "//lib/Transform/Noisy:Passes",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithToLLVM",
        ...
        "@llvm-project//mlir:MlirOptLib",
        ...
    ],
)
```

这相当于 CMake 里的：

```
add_llvm_executable(tutorial-opt tutorial-opt.cpp)
target_link_libraries(tutorial-opt PRIVATE ${LIBS})
```

但 Bazel 的 `deps` 是精确依赖图，不只是链接列表：

- 编译 `tutorial-opt.cpp` 时，允许包含的头文件来自这些 deps
- 链接时，这些库和它们的传递依赖会被带进来
- 生成库时，TableGen/PDLL 的生成 action 会先执行

`includes = ["include"]` 给编译器增加一个 include 目录；本项目源码里使用
`mlir/include/mlir/...` 这样的路径，是因为根目录也在 include 搜索路径里。

### 4.2 `lib/Dialect/Poly/BUILD`：TableGen 生成

Poly 的 BUILD 是一个完整的 TableGen 示例：

```
load("@llvm-project//mlir:tblgen.bzl",
     "gentbl_cc_library", "td_library")

td_library(
    name = "td_files",
    srcs = [
        "PolyDialect.td",
        "PolyOps.td",
        "PolyPatterns.td",
        "PolyTypes.td",
    ],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)
```

`td_library` 把 `.td` 文件作为一组源文件管理，并声明它们依赖哪些上游
TableGen 定义。它本身不生成 C++。

然后为不同产物定义 `gentbl_cc_library`：

```
gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "PolyDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "PolyDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PolyDialect.td",
    deps = [":td_files"],
)
```

`tbl_outs` 里每一对：

```
(["参数"], "输出文件名")
```

最终在 Bazel 的 output 目录里生成 `.inc` 文件，而不是写回源码目录。

对应的 CMake 写法是：

```
mlir_tablegen(PolyDialect.h.inc -gen-dialect-decls -dialect=poly)
mlir_tablegen(PolyDialect.cpp.inc -gen-dialect-defs -dialect=poly)
add_public_tablegen_target(MLIRPolyOpsIncGen)
```

最后 `cc_library` 把生成 target 作为依赖：

```
cc_library(
    name = "Poly",
    srcs = [
        "PolyDialect.cpp",
        "PolyOps.cpp",
    ],
    hdrs = [
        "PolyDialect.h",
        "PolyOps.h",
        "PolyTraits.h",
        "PolyTypes.h",
    ],
    deps = [
        ":canonicalize_inc_gen",
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)
```

Bazel 会自动把 `*_inc_gen` 产生的 `.inc` 放到编译器的 include 路径里。

### 4.3 `lib/Transform/Arith/BUILD`：pass 和 PDLL

Arith 的 BUILD 同时包含普通 C++ pattern 和 PDLL pattern：

```
gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            ["-gen-pass-decls", "-name=Arith"],
            "Passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "ArithPasses.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PDLDialectTdFiles",
        "@llvm-project//mlir:PDLInterpOpsTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "MulToAdd",
    srcs = ["MulToAdd.cpp"],
    hdrs = ["MulToAdd.h"],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "MulToAddPdllIncGen",
    tbl_outs = [
        (
            ["-x=cpp"],
            "MulToAddPdll.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-pdll",
    td_file = "MulToAdd.pdll",
    deps = [
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:ArithOpsTdFiles",
    ],
)

cc_library(
    name = "MulToAddPdll",
    srcs = ["MulToAddPdll.cpp"],
    hdrs = ["MulToAddPdll.h"],
    deps = [
        ":pass_inc_gen",
        ":MulToAddPdllIncGen",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)
```

注意 PDLL 的 `tblgen` 是：

```
tblgen = "@llvm-project//mlir:mlir-pdll"
```

而普通 TableGen 的 `tblgen` 是：

```
tblgen = "@llvm-project//mlir:mlir-tblgen"
```

这和 CMake 里的 `add_mlir_pdll_library` 与 `mlir_tablegen` 的差异对应。

### 4.4 `tests/BUILD`：自动发现 lit 测试

```
load("//bazel:lit.bzl", "glob_lit_tests")

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tests:lit.cfg.py",
        "//tests:poly_to_llvm_main.c",
        "//tools:tutorial-opt",
        "@llvm-project//clang:clang",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:count",
        "@llvm-project//llvm:llc",
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:mlir-runner",
        "@llvm-project//mlir:mlir-opt",
        "@llvm-project//mlir:mlir-translate",
        "@mlir_tutorial_pip_deps//lit",
    ],
)

glob_lit_tests()
```

`glob_lit_tests()` 在 `bazel/lit.bzl` 里定义：

```
def glob_lit_tests():
    tests = native.glob(["*.mlir"])
    for curr_test in tests:
        lit_test(src = curr_test, size = "small")
```

每个 `tests/foo.mlir` 会自动生成：

```
//tests:foo.mlir.test
//tests:foo.mlir.test.filegroup
```

`lit_test` 内部用 `py_test` 包装 lit：

```
py_test(
    name = name,
    args = ["-v", paths.join(native.package_name(), src)],
    data = ["@mlir_tutorial//tests:test_utilities", filegroup_name],
    deps = ["@mlir_tutorial_pip_deps//lit"],
    srcs = ["@llvm-project//llvm:lit"],
    main = "lit.py",
    python_version = "PY3",
    tags = tags,
)
```

所以：

- 新增一个 `tests/xxx.mlir`，不需要改 `tests/BUILD`
- 它会自动成为 `//tests:xxx.mlir.test`
- 测试使用的 `tutorial-opt`、FileCheck、mlir-opt 等来自 `test_utilities`

---

## 5. 常用命令

以下命令默认在仓库根目录执行。

### 5.1 查看版本和环境

```
bazel version
# bazel 8.3.1

bazel info workspace
bazel info output_base
bazel info bazel-bin
```

`bazel info` 显示的是 Bazel 的配置；如果它显示的输出目录不是你预期的，
检查是否在仓库根目录执行。

### 5.2 构建单个目标

```
bazel build -c fastbuild //tools:tutorial-opt
```

产物：

```
bazel-bin/tools/tutorial-opt
```

`bazel-bin` 是指向真实输出目录的符号链接，不是源码目录；不要把它加入 git。

### 5.3 构建全仓库

```
bazel build -c fastbuild //...
```

README 中写的是：

```
bazel build ...:all
```

`...` 是“所有 package”的通配符，`//...` 是更常见的等价写法。全仓库构建会
包含测试辅助 target、所有 dialect 库和 `tutorial-opt`，第一次会构建大量
LLVM target。

> 如果本机同时存在 CMake 的 `build/` 目录，确认 `.bazelignore` 里有
> `build`（本课程已经加上），否则 `//...` 可能扫描到 CMake 依赖的源码。
> 只想构建工具时用 `//tools:tutorial-opt` 更安全、更快。

本环境在 `.bazelignore` 忽略 `build` 后，实际运行
`bazel build //...` 的结果是：

```
INFO: Found 125 targets...
INFO: Build completed successfully, 21 total actions
```

大量 action 已经由前面的构建缓存命中；干净环境中的首次全量构建会重得多。

### 5.4 运行 `tutorial-opt`

最直接的方法是在构建后使用 `bazel-bin`：

```
bazel build -c fastbuild //tools:tutorial-opt
bazel-bin/tools/tutorial-opt tests/mul_to_add_pdll.mlir --mul-to-add-pdll
```

也可以用 `bazel run`：

```
bazel run -c fastbuild //tools:tutorial-opt -- \
  "$PWD/tests/mul_to_add_pdll.mlir" --mul-to-add-pdll
```

`--` 后面的参数才是传给 `tutorial-opt` 的参数；`--` 前面的是传给 Bazel 的。
`bazel run` 的工作目录可能在 Bazel 的 runfiles 目录，因此输入路径用绝对路径
更稳妥。

### 5.5 运行测试

运行一个测试：

```
bazel test -c fastbuild //tests:mul_to_add_pdll.mlir.test \
  --test_output=all
```

第一次运行测试会比只构建 `tutorial-opt` 重很多：`tests/BUILD` 的
`test_utilities` 还会拉入 clang、`mlir-opt`、`mlir-runner`、`llc` 等工具，
并构建 lit 所需的 Python 依赖。测试工具构建好之后，再运行测试会快很多。

运行 tests package 的所有测试：

```
bazel test -c fastbuild //tests:all
```

运行全仓库测试：

```
bazel test -c fastbuild //...
```

测试日志在：

```
bazel-testlogs/tests/mul_to_add_pdll.mlir.test/test.log
```

如果 `--test_output=errors`（`.bazelrc` 默认）没有显示详细信息，可以显式覆盖：

```
bazel test //tests:mul_to_add_pdll.mlir.test --test_output=all
```

### 5.6 查询依赖图

```
# 查看一个 target 的依赖
bazel query 'deps(//tools:tutorial-opt)'

# 只列出 target 标签
bazel query 'deps(//tools:tutorial-opt)' --output=label

# 查看某个 package 的所有 target
bazel query '//lib/Dialect/Poly:all'

# 查看测试 target
bazel query '//tests:all'

# 查看配置后的依赖（会考虑 select 等配置）
bazel cquery 'deps(//tools:tutorial-opt)'

# 查看 action 命令
bazel aquery '//tools:tutorial-opt'
```

`query` 工作在 loading 阶段，`cquery` 工作在 analysis 阶段，`aquery` 查看
实际 action。排查“为什么没有编译某个文件”“实际用了哪些编译参数”时非常有用。

### 5.7 清理

```
# 删除构建输出，保留外部依赖缓存
bazel clean

# 连同 output base、外部依赖缓存一起删除，下一次会重新下载 LLVM
bazel clean --expunge

# 停止 Bazel server（释放内存；下次命令会自动启动）
bazel shutdown
```

`bazel clean --expunge` 代价很高：LLVM 和 or-tools 需要重新下载/构建。只有
在缓存损坏、磁盘空间紧张或切换大版本时才使用。

### 5.8 Bzlmod 相关命令

```
bazel mod graph
bazel mod deps
bazel mod show_repo llvm-project
bazel mod tidy
```

- `mod graph`：查看模块依赖图
- `mod deps`：查看依赖来源
- `mod show_repo`：查看某个外部仓库的解析结果
- `mod tidy`：整理 `MODULE.bazel` 中的依赖（使用前先理解它会改什么）

本仓库的 `MODULE.bazel.lock` 被 `.gitignore` 忽略；第一次构建会解析依赖并生成
它。如果团队希望锁定解析结果，可以讨论是否提交 lock 文件，但当前仓库没有这样做。

---

## 6. 完整实验：用 Bazel 跑 PDLL pass

### 6.1 构建

```
cd /home/zhouzhou/npu/mlir-tutorial
bazel build //tools:tutorial-opt
```

这条命令使用 `.bazelrc` 的默认 `-c dbg`；想更快可以用
`bazel build -c fastbuild //tools:tutorial-opt`。

第一次构建会：

1. 读取 `MODULE.bazel`
2. 通过 `extensions.bzl` 下载 pinned LLVM
3. 配置 `@llvm-project`
4. 下载 BCR 依赖和 Python 工具链
5. 构建 LLVM/MLIR、or-tools 和本项目

后续只构建发生变化的 action。

在本课程环境中用默认的 `-c dbg` 首次构建时，实际结果是：

```
INFO: Found 1 target...
Target //tools:tutorial-opt up-to-date:
  bazel-bin/tools/tutorial-opt
INFO: Elapsed time: 905.803s, Critical Path: 382.93s
INFO: 5386 processes: 497 internal, 4889 linux-sandbox.
INFO: Build completed successfully, 5386 total actions
```

首次构建会占用较多磁盘（本环境的 Bazel output base 约十几 GB），
Debug 版 `tutorial-opt` 本身也很大。日常可以改用 `-c fastbuild` 或
`-c opt`，并用 `bazel clean` / `bazel clean --expunge` 管理缓存。

### 6.2 直接运行

```
bazel-bin/tools/tutorial-opt \
  tests/mul_to_add_pdll.mlir --mul-to-add-pdll
```

预期输出和 CMake 版一致：

```
module {
  func.func @just_power_of_two(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    %1 = arith.addi %0, %0 : i32
    %2 = arith.addi %1, %1 : i32
    return %2 : i32
  }
  func.func @power_of_two_plus_one(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    %1 = arith.addi %0, %0 : i32
    %2 = arith.addi %1, %1 : i32
    %3 = arith.addi %2, %arg0 : i32
    return %3 : i32
  }
}
```

### 6.3 运行对应的 lit 测试

```
bazel test //tests:mul_to_add_pdll.mlir.test --test_output=all
```

第一次运行（默认 `-c dbg`）会构建 clang、`mlir-opt`、`mlir-runner` 等测试
工具；在本课程环境中实际耗时约 495 秒。测试本身很快。成功输出：

```
-- Testing: 1 tests, 1 workers --
PASS: mlir_tutorial :: mul_to_add_pdll.mlir (1 of 1)

Testing Time: 0.08s

Total Discovered Tests: 1
  Passed: 1 (100.00%)
...
//tests:mul_to_add_pdll.mlir.test                                        PASSED in 0.6s
Executed 1 out of 1 test: 1 test passes.
```

如果失败，先看：

```
cat bazel-testlogs/tests/mul_to_add_pdll.mlir.test/test.log
```

运行 `//tests` 下的全部测试：

```
bazel test //tests:all --test_output=errors
```

本课程环境实测结果类似：

```
INFO: Found 26 targets and 25 test targets...
Executed N out of 25 tests: 25 tests pass.
```

`N` 取决于缓存命中数；最终 25 个测试全部通过。

### 6.4 验证测试自动发现

新增一个文件：

```
tests/my_new_pass.mlir
```

不需要改 `tests/BUILD`。确认 target 已出现：

```
bazel query '//tests:my_new_pass.mlir.test'
```

运行：

```
bazel test //tests:my_new_pass.mlir.test --test_output=all
```

这就是 `glob_lit_tests()` 的作用。

---

## 7. 修改本项目时怎么改 BUILD

### 7.1 给现有库增加源文件

比如给 `lib/Dialect/Poly` 增加 `PolyFoo.cpp`：

1. 把文件放到 `lib/Dialect/Poly/PolyFoo.cpp`
2. 在 `lib/Dialect/Poly/BUILD` 的 `srcs` 中加入：

```
cc_library(
    name = "Poly",
    srcs = [
        "PolyDialect.cpp",
        "PolyFoo.cpp",
        "PolyOps.cpp",
    ],
    ...
)
```
3. 运行：

```
bazel build -c fastbuild //lib/Dialect/Poly:Poly
```

如果新文件 include 了新的 MLIR 头文件，把对应的 Bazel target 加到 `deps`，
例如：

```
deps = [
    ...
    "@llvm-project//mlir:SCFDialect",
]
```

Bazel 的 C++ 规则会检查直接依赖；只在 CMake 的全局 include 里能找到，并不
代表 Bazel 知道这个依赖。

### 7.2 增加一个新的 dialect

可以参考 `lib/Dialect/Box/BUILD`。最小结构：

```
load("@llvm-project//mlir:tblgen.bzl",
     "gentbl_cc_library", "td_library")

td_library(
    name = "td_files",
    srcs = ["FooDialect.td", "FooOps.td", "FooTypes.td"],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        ...
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (["-gen-dialect-decls"], "FooDialect.h.inc"),
        (["-gen-dialect-defs"], "FooDialect.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FooDialect.td",
    deps = [":td_files"],
)

# 同理增加 types_inc_gen、ops_inc_gen
# 如果还有 canonicalizer，再加 canonicalize_inc_gen

cc_library(
    name = "Foo",
    srcs = ["FooDialect.cpp", "FooOps.cpp"],
    hdrs = ["FooDialect.h", "FooOps.h", "FooTypes.h"],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":types_inc_gen",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
```

然后：

- 如果 `tutorial-opt` 要注册这个 dialect，给 `tools/BUILD` 增加
`//lib/Dialect/Foo` 依赖
- 如果测试需要它，`tutorial-opt` 注册后，`.mlir` 测试会自动可用
- 如果新增了单独的 pass，把它放到 `lib/Transform/...`，更新 `Passes` target

### 7.3 增加一个新的 pass

假设在 `lib/Transform/Affine` 中增加 `MyPass.cpp`：

1. 把源文件放到目录
2. 在 `lib/Transform/Affine/BUILD` 的对应 cc_library 的 `srcs` 中增加
3. 如果 `Passes.h` 包含新头文件，把它加入 `hdrs`
4. 如果新 pass 依赖新的 MLIR 库，加入 `deps`
5. 如果新 pass 要暴露给 `tutorial-opt`，确认 `//lib/Transform/Affine:Passes`
依赖了包含新 pass 的库
6. 构建：

```
bazel build -c fastbuild //lib/Transform/Affine:Passes
bazel build -c fastbuild //tools:tutorial-opt
```

### 7.4 增加测试

对于 `.mlir` lit 测试，只要放在 `tests/`，`glob_lit_tests()` 会自动发现：

```
cp tests/mul_to_add_pdll.mlir tests/my_case.mlir
# 编辑 tests/my_case.mlir，确保 RUN/CHECK 正确
bazel test //tests:my_case.mlir.test --test_output=all
```

如果测试需要额外的命令行工具，把它加入 `tests/BUILD` 的
`test_utilities.data`。

### 7.5 增加外部依赖

优先使用 BCR：

```
bazel_dep(name = "some-package", version = "1.2.3")
```

如果依赖不在 BCR，可以：

- 在 `extensions.bzl` 中用 `http_archive` / `new_git_repository`
- 或用 `git_override` 覆盖某个模块的源

新增依赖后，运行：

```
bazel mod graph
bazel build -c fastbuild //tools:tutorial-opt
```

---

## 8. Bazel 和 CMake 的对照

### 8.1 Target 对照

| CMake target | Bazel label | 说明 |
| --- | --- | --- |
| `tutorial-opt` | `//tools:tutorial-opt` | 可执行文件 |
| `MLIRPoly` | `//lib/Dialect/Poly:Poly` | Poly dialect 库 |
| `MLIRBox` | `//lib/Dialect/Box:Box` | Box dialect 库 |
| `MLIRNoisy` | `//lib/Dialect/Noisy:Noisy` | Noisy dialect 库 |
| `MulToAdd` | `//lib/Transform/Arith:MulToAdd` | C++ MulToAdd 库 |
| `Passes`（Arith） | `//lib/Transform/Arith:Passes` | 聚合 pass 头文件 |
| `NoisyPasses` | `//lib/Transform/Noisy:Passes` | Noisy pass |
| `PolyToStandard` | `//lib/Conversion/PolyToStandard:PolyToStandard` | conversion pass |
| `ReduceNoiseAnalysis` | `//lib/Analysis/ReduceNoiseAnalysis:ReduceNoiseAnalysis` | ILP analysis |
| `tests/foo.mlir` | `//tests:foo.mlir.test` | lit 测试 |

### 8.2 关键差异

1. **外部依赖来源不同**
   - CMake：`externals/llvm-project` 子模块 + `FetchContent` 拉 or-tools
   - Bazel：`extensions.bzl` 拉 LLVM + BCR 拉 or-tools
2. **首次构建成本不同**
   - CMake 可以复用已经构建好的 LLVM
   - Bazel 会构建自己的 LLVM 配置；首次更慢，但缓存可以复用
3. **依赖声明严格程度不同**
   - CMake 的 include 路径往往是全局的，漏写链接在链接阶段才暴露
   - Bazel 的 `deps` 会影响头文件可见性，漏写会直接编译失败
4. **生成文件位置不同**
   - CMake：`build/lib/.../*.inc`
   - Bazel：`bazel-bin/...` 或 output base 里的生成文件
5. **测试驱动不同**
   - CMake 课程：`llvm-lit`（见 [lit 测试指南](lit.md)）
   - Bazel：`bazel test` 通过 pip 的 lit 自动运行 `.mlir` 测试
6. **默认构建模式不同**
   - CMake 课程：`CMAKE_BUILD_TYPE=Debug`
   - Bazel：`.bazelrc` 里 `-c dbg`，但可被 `-c fastbuild` / `-c opt` 覆盖

### 8.3 常用命令对照

| 目的 | CMake 主线 | Bazel 路线 |
| --- | --- | --- |
| 构建 tutorial-opt | `ninja -C build tutorial-opt` | `bazel build //tools:tutorial-opt` |
| 运行 pass | `build/tools/tutorial-opt foo.mlir --pass` | `bazel-bin/tools/tutorial-opt foo.mlir --pass` |
| 构建单个库 | `ninja -C build MLIRPoly` | `bazel build //lib/Dialect/Poly:Poly` |
| 重新生成 TableGen | 修改 `.td` 后 `ninja` | 修改 `.td` 后 `bazel build` |
| 运行测试 | `$BIN/llvm-lit -sv build/tests --filter foo` | `bazel test //tests:foo.mlir.test` |
| 全量测试 | `ninja check-mlir-tutorial` | `bazel test //...` |
| 查看生成文件 | `build/lib/...` | `bazel-bin/...` |
| 清理 | `rm -rf build` 或 Ninja clean | `bazel clean` |

### 8.4 两种构建不要混用产物

```
build/                  CMake 构建树
bazel-bin/              Bazel 构建产物符号链接
bazel-out/              Bazel output
bazel-testlogs/         Bazel 测试日志
```

它们是独立的。不要用 CMake 生成的文件去替换 Bazel 的 `bazel-bin` 文件，
也不要把 `bazel-bin` 加进 git。`.gitignore` 已经忽略了这些路径。

---

## 9. 常见问题

### 9.1 第一次构建卡在下载

Bazel 首次要拉 LLVM、or-tools 和 BCR 依赖。可以用：

```
bazel fetch -c fastbuild //tools:tutorial-opt
```

单独预取依赖，然后再 build。网络受限时，Bazel 无法使用 CMake 已经下载好的
`externals/llvm-project`；两者是两套缓存。

### 9.2 构建很慢 / 内存占用高

- 默认 `.bazelrc` 的 `-c dbg` 会生成调试信息；改用 `-c fastbuild` 或 `-c opt`
- 限制并行：

```
bazel build -c fastbuild --jobs=32 //tools:tutorial-opt
```
- 首次构建 LLVM 本身就很重；后续增量会明显变快
- 不需要时 `bazel shutdown` 停止 server

### 9.3 `undeclared inclusion` 或头文件找不到

Bazel 的 C++ target 只能使用 `deps` 中声明的头文件。找到提供头文件的 Bazel
target，并加入对应 `cc_library` 的 `deps`：

```
deps = [
    ...
    "@llvm-project//mlir:SCFDialect",
]
```

CMake 里“全局 include 路径能编过”不代表 Bazel 能编过。

### 9.4 测试失败但看不到输出

```
bazel test //tests:foo.mlir.test --test_output=all
cat bazel-testlogs/tests/foo.mlir.test/test.log
```

`--test_output=errors` 只显示失败测试的输出；`all` 显示所有测试输出。

### 9.5 生成文件在哪里

不要找源码目录里的 `PolyOps.h.inc`；Bazel 生成的文件在 output 目录：

```
bazel info bazel-bin
find "$(bazel info bazel-bin)" -name '*.inc' | head
```

CMake 的生成文件在 `build/lib/...`，Bazel 的在 `bazel-bin/...`，路径不同。

### 9.6 为什么 `externals` 不影响 Bazel

因为：

```
.bazelignore: externals, build
.bazelrc:     --deleted_packages=externals
```

Bazel 的 LLVM 来自 `@llvm-raw`，不是 `externals/llvm-project`。要升级
LLVM，修改 `extensions.bzl` 的 commit，并同步更新 CMake 子模块。

### 9.7 Bazel 版本不匹配

```
bazel version
```

应为 8.3.1。Bazelisk 会读取 `.bazelversion`。如果使用的是系统自带 Bazel，
可能版本过旧，Bzlmod 或规则行为会不同。

### 9.8 不要编辑 `bazel-*` 下的文件

`bazel-bin`、`bazel-out`、`bazel-testlogs` 都是输出。修改会在下次构建被覆盖，
也不要提交到版本库。

### 9.9 本课程为 Bazel 路线做的适配

CMake 和 Bazel 在同一工作区共存时，有两类环境问题需要处理。本课程已经做了
下面几处适配；如果你在干净的上游仓库里看到不同行为，可以先对照这里。

1. **忽略 CMake 的 `build/`**

`.bazelignore` 增加了：

```
externals
build
```

否则 `bazel query //...` 可能扫描到 `build/_deps/...` 里的第三方 BUILD
文件（例如 re2 的 Python 目录），并报 `pybind11_bazel` 无法解析。

2. **让 Bazel 的 clang 找到内建头文件**  
Bazel 构建出的 clang 在 runfiles 里，但它启动时计算的 resource directory
和 runfiles 中的 `clang/staging/include` 不一致，编译 C 文件时会找不到
`stddef.h`。`tests/lit.cfg.py` 因此做了两件事：

- 把 `+_repo_rules+llvm-project/clang` 加入 lit 的 `PATH`
- 把 `+_repo_rules+llvm-project/clang/staging/include` 加入 `C_INCLUDE_PATH`

这样 `poly_to_llvm.mlir` 和 `poly_to_llvm_eval.mlir` 里的 `clang -c ...`
才能正常工作。

3. **让自动发现的测试都有 RUN 行**

`tests/poly_constant_verifiers.mlir` 是课程新增的 verifier 用例，原本没有
`RUN:`，Bazel 的 `glob_lit_tests()` 会把它当作一个测试并报
`Test has no 'RUN:' line`。现在它有一条期望失败的测试：

```
// RUN: not tutorial-opt %s 2>&1 | FileCheck %s --check-prefix=ERROR
// ERROR: 'poly.constant' op requires coefficients to be a rank-1 tensor
```

这不是 PDLL 本身的要求，而是为了同时支持 CMake 手动测试和 Bazel
自动发现。

做完这些适配后，本环境运行：

```
bazel test //tests:all --test_output=errors
```

结果是 25 个测试全部通过。

---

## 10. 练习

### 练习 1：查询 target 依赖

运行：

```
bazel query 'deps(//tools:tutorial-opt)'
```

找出 `tutorial-opt` 依赖的：

1. Poly dialect 的 target
2. MulToAdd 的 target
3. PDLL 生成 target

<details>
<summary>参考答案</summary>

```
//lib/Dialect/Poly:Poly
//lib/Transform/Arith:MulToAdd
//lib/Transform/Arith:MulToAddPdllIncGen
```

具体输出还会包含大量 LLVM/MLIR 的传递依赖。

</details>

### 练习 2：构建并运行 PDLL pass

用 Bazel 构建 `tutorial-opt`，然后对
`tests/mul_to_add_pdll.mlir` 运行 `--mul-to-add-pdll`。

<details>
<summary>参考答案</summary>

```
bazel build -c fastbuild //tools:tutorial-opt
bazel-bin/tools/tutorial-opt \
  tests/mul_to_add_pdll.mlir --mul-to-add-pdll
```

预期看到 `just_power_of_two` 的三条加法和 `power_of_two_plus_one`
的四条加法。

</details>

### 练习 3：运行单个 lit 测试

运行 `//tests:mul_to_add_pdll.mlir.test`，并查看测试日志。

<details>
<summary>参考答案</summary>

```
bazel test -c fastbuild //tests:mul_to_add_pdll.mlir.test \
  --test_output=all
cat bazel-testlogs/tests/mul_to_add_pdll.mlir.test/test.log
```

</details>

### 练习 4：新增一个测试文件

在 `tests/` 新建 `my_test.mlir`，写一个最小 RUN 行，然后确认 Bazel 自动
生成了 target。

<details>
<summary>参考答案</summary>

```
// RUN: tutorial-opt %s --canonicalize | FileCheck %s
func.func @empty() {
  return
}
// CHECK-LABEL: func.func @empty
```

```
bazel query '//tests:my_test.mlir.test'
bazel test //tests:my_test.mlir.test --test_output=all
```

不需要修改 `tests/BUILD`，因为 `glob_lit_tests()` 会自动发现新的 `.mlir`。

</details>

### 练习 5：解释 `externals` 和 Bazel 的关系

为什么 Bazel 构建不读取 `externals/llvm-project`？如果想换 LLVM 版本，
应该改哪里？

<details>
<summary>参考答案</summary>

因为 `.bazelignore` 和 `--deleted_packages=externals` 让 Bazel 忽略这个目录；
Bazel 的 LLVM 来自 `extensions.bzl` 中的 `new_git_repository`。

换版本时修改：

```
commit = "新的 LLVM commit"
```

然后重新配置/构建。CMake 使用的子模块也应同步到同一个 commit，避免两套
构建使用不同 LLVM。

</details>

---

## 11. 验收清单

- 能说出 package、target、label 的区别
- 能读懂 `//tools:tutorial-opt`
- 能解释 `cc_library` 的 `deps` 为什么必须写全
- 知道 `.bazelversion` 固定 Bazel 8.3.1
- 知道 `.bazelrc` 默认是 `-c dbg`
- 知道 `externals` 被 Bazel 忽略
- 能解释 `MODULE.bazel` 和 `extensions.bzl` 的分工
- 能说出 `@llvm-project` 从哪里来
- 能构建 `//tools:tutorial-opt`
- 能运行 `//tests:mul_to_add_pdll.mlir.test`
- 知道新增 `.mlir` 测试会被自动发现
- 能定位 `bazel-testlogs` 中的失败日志
- 能区分 Bazel 和 CMake 的构建树

## 12. 参考

- Bazel 官方入门：
  <https://bazel.build/start>
- Bazelisk：
  <https://github.com/bazelbuild/bazelisk>
- Bzlmod：
  <https://bazel.build/external/overview>
- Bazel Central Registry：
  <https://registry.bazel.build/>
- 课程总览与环境说明：根目录 [README.md](README.md)
- [lit 测试指南](lit.md)：`glob_lit_tests()`、CMake `llvm-lit`、RUN/FileCheck
- 本项目 Bazel 配置：
  - `MODULE.bazel`
  - `extensions.bzl`
  - `.bazelrc`
  - `.bazelversion`
  - `bazel/lit.bzl`
  - `tools/BUILD`
  - `tests/BUILD`
  - `lib/**/BUILD`
- 本课程 CMake 主线：
  - [第 0 课：环境搭建](00-environment.md)
  - [第 1 课：IR 基础与测试](01-ir-basics.md)
  - [第 12 课：PDLL](12-pdll.md)