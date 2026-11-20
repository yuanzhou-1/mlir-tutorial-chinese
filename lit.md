# lit 测试指南：在 mlir-tutorial 中组织 MLIR 回归测试

> 本课程前面的文档经常出现 `// RUN:`、`FileCheck` 和 `check-mlir-tutorial`。
> 这篇文章把它们串起来，专门讲 **lit**：
>
> - lit 是什么，和 FileCheck、`not` 各负责什么
> - 一个 `.mlir` 测试从被扫描到判定的完整过程
> - 本项目 CMake 和 Bazel 两套 lit 配置
> - 如何运行单个/全部测试
> - 如何新增测试、写负向测试、调试失败
>
> 本文假设你已经了解 MLIR 测试的基本形式；FileCheck 的详细语法见
> [FileCheck 使用指南](filecheck.md)。

## 1. 学习目标

- 理解 lit 不是断言库，而是测试发现、执行和汇总工具
- 理解 `ShTest` 如何执行 `RUN:` 行
- 理解 `%s`、`%t`、`%project_source_dir` 等替换
- 理解 CMake 的 `check-mlir-tutorial` 和 Bazel 的 `//tests:*.test`
- 能阅读 `tests/lit.cmake.cfg.py` 和 `tests/lit.cfg.py`
- 能新增一个正向/负向 lit 测试
- 能根据 `FAIL`、`UNRESOLVED`、`UNSUPPORTED` 定位问题
- 能解释 `lit`、`FileCheck`、`not` 三者的分工

---

## 2. lit、FileCheck、not 各做什么

先给结论：

```
lit         发现并执行测试，收集每个测试的退出码
ShTest      根据 RUN: 行启动命令
FileCheck   检查命令输出是否符合 CHECK: 断言
not         反转一个命令的退出码，用于“期望失败”的测试
```

它们不是互相替代的关系：

```
tests/poly_verifier.mlir
  │
  │ lit 发现这个测试
  ▼
ShTest 执行：
  tutorial-opt %s --canonicalize | FileCheck %s
  │
  ├── tutorial-opt 产生 IR
  └── FileCheck 用源文件中的 CHECK 断言 IR
  │
  ▼
lit 根据整条管道的退出码报告 PASS / FAIL
```

### 2.1 lit 不是断言库

lit 本身不知道“IR 应该长什么样”。它只负责：

- 找出哪些文件是测试
- 运行测试里的命令
- 判断命令退出码
- 汇总 PASS/FAIL 数量

真正检查输出内容的是 FileCheck。一个测试可以：

```
// RUN: tutorial-opt %s --canonicalize
```

只要命令退出码为 0，lit 就认为这条测试通过，即使输出完全不符合预期。因此
MLIR 测试通常必须配合 FileCheck。

### 2.2 `not` 是单独的工具

`not` 来自 LLVM：

```
not <command>
```

如果 `<command>` 失败，`not` 返回成功；如果 `<command>` 成功，`not` 返回失败。

所以：

```
// RUN: not tutorial-opt %s 2>&1 | FileCheck %s --check-prefix=ERROR
```

表示：

1. `tutorial-opt` 必须失败
2. 它的错误输出交给 FileCheck 检查
3. 如果它意外成功，`not` 失败，lit 报 FAIL

这比简单写 `not tutorial-opt %s` 更好，因为同时检查了错误消息。

---

## 3. 一个最小 lit 测试

`tests/mul_to_add_pdll.mlir` 的开头：

```
// RUN: tutorial-opt %s --mul-to-add-pdll | FileCheck %s
```

后面是：

```
func.func @just_power_of_two(%arg: i32) -> i32 {
  %0 = arith.constant 8 : i32
  %1 = arith.muli %arg, %0 : i32
  func.return %1 : i32
}

// CHECK-LABEL: func.func @just_power_of_two(
// CHECK-SAME:    %[[ARG:.*]]: i32
// CHECK-SAME:  ) -> i32 {
// CHECK:   %[[SUM_0:.*]] = arith.addi %[[ARG]], %[[ARG]]
// CHECK:   %[[SUM_1:.*]] = arith.addi %[[SUM_0]], %[[SUM_0]]
// CHECK:   %[[SUM_2:.*]] = arith.addi %[[SUM_1]], %[[SUM_1]]
// CHECK:   return %[[SUM_2]] : i32
// CHECK: }
```

lit 读到的信息只有：

```
测试文件：tests/mul_to_add_pdll.mlir
测试格式：ShTest
RUN 命令：tutorial-opt %s --mul-to-add-pdll | FileCheck %s
```

lit 不认识 `func.func`，也不认识 `CHECK:`；它只把 `%s` 替换成测试文件路径，
执行 RUN 命令，然后看退出码。

### 3.1 `%s` 是什么

`%s` 是 lit 提供的内建替换，表示：

```
当前测试源文件的绝对路径
```

例如：

```
/home/zhouzhou/npu/mlir-tutorial/tests/mul_to_add_pdll.mlir
```

`FileCheck %s` 中的 `%s` 也指向同一个源文件，所以 FileCheck 能从源文件中读取
`CHECK:` 注释。

### 3.2 为什么要有管道

`tutorial-opt` 默认输出去 stdout；`FileCheck` 默认从 stdin 读。管道把两者接起来：

```
tutorial-opt ... | FileCheck %s
```

lit 的测试配置默认启用 `pipefail=True`，因此管道中**任意一个命令**失败都会让
整条 RUN 行失败。比如 `tutorial-opt` 崩溃、FileCheck 断言不匹配，都会让测试
失败，而不是被管道后面的命令掩盖。

---

## 4. lit 如何发现测试

`lit` 拿到一个目录后，会：

1. 找到该目录或其父目录中的 `lit.cfg.py` / `lit.site.cfg.py`
2. 读取配置中的 `config.suffixes`
3. 递归扫描目录
4. 把符合后缀的文件当作测试
5. 根据 `config.test_format` 执行

本项目的 `config.suffixes` 都是：

```
config.suffixes = [".mlir"]
```

所以 `tests/` 下的每个 `.mlir` 都是 lit 测试候选；其他文件（例如 `.c`）默认不会
被当作测试。

### 4.1 CMake 路线

`tests/CMakeLists.txt`：

```
configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cmake.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cmake.cfg.py
)

set (MLIR_TUTORIAL_TEST_DEPENDS
    FileCheck count not
    mlir-opt
    mlir-runner
    # tutorial-opt
)

add_lit_testsuite(check-mlir-tutorial "Running the MLIR tutorial regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${MLIR_TUTORIAL_TEST_DEPENDS}
)
```

逐项解释：

- `configure_lit_site_cfg` 把 `.in` 模板配置成构建树中的
`build/tests/lit.site.cfg.py`
- `MAIN_CONFIG` 指向 `tests/lit.cmake.cfg.py`
- `add_lit_testsuite` 生成 `check-mlir-tutorial` 目标
- `DEPENDS` 指定测试运行前需要的工具
- `# tutorial-opt` 是注释掉的：这个依赖没有写进 CMake target，所以第一次运行
测试前要自己先构建 `tutorial-opt`

生成的目标实际执行：

```
llvm-lit -sv build/tests
```

`llvm-lit` 是 LLVM 构建出来的 lit 启动脚本，使用 LLVM 源码树里的 lit，不要求
系统 Python 安装 lit。

### 4.2 Bazel 路线

`tests/BUILD`：

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
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:mlir-opt",
        "@llvm-project//mlir:mlir-runner",
        "@llvm-project//mlir:mlir-translate",
        "@mlir_tutorial_pip_deps//lit",
    ],
)

glob_lit_tests()
```

`glob_lit_tests()` 会把每个 `.mlir` 变成一个 `py_test`：

```
tests/poly_verifier.mlir      -> //tests:poly_verifier.mlir.test
tests/mul_to_add.mlir         -> //tests:mul_to_add.mlir.test
...
```

每个 `py_test` 的 `data` 包含 `test_utilities` 和当前测试文件。`test_utilities`
再提供 `tutorial-opt`、`mlir-opt`、`FileCheck`、`clang` 等工具。

---

## 5. 两套配置文件

### 5.1 CMake：`tests/lit.cmake.cfg.py`

关键内容：

```
config.name = "MLIR_TUTORIAL"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.project_binary_dir, "tests")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%project_source_dir", config.project_source_dir))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

config.project_tools_dir = os.path.join(config.project_binary_dir, "tools")
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.project_tools_dir, config.llvm_tools_dir]
tools = ["mlir-opt", "mlir-runner", "tutorial-opt"]
llvm_config.add_tool_substitutions(tools, tool_dirs)
```

逐项解释：

| 配置 | 作用 |
| --- | --- |
| `config.name` | 测试套件名，显示为 `MLIR_TUTORIAL :: ...` |
| `test_format = ShTest(...)` | 用 shell 风格的 RUN 行执行测试 |
| `suffixes` | 只把 `.mlir` 当测试 |
| `test_source_root` | 测试源的根目录 |
| `test_exec_root` | 测试执行/输出目录 |
| `excludes` | 跳过 `Inputs` 等辅助目录 |
| `llvm_config.add_tool_substitutions` | 让 `tutorial-opt` 等命令可从 PATH/替换找到 |
| `%project_source_dir` | 项目源码根目录替换 |

`tests/lit.cmake.site.cfg.py.in` 是模板，构建时生成
`build/tests/lit.site.cfg.py`。真实配置由 CMake 写入：

```
config.llvm_tools_dir = ".../externals/llvm-project/build/bin"
config.project_binary_dir = ".../build"
config.project_source_dir = ".../mlir-tutorial"
```

不要直接改生成出来的 `build/tests/lit.site.cfg.py`；改 `.in` 或者
`lit.cmake.cfg.py`，然后重新配置/构建。

### 5.2 Bazel：`tests/lit.cfg.py`

这个文件是 Bazel 专用配置，和 CMake 配置不是同一个文件：

```
config.name = "mlir_tutorial"
config.test_format = ShTest()
config.suffixes = [".mlir"]

runfiles_dir = Path(os.environ["RUNFILES_DIR"])

tool_relpaths = [
    "+_repo_rules+llvm-project/mlir",
    "+_repo_rules+llvm-project/llvm",
    "+_repo_rules+llvm-project/clang",
    "_main/tools",
]

config.environment["PATH"] = (
    ":".join(str(runfiles_dir.joinpath(Path(path))) for path in tool_relpaths)
    + ":"
    + os.environ["PATH"]
)

config.environment["C_INCLUDE_PATH"] = str(
    runfiles_dir.joinpath(Path("+_repo_rules+llvm-project/clang/staging/include"))
)
```

它解决的问题：

- Bazel 测试运行在 `runfiles` 目录，工具不在系统 PATH
- `mlir-opt`、`tutorial-opt`、clang 位于 runfiles 的不同子目录
- Bazel clang 的内建头文件在 `clang/staging/include`，不在它默认的
resource directory 中，需要设置 `C_INCLUDE_PATH`

`RUNFILES_DIR` 是 Bazel 的测试环境变量，脱离 Bazel 直接运行这个配置文件会
报 `KeyError`。因此：

```
CMake  -> tests/lit.cmake.cfg.py + lit.cmake.site.cfg.py.in
Bazel  -> tests/lit.cfg.py + bazel/lit.bzl
```

两套配置都设置 `ShTest` 和 `.mlir` 后缀，所以测试文件本身不需要区分路线。

### 5.3 两套配置对比

| 项目 | CMake 路线 | Bazel 路线 |
| --- | --- | --- |
| lit 可执行文件 | `externals/.../bin/llvm-lit` | Bazel `@llvm-project//llvm:lit` |
| 配置文件 | `tests/lit.cmake.cfg.py` | `tests/lit.cfg.py` |
| 测试发现 | `add_lit_testsuite` 扫描目录 | `glob_lit_tests()` 生成每个 `.mlir.test` |
| 工具路径 | `llvm_tools_dir` + `build/tools` | Bazel runfiles + `test_utilities` |
| 测试命令 | `ninja check-mlir-tutorial` | `bazel test //tests:...` |
| 额外数据 | 源目录中直接可见 | 要加入 `test_utilities` 或测试的 data |
| 测试日志 | lit 输出/构建日志 | `bazel-testlogs/.../test.log` |

---

## 6. RUN 行、替换和测试指令

### 6.1 RUN 行

`ShTest` 的 `RUN:` 行是测试的入口：

```
// RUN: tutorial-opt %s --canonicalize | FileCheck %s
```

规则：

- 必须至少有一条 `RUN:`（否则 UNRESOLVED）
- 每条 RUN 行按顺序执行
- 一条命令失败，测试失败
- 命令按 shell 语法解释管道、重定向
- `//` 是 MLIR 注释；lit 从注释文本中提取 `RUN:`

### 6.2 常用替换

| 替换 | 含义 | 例子 |
| --- | --- | --- |
| `%s` | 当前测试源文件 | `tutorial-opt %s` |
| `%t` | 当前测试专用的临时文件路径 | `... > %t` |
| `%T` | 当前测试临时目录 | 需要多文件时使用 |
| `%p` | 当前测试源目录 | `%p/Inputs/foo.mlir` |
| `%project_source_dir` | 项目源码根目录（本项目自定义） | `%project_source_dir/tests/foo.c` |
| `%PATH%` | 当前 PATH（CMake 配置自定义） | 调试用 |
| `%shlibext` | 动态库扩展名（CMake 配置） | 平台差异测试 |

本项目的典型用法是 `tests/poly_to_llvm.mlir`：

```
// RUN: clang -c %project_source_dir/tests/poly_to_llvm_main.c
```

CMake 配置把 `%project_source_dir` 设为仓库根目录；Bazel 配置把它设为
runfiles 里的 `_main` 目录，因此两种路线都能找到 C 源文件。

### 6.3 临时文件

如果测试需要保存中间结果：

```
// RUN: tutorial-opt %s --canonicalize > %t
// RUN: FileCheck %s < %t
```

`%t` 是 lit 为当前测试分配的唯一路径。不要写死源目录里的文件名，否则并行测试会
互相覆盖。

### 6.4 `not` 和负向测试

`tests/poly_constant_verifiers.mlir`：

```
// RUN: not tutorial-opt %s 2>&1 | FileCheck %s --check-prefix=ERROR

// ERROR: 'poly.constant' op requires coefficients to be a rank-1 tensor
func.func @verifiers_constants() -> !poly.poly<10> {
  %a = poly.constant dense<[1, 2, 3]> : tensor<3xi32> : !poly.poly<10>
  %b = poly.constant dense<[[1,2,3],[10, 20, 30]]> : tensor<2x3xi32> : !poly.poly<10>
  return %a : !poly.poly<10>
}
```

为什么写成这样：

- `poly.constant` 的 verifier 要求 rank-1
- 第二个常量是 `2x3`，所以 `tutorial-opt` 必须失败
- `not` 把“失败”变成成功
- `2>&1` 把诊断信息送到 stdout
- FileCheck 用 `ERROR:` 前缀检查具体错误

如果 verifier 被误删，`not` 会失败，测试就能捕捉到回归。

### 6.5 FileCheck 和前缀

FileCheck 的默认前缀是 `CHECK:`。如果测试使用：

```
// ERROR: ...
```

就必须显式写：

```
FileCheck %s --check-prefix=ERROR
```

否则会看到：

```
error: no check strings found with prefix 'CHECK:'
```

一个测试可以有多组前缀，例如同时检查 stdout 和 stderr。

### 6.6 其他 lit 指令

MLIR/LLVM 测试还支持一些注释指令：

| 指令 | 作用 |
| --- | --- |
| `REQUIRES: feature` | 没有该 feature 时标记 UNSUPPORTED |
| `UNSUPPORTED: feature` | 有该 feature 时跳过 |
| `XFAIL: feature` | 期望失败；意外成功会变 XPASS |
| `RUN:` | 必需的执行命令 |

本项目主要使用 `RUN:` 和 `not`；`REQUIRES`/`XFAIL` 在需要平台条件时有用。
例如某些测试只在 x86 上运行，可以写：

```
// REQUIRES: x86-registered-target
```

具体 feature 名由 `config.available_features` 和构建配置决定，不是任意字符串。

---

## 7. 测试状态和退出码

lit 的常见状态：

| 状态 | 含义 |
| --- | --- |
| `PASS` | 所有 RUN 成功，FileCheck 也通过 |
| `FAIL` | 某条 RUN 失败 |
| `UNRESOLVED` | 没有 RUN 行，或测试配置有问题 |
| `UNSUPPORTED` | REQUIRES/UNSUPPORTED 条件不满足 |
| `XFAIL` | 按预期失败 |
| `XPASS` | 标了 XFAIL，但意外通过 |
| `TIMEOUT` | 单测超过 `--timeout` |

`lit` 的退出码：

- 全部 PASS/XFAIL/UNSUPPORTED：0
- 任意 FAIL/UNRESOLVED/XPASS：非 0

所以 CMake 的 `check-mlir-tutorial` 和 Bazel 的 `bazel test` 都能作为 CI 步骤。

---

## 8. CMake 路线：`llvm-lit` 与 `check-mlir-tutorial`

### 8.1 运行全部测试

先构建 `tutorial-opt`：

```
cd /home/zhouzhou/npu/mlir-tutorial
ninja -C build -j 100 tutorial-opt
```

再运行 lit：

```
ninja -C build check-mlir-tutorial
```

实际输出（当前环境）：

```
-- Testing: 25 tests, 25 workers --
...
Testing Time: 0.18s

Total Discovered Tests: 25
  Passed: 25 (100.00%)
```

注意：`tests/CMakeLists.txt` 里的 `# tutorial-opt` 依赖被注释掉了，
`check-mlir-tutorial` 本身不保证会构建 `tutorial-opt`。第一次运行时请先单独
构建 `tutorial-opt`，否则可能出现 `tutorial-opt: command not found`。

### 8.2 直接运行 `llvm-lit`

`check-mlir-tutorial` 实际执行的是：

```
$BIN/llvm-lit -sv build/tests
```

其中：

```
export TOP=$PWD
export BIN=$TOP/externals/llvm-project/build/bin
```

常用选项：

```
# 只显示失败详情，简洁输出
$BIN/llvm-lit -sv build/tests

# 只运行名字匹配 poly_verifier 的测试
$BIN/llvm-lit -sv build/tests --filter poly_verifier

# 显示所有测试（包括 PASS）
$BIN/llvm-lit -sv build/tests --show-pass

# 失败时显示完整 RUN 命令和输出，并在第一个失败后停止
$BIN/llvm-lit -v build/tests --max-failures=1

# 控制并行数
$BIN/llvm-lit -sv -j 8 build/tests
```

`--filter` 是在发现测试后做正则筛选；例如：

```
-- Testing: 1 of 25 tests, 1 workers --
...
Passed: 1
```

### 8.3 `lit` 的 `-sv` 是什么

- `-s`：succinct，少打印通过的测试
- `-v`：verbose，失败时打印完整的命令和输出
- `-sv`：两者组合，是 MLIR/LLVM 常见的测试输出形式

如果要看每条 RUN 的实际命令，用 `-a` 或 `-v`：

```
$BIN/llvm-lit -v build/tests --filter poly_verifier
```

失败输出会显示：

```
RUN: at line 1
# executed command: tutorial-opt ...
```

最后打印的命令通常是失败的入口。

---

## 9. Bazel 路线

### 9.1 运行一个测试

```
cd /home/zhouzhou/npu/mlir-tutorial

bazel test //tests:mul_to_add_pdll.mlir.test \
  --test_output=all
```

成功时：

```
PASS: mlir_tutorial :: mul_to_add_pdll.mlir (1 of 1)
...
//tests:mul_to_add_pdll.mlir.test    PASSED in 0.6s
```

### 9.2 运行全部测试

```
bazel test //tests:all --test_output=errors
```

当前环境的结果：

```
INFO: Found 26 targets and 25 test targets...
Executed N out of 25 tests: 25 tests pass.
```

（`N` 取决于缓存命中数；最终 25 个测试全部通过。）

### 9.3 测试日志

```
cat bazel-testlogs/tests/mul_to_add_pdll.mlir.test/test.log
```

Bazel 的测试日志在 `bazel-testlogs` 符号链接下，不在源码目录。

### 9.4 Bazel 如何找到 `tutorial-opt`

`tests/BUILD` 的 `test_utilities` 把 `//tools:tutorial-opt` 作为 `data`；
`bazel/lit.bzl` 又把 `test_utilities` 放进每个测试的 `data`。测试运行时
`tutorial-opt` 位于 runfiles 的 `_main/tools`，由 `lit.cfg.py` 加入 PATH。

---

## 10. 新增一个 lit 测试

### 10.1 正向测试

在 `tests/` 下新建 `tests/my_pass.mlir`：

```
// RUN: tutorial-opt %s --canonicalize | FileCheck %s

func.func @demo(%x: i32) -> i32 {
  %c = arith.constant 1 : i32
  %r = arith.addi %x, %c : i32
  return %r : i32
}

// CHECK-LABEL: func.func @demo
// CHECK: arith.addi
```

CMake 路线：

```
ninja -C build tutorial-opt
ninja -C build check-mlir-tutorial
```

Bazel 路线：

```
bazel query '//tests:my_pass.mlir.test'
bazel test //tests:my_pass.mlir.test --test_output=all
```

两套路线都会自动发现它：

- CMake `add_lit_testsuite` 扫描 `tests/` 下的 `.mlir`
- Bazel `glob_lit_tests()` 在加载 BUILD 时 glob `*.mlir`

### 10.2 负向测试

如果要测试 verifier 报错：

```
// RUN: not tutorial-opt %s 2>&1 | FileCheck %s --check-prefix=ERROR

func.func @bad() -> i32 {
  %c = poly.constant dense<[[1, 2]]> : tensor<1x2xi32> : !poly.poly<4>
  return %c : !poly.poly<4>
}

// ERROR: 'poly.constant' op requires coefficients to be a rank-1 tensor
```

要点：

- `not` 期望 `tutorial-opt` 失败
- `2>&1` 把 stderr 合到 stdout
- `--check-prefix=ERROR` 与 `ERROR:` 对应
- 错误消息要足够稳定；过于具体的 SSA 名称可能随格式变化

### 10.3 使用 `%t`

如果中间结果需要被 FileCheck 读取：

```
// RUN: tutorial-opt %s --canonicalize > %t
// RUN: FileCheck %s < %t

// CHECK: ...
```

`%t` 是当前测试专属路径，适合并行运行。

### 10.4 需要额外文件时

CMake 路线的测试源目录就是 `tests/`，`%project_source_dir` 指向仓库根目录，
因此 RUN 行可以直接访问：

```
// RUN: clang -c %project_source_dir/tests/foo.c
```

Bazel 路线的测试进程只能看到 `data` 里的文件。`glob_lit_tests()` 默认只把当前
`.mlir` 和全局 `test_utilities` 放进去。如果新测试需要额外的 `.c`、`.txt` 或
输入文件，有两种做法：

1. 把该文件加入 `tests/BUILD` 的 `test_utilities.data`
2. 扩展 `bazel/lit.bzl`，让 `lit_test` 接受额外的 data（需要改宏）

例如 `tests/poly_to_llvm_main.c` 就是通过全局 `test_utilities` 提供的：

```
data = [
    "//tests:poly_to_llvm_main.c",
    ...
]
```

---

## 11. 调试 lit 测试

### 11.1 `UNRESOLVED: Test has no 'RUN:' line`

原因：文件被 lit 发现，但没有 RUN 行。解决：

- 加 `// RUN: ...`
- 或者把文件移出测试目录/改后缀
- 如果它是辅助文件，放到 `Inputs/` 等被排除的目录

本课程新增的 `tests/poly_constant_verifiers.mlir` 最初就是这个问题；现在它有一条
期望失败的 RUN 行。

### 11.2 `command not found`

常见于：

- CMake：没有先构建 `tutorial-opt`
- Bazel：工具没有加入 `test_utilities`，或没有加入 lit 的 PATH
- `clang`：Bazel clang 在 runfiles 中，需要 PATH 和头文件路径

先看 lit verbose 输出中实际执行的命令，再确认工具是否在 PATH：

```
$BIN/llvm-lit -v build/tests --filter my_test
```

Bazel：

```
bazel test //tests:my_test.mlir.test --test_output=all
```

### 11.3 `FileCheck` 找不到 CHECK 前缀

错误：

```
error: no check strings found with prefix 'CHECK:'
```

原因：RUN 行使用了 `--check-prefix=ERROR`，但文件里的断言写成了 `CHECK:`；
或者反过来。统一前缀即可。

### 11.4 相对路径找不到文件

lit 的执行目录和源目录不一定相同。不要假定当前目录就是 `tests/`：

- 用 `%s` 指当前文件
- 用 `%t` 指临时输出
- 用 `%project_source_dir` 指项目源码根目录
- 用 `%p` 指测试源目录

### 11.5 测试并行互相覆盖

错误写法：

```
// RUN: tutorial-opt %s > fixed_name.mlir
// RUN: FileCheck %s < fixed_name.mlir
```

多个测试并行时会冲突。使用 `%t`。

### 11.6 CMake 的测试目标依赖

前面提到 `tests/CMakeLists.txt` 中 `# tutorial-opt` 依赖被注释。当前安全做法：

```
ninja -C build tutorial-opt
ninja -C build check-mlir-tutorial
```

不要假设 `check-mlir-tutorial` 会自动构建 `tutorial-opt`。

### 11.7 Bazel 缓存导致“看起来没重新跑”

Bazel 会缓存测试结果。想强制重跑：

```
bazel test //tests:my_test.mlir.test \
  --cache_test_results=no --test_output=all
```

### 11.8 查看所有测试或失败详情

```
# CMake
$BIN/llvm-lit -sv build/tests --show-pass
$BIN/llvm-lit -v build/tests --filter my_test

# Bazel
bazel test //tests:all --test_output=all
```

---

## 12. 本项目的 25 个测试

当前 `tests/` 下的 `.mlir` 文件主要分为：

| 类别 | 例子 |
| --- | --- |
| Poly 语法/常量/折叠 | `poly_syntax.mlir`、`poly_constant_verifiers.mlir`、`poly_fold.mlir` |
| Poly canonicalizer/verifier/conversion | `poly_canonicalize.mlir`、`poly_verifier.mlir`、`poly_to_standard.mlir` |
| Poly 到 LLVM 的端到端测试 | `poly_to_llvm.mlir`、`poly_to_llvm_eval.mlir` |
| Affine pass | `affine_loop_unroll.mlir`、`dyn_unroll.mlir` |
| Arith/PDLL | `mul_to_add.mlir`、`mul_to_add_pdll.mlir` |
| Noisy 全局优化 | `noisy_syntax.mlir`、`noisy_reduce_noise.mlir` |
| 通用 MLIR pass/trait 演示 | `cse.mlir`、`sccp.mlir`、`code_motion.mlir` |
| Box 迷你 dialect | `box_syntax.mlir`、`box_fold.mlir`、`box_verifier.mlir` |

CMake 和 Bazel 都能运行这一整套；测试文件不需要为构建系统改写，只有辅助数据和
配置不同。

---

## 13. 练习

### 练习 1：运行全部和单个测试

用 CMake 运行全部测试；再用 lit 的 `--filter` 只运行 `poly_verifier`。

<details>
<summary>参考答案</summary>

```
cd /home/zhouzhou/npu/mlir-tutorial
ninja -C build tutorial-opt
ninja -C build check-mlir-tutorial

export BIN=$PWD/externals/llvm-project/build/bin
$BIN/llvm-lit -sv build/tests --filter poly_verifier
```

</details>

### 练习 2：用 Bazel 运行同一个测试

<details>
<summary>参考答案</summary>

```
bazel test //tests:poly_verifier.mlir.test --test_output=all
```

</details>

### 练习 3：新增一个负向测试

为一个会触发 verifier 错误的输入写测试。要求：

- 使用 `not`
- 用 `ERROR:` 检查错误消息
- 在 CMake 和 Bazel 下都能运行

<details>
<summary>参考答案</summary>

```
// RUN: not tutorial-opt %s 2>&1 | FileCheck %s --check-prefix=ERROR

func.func @bad() -> i32 {
  %c = poly.constant dense<[[1, 2]]> : tensor<1x2xi32> : !poly.poly<4>
  return %c : !poly.poly<4>
}

// ERROR: 'poly.constant' op requires coefficients to be a rank-1 tensor
```

CMake 会通过 `add_lit_testsuite` 发现；Bazel 会通过 `glob_lit_tests()` 生成
`//tests:bad.mlir.test`。

</details>

### 练习 4：解释 `%s` 和 `%project_source_dir`

为什么 `tests/poly_to_llvm.mlir` 里引用 C 文件使用
`%project_source_dir/tests/poly_to_llvm_main.c`，而不是 `tests/...`？

<details>
<summary>参考答案</summary>

lit 的执行目录通常是构建目录或 runfiles，不保证等于仓库根目录。`%s` 只是当前
测试文件，不能直接拼出项目其他源码的路径。`%project_source_dir` 明确指向项目
源码根目录；CMake 配置指向仓库根，Bazel 配置指向 runfiles 中的 `_main`。

</details>

### 练习 5：为什么 `check-mlir-tutorial` 要先构建 `tutorial-opt`

<details>
<summary>参考答案</summary>

`tests/CMakeLists.txt` 的 `MLIR_TUTORIAL_TEST_DEPENDS` 中 `# tutorial-opt`
被注释掉了，所以 `check-mlir-tutorial` 目标没有声明对 `tutorial-opt` 的依赖。
在干净构建中，lit 可能先运行，此时 `tutorial-opt` 还不存在，测试会报
`command not found`。先单独构建 `tutorial-opt` 可以避免这个问题。

</details>

### 练习 6：`lit`、FileCheck、`not` 的区别

<details>
<summary>参考答案</summary>

- lit：发现并执行测试，汇总 PASS/FAIL；不检查 IR 内容
- FileCheck：根据 `CHECK:` 等断言检查输出
- `not`：反转命令退出码，用于期望失败的测试或命令

一个测试通常把三者组合起来：

```
// RUN: not tutorial-opt %s 2>&1 | FileCheck %s --check-prefix=ERROR
```

</details>

---

## 14. 验收清单

- 能说出 lit、ShTest、FileCheck、`not` 的分工
- 能解释 `%s`、`%t`、`%project_source_dir`
- 能解释 CMake 的 `check-mlir-tutorial` 如何生成
- 能解释 Bazel 的 `glob_lit_tests()` 如何生成测试目标
- 能运行全部测试和单个测试
- 能新增正向测试
- 能新增负向测试并检查错误消息
- 能处理 `UNRESOLVED: Test has no 'RUN:' line`
- 能处理 `command not found` 和 FileCheck 前缀错误
- 能读懂 `build/tests/lit.site.cfg.py` 的来源
- 知道 CMake 和 Bazel 的 lit 配置不是同一个文件

## 15. 参考

- LLVM lit 命令指南：
  <https://llvm.org/docs/CommandGuide/lit.html>
- LLVM lit 源码与 README：
  `externals/llvm-project/llvm/utils/lit/`
- 补充：[FileCheck 使用指南](filecheck.md)
- 补充：[Bazel 使用指南](bazel.md)
- 本项目 lit 文件：
  - `tests/lit.cmake.cfg.py`
  - `tests/lit.cmake.site.cfg.py.in`
  - `tests/lit.cfg.py`
  - `tests/CMakeLists.txt`
  - `tests/BUILD`
  - `bazel/lit.bzl`
- 本课程的主线实践：
  - [第 1 课：IR 基础与测试](01-ir-basics.md)
  - [第 2 课：第一个 pass](02-first-pass.md)
  - [第 12 课：PDLL](12-pdll.md)