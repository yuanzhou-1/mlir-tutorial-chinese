一句话总结
# 第 0 课：环境搭建——从零构建 LLVM/MLIR 与 tutorial-opt

> 对应文章 1：[Build System (Getting Started)](https://jeremykun.com/2023/08/10/mlir-getting-started/)。
> 文章使用 Bazel 讲解项目组织和构建；本环境没有 Bazel，改用仓库同样维护的 CMake
> 构建，产物和 pass 行为一致。
>
> 本课完整讲解从零搭建学习环境的全过程：探测环境、拉取源码、构建 LLVM/MLIR、
> 构建并验证 `tutorial-opt`，并记录实际执行的命令、参数含义、真实耗时、产物和踩坑。
>
> 完成标志：`build/tools/tutorial-opt` 可以正常运行。

## 1. 学习目标

- 理解本项目依赖的三个层次：LLVM/MLIR 源码、MLIR 构建产物、本项目的 `tutorial-opt`
- 能独立完成一次从零构建
- 知道每个 CMake 选项为什么这样选，出问题时知道去哪里看日志

### 本课流程

1. 探测环境，确定构建路线（第 2 节）
2. 拉取 pinned LLVM 源码（第 3 节）
3. 构建 LLVM/MLIR（第 4 节）
4. 构建 `tutorial-opt`（第 5 节）
5. 验证产物（第 6 节）
6. 了解产物目录、常见问题和日常重建方式（第 7–9 节）
7. 用验收清单自检（第 10 节）

## 2. 环境探测结果

以下为 2026-10-19 的环境探测结果。

| 项目 | 结果 |
|---|---|
| OS | Ubuntu 24.04 |
| CPU / 内存 / 磁盘 | 128 核 / 503 GiB / 12 TiB 可用 |
| GCC | 13.3.0 |
| CMake | 3.28.3 |
| Ninja | 1.11.1 |
| Python | 3.12.3 |
| Bazel / Bazelisk | ❌ 未安装 |
| clang | ❌ 未安装 |
| LLVM 子模块 | ❌ 初始为空 |

### 为什么选择 CMake 而不是 Bazel？

教程文章主线使用 Bazel（Bzlmod），但仓库同时维护等价的 CMake 构建，**产出的
`tutorial-opt` 和所有 pass 行为完全一致**，学习 MLIR 本身没有区别。

当前环境没有 Bazel，而 CMake/Ninja/GCC 已齐全，因此选择 CMake，避免额外安装。
如果你想跟文章里的 Bazel 命令一一对应，可以后续安装 Bazelisk 再走 Bazel 路线。

## 3. 第一步：拉取 pinned LLVM 子模块

仓库通过 `.gitmodules` 依赖 `externals/llvm-project`，并在 `extensions.bzl` 中把
LLVM 固定到 commit `d9190f8141661bd6120dea61d28ae8940fd775d0`（2025-07-16）。

```bash
cd /home/zhouzhou/npu/mlir-tutorial
git submodule update --init --depth 1 externals/llvm-project

# 验证：必须与 pinned commit 一致
git -C externals/llvm-project rev-parse HEAD
# d9190f8141661bd6120dea61d28ae8940fd775d0
```

说明：

- `--depth 1` 只拉取该 commit 的浅克隆，节省时间和磁盘（实际约 2.2 GB）
- 如果以后需要切换/检视历史，去掉 `--depth 1` 重新初始化即可
- CMake 路线需要这份源码；Bazel 路线会自己从 git 下载同一个 commit

## 4. 第二步：构建 LLVM/MLIR

### 配置命令

```bash
LLVM_SRC=$PWD/externals/llvm-project

cmake -S $LLVM_SRC/llvm -B $LLVM_SRC/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_PARALLEL_COMPILE_JOBS=100 \
  -DLLVM_PARALLEL_LINK_JOBS=8 \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++
```

### 每个选项的含义

| 选项 | 作用 / 为什么这样选 |
|---|---|
| `CMAKE_BUILD_TYPE=Release` | 优化构建，工具运行快；开发 MLIR pass 时这个速度很重要 |
| `LLVM_ENABLE_ASSERTIONS=ON` | **开发编译器时必须保留**：IR 非法、类型不匹配时会断言失败，比静默出错好查 |
| `LLVM_ENABLE_PROJECTS=mlir` | 只构建 MLIR，不构建 clang/lldb 等，节省大量时间 |
| `LLVM_TARGETS_TO_BUILD=X86` | 只生成 X86 后端；MLIR 本身与目标无关，`mlir-runner`/`llc` 需要至少一个后端 |
| `LLVM_INSTALL_UTILS=ON` | 构建 FileCheck 等测试工具 |
| `LLVM_PARALLEL_COMPILE_JOBS=100` | 编译并发数 |
| `LLVM_PARALLEL_LINK_JOBS=8` | 链接并发数，避免大量链接同时吃内存 |

> 如果以后要跑 AArch64/GPU 相关实验，把 `X86` 改成 `X86;AArch64;NVPTX` 并重新配置。

### 编译命令

```bash
ninja -C $LLVM_SRC/build -j 100 \
  mlir-opt mlir-translate mlir-runner FileCheck count not llc
```

实际结果：

- 总目标数：**4125**
- 耗时：约 **6.5 分钟**（15:41 → 15:47，100 并发）
- 产物目录：`externals/llvm-project/build/bin/`

```bash
$BIN/mlir-opt --version
# LLVM version 22.0.0git
# Optimized build with assertions.
```

## 5. 第三步：构建 tutorial-opt

### 配置命令

```bash
cd /home/zhouzhou/npu/mlir-tutorial

cmake -S . -B build -G Ninja \
  -DLLVM_DIR=$PWD/externals/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=$PWD/externals/llvm-project/build/lib/cmake/mlir \
  -DBUILD_DEPS=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=OFF \
  -DBUILD_SAMPLES=OFF \
  -DBUILD_EXAMPLES=OFF
```

要点：

- `LLVM_DIR` / `MLIR_DIR` 指向**构建树**里的 CMake config，不需要先 `ninja install`
- `BUILD_DEPS=ON`：项目根 `CMakeLists.txt` 会用 FetchContent 拉取 **or-tools v9.11**
  （第 5 课的 ILP 求解需要它）
- `CMAKE_BUILD_TYPE=Debug`：方便用 gdb 单步调试自己的 pass；与 Release 的 LLVM 混用没问题
- `BUILD_TESTING/SAMPLES/EXAMPLES=OFF`：减少 or-tools 的构建量

> 注意：or-tools 的拉取+配置耗时约 **24 分钟**（网络下载），只发生一次。
> 实际结果是 `Configuring done (1442.4s)`。如果卡住，检查网络和
> `build/_deps/or-tools-src` 是否正在增长。

### 编译命令

```bash
ninja -C build -j 100 tutorial-opt
```

实际结果：

- 总目标数：**1792**
- 耗时：约 **2 分钟**
- 产物：`build/tools/tutorial-opt`

## 6. 验证环境

```bash
cd /home/zhouzhou/npu/mlir-tutorial
BIN=$PWD/externals/llvm-project/build/bin
TO=$PWD/build/tools/tutorial-opt

# 1) MLIR 工具可用
$BIN/mlir-opt --version

# 2) tutorial-opt 可用
$TO --help | head -20

# 3) 跑第一个 pass
$TO tests/affine_loop_unroll.mlir --affine-full-unroll | head -20
```

能看到展开后的 IR（`affine.for` 消失）即环境搭建成功。

## 7. 产物与空间占用

| 路径 | 内容 | 大小 |
|---|---|---|
| `externals/llvm-project` | LLVM 源码 + 构建树 | 4.3 GB |
| `build/` | tutorial 构建树（含 or-tools） | 7.8 GB |
| `build/tools/tutorial-opt` | 本课程的主角 | 数十 MB |
| `externals/llvm-project/build/bin/*` | mlir-opt、FileCheck、mlir-runner 等 | — |

常用工具：

```bash
BIN=$PWD/externals/llvm-project/build/bin
$BIN/mlir-opt          # MLIR 官方 pass driver（含全部内置 pass）
$BIN/mlir-translate    # MLIR <-> LLVM IR 等格式转换
$BIN/mlir-runner       # 解释执行含 LLVM dialect 的 MLIR
$BIN/FileCheck         # lit 测试的断言工具
$BIN/llc               # LLVM 后端编译器
```

## 8. 踩坑记录

1. **日志写入权限**：最初把构建日志写到系统临时目录，文件恰好属于其他用户，
   重定向时报 `Permission denied`，导致构建根本没启动。解决：日志统一写到项目内的
   `build/logs/`（例如 `build/logs/llvm-build.log`）。
2. **`ninja: error: unknown target 'clang'`**：因为配置时
   `LLVM_ENABLE_PROJECTS=mlir` 没有包含 clang。解决：先不构建 clang；第 4 课做
   端到端执行时需要 clang，届时重新配置
   `-DLLVM_ENABLE_PROJECTS='clang;mlir'` 后构建，或使用系统/apt 的 clang。
3. **lit 未安装**：`tests/` 使用 lit，但当前 Python 环境没有 `lit` 模块
   （`requirements.txt` 里是 `lit==18.1.8`）。目前我们直接运行各测试文件顶部的
   `RUN:` 命令；如果要跑 `ninja check-mlir-tutorial`，需要先安装 lit，例如：
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
4. **or-tools 配置慢**：主要耗时在 git clone 和 CMake 配置，不是编译；属正常现象。

## 9. 日常重建命令

```bash
# 改了自己的 pass 后，只重建 tutorial-opt（秒级到分钟级）
ninja -C /home/zhouzhou/npu/mlir-tutorial/build -j 100 tutorial-opt

# 想清理 tutorial 构建（不会影响 LLVM 构建）
rm -rf /home/zhouzhou/npu/mlir-tutorial/build

# LLVM 构建树建议保留；除非要换 pinned commit 或配置，不必重建
```

## 10. 验收清单

- [ ] 能说出 `LLVM_ENABLE_ASSERTIONS=ON` 对 MLIR 开发为什么重要
- [ ] 能解释 `LLVM_DIR`/`MLIR_DIR` 为什么指向构建树而不是安装目录
- [ ] 能独立重建 `tutorial-opt` 并运行一个 pass
- [ ] 知道 `mlir-opt` 和 `tutorial-opt` 的关系（同一个 driver 框架，注册的 dialect/pass 不同）
- [ ] 遇到构建错误时知道先看哪个日志

## 下一课

[第 1 课：MLIR IR 心智模型](01-ir-basics.md)