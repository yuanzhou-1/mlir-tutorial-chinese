# MLIR 手把手学习课程

本课程以 j2kun 的 **MLIR For Beginners** 文章系列为主线：**一篇文章 = 一课**。
每课先写配套学习文档，再讲解、做实验。文档中的代码全部来自本仓库，并给出可复现
命令、真实输出、源码位置和练习。

- 文章系列：<https://jeremykun.com/2023/08/10/mlir-getting-started/>
- 代码仓库：<https://github.com/j2kun/mlir-tutorial>

> 编号说明：上游 README 把 PDLL 文章也编号为 12（重复编号）。本课程按实际顺序
> 把它称为**文章 13 / 第 12 课**，保证一一对应。
>
> 课程编号关系：**第 N 课 ↔ 文章 N+1**。

## 课程总表

| 课 | 文章 | 主题 | 主要代码 / 实验 | 文档 | 状态 |
|---|---|---|---|---|---|
| 0 | 1 | Build System (Getting Started) | LLVM/MLIR 构建、`tutorial-opt` 骨架 | [00-environment.md](00-environment.md) | ✅ 完成 |
| 1 | 2 | Running and Testing a Lowering | IR 基础、ctlz lowering、lit/FileCheck、`mlir-runner` | [01-ir-basics.md](01-ir-basics.md) | ✅ 完成 |
| 2 | 3 | Writing Our First Pass | `AffineFullUnroll` 手动版与 RewritePattern 版 | [02-first-pass.md](02-first-pass.md) | ✅ 完成 |
| 3 | 4 | Using TableGen for Passes | `Passes.td`、生成的 base class、pass 注册 | [03-tablegen-passes.md](03-tablegen-passes.md) | 🟡 进行中 |
| 4 | 5 | Defining a New Dialect | Poly dialect / type / op、assemblyFormat | 04-new-dialect.md | |
| 5 | 6 | Using Traits | `Has32BitArguments`、`Pure`/`ElementwiseMappable`、LICM/sink | 05-traits.md | |
| 6 | 7 | Folders and Constant Propagation | `fold()`、常量折叠、`ConstantLike` | 06-folders.md | |
| 7 | 8 | Verifiers | `EvalOp::verify`、trait verifier、`poly_verifier.mlir` | 07-verifiers.md | |
| 8 | 9 | Canonicalizers and Declarative Rewrite Patterns | `PolyPatterns.td`、`DifferenceOfSquares` | 08-canonicalizers.md | |
| 9 | 10 | Dialect Conversion | TypeConverter、`PolyToStandard` | 09-dialect-conversion.md | |
| 10 | 11 | Lowering through LLVM | `--poly-to-llvm` pipeline、端到端执行 | 10-lowering-through-llvm.md | |
| 11 | 12 | A Global Optimization and Dataflow Analysis | Noisy dialect、IntegerRangeAnalysis、ILP | 11-global-optimization-dataflow.md | |
| 12 | 13 | Defining Patterns with PDLL | `MulToAdd.pdll`、PDLL pass | 12-pdll.md | |

## 每课内容预告

### 第 2 课：Writing Our First Pass（文章 3）

- `tutorial-opt` 的项目组织与 pass 入口
- 用 `runOnOperation()` + `walk()` 手动展开 `affine.for`
- 把同一功能封装成 `OpRewritePattern`
- `applyPatternsAndFoldGreedily` 的工作方式
- 对应代码：`lib/Transform/Affine/AffineFullUnroll.cpp`、
  `AffineFullUnrollPatternRewrite.cpp`

### 第 3 课：Using TableGen for Passes（文章 4）

- `Passes.td` 如何声明一个 pass
- `GEN_PASS_DEF_*`、生成的 base class
- pass 注册（`registerAffinePasses`）与 dependent dialects
- 自己新增一个 pass 声明

### 第 4 课：Defining a New Dialect（文章 5）

- ODS/TableGen：Dialect、Type、Op 的定义
- `assemblyFormat` 与默认 type printer/parser
- Poly 的多项式类型 `!poly.poly<N>`、`poly.add/sub/mul/eval/...`
- 测试：`tests/poly_syntax.mlir`

### 第 5 课：Using Traits（文章 6）

- `Pure`、`ElementwiseMappable`、`SameOperandsAndResultType`
- 自定义 trait `Has32BitArguments` 及 `verifyTrait`
- trait 如何让通用优化（LICM、control-flow-sink、SCCP）自动生效
- 测试：`tests/code_motion.mlir`、`control_flow_sink.mlir`、`sccp.mlir`

### 第 6 课：Folders and Constant Propagation（文章 7）

- `hasFolder`、`Op::fold`
- `arith.constant` 与 `poly.constant` 的常量传播
- `ConstantLike` trait
- 折叠与 canonicalize/CSE 的关系

### 第 7 课：Verifiers（文章 8）

- op 级 `verify()`
- trait 级 `verifyTrait()`
- 用 `-verify-diagnostics` 测试非法 IR
- 测试：`tests/poly_verifier.mlir`

### 第 8 课：Canonicalizers and Declarative Rewrite Patterns（文章 9）

- `getCanonicalizationPatterns`
- TableGen 声明式 pattern：`PolyPatterns.td`
- `DifferenceOfSquares`、`LiftConjThroughEval`
- 测试：`tests/poly_canonicalize.mlir`

### 第 9 课：Dialect Conversion（文章 10）

- `TypeConverter`：`!poly.poly<N>` → `tensor<Nxi32>`
- `OpConversionPattern` 与 `ConversionTarget`
- `applyPartialConversion`
- 测试：`tests/poly_to_standard.mlir`

### 第 10 课：Lowering through LLVM（文章 11）

- linalg、bufferization、memref、scf/cf、arith、func 的 lowering 顺序
- 注册 `--poly-to-llvm` pipeline
- `mlir-translate` + `mlir-runner` 端到端执行
- 测试：`tests/poly_to_llvm.mlir`、`poly_to_llvm_eval.mlir`

### 第 11 课：A Global Optimization and Dataflow Analysis（文章 12）

- Noisy dialect：噪声模型、`encode/decode/reduce_noise`
- `InferIntRangeInterface` 与整数范围分析
- 用 or-tools/SCIP 求解 reduce_noise 插入位置（ILP）
- 测试：`tests/noisy_*.mlir`

### 第 12 课：Defining Patterns with PDLL（文章 13）

- PDLL 语法与 C++ pattern 的对比
- `MulToAdd.pdll` → 生成 pattern
- PDLL pass 的构建与注册
- 测试：`tests/mul_to_add_pdll.mlir`

## 命测试文件说明

`tests/` 下有一部分测试用的是** MLIR 自带 pass**，不是本仓库实现的：

- `ctlz*.mlir`：`convert-math-to-funcs`（MLIR 内置 lowering，文章 2 的示例）
- `sccp.mlir`：`sccp`
- `code_motion.mlir`：`loop-invariant-code-motion`
- `control_flow_sink.mlir`：`control-flow-sink`

它们用于演示 MLIR 已有能力和 trait/analysis 的效果。本仓库自己实现的 pass 在
`lib/Transform/` 和 `lib/Conversion/` 下。

## 命令约定

除非特别说明，所有命令都在仓库根目录执行：

```bash
cd /home/zhangyuanzhou/npu/mlir-tutorial

export TOP=$PWD
export BIN=$TOP/externals/llvm-project/build/bin   # mlir-opt、FileCheck、mlir-runner 等
export TO=$TOP/build/tools/tutorial-opt            # 本项目的 pass driver
```

`tutorial-opt` 的用法与 `mlir-opt` 基本一致：

```bash
$TO <输入文件> --<pass 名>
```

## 每课文档的结构

1. **学习目标**：这一课结束后你应该能做什么
2. **概念**：必要的背景知识
3. **实验**：可复现的命令与真实输出
4. **源码**：与实验对应的代码位置
5. **练习**：动手任务与思考题（附参考答案）
6. **验收清单**：判断自己是否掌握

## 学习方式

1. 先读对应文章（链接见总表）和本课文档
2. 运行文档中的实验，先预测输出再验证
3. 做练习；卡住时回看源码，而不是直接看答案
4. 学完一课，更新总表状态，再进入下一课

## 环境说明

本环境没有 Bazel，课程使用仓库维护的 **CMake** 构建；文章 1 讲的是 Bazel，
两者产物和 pass 行为一致，差异及构建步骤见
[第 0 课文档](00-environment.md)。

- LLVM/MLIR：pinned commit `d9190f8`（LLVM 22.0.0git），构建于
  `externals/llvm-project/build`
- `tutorial-opt`：`build/tools/tutorial-opt`（Debug，含 or-tools v9.11）
- `lit` 尚未安装；测试目前通过手动执行各文件的 `RUN:` 命令完成，
  安装方法见第 1 课文档