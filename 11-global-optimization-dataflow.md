# 第 11 课：A Global Optimization and Dataflow Analysis

> 对应文章：[A Global Optimization and Dataflow Analysis](https://jeremykun.com/2023/11/15/mlir-a-global-optimization-and-dataflow-analysis/)。
>
> 本课做两件事：
>
> 1. 用 MLIR 的 `IntegerRangeAnalysis` 传播“噪声”范围，验证程序是否合法；
> 2. 把“在哪些位置插入 `reduce_noise`”建模成整数线性规划（ILP），
> 用 or-tools/SCIP 全局求解。
>
> 这是本课程里第一个**全局优化**：不是逐个 op 改写，而是看整个函数后决定
> 一组插入点。

## 1. 学习目标

- 理解 noisy integer 模型：初始噪声、最大值、加法/乘法如何增长
- 理解 `InferIntRangeInterface` 和 `IntegerRangeAnalysis`
- 理解 dataflow analysis 的 transfer function / join / fixed point
- 理解 `DataFlowSolver` 如何加载 `DeadCodeAnalysis` 和整数范围分析
- 理解 ILP 的变量、目标函数和约束
- 理解 big-M 线性化“if 决策变量 then 重置噪声”
- 理解 pass 如何根据 solver 解插入 `reduce_noise`
- 能解释四个测试为什么得到不同数量的 `reduce_noise`

## 2. Noisy integer 模型

### 2.1 基本设定

`noisy` dialect 表示“带噪声的 32 位整数”：

```
真实值藏在高位，低位是噪声
初始噪声：INITIAL_NOISE = 12
最大允许噪声：MAX_NOISE = 26
```

`!noisy.i32` 和普通 `i32` 的区别：

- 普通 `i32`：值就是值
- `!noisy.i32`：值的低位可能被噪声污染

`noisy.decode` 把 noisy 值解回 `i1`..`i5` 的普通整数；如果噪声太高，结果就不可靠。

### 2.2 噪声如何增长

| op | 噪声变化 |
| --- | --- |
| `noisy.encode` | 固定为 12 |
| `noisy.add` | `max(lhs, rhs) + 1` |
| `noisy.sub` | `max(lhs, rhs) + 1` |
| `noisy.mul` | `lhs + rhs` |
| `noisy.reduce_noise` | 重置为 12 |

为什么 add 是 `max + 1`？

两个噪声上界取较大者，相加最多再多进一位，所以再加 1。

为什么 mul 是相加？

噪声按乘法放大，近似上界是两个输入的噪声上界之和。

### 2.3 `reduce_noise` 是昂贵的

`noisy.reduce_noise` 把噪声重置为 12，但代价很高。

目标：

> 在保证任何时刻噪声不超过 26 的前提下，尽量少插入 `reduce_noise`。

这是一次全局决策：插入点会互相影响。

## 3. Noisy dialect 的结构

### 3.1 类型

`lib/Dialect/Noisy/NoisyTypes.td`：

```
def Noisy_I32 : Noisy_Type<"NoisyI32", "i32"> {
  let summary = "A type for approximate 32-bit integers.";
}
```

IR 中写成：

```
!noisy.i32
```

### 3.2 常量

`lib/Dialect/Noisy/NoisyDialect.h`：

```
constexpr int INITIAL_NOISE = 12;
constexpr int MAX_NOISE = 26;
```

### 3.3 常用 op

```
%0 = noisy.encode %x : i5 -> !noisy.i32
%1 = noisy.add %a, %b : !noisy.i32
%2 = noisy.sub %a, %b : !noisy.i32
%3 = noisy.mul %a, %b : !noisy.i32
%4 = noisy.reduce_noise %3 : !noisy.i32
%5 = noisy.decode %4 : !noisy.i32 -> i5
```

## 4. 用 `IntegerRangeAnalysis` 传播噪声

### 4.1 为什么需要 analysis

噪声不是类型上的属性：

```
!noisy.i32
```

没有写“当前噪声是 24”。

原因：

- 插入 `reduce_noise` 后，下游类型不应改变
- 控制流合并时，不同路径的噪声不同
- 噪声是随程序位置变化的

所以用 analysis 把“每个 SSA 值的噪声上界”算出来。

### 4.2 `InferIntRangeInterface`

每个相关 op 实现：

```
void AddOp::inferResultRanges(
    ArrayRef<ConstantIntRanges> inputRanges,
    SetIntRangeFn setResultRange);
```

- `inputRanges`：输入值的整数范围
- `setResultRange`：设置结果范围的回调
- `ConstantIntRanges`：包含 min/max 的范围

这是 dataflow 的 **transfer function**。

### 4.3 Noisy 的 transfer function

`lib/Dialect/Noisy/NoisyOps.cpp`：

```
ConstantIntRanges initialNoiseRange() {
  return ConstantIntRanges::fromUnsigned(APInt(32, 0),
                                         APInt(32, INITIAL_NOISE));
}

ConstantIntRanges unionPlusOne(ArrayRef<ConstantIntRanges> inputRanges) {
  auto lhsRange = inputRanges[0];
  auto rhsRange = inputRanges[1];
  auto joined = lhsRange.rangeUnion(rhsRange);
  return ConstantIntRanges::fromUnsigned(joined.umin(), joined.umax() + 1);
}
```

- `EncodeOp` / `ReduceNoiseOp`：设置 `[0, 12]`
- `AddOp` / `SubOp`：范围 union 后 `umax + 1`
- `MulOp`：`umax(lhs) + umax(rhs)`

例如：

```
encode  -> [0, 12]
add     -> [0, 13]
add     -> [0, 14]
mul     -> [0, 24]
```

### 4.4 join

当控制流合并时，dataflow 需要把不同路径的范围合并：

```
路径 A: [0, 18]
路径 B: [0, 24]
join:   [0, 24]
```

`IntegerRangeAnalysis` 使用 `rangeUnion` 做 join。

### 4.5 fixed point

Kildall 式算法：

```
初始化所有值
反复应用 transfer function
在 merge 点 join
直到范围不再变化
```

MLIR 的 `DataFlowSolver` 负责这个迭代过程。

## 5. 在 pass 中运行数据流分析

`lib/Transform/Noisy/ReduceNoiseOptimizer.cpp`：

```
DataFlowSolver solver;
// IntegerRangeAnalysis 依赖 DeadCodeAnalysis，
// 但这个依赖不会自动加载
solver.load<dataflow::DeadCodeAnalysis>();
solver.load<dataflow::IntegerRangeAnalysis>();

if (failed(solver.initializeAndRun(module))) {
  getOperation()->emitOpError() << "Failed to run the analysis.\n";
  signalPassFailure();
  return;
}
```

然后遍历 noisy op，取每个结果的范围：

```
const dataflow::IntegerValueRangeLattice *opRange =
    solver.lookupState<dataflow::IntegerValueRangeLattice>(
        op->getResult(0));

if (!opRange || opRange->getValue().isUninitialized()) {
  ...
}

ConstantIntRanges range = opRange->getValue().getValue();
if (range.umax().getZExtValue() > MAX_NOISE) {
  op->emitOpError() << "noise exceeds maximum";
  signalPassFailure();
}
```

这一步是**验证**：

> 插入 `reduce_noise` 之后，噪声是否仍然可能超过 26？

如果没有 analysis，pass 只能“觉得”自己插得对；有了 analysis，可以真正验证。

## 6. 全局优化：用 ILP 选择插入点

### 6.1 为什么不能只看单个 op

假设：

```
mul 24
  ├── branch 1: add ... -> 27
  └── branch 2: sub ... -> 27
```

如果分别在两个分支插入 `reduce_noise`，需要两个。

但如果在公共的 `mul` 后插入一个，噪声变成 12，两条分支都不再超限，只需要一个。

这是一个**全局选择**，不是局部 rewrite。

### 6.2 变量

对每个 noisy 算术 op `x`：

```
InsertReduceNoise_x ∈ {0, 1}
```

- 1：在 x 后插入 reduce_noise
- 0：不插入

对每个 SSA 值 `v`：

```
NoiseAt_v ∈ [0, 26]
```

表示该值的噪声上界。

对 add/sub，还引入：

```
Z_x
```

用来线性化 `max(lhs, rhs) + 1`。

### 6.3 目标函数

```
minimize  sum(InsertReduceNoise_x)  +  0.1 * sum(Z_x)
```

- 每个 reduce_noise 成本相同，尽量少
- 0.1*Z 是一个小惩罚，让 solver 把 Z 压到最小必要值

### 6.4 约束

#### 初始噪声

```
NoiseAt(encode result) = 12
```

#### 乘法

如果决策为 0：

```
NoiseAt_RES = NoiseAt_LHS + NoiseAt_RHS
```

如果决策为 1：

```
NoiseAt_RES = 12
```

这是一个 if-else，不能直接写成线性约束。用 big-M（`IF_THEN_AUX = 100`）线性化：

```
NoiseAt_RES >= 12 * d
NoiseAt_RES <= 12 + C * (1 - d)
NoiseAt_RES >= (NoiseAt_LHS + NoiseAt_RHS) - C * d
NoiseAt_RES <= (NoiseAt_LHS + NoiseAt_RHS) + C * d
```

当 `d = 0`，后两条把结果绑定到输入和；前两条变得宽松。

当 `d = 1`，前两条把结果绑定到 12；后两条变得宽松。

#### 加法/减法

加法的输入噪声是：

```
max(NoiseAt_LHS, NoiseAt_RHS) + 1
```

引入 `Z`：

```
Z >= NoiseAt_LHS + 1
Z >= NoiseAt_RHS + 1
Z <= 26
```

然后把乘法公式里的“输入和”替换成 `Z`。

#### reduce_noise 前的噪声

插入点之后的噪声会被重置，但插入点之前的噪声仍然必须合法：

```
输入噪声 <= 26
```

所以还要加约束，保证 reduce_noise 之前的 op 不会先爆掉。

### 6.5 求解器

`ReduceNoiseAnalysis` 使用 or-tools：

```
std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SCIP"));
```

创建变量：

```
auto decisionVar = solver->MakeIntVar(0, 1, varName);
auto ssaNoiseVar = solver->MakeNumVar(0, MAX_NOISE, varName);
```

创建约束：

```
MPConstraint *const ct = solver->MakeRowConstraint(min, max, name);
ct->SetCoefficient(var, coefficient);
```

最后：

```
solver->Solve();
```

把每个决策变量的解存进：

```
llvm::DenseMap<Operation *, bool> solution;
```

## 7. Pass 如何插入 reduce_noise

`ReduceNoiseOptimizer`：

```
ReduceNoiseAnalysis analysis(module);
OpBuilder b(&getContext());

module->walk([&](Operation *op) {
  if (!analysis.shouldInsertReduceNoise(op))
    return;

  b.setInsertionPointAfter(op);

  auto reduceOp = b.create<ReduceNoiseOp>(
      op->getLoc(), op->getResult(0));

  op->getResult(0).replaceAllUsesExcept(
      reduceOp.getResult(), {reduceOp});
});
```

关键点：

- `shouldInsertReduceNoise` 来自 ILP 解
- 在 op 后插入 `reduce_noise`
- `replaceAllUsesExcept` 把下游使用替换成 reduce 后的值
- 但 `reduce_noise` 自己的输入仍然是原值，不能替换

插入后再跑一次 IntegerRangeAnalysis 验证。

## 8. 实验

```
cd /home/zhouzhou/npu/mlir-tutorial
export TOP=$PWD
export BIN=$TOP/externals/llvm-project/build/bin
export TO=$TOP/build/tools/tutorial-opt

$TO tests/noisy_reduce_noise.mlir --noisy-reduce-noise-optimizer
```

### 8.1 乘法链

输入：

```
%2 = noisy.encode %0
%3 = noisy.encode %1
%4 = noisy.mul %2, %3
%5 = noisy.mul %4, %4
%6 = noisy.mul %5, %5
%7 = noisy.mul %6, %6
%8 = noisy.decode %7
```

输出：

```
%4 = noisy.mul %2, %3
%4_r = noisy.reduce_noise %4
%5 = noisy.mul %4_r, %4_r
%5_r = noisy.reduce_noise %5
%6 = noisy.mul %5_r, %5_r
%6_r = noisy.reduce_noise %6
%7 = noisy.mul %6_r, %6_r
%8 = noisy.decode %7
```

最后一个 mul 不需要 reduce，因为它的噪声是 24，仍然 ≤ 26。

### 8.2 加法链

```
%4 = noisy.add %2, %3
%5 = noisy.add %4, %4
%6 = noisy.add %5, %5
%7 = noisy.add %6, %6
```

噪声增长：

```
12 + 12 -> 13
13 + 13 -> 14
14 + 14 -> 15
15 + 15 -> 16
```

没有超过 26，所以不需要 `reduce_noise`。

### 8.3 mul 后接 add

```
%4 = noisy.mul %2, %3    // 24
%5 = noisy.add %4, %3    // 25
%6 = noisy.add %5, %5    // 26
%7 = noisy.add %6, %6    // 27 -> 超限
```

Solver 在 `%4` 后插入一次 `reduce_noise`，把噪声重置到 12，后续 add 就不会超限。

### 8.4 分支合并

`test_single_insertion_branching`：

```
mul 24
  ├── 4 个 add -> 27
  └── 4 个 sub -> 27
```

在公共 mul 后插入一个 reduce，两条分支都从 12 开始，总共只需要一个。

输出中只有 `CHECK-COUNT-1: noisy.reduce_noise`。

### 8.5 分支之后还要相乘

`test_double_insertion_branching`：

```
mul 24
  ├── reduce -> 12
  ├── 4 个 add -> 16
  └── 4 个 add -> 16
  └── 最后两条分支相乘 -> 32
```

公共 mul 后的 reduce 让两条分支都从 12 开始，但每条分支经过 4 个 add 后噪声
会到 16；最后把两条分支相乘得到 32，超过 26。因此每条分支还需要一次 reduce，
总共是“一个公共 reduce + 两个分支 reduce”。

## 9. 源码位置

| 内容 | 文件 |
|---|---|
| Noisy dialect 定义 | `lib/Dialect/Noisy/`（`NoisyDialect.h`、`NoisyOps.cpp`） |
| 噪声数据流分析 | `lib/Analysis/ReduceNoiseAnalysis/` |
| pass 与 ILP 求解 | `lib/Transform/Noisy/ReduceNoiseOptimizer.cpp` |
| 本课测试 | `tests/noisy_syntax.mlir`、`tests/noisy_reduce_noise.mlir` |

## 10. 当前实现的限制

代码注释里已经标出一些 FIXME：

1. **solver 不可行时没有可靠失败信号**

```
// FIXME: Should have some way to mark failure when solver is infeasible
```

`ReduceNoiseAnalysis` 构造函数没有返回值，solver 失败时不容易报错。

2. **假设所有 `reduce_noise` 已经提前删除**

```
// FIXME: assumes all reduce_noise ops have already been removed and their
// values forwarded.
```

3. **硬编码 i32 / 12 / 26**  
这个教程模型特意简化，不支持不同位宽或不同噪声参数。

4. **所有 block argument 都当作输入噪声 12**  
注释说明：教程没有控制流，真实编译器需要更通用地处理。

5. **分支测试是 SSA 图分支，不是 `cf` 控制流**  
`test_single_insertion_branching` 里的两个“分支”只是两条独立的 SSA 计算链，
不是 `cf.br` 的真实控制流。

## 11. 易混点

### `IntegerRangeAnalysis` 不是优化本身

它只负责**分析**每个值的整数范围。真正的“在哪里插入”由 ILP 决定。

### ILP 的目标不是“最少 op”

目标是尽量少插入 `reduce_noise`；插入点会改变下游噪声，所以是一个全局问题。

### 为什么不用贪心

贪心只看当前 op，可能在每个分支都插入；ILP 能看到公共上游，用一个 reduce
覆盖多条路径。

### 为什么需要 `DeadCodeAnalysis`

`IntegerRangeAnalysis` 依赖它，但依赖不会自动加载；不加载会失败或结果不完整。

### 验证 analysis 和优化 analysis 是两件事

- `IntegerRangeAnalysis`：传播噪声范围，用来验证
- `ReduceNoiseAnalysis`：解 ILP，用来决定插入点

## 12. 练习

### 练习 1：计算噪声

```
%a = noisy.encode %x : i5 -> !noisy.i32
%b = noisy.encode %y : i5 -> !noisy.i32
%c = noisy.add %a, %b : !noisy.i32
%d = noisy.add %c, %c : !noisy.i32
%e = noisy.mul %d, %d : !noisy.i32
```

求每一步的噪声上界，判断是否需要 `reduce_noise`。

<details>
<summary>参考答案</summary>

```
%a: 12
%b: 12
%c: max(12,12)+1 = 13
%d: max(13,13)+1 = 14
%e: 14+14 = 28  -> 超过 26
```

需要在 `%d` 后或 `%e` 前插入 `reduce_noise`，使 `%e` 的输入噪声变成 12，
结果变成 24。

</details>

### 练习 2：为什么分支测试只需要一个 reduce

两条分支都从同一个 `mul` 结果开始。如果在这个公共点插入 reduce，两条分支都
受益；如果分别在分支内部插入，就需要两个。

<details>
<summary>参考答案</summary>

因为噪声在公共 `mul` 后已经接近上限；在公共点重置一次，后面每条分支都能重新
从 12 开始，所以一个 reduce 就够。

</details>

### 练习 3：为什么需要 big-M

ILP 里有一个 0/1 决策变量 `d`，它控制“是否在这个 op 后插入
`reduce_noise`”。为什么不能直接写：

```
if d == 1:
  result_noise = 12
else:
  result_noise = input_noise
```

<details>
<summary>参考答案</summary>

因为线性规划只接受线性等式/不等式，不能直接表达 if-else。

big-M 用一个足够大的常数 `C`，把两个分支写成一组成对的线性约束：

- `d = 0` 时，约束等价于“结果 = 输入和”
- `d = 1` 时，约束等价于“结果 = 12”

关键是 `C` 要比所有可能的噪声值都大，这样不需要执行的分支会被“放松”到
不起作用。

</details>

### 练习 4：如果去掉最后的 noise 验证会怎样

pass 已经根据 ILP 解插入了 `reduce_noise`，为什么还要再跑一次
`IntegerRangeAnalysis`？

<details>
<summary>参考答案</summary>

ILP 是一个数学模型，可能因为约束遗漏、big-M 取值、solver 不可行或其他实现
问题而给出错误结果。插入后再用 `IntegerRangeAnalysis` 重新计算每个值的噪声，
可以确认最终 IR 真的满足 `noise <= MAX_NOISE`。

如果没有这一步，pass 可能生成非法 IR，而且错误会延迟到运行时才发现。

</details>

## 13. 验收清单

- 能说出初始噪声 12、最大噪声 26
- 能写出 encode/add/sub/mul/reduce 的噪声规则
- 能解释 `InferIntRangeInterface` 的 transfer function
- 能解释 dataflow 的 join 和 fixed point
- 能说出 `IntegerRangeAnalysis` 和 `ReduceNoiseAnalysis` 的分工
- 能解释 ILP 的决策变量和噪声变量
- 能解释 big-M 如何线性化 if-else
- 能解释为什么分支公共点插入一个 reduce 就够
- 能运行 `tests/noisy_reduce_noise.mlir`
- 能解释四个测试的不同插入数量

## 14. 下一课

[第 12 课：Defining Patterns with PDLL](12-pdll.md)

- PDLL 语法
- `MulToAdd.pdll`
- PDLL 与 C++/DRR 的对比
- 测试：`tests/mul_to_add_pdll.mlir`

## 参考

- 本课对应文章：
  <https://jeremykun.com/2023/11/15/mlir-a-global-optimization-and-dataflow-analysis/>
- MLIR 官方文档：DataFlow Analysis
  <https://mlir.llvm.org/docs/DataFlowAnalysis/>
- MLIR 官方文档：InferIntRangeInterface
  <https://mlir.llvm.org/docs/Interfaces/InferIntRangeInterface/>
- or-tools：
  <https://developers.google.com/optimization>
- 本课测试：
  - `tests/noisy_syntax.mlir`
  - `tests/noisy_reduce_noise.mlir`
- 相关源码：
  - `lib/Dialect/Noisy/`
  - `lib/Analysis/ReduceNoiseAnalysis/`
  - `lib/Transform/Noisy/ReduceNoiseOptimizer.cpp`