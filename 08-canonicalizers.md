# 第 8 课：Canonicalizers and Declarative Rewrite Patterns

> 对应文章：[Canonicalizers and Declarative Rewrite Patterns](https://jeremykun.com/2023/09/20/mlir-canonicalizers-and-declarative-rewrite-patterns/)。
>
> 上一课的 folder 只能处理“一个 op 自己能不能折叠”，不能创建新 op。
> 本课讲 canonicalizer：它可以匹配一段 IR 子图，创建/删除多个 op，
> 把 IR 重写成更规范的形式。我们重点看仓库里的 TableGen（DRR）写法。

## 1. 学习目标

- 理解 fold 和 canonicalizer 的区别
- 理解 `--canonicalize` 如何组合 fold、pattern 和 DCE
- 读懂 `PolyPatterns.td` 中的 `Pat`、`Pattern`、`Constraint`
- 理解 `$x` 绑定、`$lhs` 绑定、重复变量约束
- 理解 `HasOneUse` 这种 `CPred` 约束
- 理解 pattern 的 benefit 和 fixed-point 执行
- 能看懂生成的 `PolyCanonicalize.cpp.inc`
- 能写一个简单的声明式 pattern

## 2. 从 fold 到 canonicalizer

### 2.1 fold 的局限

`fold` 的规则：

- 只处理当前 op
- 不能创建新 op
- 不能删除其他 op
- 只能返回一个 Attribute 或一个已有 Value

所以它适合：

```text
poly.add(constant, constant) -> constant
box.unpack(box.pack(x)) -> x
```

但它做不了：

```text
x² - y²  ->  (x+y)(x-y)
```

因为这个变换要：

- 匹配一整棵子图 `sub(mul(x,x), mul(y,y))`
- 创建三个新 op：`add`、`sub`、`mul`
- 删除原来的两个 `mul` 和一个 `sub`

这属于 **DAG-to-DAG rewrite**，需要 canonicalizer。

### 2.2 canonicalizer 能做什么

canonicalizer 可以：

- 匹配多个 op 组成的子图
- 检查使用次数等额外条件
- 创建任意多个新 op
- 替换/删除匹配到的 op
- 反复运行直到 IR 不再变化（fixed point）

`--canonicalize` pass 的内部大致是：

```text
反复执行：
  1. 尝试 folder
  2. 尝试 canonicalization patterns
  3. 删除死代码
直到没有变化
```

所以 `--canonicalize` 既能做常量折叠，也能做代数重写。

## 3. 两种写法

MLIR 支持两种 canonicalizer 写法。

### 3.1 C++ 写法

在 op 的 TableGen 里加：

```tablegen
let hasCanonicalizer = 1;
```

生成：

```cpp
static void getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results,
    ::mlir::MLIRContext *context);
```

然后在 C++ 里实现一个 `OpRewritePattern`：

```cpp
struct MyPattern : public OpRewritePattern<SubOp> {
  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter &rewriter) const override {
    // 检查、创建新 op、replaceOp
  }
};

void SubOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<MyPattern>(context);
}
```

第 2 课已经见过这种写法。

### 3.2 TableGen/DRR 写法

把 pattern 直接写在 `.td` 里：

```tablegen
def MyPattern : Pat<
  (SomeOp $x),
  (OtherOp $x)
>;
```

构建时 `mlir-tblgen -gen-rewriters` 生成等价的 C++ pattern。

本仓库当前使用的是 TableGen 写法，文件是：

```text
lib/Dialect/Poly/PolyPatterns.td
```

两种写法最终都进入同一个 `RewritePatternSet`。

> DRR（Declarative Rewrite Rules）在 MLIR 中处于维护模式，但仍然广泛使用；
> 后续 PDLL 是另一种声明式方案。

## 4. `PolyPatterns.td` 逐段读

> `Pat` / `Pattern` 的完整参数、DAG 语法、约束和常见错误见
> [Pat 与 Pattern 语法](patterns.md)。

完整文件（`lib/Dialect/Poly/PolyPatterns.td`）：

```tablegen
#ifndef LIB_DIALECT_POLY_POLYPATTERNS_TD_
#define LIB_DIALECT_POLY_POLYPATTERNS_TD_

include "PolyOps.td"
include "mlir/Dialect/Complex/IR/ComplexOps.td"
include "mlir/IR/PatternBase.td"

def LiftConjThroughEval : Pat<
  (Poly_EvalOp $f, (ConjOp $z, $fastmath)),
  (ConjOp (Poly_EvalOp $f, $z), $fastmath)
>;

def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;

// Rewrites (x^2 - y^2) as (x+y)(x-y) if x^2 and y^2 have no other uses.
def DifferenceOfSquares : Pattern<
  (Poly_SubOp (Poly_MulOp:$lhs $x, $x), (Poly_MulOp:$rhs $y, $y)),
  [
    (Poly_AddOp:$sum $x, $y),
    (Poly_SubOp:$diff $x, $y),
    (Poly_MulOp:$res $sum, $diff),
  ],
  [(HasOneUse:$lhs), (HasOneUse:$rhs)]
>;

#endif
```

### 4.1 include 了什么

| include | 作用 |
|---|---|
| `PolyOps.td` | 拿到 `Poly_AddOp`、`Poly_MulOp`、`Poly_SubOp`、`Poly_EvalOp` |
| `ComplexOps.td` | 拿到 `ConjOp`，用于复共轭 pattern |
| `PatternBase.td` | 拿到 `Pat`、`Pattern`、`Constraint`、`CPred` |

### 4.2 第一个 pattern：`LiftConjThroughEval`

```tablegen
def LiftConjThroughEval : Pat<
  (Poly_EvalOp $f, (ConjOp $z, $fastmath)),
  (ConjOp (Poly_EvalOp $f, $z), $fastmath)
>;
```

它表达的是：

```text
f(conj(z))  ->  conj(f(z))
```

前提是多项式系数是实数；这里直接用这个恒等式做规范化。

#### 匹配树

左边：

```text
Poly_EvalOp
  ├── $f          (poly 操作数)
  └── ConjOp
        ├── $z    (复数操作数)
        └── $fastmath
```

右边：

```text
ConjOp
  ├── Poly_EvalOp
  │     ├── $f
  │     └── $z
  └── $fastmath
```

`$f`、`$z` 是变量绑定：

- 左边匹配到的值，右边继续使用
- `$fastmath` 是 `complex.conj` 的 fastmath 属性，重写后要保留

`Pat` 表示“产生一个结果”的 pattern：

```tablegen
Pat<source, result>
```

生成的 C++ 大致做：

1. 匹配 `poly.eval`
2. 检查第二个操作数是否由 `complex.conj` 定义
3. 创建新的 `poly.eval` 和 `complex.conj`
4. 用新的 `complex.conj` 替换原来的 `poly.eval`

### 4.3 第二个 pattern：`DifferenceOfSquares`

```tablegen
def DifferenceOfSquares : Pattern<
  (Poly_SubOp (Poly_MulOp:$lhs $x, $x), (Poly_MulOp:$rhs $y, $y)),
  [
    (Poly_AddOp:$sum $x, $y),
    (Poly_SubOp:$diff $x, $y),
    (Poly_MulOp:$res $sum, $diff),
  ],
  [(HasOneUse:$lhs), (HasOneUse:$rhs)]
>;
```

匹配：

```text
poly.sub
  ├── poly.mul x, x    ← 记为 $lhs，操作数记为 $x
  └── poly.mul y, y    ← 记为 $rhs，操作数记为 $y
```

重写为：

```text
poly.mul
  ├── poly.add x, y    ← $sum
  └── poly.sub x, y    ← $diff
```

原来的 `sub` 被新的 `mul` 替换；`$res` 是替换根 op 的结果。

#### `$lhs` 和 `$x` 的区别

```tablegen
(Poly_MulOp:$lhs $x, $x)
```

- `$lhs` 绑定**整个 `poly.mul` op**（或其结果值）
- `$x` 绑定它的第一个操作数
- 第二个参数也写 `$x`，表示“第二个操作数必须和第一个相同”
- 所以这个子树匹配的是 `x * x`，不是 `x * y`

生成代码会检查：

```cpp
if (!(*x.begin() == *x0.begin()))
  return rewriter.notifyMatchFailure(...);
```

#### `$sum`、`$diff`、`$res`

```tablegen
[
  (Poly_AddOp:$sum $x, $y),
  (Poly_SubOp:$diff $x, $y),
  (Poly_MulOp:$res $sum, $diff),
]
```

- 这是 `Pattern` 的**结果 op 列表**
- `$sum`、`$diff` 是中间结果
- 最后一个 `$res` 用来替换匹配到的根 op
- 前面的结果可以给后面的结果当操作数

`Pat` 只能产生一个结果，`Pattern` 可以产生多个结果。

#### 约束：`HasOneUse`

```tablegen
def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;
```

然后：

```tablegen
[(HasOneUse:$lhs), (HasOneUse:$rhs)]
```

含义：

- 只有当 `x²` 和 `y²` 都只有一个使用者时，才允许重写
- 如果 `x²` 还被别的 op 使用，重写会复制 `x`/`y` 相关计算，可能让 IR 变大
- 所以加约束，只在安全/有利时触发

`CPred` 里的 `$_self` 会被替换成绑定的实体。生成代码里大致是：

```cpp
if (!(((*lhs.getODSResults(0).begin()).hasOneUse())))
  return rewriter.notifyMatchFailure(...);
if (!(((*rhs.getODSResults(0).begin()).hasOneUse())))
  return rewriter.notifyMatchFailure(...);
```

注意：`HasOneUse:$lhs` 绑定的是 `poly.mul` 的结果值，`hasOneUse()` 检查它的使用次数。

## 5. 把 pattern 接进 pipeline

### 5.1 在 op 上声明 canonicalizer

`Poly_BinOp` 和 `Poly_EvalOp` 都有：

```tablegen
let hasCanonicalizer = 1;
```

这会生成：

```cpp
static void getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results,
    ::mlir::MLIRContext *context);
```

`PolyOps.cpp` 中的实现：

```cpp
void SubOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<DifferenceOfSquares>(context);
}

void EvalOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<LiftConjThroughEval>(context);
}
```

`AddOp`、`MulOp` 目前是空实现，所以它们暂时没有自己的 canonicalization pattern。

也可以用一个调用把所有生成的 pattern 加进去：

```cpp
void SomeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  populateWithGenerated(results);
}
```

生成的 `PolyCanonicalize.cpp.inc` 末尾有：

```cpp
void populateWithGenerated(RewritePatternSet &patterns) {
  patterns.add<DifferenceOfSquares>(patterns.getContext());
  patterns.add<LiftConjThroughEval>(patterns.getContext());
}
```

当前仓库选择按 op 手动添加，这样更清楚每个 pattern 属于哪个 op。

### 5.2 构建时生成 pattern

`lib/Dialect/Poly/CMakeLists.txt`：

```cmake
set(LLVM_TARGET_DEFINITIONS PolyPatterns.td)
mlir_tablegen(PolyCanonicalize.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRPolyCanonicalizationIncGen)
```

`PolyOps.cpp`：

```cpp
#include "mlir/IR/PatternMatch.h"

// Required after PatternMatch.h
#include "lib/Dialect/Poly/PolyCanonicalize.cpp.inc"
```

生成的 `.inc` 使用 `RewritePattern`、`PatternRewriter` 等类型，所以
`PatternMatch.h` 必须先 include。

## 6. Pattern 如何执行

### 6.1 benefit

生成的 pattern 构造函数大致是：

```cpp
struct DifferenceOfSquares : public RewritePattern {
  DifferenceOfSquares(MLIRContext *context)
      : RewritePattern("poly.sub", 3, context,
                       {"poly.add", "poly.mul", "poly.sub"}) {}
  ...
};
```

第二个参数 `3` 是 pattern 的 **benefit**。

TableGen `Pattern` 的默认 benefit 是**源 pattern 中的 op 数量**：

- `DifferenceOfSquares`：`sub` + 两个 `mul` = 3
- `LiftConjThroughEval`：`eval` + `conj` = 2

benefit 越高，greedy driver 越优先尝试。也可以用 `addBenefit` 调整：

```tablegen
def MyPattern : Pat<..., ..., ..., ..., (addBenefit 1)>;
```

### 6.2 fixed point

canonicalizer 会反复运行：

```text
应用 pattern
  -> 产生新 op
  -> 新 op 可能又匹配别的 pattern
  -> 再运行
直到没有 pattern 能匹配
```

所以 pattern 必须保证最终收敛。`DifferenceOfSquares` 的 `HasOneUse` 约束也
有助于避免无限制地复制/重写。

### 6.3 DCE

canonicalizer 在替换根 op 后，原来的 `sub`、两个 `mul` 通常会失去使用者，
随后被 DCE 删除。你不需要在 pattern 里手动 erase；`replaceOp` 已经处理了根 op。

## 7. 实验：运行 `tests/poly_canonicalize.mlir`

```bash
cd /home/zhouzhou/npu/mlir-tutorial
export TOP=$PWD
export BIN=$TOP/externals/llvm-project/build/bin
export TO=$TOP/build/tools/tutorial-opt

$TO tests/poly_canonicalize.mlir --canonicalize
```

输出：

```mlir
module {
  func.func @test_simple() -> !poly.poly<10> {
    %0 = poly.constant dense<[2, 4, 6]> : tensor<3xi32> : !poly.poly<10>
    return %0 : !poly.poly<10>
  }
  func.func @test_difference_of_squares(%arg0: !poly.poly<3>, %arg1: !poly.poly<3>) -> !poly.poly<3> {
    %0 = poly.add %arg0, %arg1 : !poly.poly<3>
    %1 = poly.sub %arg0, %arg1 : !poly.poly<3>
    %2 = poly.mul %0, %1 : !poly.poly<3>
    %3 = poly.add %2, %2 : !poly.poly<3>
    return %3 : !poly.poly<3>
  }
  func.func @test_difference_of_squares_other_uses(%arg0: !poly.poly<3>, %arg1: !poly.poly<3>) -> !poly.poly<3> {
    %0 = poly.mul %arg0, %arg0 : !poly.poly<3>
    %1 = poly.mul %arg1, %arg1 : !poly.poly<3>
    %2 = poly.sub %0, %1 : !poly.poly<3>
    %3 = poly.add %2, %0 : !poly.poly<3>
    return %3 : !poly.poly<3>
  }
  func.func @test_normalize_conj_through_eval(%arg0: !poly.poly<3>, %arg1: complex<f64>) -> complex<f64> {
    %0 = poly.eval %arg0, %arg1 : (!poly.poly<3>, complex<f64>) -> complex<f64>
    %1 = complex.conj %0 : complex<f64>
    return %1 : complex<f64>
  }
}
```

逐个看：

### 7.1 `@test_simple`

这是上一课的 fold：

```text
poly.from_tensor(arith.constant) -> poly.constant
poly.add(constant, constant) -> poly.constant
```

最后只剩：

```mlir
%0 = poly.constant dense<[2, 4, 6]> : tensor<3xi32> : !poly.poly<10>
return %0
```

`--canonicalize` 既调用 folder，也删除死代码。

### 7.2 `@test_difference_of_squares`

输入：

```mlir
%2 = poly.mul %0, %0
%3 = poly.mul %1, %1
%4 = poly.sub %2, %3
%5 = poly.add %4, %4
return %5
```

`DifferenceOfSquares` 匹配 `%4`：

```text
sub(mul(x,x), mul(y,y))
-> mul(add(x,y), sub(x,y))
```

输出：

```mlir
%0 = poly.add %arg0, %arg1
%1 = poly.sub %arg0, %arg1
%2 = poly.mul %0, %1
%3 = poly.add %2, %2
return %3
```

注意 outer `add %4, %4` 还在；pattern 只改写了 `sub`，没有把 outer add 消掉。

### 7.3 `@test_difference_of_squares_other_uses`

输入：

```mlir
%2 = poly.mul %0, %0
%3 = poly.mul %1, %1
%4 = poly.sub %2, %3
%5 = poly.add %4, %2
return %5
```

这里 `%2` 有两个使用者（`%4` 和 `%5`）。`HasOneUse:$lhs` 失败，所以 pattern
不触发，输出保持原样：

```mlir
%0 = poly.mul %arg0, %arg0
%1 = poly.mul %arg1, %arg1
%2 = poly.sub %0, %1
%3 = poly.add %2, %0
return %3
```

这正是 `HasOneUse` 约束的意义。

### 7.4 `@test_normalize_conj_through_eval`

输入：

```mlir
%z_bar = complex.conj %z
%ev = poly.eval %f, %z_bar
return %ev
```

`LiftConjThroughEval` 匹配：

```text
poly.eval(f, conj(z)) -> conj(poly.eval(f, z))
```

输出：

```mlir
%0 = poly.eval %arg0, %arg1
%1 = complex.conj %0
return %1
```

`complex.conj` 从 `poly.eval` 内部移到了外面。`$fastmath` 被保留到新的
`complex.conj` 上。

## 8. 看生成的 C++ pattern

`build/lib/Dialect/Poly/PolyCanonicalize.cpp.inc` 中，
`DifferenceOfSquares::matchAndRewrite` 大致做：

```cpp
// 1. 匹配根 op：poly.sub
auto castedOp0 = dyn_cast<SubOp>(op0);

// 2. 匹配两个操作数：poly.mul
auto castedOp1 = dyn_cast<MulOp>(op1);
auto castedOp2 = dyn_cast<MulOp>(op2);

// 3. 约束：每个 mul 的结果只能有一个使用者
if (!lhs.getResult().hasOneUse()) return failure();
if (!rhs.getResult().hasOneUse()) return failure();

// 4. 约束：mul 的两个操作数必须相同
if (x != x0) return failure();
if (y != y0) return failure();

// 5. 创建新 op
auto sum  = rewriter.create<AddOp>(..., x, y);
auto diff = rewriter.create<SubOp>(..., x, y);
auto res  = rewriter.create<MulOp>(..., sum, diff);

// 6. 替换根 op
rewriter.replaceOp(op0, res);
```

这段代码不是手写的，而是 TableGen 根据 `PolyPatterns.td` 生成的。理解生成的
结构有助于调试 pattern 为什么没有触发。

## 9. 源码位置

| 内容 | 文件 |
|---|---|
| DRR pattern 定义 | `lib/Dialect/Poly/PolyPatterns.td` |
| pattern 注册（`getCanonicalizationPatterns`） | `lib/Dialect/Poly/PolyOps.cpp` |
| TableGen 生成规则（`-gen-rewriters`） | `lib/Dialect/Poly/CMakeLists.txt` |
| 生成的 C++ pattern | `build/lib/Dialect/Poly/PolyCanonicalize.cpp.inc` |
| 本课测试 | `tests/poly_canonicalize.mlir` |

## 10. 常见坑

### 10.1 `hasCanonicalizer` 和 `hasFolder` 不是一回事

- `hasFolder = 1` → `fold`
- `hasCanonicalizer = 1` → `getCanonicalizationPatterns`
- 两者可以同时存在

### 10.2 `Pat` 和 `Pattern`

- `Pat<source, result>`：产生一个结果
- `Pattern<source, [results], constraints>`：产生多个结果

`DifferenceOfSquares` 需要创建 `add`、`sub`、`mul` 三个 op，所以用 `Pattern`。

### 10.3 重复变量是约束

```tablegen
(Poly_MulOp $x, $x)
```

不是“两个任意操作数”，而是“两个操作数必须相同”。生成代码会检查相等性。

### 10.4 `HasOneUse` 不是可选的装饰

如果没有 `HasOneUse`，`x²` 有多个使用者时也可能被重写，导致计算被复制，
甚至 pattern 反复触发。约束既保证正确性，也帮助收敛。

### 10.5 pattern 必须收敛

canonicalizer 会反复运行。不要写：

```text
A -> B
B -> A
```

这种循环 pattern。需要约束、benefit 或更具体的匹配条件。

### 10.6 生成的 `.inc` 需要重新构建

修改 `PolyPatterns.td` 后：

```bash
ninja -C build -j 100 tutorial-opt
```

否则运行的还是旧的 pattern。

## 11. 练习

### 练习 1：解释 `other_uses` 为什么没有重写

看 `@test_difference_of_squares_other_uses`：

```mlir
%2 = poly.mul %0, %0
%3 = poly.mul %1, %1
%4 = poly.sub %2, %3
%5 = poly.add %4, %2
```

为什么 `DifferenceOfSquares` 没有把 `%4` 改写成 `(x+y)(x-y)`？

<details>
<summary>参考答案</summary>

因为 `%2`（`x²`）除了被 `%4` 使用，还被 `%5` 使用；`HasOneUse:$lhs` 失败。
pattern 只有在两个平方都只有一个使用者时才触发，避免复制计算。

</details>

### 练习 2：`Pat` 和 `Pattern` 的区别

`LiftConjThroughEval` 使用 `Pat`，`DifferenceOfSquares` 使用 `Pattern`。
为什么？

<details>
<summary>参考答案</summary>

`LiftConjThroughEval` 只产生一个结果：`complex.conj`。
`DifferenceOfSquares` 要产生两个中间 op（`add`、`sub`）和一个最终 `mul`，
所以用 `Pattern` 的结果列表。

</details>

### 练习 3：给 pattern 提高 benefit

把 `DifferenceOfSquares` 的 benefit 提高 1：

```tablegen
def DifferenceOfSquares : Pattern<
  ...,
  [...],
  [(HasOneUse:$lhs), (HasOneUse:$rhs)],
  [],              // supplemental results
  (addBenefit 1)   // 第五个参数才是 benefit
>;
```

这会影响什么？什么时候需要这样做？

<details>
<summary>参考答案</summary>

benefit 越高，greedy driver 越先尝试这个 pattern。默认 benefit 是源 pattern
的 op 数（这里是 3）。提高 benefit 可以让它在与其他 pattern 竞争时优先执行。
只有确实希望它优先时才需要，不要随意提高，否则可能改变 IR 的规范化方向。

</details>

### 练习 4：写一个最简单的 `Pat`

匹配双重取负：

```mlir
%n  = poly.neg %x  : !poly.poly<10>
%nn = poly.neg %n  : !poly.poly<10>
return %nn : !poly.poly<10>
```

要求写一个 canonicalization pattern，把它化简成：

```mlir
return %x : !poly.poly<10>
```

不需要额外约束，直接用 `Pat` 和 `replaceWithValue` 即可。

<details>
<summary>参考答案</summary>

### 1. 让 `poly.neg` 有 canonicalizer

`Poly_NegOp` 目前没有 canonicalizer，需要加：

```tablegen
def Poly_NegOp : Op<Poly_Dialect, "neg",
                     [Pure, SameOperandsAndResultType]> {
  ...
  let hasCanonicalizer = 1;
}
```

### 2. 写 pattern

在 `PolyPatterns.td` 中加入：

```tablegen
def EliminateDoubleNeg : Pat<
  (Poly_NegOp (Poly_NegOp $x)),
  (replaceWithValue $x)
>;
```

读法：

```text
匹配：neg(neg(x))
重写：直接使用已有的 x
```

`replaceWithValue` 表示不创建新 op，而是用已有 SSA 值替换匹配结果。

### 3. 注册 pattern

在 `PolyOps.cpp` 中：

```cpp
void NegOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<EliminateDoubleNeg>(context);
}
```

### 4. 重新构建

```bash
ninja -C build -j 100 tutorial-opt
```

### 5. 验证

```mlir
func.func @double_neg(%x: !poly.poly<10>) -> !poly.poly<10> {
  %n  = poly.neg %x : !poly.poly<10>
  %nn = poly.neg %n : !poly.poly<10>
  return %nn : !poly.poly<10>
}
```

运行：

```bash
$TO build/scratch/double_neg.mlir --canonicalize
```

预期输出：

```mlir
func.func @double_neg(%arg0: !poly.poly<10>) -> !poly.poly<10> {
  return %arg0 : !poly.poly<10>
}
```

这个练习的重点是：

- `Pat` 只产生一个结果
- 结果可以直接是已有值，用 `(replaceWithValue $x)`
- 不需要约束、不需要创建新 op
- 原来的两个 `poly.neg` 失去使用者后被 DCE 删除

</details>

## 12. 验收清单

- [ ] 能说出 fold 和 canonicalizer 的区别
- [ ] 能解释 `--canonicalize` 的 fixed point 流程
- [ ] 能读懂 `Pat` / `Pattern` 的嵌套 DAG 语法
- [ ] 能解释 `$x`、`$lhs`、`$sum` 分别绑定什么
- [ ] 能解释 `HasOneUse` 约束的作用
- [ ] 知道 pattern 的 benefit 默认来自源 pattern 的 op 数
- [ ] 知道 `hasCanonicalizer` 与 `hasFolder` 的区别
- [ ] 能通过 `PolyOps.cpp` 找到 pattern 的注册位置
- [ ] 能修改 `PolyPatterns.td` 并重新构建

## 13. 下一课

**第 9 课：Dialect Conversion**

- `TypeConverter`：`!poly.poly<N>` → `tensor<Nxi32>`
- `OpConversionPattern`
- `ConversionTarget`
- `applyPartialConversion`
- 测试：`tests/poly_to_standard.mlir`

对应源码：`lib/Conversion/PolyToStandard/`。

## 参考

- 本课对应文章：
  <https://jeremykun.com/2023/09/20/mlir-canonicalizers-and-declarative-rewrite-patterns/>
- MLIR 官方文档：Canonicalization
  <https://mlir.llvm.org/docs/Canonicalization/>
- MLIR 官方文档：Declarative Rewrite Rules
  <https://mlir.llvm.org/docs/DeclarativeRewrites/>
- 补充：[Pat 与 Pattern 语法](patterns.md)
- 本课测试：`tests/poly_canonicalize.mlir`
- FileCheck 语法：[FileCheck 使用指南](filecheck.md)
- 相关源码：
  - `lib/Dialect/Poly/PolyPatterns.td`
  - `lib/Dialect/Poly/PolyOps.cpp`
  - `lib/Dialect/Poly/CMakeLists.txt`
  - `build/lib/Dialect/Poly/PolyCanonicalize.cpp.inc`