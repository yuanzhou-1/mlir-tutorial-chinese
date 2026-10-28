# 第 2 课：Writing Our First Pass

> 对应文章 3：[Writing Our First Pass](https://jeremykun.com/2023/08/10/mlir-writing-our-first-pass/)。
>
> 本课从零理解一个 MLIR pass 的组成，并对比两种实现方式：
> **手动遍历 IR** 与 **Pattern Rewrite**。最后用一个更完整的例子
> （`mul-to-add`）理解多个 pattern 如何按 benefit 协作。
>
> 说明：文章写作时使用 `PassWrapper` + `PassRegistration` 的旧写法；当前仓库已经
> 迁移到 TableGen 生成 base class（`impl::XxxBase`）。本课以仓库现状为准，
> TableGen 的细节留到第 3 课。

## 1. 学习目标

- 理解 pass 的四个要素：注册、作用对象、`runOnOperation()`、失败处理
- 读懂手动 `walk` 风格的 pass，并能运行、观察、验证
- 读懂 `OpRewritePattern` / `matchAndRewrite` 的工作方式
- 理解 greedy rewrite driver、pattern benefit、fold 三者的关系
- 能解释同一个 unroll 功能为什么两种实现输出不同
- 能独立修改 pattern 的 benefit，并通过重建 + 测试观察行为变化

## 2. Pass 在 MLIR 中是什么

Pass 是 MLIR 的**变换/分析单元**。它有几个关键特征：

1. **锚定在某种 operation 上**：可以作用于整个 module、某个 `func.func`，或任意 op
2. **有唯一的名字**：命令行里用 `--<pass-name>` 调用
3. **通过 PassManager 调度**：多个 pass 可以组成 pipeline
4. **失败是显式的**：调用 `signalPassFailure()`，而不是抛异常

在本项目中，`tutorial-opt` 就是一个“注册了一堆 pass 的 mlir-opt”：

```cpp
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<...>();
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  mlir::tutorial::registerAffinePasses();
  mlir::tutorial::registerArithPasses();
  ...
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
```

所以：

```bash
$TO input.mlir --affine-full-unroll
```

等价于“用 tutorial-opt 这个 driver 加载输入，然后运行名为 `affine-full-unroll` 的 pass”。

`mlir-opt` 与 `tutorial-opt` 的框架相同，区别只是注册了哪些 dialect/pass。

## 3. 项目组织

与文章 3 对应，pass 相关的文件分布如下：

```text
lib/Transform/Affine/
├── AffineFullUnroll.cpp                    # 手动 walk 版本
├── AffineFullUnroll.h
├── AffineFullUnrollPatternRewrite.cpp      # pattern rewrite 版本
├── AffineFullUnrollPatternRewrite.h
├── Passes.h                                # 生成注册函数
├── Passes.td                               # pass 声明（第 3 课详解）
├── CMakeLists.txt
└── BUILD                                   # Bazel 版本

tools/tutorial-opt.cpp                       # 注册所有 tutorial pass
tests/affine_loop_unroll.mlir                # FileCheck 测试
```

`Passes.td` 中的声明（简化）：

```tablegen
def AffineFullUnroll : Pass<"affine-full-unroll"> {
  let summary = "Fully unroll all affine loops";
  let dependentDialects = ["mlir::affine::AffineDialect"];
}

def AffineFullUnrollPatternRewrite : Pass<"affine-full-unroll-rewrite"> {
  let summary = "Fully unroll all affine loops using the pattern rewrite engine";
  let dependentDialects = ["mlir::affine::AffineDialect"];
}
```

这些声明会生成：

- `AffineFullUnrollBase`：pass 的基类
- `registerAffinePasses()`：把 pass 注册到全局注册表

当前仓库的 pass 实现都建立在生成代码之上。

## 4. Pass 的骨架

以手动版本为例：

```cpp
#define GEN_PASS_DEF_AFFINEFULLUNROLL
#include "lib/Transform/Affine/Passes.h.inc"

struct AffineFullUnroll : impl::AffineFullUnrollBase<AffineFullUnroll> {
  using AffineFullUnrollBase::AffineFullUnrollBase;

  void runOnOperation() {
    getOperation()->walk([&](AffineForOp op) {
      if (failed(loopUnrollFull(op))) {
        op.emitError("unrolling failed");
        signalPassFailure();
      }
    });
  }
};
```

四个要素：

| 要素 | 在代码中 |
|---|---|
| 注册 | `Passes.td` + `registerAffinePasses()` |
| 作用对象 | `getOperation()`（本 pass 由 mlir-opt 锚定在 module 上） |
| 主逻辑 | `runOnOperation()` |
| 失败处理 | `emitError(...)` + `signalPassFailure()` |

关键点：

- **`runOnOperation()` 是唯一入口**，pass manager 负责调用
- **`getOperation()` 返回被处理的 op**，本 pass 拿到的是 module
- **不要抛异常**；用 `signalPassFailure()` 让 pipeline 停止并返回错误
- **`dependentDialects`** 告诉 pass manager：这个 pass 可能创建 affine dialect 的 op，
  运行前需要加载对应 dialect

### CRTP：`AffineFullUnrollBase<AffineFullUnroll>` 是什么

你可能注意到这个继承写法：

```cpp
struct AffineFullUnroll
    : impl::AffineFullUnrollBase<AffineFullUnroll> {
  ...
};
```

**派生类把自己作为模板参数传给基类**，这种模式叫
**CRTP（Curiously Recurring Template Pattern，奇异递归模板模式）**。
"curiously recurring" 指的就是派生类的名字又出现在自己的基类列表里。

最小例子：

```cpp
template <typename Derived>
struct Base {
  void run() {
    // 把 this 静态转换成派生类，调用派生类的方法
    static_cast<Derived *>(this)->impl();
  }
};

struct Derived : Base<Derived> {
  void impl() { /* ... */ }
};
```

好处：

- **静态多态**：不需要虚函数/vtable，调用在编译期解析
- **基类知道派生类型**：可以构造、复制、查询具体类型
- **每个派生类得到独立的基类实例**：类型安全，不会混淆

回到 pass 代码，生成的基类大致是：

```cpp
template <typename DerivedT>
class AffineFullUnrollBase : public ::mlir::OperationPass<> {
public:
  AffineFullUnrollBase()
      : ::mlir::OperationPass<>(::mlir::TypeID::get<DerivedT>()) {}

  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(
        *static_cast<const DerivedT *>(this));
  }

  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  friend std::unique_ptr<::mlir::Pass> createAffineFullUnroll() {
    return std::make_unique<DerivedT>();
  }
};
```

CRTP 在这里的具体作用：

| 位置 | 用 `DerivedT` 做什么 |
|---|---|
| 构造函数 | 把派生类的 TypeID 传给 `OperationPass` |
| `clonePass()` | 复制出正确派生类型的 pass |
| `classof()` | 用派生类的 TypeID 判断类型 |
| 工厂函数 | `make_unique<DerivedT>()` 构造具体 pass |

所以：

```cpp
struct AffineFullUnroll : impl::AffineFullUnrollBase<AffineFullUnroll> {
  using AffineFullUnrollBase::AffineFullUnrollBase;
  ...
};
```

不是在“继承一个普通基类”，而是**让基类模板为这个具体 pass 生成一整套样板代码**。

`using AffineFullUnrollBase::AffineFullUnrollBase;` 则把基类构造函数引入派生类：
基类构造函数里会使用 `DerivedT`（也就是 `AffineFullUnroll`）来设置 TypeID，
因此派生类不需要自己再写一个构造函数。

> 注意：`runOnOperation()` 仍然通过 `Pass` 的虚函数接口被 PassManager 调用；
> CRTP 在这里主要用于“生成样板 + 提供静态类型信息”，不是要替代所有虚函数。

### 失败行为实验

`loopUnrollFull` 要求 trip count 在编译期可确定。仓库里有一个现成的反例
`tests/dyn_unroll.mlir`，它本身也是一条 lit 测试（用 `not` 期望 pass 失败）：

```mlir
// RUN: not tutorial-opt %s --affine-full-unroll 2>&1 | FileCheck %s
// CHECK: error: unrolling failed

// A loop with a dynamic upper bound cannot be fully unrolled.
// This test drives the pass failure path: loopUnrollFull fails,
// the pass emits an error and calls signalPassFailure().
func.func @dyn(%n: index, %buf: memref<?xi32>) -> i32 {
  %init = arith.constant 0 : i32
  %sum = affine.for %i = 0 to %n iter_args(%acc = %init) -> i32 {
    %t = affine.load %buf[%i] : memref<?xi32>
    %next = arith.addi %acc, %t : i32
    affine.yield %next : i32
  }
  return %sum : i32
}
```

运行：

```bash
$TO tests/dyn_unroll.mlir --affine-full-unroll
```

输出（节选）：

```text
tests/dyn_unroll.mlir:9:10: error: unrolling failed
  %sum = affine.for %i = 0 to %n iter_args(%acc = %init) -> i32 {
         ^
...
exit=1
```

也可以按 lit 的方式验证（`not` 命令要求被执行的命令失败）：

```bash
$BIN/not $TO tests/dyn_unroll.mlir --affine-full-unroll 2>&1 \
  | $BIN/FileCheck tests/dyn_unroll.mlir
# 无输出，退出码 0
```

这正是 `signalPassFailure()` 的效果：诊断信息 + 非零退出码。

## 5. 实现一：手动 walk

完整代码（`AffineFullUnroll.cpp`）：

```cpp
struct AffineFullUnroll : impl::AffineFullUnrollBase<AffineFullUnroll> {
  using AffineFullUnrollBase::AffineFullUnrollBase;

  void runOnOperation() {
    getOperation()->walk([&](AffineForOp op) {
      if (failed(loopUnrollFull(op))) {
        op.emitError("unrolling failed");
        signalPassFailure();
      }
    });
  }
};
```

逐行理解：

| 代码 | 含义 |
|---|---|
| `getOperation()` | 当前 pass 锚定的 op（这里是 module） |
| `->walk([&](AffineForOp op) { ... })` | 深度优先遍历所有 `affine.for`，对每个调用回调 |
| `loopUnrollFull(op)` | MLIR 提供的工具：把该循环完全展开 |
| `failed(...)` | `loopUnrollFull` 返回 `LogicalResult`，失败时进入分支 |
| `op.emitError(...)` | 在出错的 op 上生成诊断 |
| `signalPassFailure()` | 通知 pass manager：本 pass 失败，停止 pipeline |

运行：

```bash
$TO tests/affine_loop_unroll.mlir --affine-full-unroll
```

输出（节选）：

```mlir
#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 3)>
module {
  func.func @test_single_nested_loop(%arg0: memref<4xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.load %arg0[%c0] : memref<4xi32>
    %1 = arith.addi %c0_i32, %0 : i32
    %2 = affine.apply #map(%c0)
    %3 = affine.load %arg0[%2] : memref<4xi32>
    ...
  }
}
```

手动版**只展开**，不负责折叠，所以索引保留为 `affine.apply #map(%c0)`。

## 6. 实现二：OpRewritePattern

pattern 版本把“展开一个循环”封装成一个重写规则：

```cpp
struct AffineFullUnrollPattern : public OpRewritePattern<AffineForOp> {
  AffineFullUnrollPattern(mlir::MLIRContext *context)
      : OpRewritePattern<AffineForOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    // 注意：这里直接修改 IR，没有走 rewriter。
    // 对当前有限的测试成立，但不是严格的用法。
    return loopUnrollFull(op);
  }
};

struct AffineFullUnrollPatternRewrite
    : impl::AffineFullUnrollPatternRewriteBase<AffineFullUnrollPatternRewrite> {
  using AffineFullUnrollPatternRewriteBase::AffineFullUnrollPatternRewriteBase;

  void runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<AffineFullUnrollPattern>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
```

### `matchAndRewrite` 的契约

- **match**：pattern 的模板参数 `AffineForOp` 已经限定“只匹配 affine.for”
- **rewrite**：匹配成功后修改 IR；返回 `success()`
- **不匹配**：返回 `failure()`，IR 必须保持不变
- **benefit**：多个 pattern 都能匹配时，benefit 高的先执行
- **`PatternRewriter`**：所有 IR 修改应通过它完成（`create`、`replaceOp`、
  `eraseOp` 等），这样 driver 才能维护内部状态

#### 把契约拆开理解

`matchAndRewrite` 本质上是一个**带条件的重写函数**：

```text
输入：一个候选 op
  │
  ├─ 1. match：这个 op 符合我的模式吗？
  │      不符合 → return failure()，IR 一个字节都不能改
  │      符合   → 进入第 2 步
  │
  ├─ 2. rewrite：通过 PatternRewriter 修改 IR
  │
  └─ 3. return success()：告诉 driver “我改了”
```

##### match：只做判断，不改 IR

`OpRewritePattern<AffineForOp>` 的模板参数只负责回答“哪些 op 会被交给这个
pattern 尝试”——所有 `affine.for` 都会进来。但“这个具体的循环是否真的要展开”，
仍然要在函数体里判断。例如 `MulToAdd` 的 pattern 还要检查：

- 右操作数是不是常量？
- 常量是不是 2 的幂？

任何一条不满足，就 `return failure()`。**返回 failure 之前不能修改 IR**，因为
driver 会继续尝试下一个 pattern；如果 IR 已经被改过，后续 pattern 看到的就不是
原始 IR，driver 维护的工作表也会失真，debug 构建下还可能触发断言。

##### rewrite：所有修改都走 rewriter

`MulToAdd` 是正确用法的例子：

```cpp
ConstantOp newConstant = rewriter.create<ConstantOp>(...);
MulIOp newMul = rewriter.create<MulIOp>(...);
AddIOp newAdd = rewriter.create<AddIOp>(...);
rewriter.replaceOp(op, newAdd);   // 替换旧 op 的所有使用，并删除旧 op
rewriter.eraseOp(rhsDefiningOp);  // 删除不再使用的常量 op
return success();
```

`PatternRewriter` 不是普通的 `OpBuilder`。它在修改 IR 的同时会：

- 记录哪些 op 被创建、替换、删除
- 把新创建的 op 加入 driver 的工作表，保证它们会被再次匹配
- 维护 SSA use-list 和支配关系
- 在 pattern 失败时支持回滚（部分场景）
- 通知 listener / instrumentation

如果直接调用 `loopUnrollFull(op)` 修改 IR，driver 不知道发生了什么。源码注释说
“技术上不允许，但对当前有限的测试成立”，原因是这里的改写比较简单：

- 循环被删除，循环体被复制出来
- 新产生的 op 不是这个 pattern 的匹配对象，不需要再次处理
- 换成更复杂的 pattern 时，就可能漏处理或破坏 driver 状态

##### benefit：多个 pattern 的优先级

`benefit` 不是“概率”，而是**优先级**。driver 对同一个 op 会按 benefit 从高到低
尝试 pattern：

- `PowerOfTwoExpand` benefit=2，先试
- `PeelFromMul` benefit=1，后试

于是 `8*x` 一定先走“折半”路径；如果反过来，`PeelFromMul` 会先把它变成
`7*x + x`，最终得到更长的加法链。

##### 返回值只有两种

- `success()`：我匹配并成功改写了 IR
- `failure()`：我不匹配（或无法安全改写），IR 保持原样

不要返回 success 却不改 IR，也不要返回 failure 却改了 IR。driver 依赖这个约定
判断是否继续迭代。

本实现直接调用 `loopUnrollFull(op)` 修改 IR，没有经过 `rewriter`——源码注释也
明确说“技术上不允许，但对当前测试可用”。这是一个很好的反面教材：**写 pattern 时
应尽量通过 rewriter 完成修改**。

### greedy driver 做了什么

```cpp
applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
```

它会：

1. 遍历 IR，找到能匹配的 op
2. 按 benefit 从高到低尝试 pattern
3. 应用 pattern，并**递归处理新产生的 op**
4. 同时执行 fold（常量折叠等）
5. 重复直到不动点（或达到迭代上限）

所以 pattern 版本输出里的 `affine.apply` 被折叠成了常量：

```mlir
func.func @test_single_nested_loop(%arg0: memref<4xi32>) -> i32 {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %0 = affine.load %arg0[%c0] : memref<4xi32>
  %1 = affine.load %arg0[%c1] : memref<4xi32>
  ...
}
```

### 对比两种实现

| | 手动 walk | pattern rewrite |
|---|---|---|
| 入口 | `runOnOperation` + `walk` | `runOnOperation` + greedy driver |
| 匹配方式 | 自己写回调 | `OpRewritePattern<T>` 自动按类型匹配 |
| 是否自动 fold | 否 | 是 |
| 新产生的 op | 不会被自动重新处理 | driver 会递归处理 |
| 适合 | 简单、一次性的遍历 | 可组合、可复用的重写规则 |

验证：

```bash
# 手动版：留下 affine.apply
$TO tests/affine_loop_unroll.mlir --affine-full-unroll

# pattern 版：直接得到常量索引
$TO tests/affine_loop_unroll.mlir --affine-full-unroll-rewrite

# 手动版补一次 canonicalize，结果与 pattern 版接近
$TO tests/affine_loop_unroll.mlir --affine-full-unroll --canonicalize
```

### “Should we walk?”

文章特别讨论了这个问题：**不要自己在 pattern pass 里再 walk 一遍**。
`applyPatternsAndFoldGreedily` 已经负责遍历和递归处理；手动 walk 会重复工作，
还可能漏掉 pattern 新产生的 op。

## 7. 实现三：多个 pattern 协作（mul-to-add）

`lib/Transform/Arith/MulToAdd.cpp` 把乘法改写为加法，包含两个 pattern：

```cpp
struct PowerOfTwoExpand : public OpRewritePattern<MulIOp> {
  PowerOfTwoExpand(MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(MulIOp op, PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!rhsDefiningOp) return failure();

    int64_t value = rhsDefiningOp.value();
    bool is_power_of_two = (value & (value - 1)) == 0;
    if (!is_power_of_two) return failure();

    ConstantOp newConstant = rewriter.create<ConstantOp>(
        rhsDefiningOp.getLoc(),
        rewriter.getIntegerAttr(rhs.getType(), value / 2));
    MulIOp newMul = rewriter.create<MulIOp>(op.getLoc(), lhs, newConstant);
    AddIOp newAdd = rewriter.create<AddIOp>(op.getLoc(), newMul, newMul);

    rewriter.replaceOp(op, newAdd);
    rewriter.eraseOp(rhsDefiningOp);
    return success();
  }
};

struct PeelFromMul : public OpRewritePattern<MulIOp> {
  PeelFromMul(MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(MulIOp op, PatternRewriter &rewriter) const override {
    // C 不是 2 的幂：C*x -> (C-1)*x + x
    ...
  }
};
```

### 两个 pattern 的分工

| pattern | benefit | 匹配 | 改写 |
|---|---|---|---|
| `PowerOfTwoExpand` | 2 | `C` 是 2 的幂 | `C*x` → `(C/2*x) + (C/2*x)` |
| `PeelFromMul` | 1 | 任意常量 `C` | `C*x` → `(C-1)*x + x` |

benefit 更高的先执行，所以 `8*x` 先走 `PowerOfTwoExpand`，再递归展开：

```text
8*x
→ 4*x + 4*x
→ (2*x + 2*x) + (2*x + 2*x)
→ (x+x) + ...
```

注意源码里 `newMul` 只创建了一次，add 的两个操作数指向同一个 SSA 值，所以最终
IR 是共享的表达式树，而不是 8 份独立的 `x+x`。

`9*x` 则先由 `PowerOfTwoExpand` 处理不了（不是 2 的幂），落到 `PeelFromMul`：

```text
9*x → 8*x + x → (3 层展开) + x
```

### 运行

```bash
$TO tests/mul_to_add.mlir --mul-to-add
```

输出：

```mlir
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

### 为什么 benefit 很重要

如果把 `PowerOfTwoExpand` 的 benefit 改成 0（低于 `PeelFromMul`），那么 `8*x`
会先被 `PeelFromMul` 改写成 `7*x + x`，再继续 peeled 成 `6*x + x + x`……最终得到
一串很长的加法链，而不是 3 层共享的树。

这正是 benefit 的用途：**在多个合法重写中表达优先级**。

### 一个容易忽略的约定

两个 pattern 都假设常量在**右边**：

```cpp
Value rhs = op.getOperand(1);
auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
```

MLIR 的 canonicalization 会保证“常量在右”这种全局约定（文章里也提到了链接）。
如果没有 canonicalize，`3*x` 和 `x*3` 可能只有一种能被 pattern 匹配。

## 8. 实验

先设置路径：

```bash
cd /home/zhouzhou/npu/mlir-tutorial
BIN=$PWD/externals/llvm-project/build/bin
TO=$PWD/build/tools/tutorial-opt
```

### 实验 1：手动版 vs pattern 版

```bash
$TO tests/affine_loop_unroll.mlir --affine-full-unroll | grep -c affine.apply
$TO tests/affine_loop_unroll.mlir --affine-full-unroll-rewrite | grep -c affine.apply
```

预期：前者大于 0，后者为 0。解释：pattern 版的 greedy driver 会 fold。

### 实验 2：观察 pass 执行后的 IR

```bash
$TO tests/affine_loop_unroll.mlir \
  --affine-full-unroll --canonicalize \
  --mlir-print-ir-after-all 2>&1 >/dev/null | grep 'IR Dump'
```

预期输出：

```text
// -----// IR Dump After AffineFullUnroll (affine-full-unroll) //----- //
// -----// IR Dump After Canonicalizer (canonicalize) //----- //
```

### 实验 3：mul-to-add

```bash
$TO tests/mul_to_add.mlir --mul-to-add
$TO tests/mul_to_add.mlir --mul-to-add | $BIN/FileCheck tests/mul_to_add.mlir
```

第二条命令无输出、退出码 0，说明测试通过。

### 实验 4：失败路径

```bash
# 直接运行：应当失败
$TO tests/dyn_unroll.mlir --affine-full-unroll
echo $?          # 1

# 按测试方式运行：not 反转退出码，FileCheck 匹配诊断
$BIN/not $TO tests/dyn_unroll.mlir --affine-full-unroll 2>&1 \
  | $BIN/FileCheck tests/dyn_unroll.mlir
echo $?          # 0
```

预期：第一条命令打印 `unrolling failed` 诊断、退出码 1；第二条无输出、退出码 0。

### 实验 5（可选）：修改 benefit

1. 把 `lib/Transform/Arith/MulToAdd.cpp` 中 `PowerOfTwoExpand` 的 benefit 从 2
   改成 0；
2. 重建：

   ```bash
   ninja -C build -j 100 tutorial-opt
   ```

3. 重新运行 `$TO tests/mul_to_add.mlir --mul-to-add`，观察 `8*x` 的输出从
   3 条 add 变成更长的加法链；
4. 改回 benefit 2，重建，确认恢复。

## 9. 源码位置

| 内容 | 文件 |
|---|---|
| 手动 unroll pass | `lib/Transform/Affine/AffineFullUnroll.cpp` |
| pattern 版 unroll pass | `lib/Transform/Affine/AffineFullUnrollPatternRewrite.cpp` |
| pass 声明 | `lib/Transform/Affine/Passes.td` |
| 生成注册函数 | `lib/Transform/Affine/Passes.h` |
| mul-to-add 的两个 pattern | `lib/Transform/Arith/MulToAdd.cpp` |
| pass 注册入口 | `tools/tutorial-opt.cpp` |
| 测试 | `tests/affine_loop_unroll.mlir`、`tests/mul_to_add.mlir`、`tests/dyn_unroll.mlir` |

## 10. 练习

### 练习 1：预测 unroll 结果

把 `tests/affine_loop_unroll.mlir` 第一个循环的 `0 to 4` 改成 `0 to 5`，分别
运行手动版和 pattern 版，预测并验证：

- 展开后有几条 `affine.load`？
- 手动版的输出里有多少个 `affine.apply`？

### 练习 2：解释输出差异

为什么 `--affine-full-unroll-rewrite` 不需要 `--canonicalize` 就能得到常量索引，
而手动版需要？

### 练习 3：pattern benefit

`PowerOfTwoExpand` 的 benefit 是 2，`PeelFromMul` 是 1。如果把两个值交换，
`8*x` 的最终输出会变成什么样？先推理，再按实验 5 的方法验证。

### 练习 4：匹配条件

`PowerOfTwoExpand` 用 `(value & (value - 1)) == 0` 判断 2 的幂。

- 这个表达式对 `value = 0` 返回什么？
- 这会不会导致问题？可以结合 canonicalize/fold 思考。

## 11. 参考答案

<details>
<summary>练习 1</summary>

5 条 `affine.load`，索引 0..4。手动版的索引计算会留下 4 个 `affine.apply`
（对应 +1、+2、+3、+4）；pattern 版会折叠成常量。

</details>

<details>
<summary>练习 2</summary>

pattern pass 使用 `applyPatternsAndFoldGreedily`，driver 在应用 pattern 的同时
执行 fold，常量上的 `affine.apply` 当场被折叠；手动 pass 只调用
`loopUnrollFull`，不做额外 fold，所以需要再跑一次 `--canonicalize`。

</details>

<details>
<summary>练习 3</summary>

交换后 `PeelFromMul` 先匹配 `8*x`，改写成 `7*x + x`，然后 `7*x` 继续被 peeled：
`6*x + x + x` … 最终得到 8 项相加的长链，而不是共享的 3 层加法树。这演示了
benefit 决定重写优先级。

</details>

<details>
<summary>练习 4</summary>

`value = 0` 时 `0 & -1 == 0`，所以会被判定为“2 的幂”。严格来说 0 不是 2 的幂，
这会生成 `0*x + 0*x` 的重写。实际 IR 中，`muli` 乘 0 通常会被 canonicalize/fold
先消掉，所以测试没有暴露这个问题；写 pattern 时应显式排除 0。

</details>

## 12. 验收清单

- [ ] 能说出一个 pass 的四个组成要素
- [ ] 能解释 `runOnOperation()`、`getOperation()`、`signalPassFailure()` 的作用
- [ ] 能读懂手动 walk 版 unroll pass
- [ ] 能解释 `OpRewritePattern` 的 match/rewrite 契约
- [ ] 能说明 greedy driver 为什么会让输出出现常量索引
- [ ] 能解释 pattern 的 benefit 如何影响 `mul-to-add`
- [ ] 能运行两个 pass，并用 FileCheck 验证
- [ ] 知道 pattern 修改 IR 应通过 `PatternRewriter`

## 下一课

**第 3 课：Using TableGen for Passes（文章 4）**：[03-tablegen-passes.md](03-tablegen-passes.md)

- `Passes.td` 的完整语法
- `-gen-pass-decls` 生成了什么
- `GEN_PASS_DEF_*`、`impl::XxxBase` 的来龙去脉
- pass 注册与 dependent dialects
- 动手新增一个 pass 声明

对应源码：`lib/Transform/Affine/Passes.td`、`Passes.h`、`CMakeLists.txt`