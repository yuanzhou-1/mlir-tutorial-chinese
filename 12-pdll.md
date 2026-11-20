# 第 12 课：Defining Patterns with PDLL

> 对应文章：[Defining Patterns with PDLL](https://www.jeremykun.com/2024/08/04/mlir-pdll/)。
>
> 本课把第 2 课的 `mul-to-add` 从 C++ `RewritePattern` 改写成 PDLL：
> 用 `.pdll` 描述“匹配什么、改写成什么”，由 `mlir-pdll` 生成 C++ 代码，
> 再交给 PDL interpreter 执行。
>
> 学完本课，你应该能看懂 `lib/Transform/Arith/MulToAdd.pdll`，知道
> `MulToAddPdll.h.inc` 从哪里来，并能自己构建、运行和调试 PDLL pattern。

## 1. 学习目标

- 区分 PDL、PDLL、PDLInterp 三个名字
- 理解 PDLL 的 `Pattern`、`Constraint`、`let`、`rewrite` 语法
- 理解 PDLL 的 match 区和 rewrite 区
- 理解属性约束、原生约束和 `PDLResultList`
- 能读懂 `MulToAdd.pdll` 的四个 pattern
- 理解 benefit 如何决定 pattern 的优先级
- 理解 `mlir-pdll -x=cpp` 生成的代码
- 理解 pass 为什么需要 PDL/PDLInterp dialect
- 能运行 `tests/mul_to_add_pdll.mlir`
- 能对比 PDLL、C++ pattern 和 TableGen DRR 的优缺点

## 2. 为什么还需要一种 pattern 语言

前几课已经出现了三种“写 pattern”的方式：

| 方式 | 例子 | 特点 |
| --- | --- | --- |
| C++ | `MulToAdd.cpp` 的 `OpRewritePattern` | 最灵活，但样板代码多，pattern 多了以后二进制变大 |
| TableGen DRR | `PolyPatterns.td` 的 `Pat` | 声明式，简洁，但受 TableGen DAG 结构限制 |
| PDLL | `MulToAdd.pdll` | 专门为 MLIR pattern 设计的声明式语言 |

TableGen DRR 的主要限制来自 TableGen 的 DAG：

- 多结果 op 要使用 `$results__0` 这种命名约定
- 约束和它约束的对象可能隔得很远
- 替换非根 op、替换多个 op 都很别扭
- 不容易表达复杂的静态值计算

PDLL 的目标是把 pattern 写成更接近“IR 本身”的样子：匹配部分像写输入 IR，
重写部分像写输出 IR。它既可以作为 C++ pattern 的替代，也可以在将来支持
运行时加载 pattern。

### 2.1 PDL、PDLL、PDLInterp 不是同一个东西

这是初学者最容易混淆的地方：

```
PDL          = MLIR 里的一个 dialect，用 op 表示“匹配/替换”这件事
PDLL         = 写给人看的 pattern 语言（本课写的 .pdll）
PDLInterp    = 更低层的 dialect，表示 interpreter 执行 pattern 的指令
```

三者的关系：

```
MulToAdd.pdll
      │  mlir-pdll 解析
      ▼
PDL IR（pdl.pattern ...）
      │  PatternRewriter 冻结 pattern 时转换
      ▼
PDLInterp IR（pdl_interp.*）
      │  编译
      ▼
PDLByteCode
      │  由 PatternApplicator 解释执行
      ▼
真正的 IR 改写
```

关键点：

- 你通常只写 PDLL，不直接写 PDL 或 PDLInterp
- `mlir-pdll` 的 `-x=mlir` 可以看到中间生成的 PDL IR
- `-x=cpp` 生成 C++ 头文件，里面把 PDL IR 作为字符串嵌入
- 真正执行匹配的是 **PDL bytecode interpreter**

这也是为什么 pass 要声明 PDL 和 PDLInterp 两个 dependent dialect。

## 3. 复习：C++ 版 MulToAdd

在 `lib/Transform/Arith/MulToAdd.cpp` 里有两个 pattern：

```
struct PowerOfTwoExpand : public OpRewritePattern<MulIOp> {
  PowerOfTwoExpand(MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!rhsDefiningOp)
      return failure();

    int64_t value = rhsDefiningOp.value();
    bool is_power_of_two = (value & (value - 1)) == 0;
    if (!is_power_of_two)
      return failure();

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
```

另一个 `PeelFromMul` 把：

```
9*x  ->  8*x + x
```

这条规则递归地作用在 8*x 上，所以 9*x 最终会变成四次加法。

C++ 版已经能工作，那 PDLL 版的价值是什么？

- 匹配结构用声明式语法表达，不用手写 `getOperand` 和分支
- 创建/替换 op 的意图更直观
- pattern 可以被 PDL interpreter 联合执行
- 更容易做运行时加载和工具支持

下面看 PDLL 具体怎么写。

## 4. `MulToAdd.pdll` 逐段解析

完整文件在 `lib/Transform/Arith/MulToAdd.pdll`。我们分段读。

### 4.1 include ODS 定义

```
#include "mlir/Dialect/Arith/IR/ArithOps.td"
```

PDLL 的 `#include` 不是 C++ 预处理器的 include，而是 PDLL 自己的导入：

- 可以包含 `.pdll` 文件
- 也可以包含 `.td` 文件，导入 ODS 的 op、类型、属性名等信息

导入 `ArithOps.td` 后，PDLL 才知道：

- `arith.muli`、`arith.addi`、`arith.constant` 存在
- `arith.constant` 有一个名为 `value` 的属性
- 这些 op 的操作数/结果结构

没有这个 include，`op<arith.muli>` 和 `{value = ...}` 都无法正确解析。

PDLL 文件最后会被 `mlir-pdll` 处理，所以 `.td` 的 include 路径要通过
`mlir-pdll -I ...` 传入。CMake 的 `add_mlir_pdll_library` 会自动处理这些
include 路径。

### 4.2 约束：`IsPowerOfTwo`

```
Constraint IsPowerOfTwo(attr: Attr) [{
  int64_t value = cast<::mlir::IntegerAttr>(attr).getValue().getSExtValue();
  return success((value & (value - 1)) == 0);
}];
```

读法：

- 定义一个约束，名字叫 `IsPowerOfTwo`
- 输入是一个 `Attr`
- `[{ ... }]` 里是 C++ 代码
- C++ 代码返回 `LogicalResult`
- `success(...)` 表示约束成立

PDLL 的 `Attr` 会映射成 C++ 的 `::mlir::Attribute`。因为这里实际存的是
`IntegerAttr`，所以先 `cast<::mlir::IntegerAttr>`，再取整数值。

注意：这个约束没有返回值，只做判断，所以使用方式是单独一行：

```
IsPowerOfTwo(const);
```

### 4.3 返回值的约束：`Halve` 和 `MinusOne`

```
// 当前版本中，带返回值的约束需要由 C++ 实现
Constraint Halve(atttr: Attr) -> Attr;
Constraint MinusOne(attr: Attr) -> Attr;
```

这里只写了**声明**，没有写函数体。原因：

> 在本课程使用的 MLIR 版本中，PDLL 里能直接写的约束不能返回值。
> 需要在 C++ 里注册一个 native constraint。

`Halve` 把常量除以 2，`MinusOne` 把常量减 1。它们不是“判断是否匹配”，
而是“根据匹配到的属性计算一个新属性”。

在 PDLL 里，它们可以这样使用：

```
let halved: Attr = Halve(const);
```

意思是：

```
调用 Halve(const)，返回一个 Attr，绑定到 halved
```

PDLL 知道返回类型是 `Attr`，因为声明里有 `-> Attr`。

> 源码里 `Halve` 的参数名写成了 `atttr`，这是笔误；参数名在 C++ 实现里
> 不使用，所以不影响行为。实际注册和调用按约束名、类型匹配。

### 4.4 第一个 pattern：常量在左边的 2 的幂

```
Pattern PowerOfTwoExpandRhs with benefit(2) {
  let root = op<arith.muli>(
      op<arith.constant> {value = const: Attr},
      rhs: Value);

  IsPowerOfTwo(const);
  let halved: Attr = Halve(const);

  rewrite root with {
    let newConst = op<arith.constant> {value = halved};
    let newMul = op<arith.muli>(newConst, rhs);
    let newAdd = op<arith.addi>(newMul, newMul);
    replace root with newAdd;
  };
}
```

分成两段看。

#### match 区

```
let root = op<arith.muli>(
    op<arith.constant> {value = const: Attr},
    rhs: Value);

IsPowerOfTwo(const);
let halved: Attr = Halve(const);
```

它表示要匹配：

```
%c = arith.constant C : i32
%r = arith.muli %c, %arg : i32
```

并满足：

- 第一个操作数必须由 `arith.constant` 定义
- 第二个操作数是任意 `Value`，绑定到 `rhs`
- 常量的 `value` 属性绑定到 `const`
- `IsPowerOfTwo(const)` 必须成立

`op<arith.muli>(...)` 是根 op。根 op 必须被绑定，并且要连接到 rewrite 区。

嵌套的 `op<arith.constant>` 也是模式的一部分：它要求操作数确实由一条
`arith.constant` 定义，而不是任意一个值。

#### rewrite 区

```
rewrite root with {
  let newConst = op<arith.constant> {value = halved};
  let newMul = op<arith.muli>(newConst, rhs);
  let newAdd = op<arith.addi>(newMul, newMul);
  replace root with newAdd;
};
```

这段是“创建新 IR”：

```
newConst = constant (C/2)
newMul   = newConst * rhs
newAdd   = newMul + newMul
用 newAdd 替换 root
```

其中 `newMul` 使用了两次，所以 `newAdd` 的两个操作数是**同一个 SSA 值**，
不是两次独立的乘法。这正是：

```
C*x  ->  (C/2)*x + (C/2)*x
```

这里创建 op 时都省略了结果类型：

- `newConst` 的类型由 `halved` 这个整数属性推导
- `newMul` / `newAdd` 的类型由操作数和 `arith` 的类型推导规则得到
- `replace root with newAdd` 还要求替换值的类型与 root 的结果类型兼容

`rewrite root with { ... }` 本身不会自动删除 root；是里面的
`replace root with newAdd` 完成替换，原 root 失去使用者后再由 driver 清理。

#### 名字的坑

源码里的名字容易让人误解：

- `PowerOfTwoExpandRhs` 中，常量其实是第一个操作数
- `PowerOfTwoExpandLhs` 中，常量其实是第二个操作数

`Rhs/Lhs` 后缀与“常量在哪一侧”并不完全一致。读代码时不要只看名字，
必须以 `op<arith.muli>(...)` 的操作数顺序为准。

之所以有两个方向，是因为 C++ 版假设常量在右边，但输入 IR 不一定满足这个
假设；PDLL 版把两个方向都写了出来。

### 4.5 第二个 pattern：常量在右边的 2 的幂

```
Pattern PowerOfTwoExpandLhs with benefit(2) {
  let root = op<arith.muli>(
      lhs: Value,
      op<arith.constant> {value = const: Attr});

  IsPowerOfTwo(const);
  let halved: Attr = Halve(const);

  rewrite root with {
    let newConst = op<arith.constant> {value = halved};
    let newMul = op<arith.muli>(lhs, newConst);
    let newAdd = op<arith.addi>(newMul, newMul);
    replace root with newAdd;
  };
}
```

与上一个 pattern 的差异：

- 第一个操作数绑定为 `lhs`
- 第二个操作数是常量
- 重写时乘法顺序是 `lhs * newConst`

其他部分相同。

### 4.6 第三个 pattern：剥离一个常量

```
Pattern PeelFromMulRhs with benefit(1) {
  let root = op<arith.muli>(
      lhs: Value,
      op<arith.constant> {value = const: Attr});

  // 由于 PowerOfTwoExpand 的 benefit 更高，
  // 这里的 value 保证不是 2 的幂。
  let minusOne: Attr = MinusOne(const);

  rewrite root with {
    let newConst = op<arith.constant> {value = minusOne};
    let newMul = op<arith.muli>(lhs, newConst);
    let newAdd = op<arith.addi>(newMul, lhs);
    replace root with newAdd;
  };
}
```

它匹配任意“变量 * 常量”，然后把乘法拆成：

```
C*x  ->  (C-1)*x + x
```

例如：

```
9*x  ->  8*x + x
```

为什么它不检查 `IsPowerOfTwo`？

- `PowerOfTwoExpand` 的 benefit 是 2
- `PeelFromMul` 的 benefit 是 1

如果常量是 8，两个 pattern 都能匹配，但 greedy driver 会优先选 benefit 高的
`PowerOfTwoExpand`，先把 8*x 变成 4*x + 4*x。

如果常量是 9，`PowerOfTwoExpand` 的 `IsPowerOfTwo` 失败，才会轮到 `PeelFromMul`：

```
9*x -> 8*x + x
```

然后 8*x 又可以走 `PowerOfTwoExpand`，继续拆分。

> PDLL 的默认 benefit 等于 match 区中不同 op 表达式的数量。本文件里的 pattern
> 都有两个 op 表达式（一个 constant、一个 muli），所以默认是 2；
> `PeelFromMul` 显式写成 `benefit(1)`，把优先级降到 2 的幂 pattern 之后。

#### benefit 不是“正确性保证”

benefit 只影响 pattern 的选择顺序：

- 高 benefit 先尝试
- 失败才尝试低 benefit

所以 `PeelFromMul` 依赖“`PowerOfTwoExpand` 会先处理 2 的幂”这个调度约定。
如果改了 benefit 顺序，8*x 可能先被 peel 成 7*x + x，虽然理论上仍可能最终
得到正确结果，但会生成更多中间 op，甚至失去原本简洁的展开形式。

### 4.7 第四个 pattern：常量在左边的剥离

```
Pattern PeelFromMulLhs with benefit(1) {
  let root = op<arith.muli>(
      op<arith.constant> {value = const: Attr},
      rhs: Value);

  let minusOne: Attr = MinusOne(const);

  rewrite root with {
    let newConst = op<arith.constant> {value = minusOne};
    let newMul = op<arith.muli>(newConst, rhs);
    let newAdd = op<arith.addi>(newMul, rhs);
    replace root with newAdd;
  };
}
```

它和 `PeelFromMulRhs` 对称，只是常量在左边：

```
C*x  ->  (C-1)*x + x
```

四个 pattern 合在一起，覆盖了：

```
常量在左 / 常量在右
2 的幂 / 非 2 的幂
```

## 5. 原生约束的 C++ 实现

`MulToAddPdll.cpp` 里实现了两个带返回值的约束：

```
LogicalResult halveImpl(PatternRewriter &rewriter, PDLResultList &results,
                        ArrayRef<PDLValue> args) {
  Attribute attr = args[0].cast<Attribute>();
  IntegerAttr cAttr = cast<IntegerAttr>(attr);
  int64_t value = cAttr.getValue().getSExtValue();
  results.push_back(rewriter.getIntegerAttr(cAttr.getType(), value / 2));
  return success();
}

LogicalResult minusOneImpl(PatternRewriter &rewriter, PDLResultList &results,
                           ArrayRef<PDLValue> args) {
  Attribute attr = args[0].cast<Attribute>();
  IntegerAttr cAttr = cast<IntegerAttr>(attr);
  int64_t value = cAttr.getValue().getSExtValue();
  results.push_back(rewriter.getIntegerAttr(cAttr.getType(), value - 1));
  return success();
}
```

逐行解释：

1. `PatternRewriter &rewriter` 用来创建新属性
2. `PDLResultList &results` 用来返回约束的结果
3. `ArrayRef<PDLValue> args` 是 PDLL 传进来的参数
4. `args[0].cast<Attribute>()` 做类型擦除后的转换
5. `cast<IntegerAttr>` 进一步确认它是整数属性
6. 计算结果用 `results.push_back(...)` 返回

这里的返回值是一个新的 `IntegerAttr`，类型与原常量相同，值为原来的一半
或减一。

### 5.1 注册原生约束

```
void registerNativeConstraints(RewritePatternSet &patterns) {
  patterns.getPDLPatterns().registerConstraintFunction("Halve", halveImpl);
  patterns.getPDLPatterns().registerConstraintFunction("MinusOne", minusOneImpl);
}
```

把 PDLL 里的名字映射到 C++ 函数：

```
"Halve"    -> halveImpl
"MinusOne" -> minusOneImpl
```

如果没有调用 `registerNativeConstraints`，生成的 PDL pattern 在运行时找不到
`Halve`/`MinusOne`，匹配会失败。

### 5.2 两种 native constraint 函数签名

`PDLPatternModule::registerConstraintFunction` 接受两种形式：

1. 低层形式：

```
LogicalResult fn(PatternRewriter &, PDLResultList &, ArrayRef<PDLValue>);
```
2. 高层形式：

```
LogicalResult fn(PatternRewriter &, Value value, ...);
```

高层形式由框架自动把 `PDLValue` 转换成 `Value`、`Operation *` 等 C++ 类型，
但它只返回 `LogicalResult`，不能把计算结果交回 PDL。因此本仓库中带返回值的
`Halve`/`MinusOne` 使用低层形式，把结果放进 `PDLResultList`。

## 6. pass、构建和注册

### 6.1 Passes.td

```
include "mlir/Dialect/PDL/IR/PDLDialect.td"
include "mlir/Dialect/PDLInterp/IR/PDLInterpOps.td"

def MulToAddPdll : Pass<"mul-to-add-pdll"> {
  let summary = "Convert multiplications to repeated additions using pdll";
  let description = [{
    Convert multiplications to repeated additions (using pdll).
  }];
  let dependentDialects = [
    "mlir::pdl::PDLDialect",
    "mlir::pdl_interp::PDLInterpDialect",
  ];
}
```

为什么需要两个 dependent dialect？

- `PDLDialect`：生成的 C++ 代码要把 `pdl.pattern ...` 字符串解析成 PDL IR，
所以 PDL dialect 必须存在
- `PDLInterpDialect`：冻结 pattern 时，PDL IR 会 lowering 成
`pdl_interp.*` op，所以 PDLInterp dialect 也必须存在

如果忘了声明，运行 pass 时可能看到“dialect 未注册”或 PDL lowering 失败。

### 6.2 CMake

```
add_mlir_pdll_library(MulToAddPdllIncGen
  MulToAdd.pdll
  MulToAddPdll.h.inc
)

add_mlir_library(MulToAdd
    MulToAdd.cpp
    MulToAddPdll.cpp
    ...
    DEPENDS
    MLIRMulToAddPasses
    MulToAddPdllIncGen
    ...
)
```

`add_mlir_pdll_library` 会调用 `mlir-pdll`：

```
mlir-pdll -x=cpp <include paths> MulToAdd.pdll -o MulToAddPdll.h.inc
```

生成的 `MulToAddPdll.h.inc` 在构建树：

```
build/lib/Transform/Arith/MulToAddPdll.h.inc
```

不是源码目录里的文件，不要手改。

Bazel 路线的等价写法在 `lib/Transform/Arith/BUILD`：

```
gentbl_cc_library(
    name = "MulToAddPdllIncGen",
    tbl_outs = [(["-x=cpp"], "MulToAddPdll.h.inc")],
    tblgen = "@llvm-project//mlir:mlir-pdll",
    td_file = "MulToAdd.pdll",
    ...
)
```

注意这里还是 `gentbl_cc_library`，只是 `tblgen` 从 `mlir-tblgen` 换成了
`mlir-pdll`。

### 6.3 头文件

`MulToAddPdll.h`：

```
#define GEN_PASS_DECL_MULTOADDPDLL
#include "lib/Transform/Arith/Passes.h.inc"

#include "lib/Transform/Arith/MulToAddPdll.h.inc"

namespace mlir {
namespace tutorial {
...
}
}
```

它做两件事：

1. 生成 pass 的声明
2. 导入 `mlir-pdll` 生成的 pattern 类

### 6.4 pass 实现

```
struct MulToAddPdll : impl::MulToAddPdllBase<MulToAddPdll> {
  using MulToAddPdllBase::MulToAddPdllBase;

  void runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    populateGeneratedPDLLPatterns(patterns);
    registerNativeConstraints(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
```

关键三步：

1. `populateGeneratedPDLLPatterns(patterns)`：把生成的四个 PDL pattern
加入 pattern set
2. `registerNativeConstraints(patterns)`：注册 `Halve` 和 `MinusOne`
3. `applyPatternsAndFoldGreedily`：运行 greedy driver

`populateGeneratedPDLLPatterns` 是 `MulToAddPdll.h.inc` 末尾生成的函数：

```
patterns.add<PowerOfTwoExpandRhs>(...);
patterns.add<PowerOfTwoExpandLhs>(...);
patterns.add<PeelFromMulRhs>(...);
patterns.add<PeelFromMulLhs>(...);
```

### 6.5 pass 注册

`lib/Transform/Arith/Passes.h`：

```
#include "lib/Transform/Arith/MulToAdd.h"
#include "lib/Transform/Arith/MulToAddPdll.h"

#define GEN_PASS_REGISTRATION
#include "lib/Transform/Arith/Passes.h.inc"
```

`tools/tutorial-opt.cpp`：

```
mlir::tutorial::registerArithPasses();
```

因此 `tutorial-opt` 才认识：

```
--mul-to-add-pdll
```

## 7. 生成代码里有什么

### 7.1 查看 PDL IR

```
$BIN/mlir-pdll -x=mlir \
  -I externals/llvm-project/mlir/include \
  lib/Transform/Arith/MulToAdd.pdll \
  > build/scratch/lesson12/MulToAdd.pdl.mlir
```

输出开头是：

```
module {
  pdl.pattern @PowerOfTwoExpandRhs : benefit(2) {
    %0 = operands
    %1 = attribute
    %2 = types
    %3 = operation "arith.constant"(%0 : !pdl.range<value>)
         {"value" = %1} -> (%2 : !pdl.range<type>)
    %4 = result 0 of %3
    %5 = operand
    %6 = types
    %7 = operation "arith.muli"(%4, %5 : !pdl.value, !pdl.value)
         -> (%6 : !pdl.range<type>)
    apply_native_constraint "IsPowerOfTwo"(%1 : !pdl.attribute)
    %8 = apply_native_constraint "Halve"(%1 : !pdl.attribute) : !pdl.attribute
    rewrite %7 {
      %9 = operation "arith.constant" {"value" = %8}
      %10 = result 0 of %9
      %11 = operation "arith.muli"(%10, %5 : !pdl.value, !pdl.value)
      %12 = result 0 of %11
      %13 = result 0 of %11
      %14 = operation "arith.addi"(%12, %13 : !pdl.value, !pdl.value)
      replace %7 with %14
    }
  }
  ...
}
```

可以把它和 PDLL 源码对照：

| PDLL | PDL IR |
| --- | --- |
| `let root = op<arith.muli>(...)` | `%7 = operation "arith.muli"...` |
| `op<arith.constant> {value = const}` | `%3 = operation "arith.constant"...` + `%1 = attribute` |
| `IsPowerOfTwo(const)` | `apply_native_constraint "IsPowerOfTwo"` |
| `Halve(const)` | `apply_native_constraint "Halve"` |
| `op<arith.addi>(newMul, newMul)` | `%14 = operation "arith.addi"(%12, %13)` |
| `replace root with newAdd` | `replace %7 with %14` |

注意 `%12` 和 `%13` 都是 `result 0 of %11`：这正对应“同一个 `newMul`
用了两次”。

### 7.2 查看生成的 C++

```
$BIN/mlir-pdll -x=cpp \
  -I externals/llvm-project/mlir/include \
  lib/Transform/Arith/MulToAdd.pdll \
  > build/scratch/lesson12/MulToAdd.direct.h.inc
```

生成的代码大致结构：

```
static ::llvm::LogicalResult IsPowerOfTwoPDLFn(
    ::mlir::PatternRewriter &rewriter, ::mlir::Attribute attr) {
  int64_t value = cast<::mlir::IntegerAttr>(attr).getValue().getSExtValue();
  return success((value & (value - 1)) == 0);
}

namespace {
struct PowerOfTwoExpandRhs : ::mlir::PDLPatternModule {
  template <typename... ConfigsT>
  PowerOfTwoExpandRhs(::mlir::MLIRContext *context, ConfigsT &&...configs)
      : ::mlir::PDLPatternModule(
            ::mlir::parseSourceString<::mlir::ModuleOp>(
                R"mlir(pdl.pattern @PowerOfTwoExpandRhs : benefit(2) {
                  ...
                })mlir",
                context),
            std::forward<ConfigsT>(configs)...) {
    registerConstraintFunction("IsPowerOfTwo", IsPowerOfTwoPDLFn);
  }
};
} // namespace

template <typename... ConfigsT>
void populateGeneratedPDLLPatterns(::mlir::RewritePatternSet &patterns,
                                   ConfigsT &&...configs) {
  patterns.add<PowerOfTwoExpandRhs>(patterns.getContext(), configs...);
  patterns.add<PowerOfTwoExpandLhs>(patterns.getContext(), configs...);
  patterns.add<PeelFromMulRhs>(patterns.getContext(), configs...);
  patterns.add<PeelFromMulLhs>(patterns.getContext(), configs...);
}
```

重点：

- PDL IR 被作为原始字符串嵌入 C++，运行时才解析
- inline 的 `IsPowerOfTwo` 被翻译成一个静态 C++ 函数并自动注册
- 带返回值的 `Halve`/`MinusOne` 没有自动实现，需要用户在 `MulToAddPdll.cpp`
里注册
- 每个 pattern 类都继承 `PDLPatternModule`，但最终都会被当作 pattern 使用

### 7.3 从 pattern set 到 bytecode

当 `applyPatternsAndFoldGreedily` 把 `RewritePatternSet` 冻结成
`FrozenRewritePatternSet` 时，MLIR 会做：

```
RewritePatternSet
   │ 取出 PDLPatternModule
   ▼
PDL IR
   │ convertPDLToPDLInterp
   ▼
PDLInterp IR
   │ 编译
   ▼
PDLByteCode
```

相关代码在：

- `mlir/lib/Rewrite/FrozenRewritePatternSet.cpp`
- `mlir/lib/Rewrite/PatternApplicator.cpp`
- `mlir/lib/Conversion/PDLToPDLInterp/`

`PatternApplicator` 会先让 PDL bytecode 匹配，再把 PDL 匹配结果和原生 C++
pattern 按 benefit 放在一起选择。因此 PDLL 和 C++ pattern 可以混用。

## 8. 实验

以下命令都在仓库根目录执行：

```
cd /home/zhouzhou/npu/mlir-tutorial

export TOP=$PWD
export BIN=$TOP/externals/llvm-project/build/bin
export TO=$TOP/build/tools/tutorial-opt
```

### 8.1 构建 tutorial-opt

```
ninja -C build -j 100 tutorial-opt
```

如果 `.pdll` 有改动，这条命令会先运行 `mlir-pdll` 重新生成
`build/lib/Transform/Arith/MulToAddPdll.h.inc`，然后编译。

### 8.2 确认 pass 已注册

```
$TO --help | grep mul-to-add
```

输出：

```
--mul-to-add        - Convert multiplications to repeated additions
--mul-to-add-pdll   - Convert multiplications to repeated additions using pdll
```

### 8.3 运行测试

```
$TO tests/mul_to_add_pdll.mlir --mul-to-add-pdll
```

真实输出：

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

### 8.4 运行 FileCheck

```
$TO tests/mul_to_add_pdll.mlir --mul-to-add-pdll \
  | $BIN/FileCheck tests/mul_to_add_pdll.mlir
```

没有输出就是通过。

测试里的 `CHECK` 断言了：

- `@just_power_of_two` 产生三条加法
- `@power_of_two_plus_one` 产生四条加法
- 每一层都复用上一层的结果

### 8.5 为什么 8 是三条加法

```
8x
 -> 4x + 4x              (一条 addi)
 -> (2x + 2x) + (2x + 2x)  (但两个 4x 是同一个 SSA 值)
 -> (x + x) * 4           (同样复用)
```

更准确地说，greedy driver 的中间步骤会得到：

```
%a = arith.addi %x, %x       // 2x
%b = arith.addi %a, %a       // 4x
%c = arith.addi %b, %b       // 8x
```

因为 `newMul` 在 PDLL 里被用了两次，所以 4x 只计算一次，两个使用点共享它。
最终只需要三条 `addi`。

### 8.6 为什么 9 是四条加法

```
9x
 -> 8x + x                  (PeelFromMul，第一条 addi)
 -> (4x + 4x) + x           (PowerOfTwoExpand，第二条 addi)
 -> ((2x + 2x) + (2x + 2x)) + x
 -> ((x + x) + (x + x)) + x (第三、四条 addi，复用中间值)
```

输出正好是四条：

```
%0 = arith.addi %arg0, %arg0 : i32   // 2x
%1 = arith.addi %0, %0 : i32         // 4x
%2 = arith.addi %1, %1 : i32         // 8x
%3 = arith.addi %2, %arg0 : i32      // 8x + x
```

### 8.7 补充实验：常量在左边

测试文件只覆盖了常量在右边的情况。先创建临时目录：

```
mkdir -p build/scratch/lesson12
```

下面这份临时输入放到 `build/scratch/lesson12/pdll_cases.mlir`，不会出现在
仓库测试里：

```
func.func @lhs_constant(%arg: i32) -> i32 {
  %c = arith.constant 8 : i32
  %m = arith.muli %c, %arg : i32
  return %m : i32
}

func.func @lhs_9(%arg: i32) -> i32 {
  %c = arith.constant 9 : i32
  %m = arith.muli %c, %arg : i32
  return %m : i32
}

func.func @no_constant(%arg: i32, %other: i32) -> i32 {
  %m = arith.muli %arg, %other : i32
  return %m : i32
}
```

运行：

```
$TO build/scratch/lesson12/pdll_cases.mlir --mul-to-add-pdll
```

真实输出：

```
module {
  func.func @lhs_constant(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    %1 = arith.addi %0, %0 : i32
    %2 = arith.addi %1, %1 : i32
    return %2 : i32
  }
  func.func @lhs_9(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    %1 = arith.addi %0, %0 : i32
    %2 = arith.addi %1, %1 : i32
    %3 = arith.addi %2, %arg0 : i32
    return %3 : i32
  }
  func.func @no_constant(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.muli %arg0, %arg1 : i32
    return %0 : i32
  }
}
```

可以看到：

- `8*x` 仍然变成三条 `addi`（走常量在左的 pattern）
- `9*x` 变成四条 `addi`（先 peel，再展开 8*x）
- `arg * other` 保持不变（没有常量，不匹配任何 pattern）

这说明 PDLL 版比原来的 C++ 版多覆盖了常量在左边的输入。

### 8.8 补充实验：1 和 0

如果输入是：

```
func.func @times_one(%arg: i32) -> i32 {
  %c = arith.constant 1 : i32
  %m = arith.muli %arg, %c : i32
  return %m : i32
}
```

会得到：

```
return %arg : i32
```

`x*1` 和 `x*0` 会被 `arith.muli` 的 folder 直接折叠。这个实验说明：

- greedy driver 里 pattern 和 folder 是协同工作的
- 不要只根据 pattern 的数学形式推断最终 IR，folding 也可能先改变输入

## 9. 源码位置

| 内容 | 文件 |
|---|---|
| PDLL pattern 源文件 | `lib/Transform/Arith/MulToAdd.pdll` |
| 原生约束与 pass 实现 | `lib/Transform/Arith/MulToAddPdll.cpp` |
| pass 定义 | `lib/Transform/Arith/Passes.td` |
| 构建配置 | `lib/Transform/Arith/CMakeLists.txt`、`lib/Transform/Arith/BUILD` |
| 生成的 C++ 代码 | `build/lib/Transform/Arith/MulToAddPdll.h.inc`（构建产物） |
| 本课测试 | `tests/mul_to_add_pdll.mlir` |

## 10. PDLL、C++、DRR 对比

| 维度 | C++ `OpRewritePattern` | TableGen DRR | PDLL |
| --- | --- | --- | --- |
| 表达能力 | 最强，任意 C++ | 受 DAG 限制 | 介于两者之间，持续发展 |
| 样板代码 | 多 | 少 | 少 |
| 多结果 op | 直接处理 | 需要特殊命名 | 更自然 |
| 静态值运算 | 直接写 C++ | 可用 `CPred` | 当前常需 native constraint |
| 编译期检查 | C++ 编译器 | TableGen 检查 | PDLL 编译器检查 |
| 运行方式 | 编译成 C++ | 编译成 C++ | PDL bytecode interpreter |
| 运行时加载 | 困难 | 困难 | 设计目标之一 |
| 二进制大小 | 每个 pattern 都进二进制 | 同 C++ | bytecode 通常更小 |
| 调试 | 断点/日志直接 | 生成的 C++ | 可看 PDL/PDLInterp/bytecode |
| 区域、dialect conversion | 支持 | 有限 | 当前版本仍有限制 |

选择建议：

- 需要复杂控制流、region、dialect conversion：先用 C++
- 简单到中等复杂、想减少样板：DRR 或 PDLL
- 想尝试运行时 pattern、工具链支持、bytecode 共享：PDLL
- 当前项目已经有 PDLL 基础设施，`mul-to-add-pdll` 是很好的模板

## 11. 常见坑与限制

### 11.1 忘记 include ODS

如果 `.pdll` 没有：

```
#include "mlir/Dialect/Arith/IR/ArithOps.td"
```

`op<arith.muli>` 和 `{value = ...}` 无法正确解析。`mlir-pdll` 也需要正确的
`-I` 路径。

### 11.2 忘记注册 native constraint

`Halve` 和 `MinusOne` 只是声明，必须：

```
registerNativeConstraints(patterns);
```

否则运行时找不到实现，pattern 无法匹配。

### 11.3 忘记 dependent dialect

`Passes.td` 必须声明：

```
let dependentDialects = [
  "mlir::pdl::PDLDialect",
  "mlir::pdl_interp::PDLInterpDialect",
];
```

原因见第 6.1 节：PDL 字符串解析和 PDL→PDLInterp lowering 都需要 dialect。

### 11.4 benefit 顺序

`PowerOfTwoExpand` 的 benefit 必须是 2，`PeelFromMul` 必须是 1。
顺序反过来会让 8*x 先被 peel 成 7*x + x，展开效果变差。

### 11.5 变量重用不是“复制”

```
let newMul = op<arith.muli>(newConst, rhs);
let newAdd = op<arith.addi>(newMul, newMul);
```

两个 `newMul` 是同一个 SSA 值。若想创建两个独立乘法，必须写两个不同的
`op<arith.muli>` 表达式。

### 11.6 `replace` 的类型约束

`replace root with newAdd` 要求替换值的数量和类型与 root 的结果兼容。
这里 `arith.addi` 和 `arith.muli` 都是 `i32`，所以没问题。

### 11.7 PDLL 不是万能的

在本课程对应的版本中：

- PDLL 不能直接做算术/比较等静态表达式
- region 支持有限
- dialect conversion 不在 PDLL 的支持范围内
- 带返回值的约束必须写 native C++（本版本）

所以 PDLL 是“减少样板 + 运行时 pattern”的工具，不是 C++ 的完全替代。

### 11.8 生成文件不要手改

```
build/lib/Transform/Arith/MulToAddPdll.h.inc
```

是构建产物。修改 `.pdll` 后重新构建，它会自动更新。

## 12. 练习

### 练习 1：预测 8x 和 9x 的输出

对于：

```
%c8 = arith.constant 8 : i32
%r8 = arith.muli %arg, %c8 : i32
```

和：

```
%c9 = arith.constant 9 : i32
%r9 = arith.muli %arg, %c9 : i32
```

分别需要几条 `arith.addi`？为什么？

<details>
<summary>参考答案</summary>

8x 需要 3 条，9x 需要 4 条。

8x：

```
8x -> 4x + 4x
4x -> 2x + 2x
2x -> x + x
```

9x：

```
9x -> 8x + x
8x -> 4x + 4x
4x -> 2x + 2x
2x -> x + x
```

因为 `newMul` 在 PDLL 里被复用，两个 4x 共享同一个 SSA 值，所以 8x 不是
“每个分支各展开一次”的 7 条加法，而是 3 条。

</details>

### 练习 2：为常量在左边的 9x 增加测试

在 `tests/mul_to_add_pdll.mlir` 中增加一个函数：

```
func.func @lhs_power_of_two_plus_one(%arg: i32) -> i32 {
  %0 = arith.constant 9 : i32
  %1 = arith.muli %0, %arg : i32
  return %1 : i32
}
```

写出 FileCheck 断言，并说明它走哪两个 pattern。

<details>
<summary>参考答案</summary>

期望输出：

```
func.func @lhs_power_of_two_plus_one(%arg0: i32) -> i32 {
  %0 = arith.addi %arg0, %arg0 : i32
  %1 = arith.addi %0, %0 : i32
  %2 = arith.addi %1, %1 : i32
  %3 = arith.addi %2, %arg0 : i32
  return %3 : i32
}
```

对应的检查可以写成：

```
// CHECK-LABEL: func.func @lhs_power_of_two_plus_one(
// CHECK-SAME:    %[[ARG:.*]]: i32
// CHECK-SAME:  ) -> i32 {
// CHECK:   %[[SUM_0:.*]] = arith.addi %[[ARG]], %[[ARG]]
// CHECK:   %[[SUM_1:.*]] = arith.addi %[[SUM_0]], %[[SUM_0]]
// CHECK:   %[[SUM_2:.*]] = arith.addi %[[SUM_1]], %[[SUM_1]]
// CHECK:   %[[SUM_3:.*]] = arith.addi %[[SUM_2]], %[[ARG]]
// CHECK:   return %[[SUM_3]] : i32
// CHECK: }
```

它先走 `PeelFromMulLhs`（9x -> 8x + x），再走
`PowerOfTwoExpandLhs`（8x -> 4x + 4x -> ...）。

</details>

### 练习 3：为什么 Peel 不检查 2 的幂

`PeelFromMulRhs` 没有调用 `IsPowerOfTwo`，它为什么仍然安全？如果两个 pattern
的 benefit 反过来会怎样？

<details>
<summary>参考答案</summary>

`PowerOfTwoExpand` 的 benefit 是 2，`PeelFromMul` 的 benefit 是 1。
greedy driver 优先尝试高 benefit 的 pattern，所以 8x 会先被
`PowerOfTwoExpand` 处理。只有 2 的幂检查失败时，才会落到 `PeelFromMul`。

如果 benefit 反过来，8x 可能先被 peel 成 7x + x，虽然仍可能最终得到正确结果，
但会生成更多中间 op，展开形式变差。

</details>

### 练习 4：解释生成代码的执行链

`MulToAddPdll.cpp` 只调用了：

```
populateGeneratedPDLLPatterns(patterns);
registerNativeConstraints(patterns);
applyPatternsAndFoldGreedily(...)
```

请说明从 `.pdll` 到真正执行的完整链路。

<details>
<summary>参考答案</summary>

```
MulToAdd.pdll
  -> mlir-pdll -x=cpp
  -> MulToAddPdll.h.inc
     - 每个 pattern 是 PDLPatternModule 子类
     - 构造时解析内嵌的 PDL IR 字符串
  -> populateGeneratedPDLLPatterns 加入 RewritePatternSet
  -> registerNativeConstraints 注册 Halve/MinusOne
  -> applyPatternsAndFoldGreedily 冻结 pattern set
  -> PDL IR -> PDLInterp IR -> PDLByteCode
  -> PatternApplicator 解释执行
```

</details>

### 练习 5：为什么 native constraint 用 `PDLResultList`

`halveImpl` 的签名是：

```
LogicalResult halveImpl(PatternRewriter &rewriter, PDLResultList &results,
                        ArrayRef<PDLValue> args)
```

为什么不是直接返回 `Attribute`？

<details>
<summary>参考答案</summary>

PDL 约束是运行时解释执行的，结果需要统一放入 PDL 的结果列表中，供后续
PDL IR 使用。`PDLResultList` 是解释器和原生函数之间的通用返回值通道。

高层注册接口也支持更简单的 C++ 函数签名，但带返回值的约束使用低层接口更直接，
也是本仓库采用的方式。

</details>

## 13. 验收清单

- 能区分 PDL、PDLL、PDLInterp
- 能读懂 `Pattern` 的 match 区和 rewrite 区
- 能解释 `op<arith.muli>` 的嵌套匹配
- 能解释 `{value = const: Attr}` 绑定的是什么
- 能解释 `IsPowerOfTwo` 和 `Halve` 的区别
- 能解释 `PDLResultList` 的作用
- 能说出四个 pattern 的分工
- 能解释 benefit 2/1 的顺序
- 能说出 `add_mlir_pdll_library` 生成了什么
- 能找到 `build/lib/Transform/Arith/MulToAddPdll.h.inc`
- 能运行 `tests/mul_to_add_pdll.mlir`
- 能解释为什么 pass 需要 PDL/PDLInterp dialect
- 能对比 PDLL 与 C++/DRR 的优缺点

## 14. 课程主线到这里完成

第 12 课对应教程的最后一篇文章。到这里，本课程覆盖了：

- 构建环境和 `tutorial-opt`
- IR、pass、TableGen pass
- 自定义 dialect / type / op
- traits、fold、verifier、canonicalizer
- dialect conversion、LLVM lowering
- dataflow analysis 和全局优化
- PDLL pattern

接下来可以自己练习：

1. 给 PDLL 增加一个 pattern，例如 `x * 2 -> x + x`
2. 给 `tests/mul_to_add_pdll.mlir` 增加常量在左、常量在右的回归测试
3. 用 `mlir-pdll -x=mlir` 对照阅读 PDL IR
4. 给原生约束增加错误处理，例如只接受 `i32` 常量
5. 尝试用 PDLL LSP 获得补全和跳转

## 参考

- 本课对应文章：
  <https://www.jeremykun.com/2024/08/04/mlir-pdll/>
- MLIR 官方文档：PDLL
  <https://mlir.llvm.org/docs/PDLL/>
- MLIR 官方文档：PDL dialect
  <https://mlir.llvm.org/docs/Dialects/PDL/>
- MLIR 官方文档：PDLInterp dialect
  <https://mlir.llvm.org/docs/Dialects/PDLInterp/>
- MLIR 官方文档：Pattern Rewriting
  <https://mlir.llvm.org/docs/PatternRewriter/>
- MLIR 官方文档：TableGen DRR
  <https://mlir.llvm.org/docs/DeclarativeRewrites/>
- 本课测试：`tests/mul_to_add_pdll.mlir`
- 相关源码：
  - `lib/Transform/Arith/MulToAdd.pdll`
  - `lib/Transform/Arith/MulToAddPdll.h`
  - `lib/Transform/Arith/MulToAddPdll.cpp`
  - `lib/Transform/Arith/Passes.td`
  - `lib/Transform/Arith/CMakeLists.txt`
  - `lib/Transform/Arith/BUILD`
  - `build/lib/Transform/Arith/MulToAddPdll.h.inc`（构建产物）
- 前序参考：
  - [Pat 与 Pattern 语法](patterns.md)
  - [FileCheck 使用指南](filecheck.md)
  - [MLIR 属性类型速查](attributes.md)