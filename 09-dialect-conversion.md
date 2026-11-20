# 第 9 课：Dialect Conversion

> 对应文章：[Dialect Conversion](https://jeremykun.com/2023/10/23/mlir-dialect-conversion/)。
>
> 前几课我们在 `poly` dialect 内部做优化。本课开始“往下走”：把
> `!poly.poly<N>` 和 `poly.*` 运算 lowering 成标准 dialect 的
> `tensor`、`arith`、`scf`。
>
> 这一过程不是普通 rewrite pattern 能直接完成的，因为**类型也要变**。

## 1. 学习目标

- 理解为什么 dialect conversion 比普通 rewrite 多了“类型转换”问题
- 理解 `TypeConverter`、`ConversionTarget`、`OpConversionPattern`
- 理解 `OpAdaptor` 和 `ConversionPatternRewriter` 的作用
- 理解 `applyPartialConversion` 的流程
- 看懂 `poly.add/sub/mul/eval/from_tensor/to_tensor/constant` 的 lowering
- 理解为什么还要转换 `func.func` / `return` / `call` 的类型
- 能新增一个自己的 conversion pattern

## 2. 为什么不能直接写普通 rewrite

假设只把：

```mlir
%2 = poly.add %0, %1 : !poly.poly<10>
```

改写成：

```mlir
%2 = arith.addi %0, %1 : !poly.poly<10>
```

是不行的，因为 `arith.addi` 不认识 `!poly.poly<10>`。

真正的 lowering 要同时做两件事：

```text
!poly.poly<10>  ->  tensor<10xi32>
poly.add        ->  arith.addi
```

问题在于：

1. `poly.add` 的结果类型变了
2. 所有使用这个结果的下游 op 也必须跟着变
3. 中间状态下，IR 可能同时混用新旧类型
4. MLIR 在 pass 之间会验证 IR，不能长期保留这种混合状态

如果每个 rewrite pattern 都自己检查“我的操作数转换了没有”，代码会非常复杂。

MLIR 的 **dialect conversion framework** 就是来解决这个问题的：

- 单独描述类型转换
- 单独描述每个 op 怎么转换
- 框架负责排序、追踪、插入临时的类型转换
- 最后要求所有非法 op 都被消除

## 3. 核心组件

| 组件 | 作用 |
|---|---|
| `TypeConverter` | 定义旧类型到新类型的映射 |
| `ConversionTarget` | 定义哪些 op/dialect 合法、哪些非法 |
| `OpConversionPattern` | 每个 op 的 lowering 规则 |
| `ConversionPatternRewriter` | conversion 专用的 rewriter |
| `applyPartialConversion` | 驱动整个转换过程 |

本仓库的 lowering 是：

```text
!poly.poly<N>  ->  tensor<Nxi32>

poly.add       ->  arith.addi
poly.sub       ->  arith.subi
poly.mul       ->  scf.for + tensor.extract/tensor.insert + arith
poly.eval      ->  scf.for + Horner
poly.from_tensor -> tensor.pad（必要时）
poly.to_tensor ->  直接使用输入 tensor
poly.constant  ->  arith.constant + poly.from_tensor
```

## 4. `TypeConverter`

`lib/Conversion/PolyToStandard/PolyToStandard.cpp:19`：

```cpp
class PolyToStandardTypeConverter : public TypeConverter {
 public:
  PolyToStandardTypeConverter(MLIRContext *ctx) {
    // 1. 默认：类型不变
    addConversion([](Type type) { return type; });

    // 2. 专门处理 PolynomialType
    addConversion([ctx](PolynomialType type) -> Type {
      int degreeBound = type.getDegreeBound();
      IntegerType elementTy = IntegerType::get(
          ctx, 32, IntegerType::SignednessSemantics::Signless);
      return RankedTensorType::get({degreeBound}, elementTy);
    });
  }
};
```

效果：

```text
!poly.poly<10>  ->  tensor<10xi32>
```

### 4.1 顺序很重要

TypeConverter 会**优先尝试最近添加的 conversion**。

这里先加“原样返回”，再加 `PolynomialType` 的转换，所以：

```text
PolynomialType -> tensor<Nxi32>   // 先尝试
其他类型       -> 保持不变         // 兜底
```

如果把顺序反过来，`PolynomialType` 可能先被“原样返回”匹配，永远不会转换成
tensor。

### 4.2 没有 materialization hooks

代码注释解释了：

> 这个 lowering 在一个 pass 内完成，所以不需要自定义 materialization。
> 如果跨多个 pass，中间的类型冲突可能需要 materializer。

要理解这句话，需要先理解转换过程中“新旧类型暂时对不上”是什么意思。

#### 类型冲突从哪里来

假设只转换 `poly.add`：

```mlir
func.func @f(%a: !poly.poly<10>, %b: !poly.poly<10>) {
  %c = poly.add %a, %b : !poly.poly<10>
  return
}
```

`ConvertAdd` 创建的是：

```mlir
%c_new = arith.addi %a_new, %b_new : tensor<10xi32>
```

但函数参数仍然是旧类型：

```mlir
%a : !poly.poly<10>
%b : !poly.poly<10>
```

`arith.addi` 需要 `tensor<10xi32>`，而 `%a` 还是 `!poly.poly<10>`，类型不一致。

#### `unrealized_conversion_cast` 是临时桥

为了继续转换，框架会插入：

```mlir
%a_new = builtin.unrealized_conversion_cast %a
       : !poly.poly<10> to tensor<10xi32>

%b_new = builtin.unrealized_conversion_cast %b
       : !poly.poly<10> to tensor<10xi32>

%c_new = arith.addi %a_new, %b_new : tensor<10xi32>
```

这个 cast 表示：

> “这里暂时需要一个新类型的值，但真正的转换方式还没决定。”

它没有运行时语义，不能拿来执行程序；它只是 conversion 过程中的临时占位符。

#### 如果 pass 结束时它还留着，就失败

如果函数签名没有被转换，pass 结束时 IR 里仍然有：

```mlir
func.func @f(%a: !poly.poly<10>, %b: !poly.poly<10>) {
  %a_new = builtin.unrealized_conversion_cast %a
         : !poly.poly<10> to tensor<10xi32>
  ...
}
```

`!poly.poly<10>` 还在函数签名里，临时 cast 也无法消除。

于是 conversion 框架报告：

```text
error: failed to legalize unresolved materialization
       from '!poly.poly<10>' to 'tensor<10xi32>'
       that remained live after conversion
```

意思是：

> “我本来想用临时 cast 撑过去，但到最后都没办法把它变成真正的 IR。”

这就是 pass 失败。

#### 本 pass 为什么不需要 materializer

本 pass 不仅转换了 `poly.add`，还转换了：

- `func.func` 的参数/返回值
- `func.return`
- `func.call`
- 分支的参数
- 所有 `poly.*` op

所以转换完成后：

```text
函数参数：tensor<10xi32>
arith.addi 操作数：tensor<10xi32>
return 类型：tensor<10xi32>
```

没有新旧类型混用，临时 cast 自然消失。

因此代码注释说：

> 这个 lowering 在一个 pass 内完成，所以不需要自定义 materialization。

#### 什么时候需要 materializer

如果 lowering 被拆成多个 pass，或者某些类型边界必须保留，就可能需要。

例如：

- 第一个 pass 只转换一部分 poly op
- 第二个 pass 再转换剩下的
- 中间 pass 之间 IR 必须合法
- 或者一个外部函数边界暂时还要保持 `!poly.poly<N>`

这时可以注册 materializer，告诉框架：

> “如果你需要把 `tensor<Nxi32>` 变成 `!poly.poly<N>`，就用 `poly.from_tensor`。”

概念上类似：

```cpp
typeConverter.addTargetMaterialization(
    [](OpBuilder &builder, Type resultType,
       ValueRange inputs, Location loc) -> Value {
      // 如果 resultType 是 !poly.poly<N>
      // 就创建 poly.from_tensor inputs[0]
    });
```

这样框架就不会留下 `unrealized_conversion_cast`，而是插入真正的
`poly.from_tensor` 或 `poly.to_tensor`。

#### 两种 materialization

| 类型 | 方向 | 例子 |
|---|---|---|
| target materialization | 旧类型 → 新类型 | `!poly.poly<N>` → `tensor<Nxi32>` |
| source materialization | 新类型 → 旧类型 | `tensor<Nxi32>` → `!poly.poly<N>` |

它们都是“用真实 op 把值转换成目标类型”的钩子。

本 pass 不需要它们，因为所有边界都在同一个 pass 内被转换干净。

#### 一句话理解

```text
unrealized_conversion_cast  = 转换过程中的临时桥
materializer                = 把临时桥换成真正的 op
最终还有临时桥              = pass 失败
```

## 5. `ConversionTarget`

> 函数签名、`return`、`call`、分支的辅助转换函数详见
> [Dialect Conversion 的类型边界辅助函数](conversion-helpers.md)。

`PolyToStandard.cpp:260`：

```cpp
ConversionTarget target(*context);
target.addLegalDialect<arith::ArithDialect>();
target.addIllegalDialect<PolyDialect>();
```

含义：

- `arith` 最终是合法的
- `poly` 最终必须消失

只写这两行还不够。函数签名里也有 `!poly.poly`：

```mlir
func.func @f(%0: !poly.poly<10>) -> !poly.poly<10>
```

如果只转换 `poly.add`，函数参数和返回值仍然是旧类型，`arith.addi` 的操作数是
tensor，二者对不上，会留下 unresolved materialization，pass 失败。

所以要注册函数/调用/返回相关的转换辅助：

```cpp
populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
    patterns, typeConverter);

target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
  return typeConverter.isSignatureLegal(op.getFunctionType()) &&
         typeConverter.isLegal(&op.getBody());
});

populateReturnOpTypeConversionPattern(patterns, typeConverter);
target.addDynamicallyLegalOp<func::ReturnOp>(
    [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

populateCallOpTypeConversionPattern(patterns, typeConverter);
target.addDynamicallyLegalOp<func::CallOp>(
    [&](func::CallOp op) { return typeConverter.isLegal(op); });

populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
```

这些 helper 会生成 pattern，把函数签名、return、call、分支的参数/结果类型也一起
转换。

### 5.1 为什么要 dependent dialects

`PolyToStandard.td`：

```tablegen
let dependentDialects = [
  "mlir::arith::ArithDialect",
  "mlir::tutorial::poly::PolyDialect",
  "mlir::tensor::TensorDialect",
  "mlir::scf::SCFDialect",
];
```

lowering 会创建 `arith`、`tensor`、`scf` 的 op；这些 dialect 必须提前加载，
否则框架拿到的是未注册的 op，无法正确验证/创建。

## 6. `OpConversionPattern`

### 6.1 参数的含义

```cpp
LogicalResult matchAndRewrite(
    AddOp op, AddOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const override
```

- `op`：**原始 op**，操作数/结果还是旧类型
- `adaptor`：**已经类型转换后的操作数**
  - `adaptor.getLhs()` 返回 `tensor<10xi32>`
  - `op.getLhs()` 仍然是 `!poly.poly<10>`
- `rewriter`：conversion 专用 rewriter

转换后的 op 应该使用 `adaptor` 的值，而不是 `op` 的原始值。

### 6.2 add / sub

```cpp
struct ConvertAdd : public OpConversionPattern<AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    arith::AddIOp addOp = rewriter.create<arith::AddIOp>(
        op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op.getOperation(), addOp);
    return success();
  }
};
```

因为 `arith.addi` 支持 tensor 的逐元素加法，所以这里不需要手写循环。

`sub` 同理，只是换成 `arith.subi`。

### 6.3 mul：双重循环

`poly.mul` 是多项式乘法，要计算：

```text
result[(i + j) % N] += lhs[i] * rhs[j]
```

`ConvertMul` 的步骤：

1. 取转换后的 tensor 类型，得到 `N`
2. 创建全零 tensor 作为累加器
3. 创建外层 `scf.for i = 0..N-1`
4. 内层 `scf.for j = 0..N-1`
5. 计算目标下标 `(i + j) % N`
6. `tensor.extract` 取两个系数
7. `arith.muli` 相乘
8. `tensor.extract` 取累加值，`arith.addi` 加上
9. `tensor.insert` 写回
10. `scf.yield` 传递新的 tensor

关键代码：

```cpp
auto destIndex = b.create<arith::RemUIOp>(
    b.create<arith::AddIOp>(p0Index, p1Index), numTermsOp);

auto mulOp = b.create<arith::MulIOp>(
    b.create<tensor::ExtractOp>(p0, ValueRange(p0Index)),
    b.create<tensor::ExtractOp>(p1, ValueRange(p1Index)));

auto result = b.create<arith::AddIOp>(
    mulOp,
    b.create<tensor::ExtractOp>(accumTensor, destIndex.getResult()));

auto stored = b.create<tensor::InsertOp>(
    result, accumTensor, destIndex.getResult());
```

这里的 `% N` 对应 `x^N = 1` 的环语义。

### 6.4 eval：Horner 法

`poly.eval` 使用 Horner：

```text
result = 0
for i = 1..N:
  result = point * result + coeff[N - i]
```

`ConvertEval` 创建 `scf.for`，循环变量从 1 到 N：

```cpp
auto coeffIndex = b.create<arith::SubIOp>(numTermsOp, loopIndex);
auto mulOp = b.create<arith::MulIOp>(point, accum);
auto result = b.create<arith::AddIOp>(
    mulOp,
    b.create<tensor::ExtractOp>(poly, coeffIndex.getResult()));
```

当前实现只处理 `i32` 的 point/output，不支持 `complex`。

### 6.5 from_tensor / to_tensor

`poly.to_tensor` 在类型转换后就是“把 polynomial tensor 原样交出去”：

```cpp
rewriter.replaceOp(op, adaptor.getInput());
```

`poly.from_tensor` 需要处理系数长度不足的情况：

```text
输入 tensor<3xi32>，结果 poly<10>
-> tensor.pad 到 tensor<10xi32>
```

`ConvertFromTensor` 里：

```cpp
if (inputShape < resultShape) {
  // 右侧补零
  coeffValue = b.create<tensor::PadOp>(
      resultTensorTy, coeffValue, low, high,
      b.create<arith::ConstantOp>(
          rewriter.getIntegerAttr(resultEltTy, 0)),
      /*nofold=*/false);
}
rewriter.replaceOp(op, coeffValue);
```

`low` 是 0，`high` 是 `resultShape - inputShape`，所以是在高位补零。

### 6.6 constant：先变成 arith.constant，再递归转换

`ConvertConstant`：

```cpp
auto constOp = b.create<arith::ConstantOp>(adaptor.getCoefficients());
auto fromTensorOp =
    b.create<FromTensorOp>(op.getResult().getType(), constOp);
rewriter.replaceOp(op, fromTensorOp.getResult());
```

它先创建：

```mlir
%cst = arith.constant dense<[2, 3, 4]> : tensor<3xi32>
```

然后创建一个 **新的 `poly.from_tensor`**：

```mlir
%p = poly.from_tensor %cst : tensor<3xi32> -> !poly.poly<10>
```

这个 `poly.from_tensor` 也是非法的，但 conversion 框架会继续用 `ConvertFromTensor`
处理它，最终得到 `tensor.pad`。

这是 dialect conversion 的一个特点：pattern 可以创建“稍后还会被转换”的 op，
但必须确保框架最终能把它合法化。

## 7. 驱动器

最后：

```cpp
if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
  signalPassFailure();
}
```

`applyPartialConversion` 负责：

- 遍历 IR
- 对非法 op 尝试匹配 conversion patterns
- 自动转换 operand 类型
- 处理临时的 `unrealized_conversion_cast`
- 最后检查是否还有非法 op

失败时 `signalPassFailure()` 让 pass 以非零状态结束。

> 这里用 `applyPartialConversion` 而不是 `applyFullConversion`，一个重要原因是
> 前者失败时的诊断信息更详细。

## 8. 实验

运行：

```bash
cd /home/zhouzhou/npu/mlir-tutorial
export TOP=$PWD
export BIN=$TOP/externals/llvm-project/build/bin
export TO=$TOP/build/tools/tutorial-opt

$TO tests/poly_to_standard.mlir --poly-to-standard
```

### 8.1 add / sub

输入：

```mlir
%2 = poly.add %0, %1 : !poly.poly<10>
```

输出：

```mlir
%0 = arith.addi %arg0, %arg1 : tensor<10xi32>
```

函数参数也变成了 `tensor<10xi32>`。

### 8.2 to_tensor / from_tensor

`to_tensor` 被消除，直接返回输入：

```mlir
func.func @test_lower_to_tensor(%arg0: tensor<10xi32>) -> tensor<10xi32> {
  return %arg0 : tensor<10xi32>
}
```

`from_tensor` 在形状相同时同样直接返回；在 `poly<20>` 但输入是
`tensor<10xi32>` 时，会生成 `tensor.pad`。

### 8.3 mul

`poly.mul` 被展开成嵌套 `scf.for`：

```mlir
%0 = scf.for ... iter_args(%arg3 = %cst) -> (tensor<10xi32>) {
  %1 = scf.for ... iter_args(%arg5 = %arg3) -> (tensor<10xi32>) {
    %2 = arith.addi %arg2, %arg4 : index
    %3 = arith.remui %2, %c10 : index
    %extracted = tensor.extract %arg1[%arg4]
    %extracted_0 = tensor.extract %arg0[%arg2]
    %4 = arith.muli %extracted_0, %extracted
    %extracted_1 = tensor.extract %arg5[%3]
    %5 = arith.addi %4, %extracted_1
    %inserted = tensor.insert %5 into %arg5[%3]
    scf.yield %inserted
  }
  scf.yield %1
}
```

### 8.4 eval

`poly.eval` 变成一个 Horner 循环：

```mlir
%0 = scf.for %arg2 = %c1 to %c11 step %c1
     iter_args(%arg3 = %c0_i32) -> (i32) {
  %1 = arith.subi %c10, %arg2 : index
  %2 = arith.muli %arg1, %arg3 : i32
  %extracted = tensor.extract %arg0[%1]
  %3 = arith.addi %2, %extracted
  scf.yield %3
}
```

### 8.5 constant + fold

测试 `@test_lower_add_and_fold` 里：

```mlir
%0 = poly.constant dense<[2, 3, 4]> : !poly.poly<10>
%1 = poly.constant dense<[3, 4, 5]> : !poly.poly<10>
%2 = poly.add %0, %1 : !poly.poly<10>
```

会先 lower 成 arith constant，再在 conversion 过程中尝试折叠。输出中能看到：

```text
arith.constant dense<[2, 3, 4]>
arith.constant dense<[3, 4, 5]>
arith.constant dense<[5, 7, 9]>
```

以及为每个短系数张量生成的 `tensor.pad`。

### 8.6 验证测试

```bash
$TO tests/poly_to_standard.mlir --poly-to-standard \
  | $BIN/FileCheck tests/poly_to_standard.mlir
```

无输出表示通过。

## 9. 源码位置

| 内容 | 文件 |
|---|---|
| `PolyToStandardTypeConverter` | `lib/Conversion/PolyToStandard/PolyToStandard.cpp:19` |
| `ConversionTarget` 与 helper 注册 | `lib/Conversion/PolyToStandard/PolyToStandard.cpp:260` |
| 各 op 的 `Convert*` pattern | `lib/Conversion/PolyToStandard/PolyToStandard.cpp` |
| pass 声明与 `dependentDialects` | `lib/Conversion/PolyToStandard/PolyToStandard.td` |
| 本课测试 | `tests/poly_to_standard.mlir` |

## 10. 当前实现的限制

### 10.1 没有 `poly.neg` 的 conversion

`PolyToStandard.cpp` 注册的 pattern 里没有 `ConvertNeg`：

```cpp
patterns.add<ConvertAdd, ConvertConstant, ConvertSub, ConvertEval,
             ConvertMul, ConvertFromTensor, ConvertToTensor>(...);
```

所以：

```bash
$TO build/scratch/poly_neg_to_standard.mlir --poly-to-standard
```

会失败：

```text
error: failed to legalize operation 'poly.neg' that was explicitly marked illegal
```

这是本课的一个自然练习：补一个 `ConvertNeg`。

### 10.2 eval 只支持 i32

`ConvertEval` 使用 `arith` 的整数循环，没有处理 `complex`。第 7 课的 verifier
允许 complex，但 lowering 还没有实现。

### 10.3 from_tensor 只处理“输入比结果短”

`ConvertFromTensor` 只在 `inputShape < resultShape` 时补零；如果输入更长，
当前代码没有截断或报错逻辑。

## 11. 常见坑

### 11.1 忘记 dependent dialects

如果 pass 创建 `arith`/`tensor`/`scf` op 但没有声明：

```tablegen
let dependentDialects = [...];
```

运行时会遇到 dialect 未加载/op 未注册的问题。

### 11.2 忘记转换函数签名

只转换 `poly.add` 不够；函数参数、返回值、`return`、`call` 都要一起转换。
否则会留下：

```mlir
builtin.unrealized_conversion_cast
```

最终 pass 失败。

### 11.3 用 `op.getLhs()` 而不是 `adaptor.getLhs()`

`op` 的操作数还是 `!poly.poly`，`adaptor` 的才是 `tensor`。创建 `arith.addi`
时必须用 adaptor。

### 11.4 把 `ConversionPatternRewriter` 写成 `PatternRewriter`

dialect conversion 需要额外的类型/转换跟踪能力，必须使用
`ConversionPatternRewriter`。

### 11.5 以为 `applyPartialConversion` 会容忍非法 op

它只是 partial conversion；如果 pass 结束后还有非法 op，仍然失败。

### 11.6 TypeConverter 的注册顺序

最近添加的 conversion 先尝试。专用转换要放在兜底转换之后添加。

## 12. 练习

### 练习 1：补一个 `ConvertNeg`

要求把：

```mlir
%n = poly.neg %x : !poly.poly<10>
```

lower 成：

```mlir
%zero = arith.constant dense<0> : tensor<10xi32>
%n = arith.subi %zero, %x : tensor<10xi32>
```

提示：使用 `adaptor.getInput()` 和 `DenseElementsAttr::get`。

<details>
<summary>参考答案</summary>

```cpp
struct ConvertNeg : public OpConversionPattern<NegOp> {
  ConvertNeg(MLIRContext *context)
      : OpConversionPattern<NegOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      NegOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto tensorTy = cast<RankedTensorType>(input.getType());

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto zero = b.create<arith::ConstantOp>(
        tensorTy, DenseElementsAttr::get(tensorTy, 0));
    auto sub = b.create<arith::SubIOp>(zero, input);

    rewriter.replaceOp(op, sub.getResult());
    return success();
  }
};
```

然后在 pattern 列表中加入 `ConvertNeg`：

```cpp
patterns.add<ConvertAdd, ConvertConstant, ConvertSub, ConvertEval,
             ConvertMul, ConvertFromTensor, ConvertToTensor,
             ConvertNeg>(typeConverter, context);
```

测试：

```mlir
func.func @neg(%x: !poly.poly<10>) -> !poly.poly<10> {
  %n = poly.neg %x : !poly.poly<10>
  return %n : !poly.poly<10>
}
```

输出应包含 `arith.subi`，且不再有 `poly.neg`。

</details>

### 练习 2：为什么需要函数转换 helper？

如果只注册 `ConvertAdd`，运行：

```mlir
func.func @f(%a: !poly.poly<10>, %b: !poly.poly<10>)
    -> !poly.poly<10> {
  %c = poly.add %a, %b : !poly.poly<10>
  return %c : !poly.poly<10>
}
```

会发生什么？为什么？

<details>
<summary>参考答案</summary>

`ConvertAdd` 会创建 `arith.addi`，它的操作数是转换后的
`tensor<10xi32>`；但 `func.func` 的参数和返回值仍然是
`!poly.poly<10>`。这会在函数参数和 arith 操作数之间产生类型冲突，留下
`builtin.unrealized_conversion_cast`，最终无法合法化。

所以要使用 `populateFunctionOpInterfaceTypeConversionPattern`、
`populateReturnOpTypeConversionPattern` 等 helper，把函数签名和 return/call
一起转换。

</details>

### 练习 3：解释 `tensor.pad`

`poly.from_tensor` 把：

```text
tensor<3xi32> -> !poly.poly<10>
```

lower 成什么？为什么需要 pad？

<details>
<summary>参考答案</summary>

`!poly.poly<10>` 对应 `tensor<10xi32>`，而输入只有 3 个系数。需要在高位补
7 个 0：

```mlir
%padded = tensor.pad %input low[0] high[7]
  { tensor.yield %zero }
  : tensor<3xi32> to tensor<10xi32>
```

低位对应低次项，不能补在低位，否则系数含义会改变；所以 `low = 0`,
`high = 10 - 3 = 7`。

</details>

## 13. 验收清单

- [ ] 能解释为什么 lowering 需要同时转换类型和 op
- [ ] 能说出 `TypeConverter`、`ConversionTarget`、`OpConversionPattern` 的分工
- [ ] 能解释 `op` 和 `adaptor` 中操作数类型的不同
- [ ] 能解释为什么要转换 `func`/`return`/`call`
- [ ] 能看懂 `poly.add/sub` 的 lowering
- [ ] 能解释 `poly.mul` 的双重循环和取模
- [ ] 能解释 `poly.eval` 的 Horner 循环
- [ ] 能解释 `poly.constant` 为什么先生成 `poly.from_tensor`
- [ ] 能运行 `tests/poly_to_standard.mlir`
- [ ] 能补一个 `ConvertNeg`

## 14. 下一课

**第 10 课：Lowering through LLVM**：[10-lowering-through-llvm.md](10-lowering-through-llvm.md)

- `--poly-to-llvm` pipeline
- `mlir-translate` + `mlir-runner` 端到端执行
- linalg、bufferization、memref、LLVM lowering 顺序
- 测试：`tests/poly_to_llvm.mlir`、`poly_to_llvm_eval.mlir`

## 参考

- 本课对应文章：
  <https://jeremykun.com/2023/10/23/mlir-dialect-conversion/>
- MLIR 官方文档：Dialect Conversion
  <https://mlir.llvm.org/docs/DialectConversion/>
- 补充：[Dialect Conversion 的类型边界辅助函数](conversion-helpers.md)
- 本课测试：`tests/poly_to_standard.mlir`
- 相关源码：
  - `lib/Conversion/PolyToStandard/PolyToStandard.cpp`
  - `lib/Conversion/PolyToStandard/PolyToStandard.td`
  - `lib/Conversion/PolyToStandard/PolyToStandard.h`
  - `tools/tutorial-opt.cpp`
- FileCheck 语法：[FileCheck 使用指南](filecheck.md)