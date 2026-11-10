# 第 6 课：Folders and Constant Propagation

> 对应文章：[Folders and Constant Propagation](https://jeremykun.com/2023/09/11/mlir-folders/)。
>
> 上一课我们给 `poly` 加上了 `Pure`、`ElementwiseMappable` 等 trait，LICM、
> control-flow-sink、CSE 已经能作用在 poly 程序上。但 SCCP 仍然不能把
> `poly.mul` / `poly.add` 算成常量——因为它不知道这些 op 的**常量语义**。
>
> 本课补上折叠（folding）所需的三块基础设施：常量 op、每个 op 的 folder、
> dialect 的常量物化钩子。学完后，`--canonicalize` 和 `-sccp` 都能把
> 多项式运算在编译期算出来。

## 1. 学习目标

- 理解 fold 是什么，以及它不是 pass 的原因
- 理解 `OpFoldResult` 的三种返回值
- 理解 `FoldAdaptor` 如何把“常量操作数”变成 `Attribute`
- 理解 `ConstantLike`、folder、`materializeConstant` 三者的分工
- 能读懂 `poly.constant/add/sub/mul/from_tensor` 的 fold 实现
- 能区分 `--canonicalize`、`-sccp`、`-cse` 的职责
- 能自己给一个 op 增加 folder 并写测试

## 2. 先看现象：同一段 IR，不同 pass 的结果

### 2.1 输入

取本课测试文件 `tests/poly_fold.mlir` 里的第一个函数：

```mlir
func.func @add_constants() -> !poly.poly<10> {
  %a = poly.constant dense<[1, 2, 3]> : tensor<3xi32> : !poly.poly<10>
  %b = poly.constant dense<[10, 20, 30]> : tensor<3xi32> : !poly.poly<10>
  %sum = poly.add %a, %b : !poly.poly<10>
  return %sum : !poly.poly<10>
}
```

语义上就是：

```text
(1 + 2x + 3x²) + (10 + 20x + 30x²)
= 11 + 22x + 33x²
```

三个操作数都是编译期已知的，理论上不需要在运行时做加法。

### 2.2 `--canonicalize`：局部折叠并清理死代码

```bash
cd /home/zhouzhou/npu/mlir-tutorial
export TOP=$PWD
export BIN=$TOP/externals/llvm-project/build/bin
export TO=$TOP/build/tools/tutorial-opt

$TO tests/poly_fold.mlir --canonicalize
```

只观察 `@add_constants` 的输出：

```mlir
func.func @add_constants() -> !poly.poly<10> {
  %0 = poly.constant dense<[11, 22, 33]> : tensor<3xi32> : !poly.poly<10>
  return %0 : !poly.poly<10>
}
```

结果：

- `poly.add` 被替换成一条新的 `poly.constant`
- 原来的两条 `poly.constant` 失去使用者，被删掉
- 只剩一条常量 + `return`

### 2.3 `-sccp`：传播常量，但不负责清理全部死代码

SCCP 是一个 Function 级 pass，测试里用显式 pipeline 调用：

```bash
$TO -pass-pipeline="builtin.module(func.func(sccp))" tests/poly_fold.mlir
```

同样只看 `@add_constants`：

```mlir
func.func @add_constants() -> !poly.poly<10> {
  %0 = poly.constant dense<[11, 22, 33]> : tensor<3xi32> : !poly.poly<10>
  %1 = poly.constant dense<[10, 20, 30]> : tensor<3xi32> : !poly.poly<10>
  %2 = poly.constant dense<[1, 2, 3]> : tensor<3xi32> : !poly.poly<10>
  return %0 : !poly.poly<10>
}
```

`poly.add` 同样被算成了 `[11, 22, 33]`，但两条原始常量还留着。它们已经没人使用，
是死代码。原因不是 SCCP “算错了”，而是：

- SCCP 的职责是**替换常量值**
- 它不做通用的 dead-code elimination（DCE）
- 清理死代码交给 `--canonicalize`、`--symbol-dce` 等 pass

这也是文章里特别提醒的两点之一：**SCCP 不会删除死代码**。

### 2.4 `-cse`：只做公共子表达式消除

CSE（common subexpression elimination）不是常量折叠。它只把**结构相同**的 op
合并成一个。以 `tests/cse.mlir` 为例：

```bash
$TO tests/cse.mlir -cse
```

输出：

```mlir
func.func @test_simple_cse() -> !poly.poly<10> {
  %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %0 = poly.from_tensor %cst : tensor<3xi32> -> !poly.poly<10>
  %1 = poly.mul %0, %0 : !poly.poly<10>
  %2 = poly.add %1, %1 : !poly.poly<10>
  return %2 : !poly.poly<10>
}
```

输入里有两个完全相同的 `poly.mul %p0, %p0`，CSE 把它们合并成一个 `%1`，但它
**没有把** `poly.mul` 算成常量，也没有把 `poly.from_tensor` 变成 `poly.constant`。
CSE 只看“这两个 op 是否等价”，不执行 folder。

如果对同一份输入运行 `--canonicalize`：

```bash
$TO tests/cse.mlir --canonicalize
```

会直接得到：

```mlir
func.func @test_simple_cse() -> !poly.poly<10> {
  %0 = poly.constant dense<[2, 8, 20, 24, 18]> : tensor<5xi32> : !poly.poly<10>
  return %0 : !poly.poly<10>
}
```

因为 `(1 + 2x + 3x²)² = 1 + 4x + 10x² + 12x³ + 9x⁴`，而这个结果在
`x¹⁰ = 1` 的环里没有发生回绕。

### 2.5 三者对比

| 工具 | 做什么 | 是否求值常量 | 是否通过控制流传播 | 是否负责 DCE |
|---|---|---|---|---|
| `fold` | 单个 op 的局部常量计算/等价替换 | 是 | 否 | 否 |
| `--canonicalize` | 反复折叠 + 规范化 pattern | 是，局部 | 基本否 | 是 |
| `-sccp` | 数据流分析，把常量传播到使用点 | 是，借助 fold | 是 | 不完整 |
| `-cse` | 删除重复的等价 op | 否 | 否 | 可能顺带清理 |

> “是否通过控制流传播”是 SCCP 与 canonicalize 的核心区别。本课的 poly 例子都是
> 直线代码；SCCP 的名字里有 *conditional*，因为它能在 `if` / 循环的控制流图上
> 推导哪些值仍然是常量。后面的课程遇到控制流时可以再观察这一点。

## 3. Fold 是什么

### 3.1 定义

fold 是 op 自己提供的一个**局部查询**：

> 如果这个 op 的所有输入都已经是编译期常量，我能不能直接算出结果？
> 或者，这个 op 是不是等价于某个已经存在的 SSA 值？

它**不是 pass**，也不会主动遍历 IR。它只是被 canonicalizer、SCCP、
`OperationFolder` 等基础设施调用。

在 TableGen 里写：

```tablegen
let hasFolder = 1;
```

ODS 就会为这个 op 生成一个 `fold` 声明，并注册到 `OperationName` 的 fold hook 上。

### 3.2 `OpFoldResult` 的三种返回值

`fold` 的返回类型是 `OpFoldResult`。它本质上是：

```cpp
class OpFoldResult : public PointerUnion<Attribute, Value> { ... };
```

也就是“一个 `Attribute` 或一个 `Value`”。因此有三种情况：

| 返回值 | 含义 | 后续动作 |
|---|---|---|
| `Attribute` | 这个 op 的结果是一个编译期常量 | 需要 dialect 把它“物化”成一个常量 op |
| `Value` | 这个 op 等价于某个已有 SSA 值 | 直接把结果替换成这个值 |
| 空（`{}` / `nullptr`） | 这次无法折叠 | 保持原样，交给其他 pattern |

前两种都有真实例子。

**返回 Attribute：多项式常量**

```cpp
// lib/Dialect/Poly/PolyOps.cpp:18
OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor) {
  return constFoldBinaryOp<IntegerAttr, APInt, void>(
      adaptor.getOperands(), [&](APInt a, APInt b) { return a + b; });
}
```

`poly.add` 算出的是系数的张量属性，例如 `dense<[11, 22, 33]>`。这个属性还不是
IR 中的值，需要下一步物化。

**返回 Value：box 的拆包**

我们之前的迷你 dialect 里有一个更直观的例子：

```cpp
// lib/Dialect/Box/BoxOps.cpp:30
OpFoldResult UnpackOp::fold(UnpackOp::FoldAdaptor) {
  if (auto pack = getBox().getDefiningOp<PackOp>())
    return pack.getInput();
  return {};
}
```

`box.unpack(box.pack(x))` 不需要产生新常量，它直接等价于 `x`。所以返回
`pack.getInput()`，类型是 `Value`。

运行：

```bash
$TO tests/box_fold.mlir --canonicalize
```

输出：

```mlir
module {
  func.func @roundtrip(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
  func.func @box_of_poly(%arg0: !poly.poly<5>) -> !poly.poly<5> {
    return %arg0 : !poly.poly<5>
  }
}
```

`box.pack` / `box.unpack` 都消失了，直接返回参数。

**返回空：折叠失败**

```cpp
OpFoldResult FromTensorOp::fold(FromTensorOp::FoldAdaptor adaptor) {
  // 如果输入不是 DenseIntElementsAttr，dyn_cast_or_null 返回空指针，
  // 也就是“这次不折叠”。
  return dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getInput());
}
```

注意：空返回值不是错误，只是“我处理不了，别替换我”。

### 3.3 fold 的边界

fold 的约束比普通 rewrite pattern 严格得多：

- **不能创建新 op**
- **不能删除其他 op**
- 只能使用已有的 SSA 值，或者返回一个 `Attribute`
- 只应关心当前 op 本身，不应做跨 block 的分析

因此，像 `(x+y)(x-y) -> x²-y²` 这种需要新建多个 op 的代数变换，不适合放在
`fold` 里，而应该放在 canonicalization pattern 里（下一课会看到
`DifferenceOfSquares`）。

## 4. Fold 的调用链

理解这条链，比记住 API 更重要。

### 4.1 收集常量操作数

`Operation::fold` 有一个重载，会先检查每个操作数是不是常量：

```cpp
// externals/llvm-project/mlir/lib/IR/Operation.cpp（简化）
LogicalResult Operation::fold(SmallVectorImpl<OpFoldResult> &results) {
  SmallVector<Attribute> constants;
  constants.assign(getNumOperands(), Attribute());

  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    matchPattern(getOperand(i), m_Constant(&constants[i]));

  return fold(constants, results);
}
```

关键点：

- 结果数组的长度等于操作数个数
- 是常量的操作数被填入对应的 `Attribute`
- 不是常量的位置保持空 `Attribute`
- `m_Constant` 只会匹配带 `ConstantLike` trait 的 op

所以 folder 拿到的不是原始 `Value`，而是“已知常量值或空”的数组。

`m_Constant` 本身也很直接：

```cpp
// externals/llvm-project/mlir/include/mlir/IR/Matchers.h（简化）
bool match(Operation *op) {
  if (!op->hasTrait<OpTrait::ConstantLike>())
    return false;

  SmallVector<OpFoldResult, 1> foldedOp;
  LogicalResult result = op->fold(/*operands=*/{}, foldedOp);
  assert(succeeded(result) && "expected ConstantLike op to be foldable");

  if (auto attr = dyn_cast<AttrT>(cast<Attribute>(foldedOp.front())))
    ...
}
```

这解释了两件事：

1. `ConstantLike` op 自己必须能 fold 成一个 `Attribute`，否则这里的断言会失败。
2. `poly.constant` 必须实现 `fold` 返回系数属性，否则其他 op 看不到它是常量。

### 4.2 `FoldAdaptor`：把 Value 视角换成 Attribute 视角

ODS 为每个带 folder 的 op 生成：

```cpp
using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
::mlir::OpFoldResult fold(FoldAdaptor adaptor);
```

以 `poly.add` 为例，普通 op 类的 `getLhs()` 返回 `Value`：

```cpp
::mlir::TypedValue<::mlir::Type> getLhs();
```

但在 `FoldAdaptor` 里，`getLhs()` 返回的是 `Attribute`：

```cpp
// 概念上等价于
Attribute getLhs();
```

因为 adaptor 的底层容器是 `ArrayRef<Attribute>`，不是 `ValueRange`。
如果某个操作数不是常量，对应位置就是空 `Attribute`。

这就是为什么 `AddOp::fold` 可以直接写：

```cpp
adaptor.getOperands()   // ArrayRef<Attribute>
```

而不需要自己去找定义操作数的 op。

> 小细节：`poly.constant` 的 `coefficients` 是**属性**而不是 SSA 操作数，
> 所以 `adaptor.getCoefficients()` 直接读 op 的属性，与操作数数组无关。

### 4.3 折叠结果如何变成 IR

如果 folder 返回 `Attribute`，调用者需要把它变成一个真正的 op。以 greedy
rewrite driver（`--canonicalize` 使用）为例：

```cpp
// externals/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp（简化）
if (auto value = dyn_cast<Value>(ofr)) {
  replacements.push_back(value);
  continue;
}

Operation *constOp = op->getDialect()->materializeConstant(
    rewriter, cast<Attribute>(ofr), resultType, op->getLoc());

if (!constOp) {
  // 物化失败，放弃这次折叠
  ...
}

assert(constOp->hasTrait<OpTrait::ConstantLike>());
assert(constOp->getResultTypes()[0] == resultType);
replacements.push_back(constOp->getResult(0));
```

三个断言/检查值得记住：

- 物化出来的 op 必须是 `ConstantLike`
- 结果类型必须与 folder 声明的结果类型一致
- 物化失败时，这次折叠不会被应用

SCCP 使用 `OperationFolder`，逻辑类似，但还会对常量做去重和插入位置管理。

### 4.4 完整链路

以 `poly.add` 为例：

```text
poly.constant [1,2,3] ──fold──> DenseIntElementsAttr [1,2,3]
poly.constant [10,20,30] ──fold──> DenseIntElementsAttr [10,20,30]
                              │
                              ▼
                  Operation::fold 收集两个 Attribute
                              │
                              ▼
                  AddOp::fold(adaptor) 计算 [11,22,33]
                              │
                              ▼
                  PolyDialect::materializeConstant
                              │
                              ▼
                  poly.constant [11,22,33]
                              │
                              ▼
                  替换旧 add；旧常量变成死代码
```

## 5. 让 poly 支持常量折叠：三件事

文章把所需工作概括为三条：

1. 增加一个常量 op
2. 增加 materialization hook
3. 给每个 op 增加 folder

我们逐一看本仓库的实现。

### 5.1 常量 op：`poly.constant`

定义在 `lib/Dialect/Poly/PolyOps.td:66`：

```tablegen
def Poly_ConstantOp : Op<Poly_Dialect, "constant", [Pure, ConstantLike]> {
  let summary = "Define a constant polynomial via an attribute.";
  let arguments = (ins AnyIntElementsAttr:$coefficients);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$coefficients attr-dict `:` qualified(type($output))";
  let hasFolder = 1;
}
```

三个关键点：

- `ConstantLike`：告诉通用基础设施“我是常量生产者”
- `AnyIntElementsAttr`：常量是**属性**，不是 SSA 操作数；因此这个 op 没有
  SSA 输入
- `hasFolder = 1`：生成 `ConstantOp::fold`，让 `m_Constant` 能取出系数属性

`ConstantLike` 在 MLIR 里的定义要求：

```cpp
static_assert(hasTrait<OneResult>());
static_assert(hasTrait<ZeroOperands>());
```

也就是说，常量 op 必须“零个 SSA 操作数、一个结果”。`poly.constant` 正好满足。

它的 folder 非常简单：

```cpp
// lib/Dialect/Poly/PolyOps.cpp:14
OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor) {
  return adaptor.getCoefficients();
}
```

返回的就是 op 自带的系数属性。这个 folder 看似“什么都没做”，但没有它，
`m_Constant` 就无法把 `poly.constant` 识别为常量。

> 语法提示：`poly.constant` 的 assemblyFormat 里，系数属性自己会打印类型，
> 后面还要再打印结果类型，所以会出现两个冒号：
>
> ```mlir
> %p = poly.constant dense<[1, 2, 3]> : tensor<3xi32> : !poly.poly<10>
> ```
>
> 第一个 `tensor<3xi32>` 是系数属性的类型，第二个 `!poly.poly<10>` 是结果类型。

### 5.2 给运算 op 加 folder

`Poly_BinOp` 这个 TableGen 基类统一加了：

```tablegen
// lib/Dialect/Poly/PolyOps.td:20
class Poly_BinOp<string mnemonic>
    : Op<Poly_Dialect, mnemonic,
         [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  ...
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}
```

`hasFolder` 和 `hasCanonicalizer` 是两件不同的事：

- `hasFolder = 1`：生成 `fold`，做局部常量计算/等价替换
- `hasCanonicalizer = 1`：允许这个 op 注册 canonicalization pattern
  （例如 `poly.sub` 的 `DifferenceOfSquares`）

`poly.from_tensor` 也单独加了 `hasFolder = 1`（`PolyOps.td:45`）。

#### add / sub：`constFoldBinaryOp`

```cpp
// lib/Dialect/Poly/PolyOps.cpp:18
OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor) {
  return constFoldBinaryOp<IntegerAttr, APInt, void>(
      adaptor.getOperands(), [&](APInt a, APInt b) { return a + b; });
}
```

`constFoldBinaryOp` 是 MLIR 提供的公共 helper，行为可以概括为：

1. 两个操作数都必须是 `Attribute`
2. 两个属性必须类型相同
3. 对每个元素执行你给的 lambda
4. 返回结果属性；如果任一条件不满足，返回空

模板参数：

| 参数 | 本代码中的值 | 含义 |
|---|---|---|
| `AttrElementT` | `IntegerAttr` | 元素属性类型 |
| `ElementValueT` | `APInt` | 元素 C++ 值类型 |
| `PoisonAttr` | `void` | 不启用 poison 传播 |

`sub` 完全对称，只把 lambda 换成 `a - b`。

注意：如果两个操作数里有一个不是常量，`adaptor.getOperands()` 中对应项是空，
helper 直接返回空，所以不会错误折叠。

#### from_tensor：属性透传

```cpp
// lib/Dialect/Poly/PolyOps.cpp:63
OpFoldResult FromTensorOp::fold(FromTensorOp::FoldAdaptor adaptor) {
  // Returns null if the cast failed, which corresponds to a failed fold.
  return dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getInput());
}
```

`poly.from_tensor` 的输入是 `tensor<...xi32>`，结果类型是 `!poly.poly<N>`。
如果输入已经是常量张量，它把张量属性直接返回。之后由 materializer 把这个属性
包成 `poly.constant`。

如果输入是运行时的 tensor（例如函数参数），`adaptor.getInput()` 是空，
`dyn_cast_or_null` 返回空，折叠失败。

> 属性类型辨析：`DenseElementsAttr`、`AnyIntElementsAttr`、
> `SplatElementsAttr`、`AttrElementT`、`ElementsAttr` 分别属于运行时对象、
> ODS 约束、接口和模板参数，详见
> [MLIR 属性类型速查](attributes.md)。

### 5.3 mul：手写卷积

`poly.mul` 不能简单地逐元素相乘，因为它要做多项式乘法。实现位于
`lib/Dialect/Poly/PolyOps.cpp:28`：

```cpp
OpFoldResult MulOp::fold(MulOp::FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getOperands()[0]);
  auto rhs = dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getOperands()[1]);
  if (!lhs || !rhs) return nullptr;

  auto degree =
      llvm::cast<PolynomialType>(getResult().getType()).getDegreeBound();
  auto maxIndex = lhs.size() + rhs.size() - 1;

  SmallVector<APInt, 8> result;
  result.reserve(maxIndex);
  for (int i = 0; i < maxIndex; ++i)
    result.push_back(APInt((*lhs.begin()).getBitWidth(), 0));

  int i = 0;
  for (auto lhsIt = lhs.value_begin<APInt>(); lhsIt != lhs.value_end<APInt>();
       ++lhsIt) {
    int j = 0;
    for (auto rhsIt = rhs.value_begin<APInt>(); rhsIt != rhs.value_end<APInt>();
         ++rhsIt) {
      // index is modulo degree because poly's semantics are defined modulo
      // x^N = 1.
      result[(i + j) % degree] += *rhsIt * (*lhsIt);
      ++j;
    }
    ++i;
  }

  return DenseIntElementsAttr::get(
      RankedTensorType::get(static_cast<int64_t>(result.size()),
                            IntegerType::get(getContext(), 32)),
      result);
}
```

逐段理解：

1. **取两个系数张量**
   - 任一不是 `DenseIntElementsAttr` 就返回空
   - 这同时处理了“运行时值”和“不是整数系数”的情况

2. **读取次数上界**
   - `degree` 是 `!poly.poly<N>` 里的 `N`
   - `maxIndex = lhs.size() + rhs.size() - 1` 是两个多项式乘积的长度上界

3. **初始化结果数组**
   - 每个元素都是和输入同 bit width 的 `APInt(0)`

4. **双重循环**
   - `lhs[i] * rhs[j]` 贡献到结果的第 `i + j` 次项
   - 因为环定义是 `x^N = 1`，所以下标对 `degree` 取模
   - 例如 `x^{N}` 回绕到 `x^0`，`x^{N+1}` 回绕到 `x^1`

5. **构造结果属性**
   - 结果元素的 bit width 固定为 32
   - 张量长度是 `maxIndex`，不是 `degree`

#### 例子一：不触发回绕

```mlir
func.func @mul_constants() -> !poly.poly<4> {
  %a = poly.constant dense<[1, 1]> : tensor<2xi32> : !poly.poly<4>
  %square = poly.mul %a, %a : !poly.poly<4>
  return %square : !poly.poly<4>
}
```

`(1 + x)² = 1 + 2x + x²`，在 `x⁴ = 1` 的环里没有回绕：

```mlir
%0 = poly.constant dense<[1, 2, 1]> : tensor<3xi32> : !poly.poly<4>
```

#### 例子二：触发回绕

```mlir
%a = poly.constant dense<[1, 0, 0, 1]> : tensor<4xi32> : !poly.poly<4>
%b = poly.constant dense<[1, 1]> : tensor<2xi32> : !poly.poly<4>
%c = poly.mul %a, %b : !poly.poly<4>
```

这里：

```text
a = 1 + x³
b = 1 + x
a*b = 1 + x + x³ + x⁴
x⁴ = 1
=> 2 + x + x³
```

折叠结果：

```mlir
%0 = poly.constant dense<[2, 1, 0, 1, 0]> : tensor<5xi32> : !poly.poly<4>
```

注意结果张量长度是 5（`4 + 2 - 1`），最后一位是 0。这是当前 fold 实现的
自然结果：它按两个输入长度的乘积上界分配数组，再对下标取模。多项式语义仍然
正确，`poly.constant` 也没有要求系数长度必须小于等于 `degreeBound`。

### 5.4 常量物化：`materializeConstant`

folder 返回 `Attribute` 后，IR 里还没有对应的 SSA 值。需要 dialect 提供：

```tablegen
// lib/Dialect/Poly/PolyDialect.td:14
let hasConstantMaterializer = 1;
```

ODS 生成声明：

```cpp
// build/lib/Dialect/Poly/PolyDialect.h.inc
::mlir::Operation *materializeConstant(::mlir::OpBuilder &builder,
                                       ::mlir::Attribute value,
                                       ::mlir::Type type,
                                       ::mlir::Location loc) override;
```

实现位于 `lib/Dialect/Poly/PolyDialect.cpp:29`：

```cpp
Operation *PolyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  auto coeffs = dyn_cast<DenseIntElementsAttr>(value);
  if (!coeffs)
    return nullptr;
  return builder.create<ConstantOp>(loc, type, coeffs);
}
```

- `value`：folder 算出来的属性，例如 `dense<[11, 22, 33]>`
- `type`：结果应该是什么类型，这里是 `!poly.poly<10>`
- 返回 `nullptr`：告诉调用者“我物化不了”，这次折叠放弃

为什么需要 `type`？因为同一个属性可能对应多种结果类型。例如
`arith.constant` 可以用一个 hex 属性配合不同的张量形状/元素类型来解释；
物化器必须知道目标类型才能构造正确的 op。

### 5.5 三块基础设施的分工

| 组件 | 位置 | 作用 |
|---|---|---|
| `poly.constant` + `ConstantLike` | `PolyOps.td` | 生产常量，让 `m_Constant` 能识别 |
| 各 op 的 `fold` | `PolyOps.cpp` | 根据常量操作数计算新属性/返回已有值 |
| `materializeConstant` | `PolyDialect.cpp` | 把属性变成 `poly.constant` op |

缺任何一块，常量折叠都无法完整工作：

- 没有常量 op：folder 算出的属性没有载体
- 没有 folder：常量 op 能被识别，但 `poly.add` 不知道该怎么算
- 没有 materializer：folder 返回了属性，但无法变成 IR 中的值

## 6. 实验

先设置路径（第 2 节已设置过，可跳过）：

```bash
cd /home/zhouzhou/npu/mlir-tutorial
BIN=$PWD/externals/llvm-project/build/bin
TO=$PWD/build/tools/tutorial-opt
```

### 6.1 `--canonicalize`：折叠 + 清理

运行完整测试文件：

```bash
$TO tests/poly_fold.mlir --canonicalize
```

输出（节选，省略了 `@mul_wrap`）：

```mlir
module {
  func.func @add_constants() -> !poly.poly<10> {
    %0 = poly.constant dense<[11, 22, 33]> : tensor<3xi32> : !poly.poly<10>
    return %0 : !poly.poly<10>
  }
  func.func @mul_constants() -> !poly.poly<4> {
    %0 = poly.constant dense<[1, 2, 1]> : tensor<3xi32> : !poly.poly<4>
    return %0 : !poly.poly<4>
  }
  func.func @from_tensor_fold() -> !poly.poly<10> {
    %0 = poly.constant dense<[2, 4, 6]> : tensor<3xi32> : !poly.poly<10>
    return %0 : !poly.poly<10>
  }
  func.func @non_constant(%arg0: !poly.poly<10>) -> !poly.poly<10> {
    %0 = poly.add %arg0, %arg0 : !poly.poly<10>
    return %0 : !poly.poly<10>
  }
  func.func @eval_has_no_folder() -> i32 {
    %0 = poly.constant dense<[1, 2, 3]> : tensor<3xi32> : !poly.poly<10>
    %c2_i32 = arith.constant 2 : i32
    %1 = poly.eval %0, %c2_i32 : (!poly.poly<10>, i32) -> i32
    return %1 : i32
  }
}
```

逐个看：

**`@add_constants`**

- `poly.constant` 的 folder 返回系数属性
- `poly.add` 的 folder 用 `constFoldBinaryOp` 逐元素相加
- materializer 产生 `[11, 22, 33]`
- 原来的两个常量失去使用者，被 canonicalizer 删除

**`@mul_constants`**

- `[1, 1] * [1, 1] = [1, 2, 1]`
- 这是一次真实的多项式卷积，不是逐元素相乘

**`@mul_wrap`**（测试文件中的回绕用例）

- `[1, 0, 0, 1]` 表示 `1 + x³`，`[1, 1]` 表示 `1 + x`
- 乘积是 `1 + x + x³ + x⁴`
- 在 `x⁴ = 1` 的环里，`x⁴` 回绕成常数项，结果是 `[2, 1, 0, 1, 0]`
- 这说明 `MulOp::fold` 的 `(i + j) % degree` 确实生效

**`@from_tensor_fold`**

这里经历了多步：

```text
arith.constant dense<[1,2,3]>
        │  arith.constant 的 folder
        ▼
DenseIntElementsAttr [1,2,3]
        │  FromTensorOp::fold
        ▼
DenseIntElementsAttr [1,2,3]  (仍然只是属性)
        │  materializeConstant
        ▼
poly.constant [1,2,3]
        │  AddOp::fold
        ▼
DenseIntElementsAttr [2,4,6]
        │  materializeConstant
        ▼
poly.constant [2,4,6]
```

`poly.from_tensor` 本身被消除了，因为它的结果被常量 `poly.constant` 取代。

**`@non_constant`**

`%arg0` 是函数参数，不是常量。`adaptor.getOperands()` 里对应位置为空，
`constFoldBinaryOp` 返回空，所以 `poly.add` 保留。

注意：`poly.add %p, %p` 从代数上等于 `2 * p`，但当前代码没有实现
`x + x -> 2x` 这种规则。这种“有名字的代数恒等式”更适合放在
canonicalization pattern 里，而不是 fold 里。

**`@eval_has_no_folder`**

`poly.eval` 的 `.td` 定义（`PolyOps.td:57`）有：

```tablegen
let hasVerifier = 1;
let hasCanonicalizer = 1;
```

但没有 `let hasFolder = 1;`。因此：

- `poly.eval` 的常量参数虽然能被识别为 `Attribute`
- 但 `EvalOp` 自己没有 folder，无法计算 `1 + 2*2 + 3*2² = 17`
- canonicalizer 只能应用 `LiftConjThroughEval` 这样的 pattern
- 最终 `poly.eval` 保持原样

这正好说明：**仅有常量输入不够，op 必须自己声明 folder**。

### 6.2 SCCP：替换常量，但留下死代码

```bash
$TO -pass-pipeline="builtin.module(func.func(sccp))" tests/poly_fold.mlir
```

输出中 `@add_constants`：

```mlir
func.func @add_constants() -> !poly.poly<10> {
  %0 = poly.constant dense<[11, 22, 33]> : tensor<3xi32> : !poly.poly<10>
  %1 = poly.constant dense<[10, 20, 30]> : tensor<3xi32> : !poly.poly<10>
  %2 = poly.constant dense<[1, 2, 3]> : tensor<3xi32> : !poly.poly<10>
  return %0 : !poly.poly<10>
}
```

与 canonicalize 的差别：

- SCCP 把 `return` 的操作数换成了新常量
- 它没有把原来两条常量删掉
- 它也没有做通用的 DCE

`@from_tensor_fold` 的 SCCP 输出还会保留：

```mlir
%1 = poly.constant dense<[1, 2, 3]> : tensor<3xi32> : !poly.poly<10>
%cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
```

这些都是死代码。生产环境通常会再跑一次 `--canonicalize` 或 `--symbol-dce`
做清理。

SCCP 还有一个 canonicalize 不具备的能力：它能在控制流上传播常量。虽然本课的
poly 例子都是直线代码，但你可以把 SCCP 理解为“带数据流分析的、跨 block 的
常量折叠”。

### 6.3 CSE：只合并重复 op

```bash
$TO tests/cse.mlir -cse
```

输出：

```mlir
func.func @test_simple_cse() -> !poly.poly<10> {
  %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %0 = poly.from_tensor %cst : tensor<3xi32> -> !poly.poly<10>
  %1 = poly.mul %0, %0 : !poly.poly<10>
  %2 = poly.add %1, %1 : !poly.poly<10>
  return %2 : !poly.poly<10>
}
```

输入中两个相同的 `poly.mul` 被合并成一个。CSE 之后仍然保留
`poly.from_tensor` 和 `poly.add`，说明 CSE 没有做常量求值。

对比：

```bash
$TO tests/cse.mlir --canonicalize
```

输出：

```mlir
func.func @test_simple_cse() -> !poly.poly<10> {
  %0 = poly.constant dense<[2, 8, 20, 24, 18]> : tensor<5xi32> : !poly.poly<10>
  return %0 : !poly.poly<10>
}
```

这里 `poly.from_tensor`、两个 `poly.mul`、`poly.add` 全部被折叠/清理，只剩一个
常量。

### 6.4 测试文件

新增的 `tests/poly_fold.mlir` 把上面的行为固定成回归测试：

```mlir
// RUN: tutorial-opt %s --canonicalize | FileCheck %s --check-prefix=CANON
// RUN: tutorial-opt -pass-pipeline="builtin.module(func.func(sccp))" %s | FileCheck %s --check-prefix=SCCP

// CANON-LABEL: func.func @add_constants
// CANON: poly.constant dense<[11, 22, 33]>
// CANON-NOT: poly.add
// SCCP-LABEL: func.func @add_constants
// SCCP: poly.constant dense<[11, 22, 33]>
// SCCP-NOT: poly.add
func.func @add_constants() -> !poly.poly<10> {
  %a = poly.constant dense<[1, 2, 3]> : tensor<3xi32> : !poly.poly<10>
  %b = poly.constant dense<[10, 20, 30]> : tensor<3xi32> : !poly.poly<10>
  %sum = poly.add %a, %b : !poly.poly<10>
  return %sum : !poly.poly<10>
}

// CANON-LABEL: func.func @mul_constants
// CANON: poly.constant dense<[1, 2, 1]>
// CANON-NOT: poly.mul
// SCCP-LABEL: func.func @mul_constants
// SCCP: poly.constant dense<[1, 2, 1]>
// SCCP-NOT: poly.mul
func.func @mul_constants() -> !poly.poly<4> {
  %a = poly.constant dense<[1, 1]> : tensor<2xi32> : !poly.poly<4>
  %b = poly.constant dense<[1, 1]> : tensor<2xi32> : !poly.poly<4>
  %c = poly.mul %a, %b : !poly.poly<4>
  return %c : !poly.poly<4>
}

// CANON-LABEL: func.func @mul_wrap
// CANON: poly.constant dense<[2, 1, 0, 1, 0]>
// CANON-NOT: poly.mul
// SCCP-LABEL: func.func @mul_wrap
// SCCP: poly.constant dense<[2, 1, 0, 1, 0]>
// SCCP-NOT: poly.mul
func.func @mul_wrap() -> !poly.poly<4> {
  %a = poly.constant dense<[1, 0, 0, 1]> : tensor<4xi32> : !poly.poly<4>
  %b = poly.constant dense<[1, 1]> : tensor<2xi32> : !poly.poly<4>
  %c = poly.mul %a, %b : !poly.poly<4>
  return %c : !poly.poly<4>
}

// CANON-LABEL: func.func @from_tensor_fold
// CANON: poly.constant dense<[2, 4, 6]>
// CANON-NOT: poly.from_tensor
// SCCP-LABEL: func.func @from_tensor_fold
// SCCP: poly.constant dense<[2, 4, 6]>
// SCCP-NOT: poly.from_tensor
func.func @from_tensor_fold() -> !poly.poly<10> {
  %t = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %p = poly.from_tensor %t : tensor<3xi32> -> !poly.poly<10>
  %sum = poly.add %p, %p : !poly.poly<10>
  return %sum : !poly.poly<10>
}

// CANON-LABEL: func.func @non_constant
// CANON: poly.add
// SCCP-LABEL: func.func @non_constant
// SCCP: poly.add
func.func @non_constant(%p: !poly.poly<10>) -> !poly.poly<10> {
  %sum = poly.add %p, %p : !poly.poly<10>
  return %sum : !poly.poly<10>
}

// CANON-LABEL: func.func @eval_has_no_folder
// CANON: poly.eval
// SCCP-LABEL: func.func @eval_has_no_folder
// SCCP: poly.eval
func.func @eval_has_no_folder() -> i32 {
  %p = poly.constant dense<[1, 2, 3]> : tensor<3xi32> : !poly.poly<10>
  %x = arith.constant 2 : i32
  %y = poly.eval %p, %x : (!poly.poly<10>, i32) -> i32
  return %y : i32
}
```

手动运行：

```bash
$TO tests/poly_fold.mlir --canonicalize | $BIN/FileCheck tests/poly_fold.mlir --check-prefix=CANON
$TO -pass-pipeline="builtin.module(func.func(sccp))" tests/poly_fold.mlir \
  | $BIN/FileCheck tests/poly_fold.mlir --check-prefix=SCCP
```

两条命令都没有输出，表示通过。`tests/` 下的 `.mlir` 由 lit 收集（本环境 lit
未安装，手动执行 `RUN:` 命令即可，见第 0 课的环境说明）。

## 7. 源码位置

| 内容 | 文件 |
|---|---|
| `hasFolder` / `hasConstantMaterializer` 声明 | `lib/Dialect/Poly/PolyOps.td`、`PolyDialect.td` |
| 各 op 的 fold 实现 | `lib/Dialect/Poly/PolyOps.cpp` |
| `materializeConstant` 实现 | `lib/Dialect/Poly/PolyDialect.cpp` |
| box 的 fold（返回 `Value` 的例子） | `lib/Dialect/Box/BoxOps.cpp` |
| 折叠测试 | `tests/poly_fold.mlir` |
| CSE 测试 | `tests/cse.mlir` |
| box 折叠测试 | `tests/box_fold.mlir` |

## 8. 常见问题

### 8.1 为什么 `poly.constant` 还要有 folder？

因为通用基础设施不是通过“op 名字叫 constant”来判断常量的，而是通过：

- `ConstantLike` trait
- 调用这个 op 的 folder，拿到 `Attribute`

`ConstantOp::fold` 返回系数属性；没有它，`m_Constant` 在 debug 构建里会触发
“expected ConstantLike op to be foldable” 断言，其他 op 也无法把
`poly.constant` 当作常量输入。

### 8.2 为什么 `hasFolder` 和 `hasCanonicalizer` 都存在？

它们解决不同层次的问题：

- folder：只在当前 op 上做局部计算，不创建新 op
- canonicalizer：可以创建/删除 op、重写多个 op，适合代数恒等式

`poly.sub` 的 `DifferenceOfSquares` 就是 canonicalization pattern，而不是
folder。下一课会详细讲它。

### 8.3 如果只加 folder，不加 materializer 会怎样？

folder 能返回 `Attribute`，但调用者无法把它变成 SSA 值。greedy driver 会调用
`materializeConstant`；返回 `nullptr` 时会放弃这次折叠，IR 看起来“什么都没发生”。

### 8.4 为什么 `from_tensor` 的 fold 返回的是 tensor 属性，而不是 poly 属性？

因为它的输入是 tensor，结果类型是 `!poly.poly<N>`。folder 只负责“计算结果
数据”；把 tensor 属性包装成特定结果类型的工作交给 `materializeConstant`。
物化器拿到 `type = !poly.poly<N>` 后，构造 `poly.constant`。

### 8.5 `OpFoldResult` 返回 `Value` 和返回 `Attribute` 有什么区别？

- `Value`：结果已经存在于 IR 中，直接替换使用点，不需要新建 op
- `Attribute`：结果只是编译期数据，需要 dialect 物化成常量 op

`box.unpack` 的 `unpack(pack(x)) -> x` 适合返回 `Value`；
`poly.add` 的 `[1,2,3] + [10,20,30] -> [11,22,33]` 适合返回 `Attribute`。

### 8.6 为什么 SCCP 的输出里还有死常量？

SCCP 替换值，但不做通用 DCE。它的实现只会在“结果全部被替换且 op 明显无副作用”
时删掉部分 op；常量 op 往往被保留。需要额外的 canonicalize/DCE 清理。

### 8.7 fold 可以返回一个新创建的 `Value` 吗？

不可以。fold 不能创建新 op，所以不能通过 `builder.create` 产生新的 SSA 值。
它只能返回**已经存在**的 `Value`，或者返回 `Attribute` 让调用者物化。

## 9. 练习

### 练习 1：解释 `poly.eval` 为什么没有折叠

输入：

```mlir
%p = poly.constant dense<[1, 2, 3]> : tensor<3xi32> : !poly.poly<10>
%x = arith.constant 2 : i32
%y = poly.eval %p, %x : (!poly.poly<10>, i32) -> i32
```

运行 `--canonicalize` 后 `poly.eval` 仍然存在。请回答：

1. 它的两个输入是不是常量？
2. 为什么仍然不能折叠？
3. 如果要让它折叠，需要改哪些地方？

<details>
<summary>参考答案</summary>

1. 是。`poly.constant` 和 `arith.constant` 都带 `ConstantLike`，都能被
   `m_Constant` 取出属性。
2. `poly.eval` 没有 `hasFolder = 1`，也没有实现 `EvalOp::fold`。常量输入只是
   提供了“可用的 Attribute”，但 op 自己没有计算逻辑。
3. 在 `PolyOps.td` 的 `Poly_EvalOp` 中加 `let hasFolder = 1;`，然后在
   `PolyOps.cpp` 实现 `OpFoldResult EvalOp::fold(EvalOp::FoldAdaptor adaptor)`：
   - 从 `adaptor.getPolynomial()` 取系数张量
   - 从 `adaptor.getPoint()` 取 `IntegerAttr` 或 `ComplexAttr`
   - 用 Horner 法计算
   - 返回 `IntegerAttr` 或 `ComplexAttr`

   注意：`poly.eval` 的结果类型可能是 `i32` 或 `complex<f64>`，所以 folder
   要同时处理整数和复数；如果只实现整数分支，遇到复数就返回空。

</details>

### 练习 2：给 `poly.neg` 增加 folder

`poly.neg` 目前在 `PolyOps.td:74`，只有 `Pure` 和 `SameOperandsAndResultType`，
没有 folder。请：

1. 在 `.td` 中加 `hasFolder = 1`
2. 实现 `NegOp::fold`
3. 构造一个 `poly.constant` + `poly.neg` 的测试，确认 `--canonicalize` 能把它
   折叠成取反后的常量

<details>
<summary>参考答案</summary>

`.td`：

```tablegen
def Poly_NegOp : Op<Poly_Dialect, "neg", [Pure, SameOperandsAndResultType]> {
  let summary = "Negation of a polynomial.";
  let arguments = (ins Polynomial:$input);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($output))";
  let hasFolder = 1;   // 新增
}
```

`PolyOps.cpp`：

```cpp
OpFoldResult NegOp::fold(NegOp::FoldAdaptor adaptor) {
  auto coeffs = dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getInput());
  if (!coeffs)
    return {};

  SmallVector<APInt> negated;
  negated.reserve(coeffs.getNumElements());
  for (APInt value : coeffs.getValues<APInt>())
    negated.push_back(-value);

  return DenseIntElementsAttr::get(coeffs.getType(), negated);
}
```

测试：

```mlir
func.func @neg_constant() -> !poly.poly<10> {
  %p = poly.constant dense<[1, 2, 3]> : tensor<3xi32> : !poly.poly<10>
  %n = poly.neg %p : !poly.poly<10>
  return %n : !poly.poly<10>
}
```

`--canonicalize` 后应得到：

```mlir
%0 = poly.constant dense<[-1, -2, -3]> : tensor<3xi32> : !poly.poly<10>
return %0 : !poly.poly<10>
```

注意 `APInt` 的取反按位宽取模；对于 i32，`-1` 的存储形式是 32 位全 1，
打印成有符号十进制仍然是 `-1`。

</details>

### 练习 3：为什么 `%p + %p` 没有折叠？

```mlir
func.func @non_constant(%p: !poly.poly<10>) -> !poly.poly<10> {
  %sum = poly.add %p, %p : !poly.poly<10>
  return %sum : !poly.poly<10>
}
```

请解释原因，并说明如果要实现 `x + x -> 2x`，应该放在 folder 还是
canonicalization pattern 里。

<details>
<summary>参考答案</summary>

`%p` 是函数参数，不是常量，`adaptor.getOperands()` 中对应位置为空，
`constFoldBinaryOp` 直接返回空。当前 `AddOp::fold` 只处理“两个输入都是
常量属性”的情况。

`x + x -> 2x` 需要识别两个操作数相同，并构造一个新的常量/乘法 op；这更适合
放在 canonicalization pattern 里（或 fold 的一种特殊等价替换，但要注意不能
随意创建新 op）。当前仓库没有实现这条规则。

</details>

## 10. 验收清单

- [ ] 能解释 fold 不是 pass，而是 op 提供的局部查询
- [ ] 能说出 `OpFoldResult` 的三种返回值
- [ ] 能解释 `FoldAdaptor` 为什么能把 `getLhs()` 变成 `Attribute`
- [ ] 能说明 `ConstantLike` 的作用和两个结构要求
- [ ] 能解释 `poly.constant` 的 folder 为什么是必需的
- [ ] 能读懂 `poly.add`、`sub`、`mul`、`from_tensor` 的 folder
- [ ] 能解释 `materializeConstant` 在折叠链路中的位置
- [ ] 能区分 canonicalize、SCCP、CSE 的行为
- [ ] 能运行 `tests/poly_fold.mlir` 的两个 RUN 命令
- [ ] 能独立给 `poly.neg` 增加 folder

## 11. 下一课

**第 7 课：Verifiers**：[07-verifiers.md](07-verifiers.md)

- op 级 `verify()`
- trait 级 `verifyTrait()`
- 用 `-verify-diagnostics` 测试非法 IR
- `poly.eval` 为什么要求 point 是 i32 或 complex
- 测试：`tests/poly_verifier.mlir`

对应源码：`lib/Dialect/Poly/PolyOps.cpp`、`lib/Dialect/Poly/PolyTraits.h`。

## 参考

- 本课对应文章：<https://jeremykun.com/2023/09/11/mlir-folders/>
- MLIR 官方文档：Operation Folding
  <https://mlir.llvm.org/docs/Canonicalization/#canonicalization>
- 补充：[MLIR 属性类型速查](attributes.md)
- 补充：[FileCheck 使用指南](filecheck.md)
- 本课测试：`tests/poly_fold.mlir`
- 相关源码：
  - `lib/Dialect/Poly/PolyOps.td`
  - `lib/Dialect/Poly/PolyOps.cpp`
  - `lib/Dialect/Poly/PolyDialect.td`
  - `lib/Dialect/Poly/PolyDialect.cpp`