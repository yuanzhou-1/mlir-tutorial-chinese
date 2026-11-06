# MLIR 属性类型速查：ElementsAttr、DenseElementsAttr 与 ODS 约束

> 本文补充第 6 课 [Folders and Constant Propagation](06-folders.md)。
>
> 读 fold 代码时经常会同时遇到这几个名字：
> `ElementsAttr`、`DenseElementsAttr`、`DenseIntElementsAttr`、
> `SplatElementsAttr`、`AnyIntElementsAttr`、`AttrElementT`。
> 它们看起来都是“属性”，但其实分属三个不同层次：MLIR 运行时对象、
> C++ 接口/模板、TableGen 约束。本文把它们拆开。

## 1. 先看一行代码里的五个名字

`poly.constant` 的定义：

```tablegen
def Poly_ConstantOp : Op<Poly_Dialect, "constant", [Pure, ConstantLike]> {
  let arguments = (ins AnyIntElementsAttr:$coefficients);
  let results = (outs Polynomial:$output);
  let hasFolder = 1;
}
```

对应的 folder：

```cpp
// lib/Dialect/Poly/PolyOps.cpp:14
OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor) {
  return adaptor.getCoefficients();
}
```

对应的加法 folder：

```cpp
// lib/Dialect/Poly/PolyOps.cpp:18
OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor) {
  return constFoldBinaryOp<IntegerAttr, APInt, void>(
      adaptor.getOperands(), [&](APInt a, APInt b) { return a + b; });
}
```

对应的物化器：

```cpp
// lib/Dialect/Poly/PolyDialect.cpp:29
Operation *PolyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  auto coeffs = dyn_cast<DenseIntElementsAttr>(value);
  if (!coeffs)
    return nullptr;
  return builder.create<ConstantOp>(loc, type, coeffs);
}
```

五个名字分别出现在：

```text
AnyIntElementsAttr   -> TableGen 参数约束
DenseIntElementsAttr -> C++ 运行时属性类型
AttrElementT         -> constFoldBinaryOp 的模板参数
IntegerAttr          -> 标量属性类型（模板实参）
ElementsAttr         -> 属性接口
```

它们不是同一层的东西。

---

## 2. 三层分类

| 层次 | 名字 | 用途 |
|---|---|---|
| MLIR 运行时对象 | `Attribute`、`IntegerAttr`、`DenseElementsAttr`、`DenseIntElementsAttr`、`SplatElementsAttr` | 真正存放常量数据 |
| MLIR C++ 接口 | `ElementsAttr` | 表示“元素容器”的统一接口 |
| TableGen/ODS 约束 | `AnyIntElementsAttr`、`AnyI32ElementsAttr`、`ElementsAttr`（ODS 版） | 描述 op 允许哪些属性 |
| C++ 模板参数 | `AttrElementT`、`ElementValueT`、`PoisonAttr` | 让 `constFoldBinaryOp` 泛型化 |

> 最容易混淆的一点：
>
> `AnyIntElementsAttr` 不是 C++ 类，不能写 `isa<AnyIntElementsAttr>(attr)`。
> 它是 TableGen 里的约束名，写 `.td` 时使用。

---

## 3. `Attribute`：所有属性的根

MLIR 里的编译期数据都叫 Attribute：

```mlir
42 : i32
dense<[1, 2, 3]> : tensor<3xi32>
"hello"
```

对应的 C++ 对象分别是：

```cpp
IntegerAttr
DenseIntElementsAttr
StringAttr
```

它们都可以被装进一个类型擦除的：

```cpp
Attribute attr;
```

`Attribute` 有点像 `std::any` 或 `void *` 的 MLIR 版本：

- 本身不暴露具体数据布局
- 通过 `isa` / `dyn_cast` / `cast` 恢复具体类型
- 是不可变的、由 `MLIRContext` 管理的编译期数据

### Attribute 和 Value 的区别

| | `Attribute` | `Value` |
|---|---|---|
| 名字 | 无 | `%a`、`%sum` |
| 是否有 use-def | 否 | 是 |
| 是否在 IR 中占一个 SSA 位置 | 否 | 是 |
| 例子 | `dense<[1,2,3]>` | `%p` |
| 生命周期 | 编译期 | IR 中存在期间 |

fold 返回的是 Attribute，不是 Value：

```cpp
OpFoldResult AddOp::fold(...) {
  return constFoldBinaryOp(...);  // Attribute
}
```

---

## 4. `ElementsAttr`：元素容器接口

`ElementsAttr` 不是一种具体的存储格式，而是一个**属性接口**：

> 这个属性表示一个 tensor 或 vector，里面有一串元素。

它提供一组通用操作：

```cpp
int64_t getNumElements();
Type getElementType();
bool isSplat();
auto getValues<T>();
auto try_value_begin<T>();
```

具体实现可以是：

- `DenseElementsAttr`：稠密存储
- `SparseElementsAttr`：稀疏存储
- `DenseResourceElementsAttr`：外部资源存储

所以 `ElementsAttr` 负责“统一视角”，具体类负责“怎么存”。

`constFoldBinaryOp` 使用 `ElementsAttr` 的原因就是：它不关心底层是 dense 还是
sparse，只要能按元素迭代即可。

简化后的代码：

```cpp
if (isa<ElementsAttr>(operands[0]) &&
    isa<ElementsAttr>(operands[1])) {
  auto lhs = cast<ElementsAttr>(operands[0]);
  auto rhs = cast<ElementsAttr>(operands[1]);
  ...
}
```

---

## 5. `DenseElementsAttr`：稠密张量属性

`DenseElementsAttr` 是 `ElementsAttr` 接口最常见的实现：

```mlir
dense<[1, 2, 3]> : tensor<3xi32>
```

特点：

- 元素在内存中连续存放
- 适合常量张量
- 元素可以是整数、浮点、复数
- 是不可变的属性

对应 C++ 类：

```cpp
class DenseElementsAttr : public Attribute { ... };
```

### 5.1 `DenseIntElementsAttr`：整数版本

`DenseIntElementsAttr` 是 `DenseElementsAttr` 的整数特化：

```mlir
dense<[1, 2, 3]> : tensor<3xi32>
```

严格的继承链是：

```text
DenseIntElementsAttr
  -> DenseIntOrFPElementsAttr
     -> DenseElementsAttr
```

中间的 `DenseIntOrFPElementsAttr` 同时服务于整数和浮点版本，平时使用时不需要
直接写它。

它支持按 `APInt` 迭代：

```cpp
DenseIntElementsAttr coeffs = ...;
for (APInt value : coeffs.getValues<APInt>()) {
  // value 是一个整数系数
}
```

在 poly 代码里：

```cpp
auto coeffs = dyn_cast<DenseIntElementsAttr>(value);
```

就是在问：

> “这个 Attribute 是不是一个稠密整数张量？”

### 5.2 `DenseIntElementsAttr` 和 `IntegerAttr` 的区别

| | `IntegerAttr` | `DenseIntElementsAttr` |
|---|---|---|
| 表示 | 一个标量整数 | 一个整数 tensor/vector |
| MLIR 例子 | `42 : i32` | `dense<[1,2,3]> : tensor<3xi32>` |
| 元素访问 | `getValue()` 得到一个 `APInt` | `getValues<APInt>()` 得到一串 |
| 在 poly 中 | 可能出现在 `poly.eval` 的 point | poly 的系数 |

`poly.constant` 的系数是 `DenseIntElementsAttr`，不是 `IntegerAttr`。

---

## 6. `SplatElementsAttr`：所有元素相同的视图

`SplatElementsAttr` 表示：

> 一个 `DenseElementsAttr`，其中所有元素的值都相同。

例如：

```mlir
dense<1> : tensor<100xi32>
```

它不是 100 个不同的 1，而是“100 个 1”的 splat 表示。

### 6.1 它不是独立的存储格式

看 MLIR 的定义：

```cpp
class SplatElementsAttr : public DenseElementsAttr {
public:
  static bool classof(Attribute attr) {
    auto denseAttr = dyn_cast<DenseElementsAttr>(attr);
    return denseAttr && denseAttr.isSplat();
  }
};
```

所以：

```cpp
DenseIntElementsAttr attr = ...;

isa<DenseElementsAttr>(attr);  // true
isa<SplatElementsAttr>(attr);  // 可能是 true，也可能是 false
```

一个对象可以同时是：

```text
DenseElementsAttr
DenseIntElementsAttr
SplatElementsAttr
ElementsAttr
```

只要它满足各自的条件。

### 6.2 为什么 helper 要单独判断 splat？

如果两个操作数都是 splat：

```mlir
dense<1> : tensor<100xi32>
dense<2> : tensor<100xi32>
```

就不需要真的迭代 100 次。`constFoldBinaryOp` 可以只算一次：

```cpp
calculate(1, 2);  // 得到 3
// 直接构造 dense<3> : tensor<100xi32>
```

对应源码里的分支：

```cpp
if (isa<SplatElementsAttr>(operands[0]) &&
    isa<SplatElementsAttr>(operands[1])) {
  auto lhs = cast<SplatElementsAttr>(operands[0]);
  auto rhs = cast<SplatElementsAttr>(operands[1]);
  ...
  auto elementResult = calculate(lhs.getSplatValue<ElementValueT>(),
                                 rhs.getSplatValue<ElementValueT>());
  ...
}
```

在我们的 `poly_fold.mlir` 的 SCCP 输出里可能看到：

```mlir
%1 = poly.constant dense<1> : tensor<2xi32> : !poly.poly<4>
```

这就是一个 splat。

---

## 7. `AnyIntElementsAttr`：ODS 约束，不是 C++ 类

这是最容易误解的名字。

### 7.1 它出现在 `.td` 里

```tablegen
let arguments = (ins AnyIntElementsAttr:$coefficients);
```

### 7.2 它的定义是 TableGen 约束

简化自 `CommonAttrConstraints.td`：

```tablegen
class IntElementsAttrBase<Pred condition, string summary> :
    ElementsAttrBase<And<[CPred<"::llvm::isa<::mlir::DenseIntElementsAttr>($_self)">,
                          condition]>,
                     summary> {
  let storageType = "::mlir::DenseIntElementsAttr";
  let returnType = "::mlir::DenseIntElementsAttr";
}

def AnyIntElementsAttr : IntElementsAttrBase<CPred<"true">,
                                             "integer elements attribute">;
```

它表达的是：

> 接受任意整数位宽的 `DenseIntElementsAttr`。

“Any” 指的是**任意整数位宽**：

```mlir
dense<[1, 2, 3]> : tensor<3xi8>
dense<[1, 2, 3]> : tensor<3xi32>
dense<[1, 2, 3]> : tensor<3xi64>
```

不是“任意存储形式”。底层仍然要求是 dense 整数张量。

### 7.3 ODS 会为它生成 C++ 访问器

对于：

```tablegen
let arguments = (ins AnyIntElementsAttr:$coefficients);
```

ODS 生成：

```cpp
DenseIntElementsAttr getCoefficients();
```

所以：

```text
TableGen: AnyIntElementsAttr
        ↓
C++:      DenseIntElementsAttr
```

这就是为什么：

- `.td` 里写 `AnyIntElementsAttr`
- `.cpp` 里用 `DenseIntElementsAttr`

### 7.4 不能对 `AnyIntElementsAttr` 做 `isa`

```cpp
isa<AnyIntElementsAttr>(attr);  // 错误：这不是 C++ 类型
```

`AnyIntElementsAttr` 只存在于 TableGen 世界里。

类似的约束还有：

```tablegen
AnyI32ElementsAttr   // 只接受 i32 元素的 dense 整数张量
AnyI64ElementsAttr   // 只接受 i64
ElementsAttr         // 任意 ElementsAttr（ODS 版）
```

---

## 8. `AttrElementT`：C++ 模板参数

`AttrElementT` 既不是运行时对象，也不是 TableGen 约束，它只是
`constFoldBinaryOp` 的模板参数名。

调用：

```cpp
constFoldBinaryOp<IntegerAttr, APInt, void>(...)
```

对应：

```text
AttrElementT = IntegerAttr
ElementValueT = APInt
PoisonAttr   = void
```

### 8.1 三个模板参数分别做什么

| 参数 | 本代码中的值 | 作用 |
|---|---|---|
| `AttrElementT` | `IntegerAttr` | 当操作数是**标量属性**时，用哪种属性类型 |
| `ElementValueT` | `APInt` | 当操作数是**张量属性**时，每个元素的 C++ 类型 |
| `PoisonAttr` | `void` | 不启用 poison 传播 |

### 8.2 为什么 poly 用 `IntegerAttr`，但实际属性是 `DenseIntElementsAttr`？

因为 `constFoldBinaryOp` 是通用 helper，它同时支持两类输入：

```cpp
// 分支 1：两个标量 IntegerAttr
if (isa<AttrElementT>(operands[0]) &&
    isa<AttrElementT>(operands[1])) {
  auto lhs = cast<AttrElementT>(operands[0]);
  auto rhs = cast<AttrElementT>(operands[1]);
  auto result = calculate(lhs.getValue(), rhs.getValue());
}

// 分支 2：两个 tensor/vector ElementsAttr
if (isa<ElementsAttr>(operands[0]) &&
    isa<ElementsAttr>(operands[1])) {
  auto lhs = cast<ElementsAttr>(operands[0]);
  ...
  for (APInt a, b : ...) {
    calculate(a, b);
  }
}
```

对 poly：

```text
[1,2,3] 和 [10,20,30] 都是 DenseIntElementsAttr
-> 不是标量 IntegerAttr
-> 跳过分支 1
-> 进入 ElementsAttr 分支
-> 用 APInt 逐元素计算
```

所以 `IntegerAttr` 这个模板实参在 poly 的这条路径里**不会**用于实际数据，
它只是让 helper 的标量分支有类型可用。

---

## 9. 完整链条

以 `poly.add` 的常量为：

```text
TableGen 约束：
  AnyIntElementsAttr
        │
        ▼
生成的 C++ 访问器：
  DenseIntElementsAttr getCoefficients()
        │
        ▼
实际属性对象：
  DenseIntElementsAttr([1,2,3])
  DenseIntElementsAttr([10,20,30])
        │
        ▼
FoldAdaptor：
  ArrayRef<Attribute>
  [ attr1, attr2 ]
        │
        ▼
constFoldBinaryOp<IntegerAttr, APInt, void>
        │
        ├─ isa<IntegerAttr>?          否
        ├─ isa<SplatElementsAttr>?    否
        └─ isa<ElementsAttr>?         是
                │
                ▼
        按 APInt 逐元素调用 a + b
                │
                ▼
        DenseElementsAttr::get(...)
                │
                ▼
        Attribute([11,22,33])
```

---

## 10. C++ 里怎么判断和取值

### 10.1 判断接口

```cpp
Attribute attr = ...;

if (isa<ElementsAttr>(attr)) {
  // attr 是某种元素容器：dense、sparse、resource...
}
```

### 10.2 判断具体类

```cpp
if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
  // dense 是稠密元素属性
}

if (auto ints = dyn_cast<DenseIntElementsAttr>(attr)) {
  // ints 是稠密整数元素属性
}
```

### 10.3 判断 splat

```cpp
if (isa<SplatElementsAttr>(attr)) {
  // 所有元素都相同
}
```

### 10.4 读取元素

```cpp
DenseIntElementsAttr coeffs = ...;

for (APInt value : coeffs.getValues<APInt>()) {
  // 逐个系数
}
```

### 10.5 读取属性类型

```cpp
Type t = coeffs.getType();
// t 通常是 RankedTensorType，例如 tensor<3xi32>
```

注意：这里的 `tensor<3xi32>` 是 **Attribute 自己的类型**，不是 op 的结果类型
`!poly.poly<10>`。`materializeConstant` 会同时拿到这两个类型。

---

## 11. 常见混淆

### 11.1 `AnyIntElementsAttr` 不是 C++ 类

- `.td` 里可以用
- C++ 里不能用 `isa` / `dyn_cast`
- 它最终生成 `DenseIntElementsAttr` 访问器

### 11.2 `AttrElementT` 不是属性类型

它只是模板参数名。真正可以 `isa` 的是它的实参，例如 `IntegerAttr`。

### 11.3 `ElementsAttr` 不是具体存储

它是接口。`DenseElementsAttr` 是它的一种实现。

### 11.4 `SplatElementsAttr` 不是另一种存储

它是 `DenseElementsAttr` 的一个更具体的视图。一个 `DenseIntElementsAttr`
如果是 splat，可以同时满足 `DenseElementsAttr`、`DenseIntElementsAttr`、
`SplatElementsAttr`、`ElementsAttr`。

### 11.5 `DenseIntElementsAttr` 不是 `IntegerAttr`

- `IntegerAttr`：一个标量整数
- `DenseIntElementsAttr`：一个整数 tensor/vector

### 11.6 Attribute 的类型不是 op 的结果类型

```mlir
%p = poly.constant dense<[1,2,3]>
     : tensor<3xi32> : !poly.poly<10>
```

这里：

- Attribute 的类型：`tensor<3xi32>`
- op 的结果类型：`!poly.poly<10>`

`materializeConstant` 用 `type` 参数把两者组合起来。

---

## 12. 练习

### 练习 1：识别类型

给定：

```mlir
%p = poly.constant dense<[1, 2, 3]>
     : tensor<3xi32> : !poly.poly<10>
```

回答：

1. `coefficients` 的 C++ 动态类型是什么？
2. 这个 Attribute 自己的类型是什么？
3. `%p` 的 SSA 类型是什么？
4. `AnyIntElementsAttr` 在这一行里起作用吗？

<details>
<summary>参考答案</summary>

1. `DenseIntElementsAttr`
2. `tensor<3xi32>`
3. `!poly.poly<10>`
4. 它只存在于 `poly.constant` 的 TableGen 定义里，负责约束和生成
   `DenseIntElementsAttr` 访问器；运行时对象本身不是 `AnyIntElementsAttr`。

</details>

### 练习 2：判断 `isa`

对于：

```mlir
dense<1> : tensor<100xi32>
```

下面哪些为真？

```cpp
isa<ElementsAttr>(attr)
isa<DenseElementsAttr>(attr)
isa<DenseIntElementsAttr>(attr)
isa<SplatElementsAttr>(attr)
```

<details>
<summary>参考答案</summary>

四个都为真。它是一个稠密整数 splat 属性，因此同时满足接口
`ElementsAttr`、具体类 `DenseElementsAttr`/`DenseIntElementsAttr`，以及
splat 视图 `SplatElementsAttr`。

</details>

### 练习 3：为什么模板实参是 `IntegerAttr`？

```cpp
constFoldBinaryOp<IntegerAttr, APInt, void>(...)
```

既然 poly 的系数是 `DenseIntElementsAttr`，为什么第一个模板参数不是
`DenseIntElementsAttr`？

<details>
<summary>参考答案</summary>

`AttrElementT` 描述的是“标量元素属性”的类型，用于 helper 的第一个分支；
`ElementValueT=APInt` 描述的是“张量元素值”的 C++ 类型，用于 dense 分支。
poly 的系数是 dense 张量，所以走 dense 分支，实际使用 `APInt`；
`IntegerAttr` 只是通用 helper 为了支持标量常量而存在的模板参数。

</details>

### 练习 4：把约束改窄

如果把 `Poly_ConstantOp` 的约束从：

```tablegen
AnyIntElementsAttr
```

改成：

```tablegen
AnyI32ElementsAttr
```

会发生什么？

<details>
<summary>参考答案</summary>

`AnyI32ElementsAttr` 要求元素类型是 32 位整数。于是：

```mlir
poly.constant dense<[1,2,3]> : tensor<3xi8> : !poly.poly<10>
```

会被 verifier 拒绝；i32 仍然合法。`fold` 的 C++ 代码不需要改，因为生成的
访问器仍然是 `DenseIntElementsAttr`，只是验证条件变窄了。

</details>

---

## 13. 参考

- 第 6 课：[Folders and Constant Propagation](06-folders.md)
- MLIR Builtin Attributes 文档：
  <https://mlir.llvm.org/docs/Dialects/Builtin/>
- MLIR 属性接口定义：
  `externals/llvm-project/mlir/include/mlir/IR/BuiltinAttributeInterfaces.td`
- MLIR 属性类：
  `externals/llvm-project/mlir/include/mlir/IR/BuiltinAttributes.h`
- ODS 属性约束：
  `externals/llvm-project/mlir/include/mlir/IR/CommonAttrConstraints.td`
- 公共 folder：
  `externals/llvm-project/mlir/include/mlir/Dialect/CommonFolders.h`
- 本仓库中的使用：
  - `lib/Dialect/Poly/PolyOps.td`
  - `lib/Dialect/Poly/PolyOps.cpp`
  - `lib/Dialect/Poly/PolyDialect.cpp`