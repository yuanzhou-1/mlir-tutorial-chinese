# 第 7 课：Verifiers

> 对应文章：[Verifiers](https://jeremykun.com/2023/09/13/mlir-verifiers/)。
>
> 上一课我们让 folder 能算出常量，但 folder 能算的前提是 IR 本身是合法的。
> 本课讲 MLIR 的 verifier：谁在什么时候检查 IR、三层验证如何分工、如何写
> trait verifier 和 op 级 verifier，以及如何测试错误诊断。

## 1. 学习目标

- 理解 verifier 的职责，以及它在 pass/parser 中的位置
- 分清三层验证：
  - ODS 类型/属性约束与 `AllTypesMatch`
  - trait 的 `verifyTrait`
  - op 自己的 `verify()`
- 读懂 `Has32BitArguments` 这个自定义 trait
- 读懂 `EvalOp::verify()`，理解 `emitOpError`
- 理解为什么 trait verifier 通常先于 op 级 verifier
- 会用 `-verify-diagnostics` 和 FileCheck 测试错误信息
- 能自己加一个 op verifier

## 2. Verifier 在做什么

Verifier 的任务是检查 IR 是否**良构（well-formed）**：

- 类型是否满足 op 的约束
- 操作数/结果的数量和类型是否正确
- 属性是否合法
- 某些语义不变量是否成立

它和 folder/canonicalizer 不同：

| | verifier | folder/canonicalizer |
|---|---|---|
| 目的 | 判断 IR 是否合法 | 改写 IR |
| 失败时 | 发出诊断，停止 | 返回空/不匹配，IR 保持 |
| 是否修改 IR | 不应修改 | 会修改 |
| 何时运行 | 解析后、pass 前后 | canonicalize/SCCP 等驱动 |

MLIR 的 pass pipeline 默认会在 pass 之后运行 verifier（除非显式关闭），
`mlir-opt` / `tutorial-opt` 在解析输入后也会验证。verifier 的意义是：

> 如果每个 pass 都保证输出合法，那么后面的 pass 就可以依赖这些不变量，
> 不用到处写防御性检查。

注意：verifier 里不应该创建 op、删除 op、修改操作数。它只检查，不修复。

## 3. 三层验证

以 `poly.eval` 为例，它的合法性由三层共同保证：

```text
第一层：ODS 类型/属性约束 + 生成的类型检查
        PolyOrContainer、IntOrComplex、AllTypesMatch
                    │
                    ▼
第二层：trait 的 verifyTrait
        Has32BitArguments
                    │
                    ▼
第三层：op 自己的 verify()
        EvalOp::verify()
```

### 3.1 第一层：ODS 约束

`Poly_EvalOp` 的定义（`lib/Dialect/Poly/PolyOps.td:57`）：

```tablegen
def Poly_EvalOp : Op<Poly_Dialect, "eval",
    [AllTypesMatch<["point", "output"]>, Has32BitArguments]> {
  let summary = "Evaluates a Polynomial at a given input value.";
  let arguments = (ins Polynomial:$polynomial, IntOrComplex:$point);
  let results = (outs IntOrComplex:$output);
  let assemblyFormat = "$polynomial `,` $point attr-dict
      `:` `(` qualified(type($polynomial)) `,` type($point) `)`
      `->` type($output)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}
```

其中：

- `Polynomial`：第一个操作数必须是 `!poly.poly<N>`
- `IntOrComplex`：point 必须是整数或复数
- `AllTypesMatch<["point", "output"]>`：point 和 output 类型必须相同
- `hasVerifier = 1`：生成 `EvalOp::verify()` 声明

ODS 会生成 `verifyInvariantsImpl()`，其中包含类型约束和 `AllTypesMatch` 检查。
例如 `point` 是 `i32`、`output` 是 `i64` 时，会得到：

```text
error: 'poly.eval' op failed to verify that all of {point, output} have same type
```

#### `SameOperandsAndResultType` 也是同类检查

`Poly_BinOp` 使用：

```tablegen
class Poly_BinOp<string mnemonic>
    : Op<Poly_Dialect, mnemonic,
         [Pure, ElementwiseMappable, SameOperandsAndResultType]> { ... }
```

如果绕过 pretty 语法，用 generic form 写出类型不一致的 `poly.add`：

```mlir
func.func @add_mismatch(%p: !poly.poly<10>, %q: !poly.poly<5>)
    -> !poly.poly<10> {
  %r = "poly.add"(%p, %q)
       : (!poly.poly<10>, !poly.poly<5>) -> !poly.poly<10>
  return %r : !poly.poly<10>
}
```

会得到：

```text
error: 'poly.add' op requires the same type for all operands and results
```

这来自 `SameOperandsAndResultType` 生成的 trait verifier。

### 3.2 第二层：trait 的 `verifyTrait`

`Has32BitArguments` 是一个自定义 native trait，定义在 `PolyOps.td:16`：

```tablegen
def Has32BitArguments : NativeOpTrait<"Has32BitArguments"> {
  let cppNamespace = "::mlir::tutorial::poly";
}
```

它让生成的 `EvalOp` 类多继承一个 C++ trait：

```cpp
class EvalOp : public ::mlir::Op<
    EvalOp,
    ...,
    ::mlir::OpTrait::OpInvariants,
    ::mlir::tutorial::poly::Has32BitArguments,
    ...> { ... };
```

C++ 实现位于 `lib/Dialect/Poly/PolyTraits.h:9`：

```cpp
template <typename ConcreteType>
class Has32BitArguments
    : public OpTrait::TraitBase<ConcreteType, Has32BitArguments> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    for (auto type : op->getOperandTypes()) {
      // OK to skip non-integer operand types
      if (!type.isIntOrIndex()) continue;

      if (!type.isInteger(32)) {
        return op->emitOpError()
               << "requires each numeric operand to be a 32-bit integer";
      }
    }

    return success();
  }
};
```

逐点理解：

- `verifyTrait(Operation *op)` 是静态函数，拿到的是**通用 Operation**
- 因此它不能直接调用 `getPoint()`；只能遍历 `getOperandTypes()`
- 非整数类型（例如 `!poly.poly<10>`、`complex<f64>`）直接跳过
- 整数类型必须是 32 位

这个 trait 是通用的：任何 op 加上 `Has32BitArguments`，都会检查它的整数操作数。

### 3.3 第三层：op 自己的 `verify()`

`hasVerifier = 1` 生成：

```cpp
LogicalResult EvalOp::verify();
```

实现位于 `lib/Dialect/Poly/PolyOps.cpp:68`：

```cpp
LogicalResult EvalOp::verify() {
  auto pointTy = getPoint().getType();
  bool isSignlessInteger = pointTy.isSignlessInteger(32);
  auto complexPt = dyn_cast<ComplexType>(pointTy);
  return isSignlessInteger || complexPt
             ? success()
             : emitOpError("argument point must be a 32-bit "
                           "integer, or a complex number");
}
```

它比 trait 更具体：

- `isSignlessInteger(32)`：必须是 **signless** 的 32 位整数
- 或者是任意 `ComplexType`
- 否则 `emitOpError` 发出诊断

注意 `isInteger(32)` 和 `isSignlessInteger(32)` 的差别：

- `isInteger(32)`：位宽是 32，允许 signed/unsigned/signless
- `isSignlessInteger(32)`：位宽是 32 且必须是 signless

所以 `si32` 能通过 trait，但会被 op verifier 拒绝。这正是分层验证的意义：
trait 检查通用规则，op verifier 检查更具体的语义。

## 4. 验证顺序

生成的 `EvalOp` 类大致是：

```cpp
class EvalOp : public ::mlir::Op<
    EvalOp,
    ...,
    ::mlir::OpTrait::OpInvariants,
    ::mlir::tutorial::poly::Has32BitArguments,
    ...> { ... };
```

`Op::verifyInvariants` 的调度逻辑（简化自 `OpDefinition.h`）：

```cpp
return failure(
    failed(verifyTraits<Traits<ConcreteType>...>(op)) ||
    failed(cast<ConcreteType>(op).verify()));
```

而 `verifyTraits` 会按 trait 列表顺序调用每个 trait 的 `verifyTrait`：

```cpp
return success((succeeded(verifyTrait<Ts>(op)) && ...));
```

其中 `OpInvariants` 这个 trait 会调用生成的 `verifyInvariantsImpl()`；
它负责 ODS 类型约束和 `AllTypesMatch`。

因此顺序是：

```text
1. ODS 类型约束 / AllTypesMatch   (verifyInvariantsImpl)
2. Has32BitArguments              (verifyTrait)
3. EvalOp::verify()               (op-specific verify)
```

并且是短路求值：前一层失败，后面的就不会执行。

### 为什么 i64 只看到 trait 的错误？

先设置路径（后续实验都用同一组变量）：

```bash
cd /home/zhouzhou/npu/mlir-tutorial
export TOP=$PWD
export BIN=$TOP/externals/llvm-project/build/bin
export TO=$TOP/build/tools/tutorial-opt

$TO tests/poly_verifier.mlir 2>&1
```

输出：

```text
tests/poly_verifier.mlir:8:8: error: 'poly.eval' op requires each numeric operand to be a 32-bit integer
  %0 = poly.eval %arg0, %cst : (!poly.poly<10>, i64) -> i64
       ^
```

这里只出现了 `Has32BitArguments` 的错误，没有出现 `EvalOp::verify` 的错误。
因为 trait 先失败并短路，op verifier 没有再执行。

`tests/poly_verifier.mlir` 的 CHECK：

```mlir
// CHECK: to be a 32-bit integer
```

它只匹配 `to be a 32-bit integer` 这个片段，所以 trait 和 op verifier 的消息
都能匹配。测试注释也说明这是“有点脆弱”的写法。

## 5. 测试错误诊断

> FileCheck 的完整语法见 [FileCheck 使用指南](filecheck.md)。

### 5.1 FileCheck 检查 stderr

`tests/poly_verifier.mlir`：

```mlir
// RUN: tutorial-opt %s 2>%t; FileCheck %s < %t

func.func @test_invalid_evalop(%arg0: !poly.poly<10>, %cst: i64) -> i64 {
  // CHECK: to be a 32-bit integer
  %0 = poly.eval %arg0, %cst : (!poly.poly<10>, i64) -> i64
  return %0 : i64
}
```

`2>%t` 把 stderr 重定向到临时文件，因为 verifier 的错误是诊断信息，不是正常
stdout。然后 FileCheck 检查这个文件。

运行：

```bash
$TO tests/poly_verifier.mlir 2>&1 | $BIN/FileCheck tests/poly_verifier.mlir
```

无输出、退出码 0，表示通过。

### 5.2 `-verify-diagnostics`

另一种写法是把期望的错误直接写在输入文件里：

```mlir
func.func @eval_i64(%p: !poly.poly<10>, %x: i64) -> i64 {
  // expected-error @+1 {{requires each numeric operand to be a 32-bit integer}}
  %r = poly.eval %p, %x : (!poly.poly<10>, i64) -> i64
  return %r : i64
}
```

运行：

```bash
$TO build/scratch/poly_verifier_diag.mlir -verify-diagnostics
```

退出码 0 表示“期望的错误都出现了，且没有多余错误”。

`expected-error @+1` 表示错误出现在下一行；`{{...}}` 是精确匹配的消息模式。

### 5.3 两种测试方式对比

| 方式 | 适合 |
|---|---|
| `2>%t` + FileCheck | 需要灵活匹配片段、检查多行输出 |
| `-verify-diagnostics` | 错误位置、消息、数量都要精确匹配 |

## 6. 实验

### 6.1 现有 verifier 测试

```bash
$TO tests/poly_verifier.mlir 2>&1
```

输出见 4 节。`i64` 被 `Has32BitArguments` 拒绝。

### 6.2 自定义 verifier 的真正作用：`si32`

`si32` 是 32 位有符号整数：

- `Has32BitArguments` 的 `isInteger(32)` 为真，trait 通过
- `EvalOp::verify` 的 `isSignlessInteger(32)` 为假，custom verifier 拒绝

输入：

```mlir
func.func @eval_si32(%p: !poly.poly<10>, %x: si32) -> si32 {
  %r = poly.eval %p, %x : (!poly.poly<10>, si32) -> si32
  return %r : si32
}
```

运行：

```bash
$TO build/scratch/poly_verifier_si32.mlir 2>&1
```

输出：

```text
error: 'poly.eval' op argument point must be a 32-bit integer, or a complex number
```

这次没有 trait 错误，因为 trait 已经通过了；错误来自 op 级 verifier。
这是本课最重要的实验：它证明两层 verifier 都在工作，而且顺序正确。

### 6.3 complex 类型

```mlir
func.func @eval_complex(%p: !poly.poly<10>, %z: complex<f64>)
    -> complex<f64> {
  %r = poly.eval %p, %z : (!poly.poly<10>, complex<f64>) -> complex<f64>
  return %r : complex<f64>
}
```

这个 op 是合法的：

- `Has32BitArguments` 只检查整数，跳过 `complex<f64>`
- `EvalOp::verify` 接受任意 `ComplexType`

当前实现也接受 `complex<f32>`，因为代码只检查 `dyn_cast<ComplexType>`，没有限制
元素类型。如果业务上只允许 `complex<f64>`，可以在 `EvalOp::verify` 里再加检查。

### 6.4 `AllTypesMatch` 失败

```mlir
func.func @eval_mismatch(%p: !poly.poly<10>, %x: i32) -> i64 {
  %r = poly.eval %p, %x : (!poly.poly<10>, i32) -> i64
  return %r : i64
}
```

运行：

```bash
$TO build/scratch/poly_verifier_cases.mlir 2>&1
```

会看到：

```text
error: 'poly.eval' op failed to verify that all of {point, output} have same type
```

这来自 `AllTypesMatch<["point", "output"]>` 生成的检查。

### 6.5 `SameOperandsAndResultType` 失败

用 generic form 写一个类型不一致的 `poly.add`：

```mlir
func.func @add_mismatch(%p: !poly.poly<10>, %q: !poly.poly<5>)
    -> !poly.poly<10> {
  %r = "poly.add"(%p, %q)
       : (!poly.poly<10>, !poly.poly<5>) -> !poly.poly<10>
  return %r : !poly.poly<10>
}
```

运行：

```bash
$TO build/scratch/poly_verifier_add_generic.mlir 2>&1
```

输出：

```text
error: 'poly.add' op requires the same type for all operands and results
```

这是 `SameOperandsAndResultType` 生成的 trait verifier。

## 7. 源码位置

| 内容 | 文件 |
|---|---|
| `EvalOp` 的 ODS 定义、`AllTypesMatch`、`hasVerifier` | `lib/Dialect/Poly/PolyOps.td:57` |
| 自定义 trait `Has32BitArguments` | `lib/Dialect/Poly/PolyTraits.h:9` |
| `Has32BitArguments` 的 TableGen 声明 | `lib/Dialect/Poly/PolyOps.td:16` |
| `EvalOp::verify()` | `lib/Dialect/Poly/PolyOps.cpp:68` |
| 生成的 `verifyInvariantsImpl` / `verifyInvariants` | `build/lib/Dialect/Poly/PolyOps.cpp.inc` |
| 现有测试 | `tests/poly_verifier.mlir` |

## 8. 常见问题

### 8.1 为什么 `return failure()` 不够？

```cpp
// 不推荐
return failure();
```

这样 verifier 会失败，但不会输出任何诊断，用户只看到“失败”而没有原因。
应该用：

```cpp
return emitOpError("...");
```

`emitOpError` 会返回一个可转换为 `failure()` 的诊断对象。

### 8.2 为什么不能把 trait 写成只适用于 EvalOp？

trait 的签名是：

```cpp
static LogicalResult verifyTrait(Operation *op)
```

它拿到的是通用 `Operation*`，不知道具体 op 的命名访问器。若在 trait 里
`cast<EvalOp>(op).getPoint()`，这个 trait 就只能用于 EvalOp，失去通用性。
通用规则放 trait，具体规则放 op 的 `verify()`。

### 8.3 `isInteger(32)` 和 `isSignlessInteger(32)` 有什么区别？

- `isInteger(32)`：位宽 32，signed/unsigned/signless 都算
- `isSignlessInteger(32)`：位宽 32 且必须是 signless

`si32` 通过前者，不通过后者。这正是 6.2 实验的基础。

### 8.4 为什么 i64 只报 trait 错误？

因为验证顺序是：

```text
ODS -> trait -> op verify
```

trait 先失败并短路，后面的 `EvalOp::verify` 不会运行。

### 8.5 verifier 可以修改 IR 吗？

不应该。verifier 的职责是检查，不是修复。需要修复 IR 的逻辑应该放在
canonicalizer、rewrite pattern 或专门的 pass 里。

## 9. 练习

### 练习 1：给 `poly.constant` 增加 verifier

目前 `poly.constant` 接受任意形状的整数张量：

```mlir
%c = poly.constant dense<[[1, 2], [3, 4]]>
     : tensor<2x2xi32> : !poly.poly<10>
```

这在语法上能通过，但多项式系数应该是一维张量。请：

1. 在 `Poly_ConstantOp` 中加 `let hasVerifier = 1;`
2. 实现 `ConstantOp::verify()`，要求 `coefficients` 的 rank 为 1
3. 用 `tutorial-opt` 验证 2-D 系数被拒绝

<details>
<summary>参考答案</summary>

`.td`：

```tablegen
def Poly_ConstantOp : Op<Poly_Dialect, "constant",
                          [Pure, ConstantLike]> {
  ...
  let hasFolder = 1;
  let hasVerifier = 1;   // 新增
}
```

`PolyOps.cpp`：

```cpp
LogicalResult ConstantOp::verify() {
  auto coefficients = getCoefficients();
  if (coefficients.getType().getRank() != 1)
    return emitOpError("coefficients must be a 1-D tensor");
  return success();
}
```

修改 `.td` 和 `.cpp` 后必须重新构建，否则运行的还是旧的
`tutorial-opt`，新 verifier 不会生效：

```bash
ninja -C build -j 100 tutorial-opt
```

`.td` 改动会触发 TableGen 重新生成 `PolyOps.cpp.inc` / `PolyOps.h.inc`，
CMake/Ninja 会自动处理这个依赖。

测试：

```mlir
func.func @bad_constant() -> !poly.poly<10> {
  %c = poly.constant dense<[[1, 2], [3, 4]]>
       : tensor<2x2xi32> : !poly.poly<10>
  return %c : !poly.poly<10>
}
```

运行：

```bash
$TO build/scratch/poly_constant_2d.mlir 2>&1
```

应看到：

```text
error: 'poly.constant' op coefficients must be a 1-D tensor
```

</details>

### 练习 2：解释为什么 `si32` 报的是另一条错误

请按验证顺序解释：

- `i64` 为什么报 `Has32BitArguments` 的错误
- `si32` 为什么报 `EvalOp::verify` 的错误

<details>
<summary>参考答案</summary>

- `i64`：`isInteger(32)` 为假，trait 失败并短路，op verifier 不执行。
- `si32`：`isInteger(32)` 为真，trait 通过；随后 `EvalOp::verify` 的
  `isSignlessInteger(32)` 为假，op verifier 发出自己的错误。

这说明验证顺序是 ODS -> trait -> op verify，且前一层失败会阻止后一层执行。

</details>

### 练习 3：只允许 `complex<f64>`

当前 `EvalOp::verify` 接受任意 `ComplexType`。如果要限制为 `complex<f64>`，应该怎么改？

<details>
<summary>参考答案</summary>

```cpp
auto complexPt = dyn_cast<ComplexType>(pointTy);
bool isComplexF64 = complexPt && complexPt.getElementType().isF64();
```

然后：

```cpp
return isSignlessInteger || isComplexF64
           ? success()
           : emitOpError("argument point must be a 32-bit integer, "
                         "or a complex<f64>");
```

测试 `complex<f32>` 应被拒绝，`complex<f64>` 仍然通过。

</details>

## 10. 验收清单

- [ ] 能解释 verifier 和 folder/canonicalizer 的区别
- [ ] 能说出三层验证及执行顺序
- [ ] 能解释 `AllTypesMatch` 生成什么检查
- [ ] 能读懂 `Has32BitArguments::verifyTrait`
- [ ] 能读懂 `EvalOp::verify`
- [ ] 能解释 `isInteger(32)` 与 `isSignlessInteger(32)` 的差别
- [ ] 能解释 `i64` 和 `si32` 为什么报不同错误
- [ ] 能用 `2>%t` + FileCheck 测试诊断
- [ ] 能用 `-verify-diagnostics` 测试诊断
- [ ] 能写一个 op 级 verifier

## 11. 下一课

**第 8 课：Canonicalizers and Declarative Rewrite Patterns**

- `getCanonicalizationPatterns`
- TableGen 声明式 pattern：`PolyPatterns.td`
- `DifferenceOfSquares`、`LiftConjThroughEval`
- 测试：`tests/poly_canonicalize.mlir`

补充阅读：[Pat 与 Pattern 语法](patterns.md)。

对应源码：`lib/Dialect/Poly/PolyPatterns.td`、`lib/Dialect/Poly/PolyOps.cpp`。

## 参考

- 本课对应文章：<https://jeremykun.com/2023/09/13/mlir-verifiers/>
- MLIR 官方文档：Verifying Operations
  <https://mlir.llvm.org/docs/DefiningDialects/Operations/#verification>
- 本课测试：`tests/poly_verifier.mlir`
- 相关源码：
  - `lib/Dialect/Poly/PolyOps.td`
  - `lib/Dialect/Poly/PolyOps.cpp`
  - `lib/Dialect/Poly/PolyTraits.h`