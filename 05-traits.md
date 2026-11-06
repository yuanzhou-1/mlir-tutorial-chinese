# 第 5 课：Using Traits

> 对应文章：[Using Traits](https://jeremykun.com/2023/09/07/mlir-using-traits/)。
>
> 本课解释 MLIR 的 trait 机制：它是 op 的**编译期属性/能力标记**，能让通用 pass
> （LICM、control-flow-sink、CSE、DCE、SCCP 等）知道哪些操作是安全的、可移动的、
> 可折叠的。我们还会实现一个自定义 native trait `Has32BitArguments`。

## 1. 学习目标

- 理解 trait 是什么，以及 trait / interface / verifier / attribute 的区别
- 理解 `Pure` 如何让 LICM、control-flow-sink、CSE、DCE 生效
- 理解 `SameOperandsAndResultType`、`AllTypesMatch`、`ElementwiseMappable`
- 理解 `ConstantLike` 与常量传播的关系
- 能读懂并实现一个自定义 trait `Has32BitArguments`
- 能运行并解释 LICM、control-flow-sink、CSE、SCCP 的实验结果

## 2. Trait 是什么

Trait 是附加在 op 上的**编译期属性/能力标记**：

```tablegen
class Poly_BinOp<string mnemonic>
    : Op<Poly_Dialect, mnemonic,
         [Pure, ElementwiseMappable, SameOperandsAndResultType]> { ... }
```

这些名字会变成生成 C++ 类继承的 trait 基类/接口：

```cpp
class AddOp
  : public ::mlir::Op<AddOp,
      ...,
      ::mlir::ConditionallySpeculatable::Trait,
      ::mlir::OpTrait::AlwaysSpeculatableImplTrait,
      ::mlir::MemoryEffectOpInterface::Trait,
      ::mlir::OpTrait::Elementwise,
      ::mlir::OpTrait::Scalarizable,
      ::mlir::OpTrait::Vectorizable,
      ::mlir::OpTrait::Tensorizable,
      ::mlir::OpTrait::SameOperandsAndResultType,
      ...> { ... };
```

#### 这个类声明怎么读

```cpp
class AddOp : public ::mlir::Op<AddOp, Trait1, Trait2, ...> { ... };
```

- `::mlir::Op<...>` 是所有 MLIR op 的 **CRTP 基类**：
  - 第一个参数 `AddOp` 是具体 op 类型
  - 后面的参数是 trait/interface，基类会多重继承它们
- TableGen 里的 `[Pure, ElementwiseMappable, SameOperandsAndResultType]`
  被展开成多个 C++ trait
- ODS 还会自动加一些结构 trait

| 生成的 trait | 来源 | 作用 |
|---|---|---|
| `ConditionallySpeculatable::Trait` | `Pure` | 可推测执行接口 |
| `AlwaysSpeculatableImplTrait` | `Pure` | 实现“总是可推测执行” |
| `MemoryEffectOpInterface::Trait` | `Pure` | 内存副作用接口 |
| `Elementwise` | `ElementwiseMappable` | 逐元素语义标记 |
| `Scalarizable` | `ElementwiseMappable` | 可标量化 |
| `Vectorizable` | `ElementwiseMappable` | 可向量化 |
| `Tensorizable` | `ElementwiseMappable` | 可张量化 |
| `SameOperandsAndResultType` | 显式声明 | 操作数/结果类型一致 |
| `InferTypeOpInterface::Trait` | `SameOperandsAndResultType` | 支持返回类型推导 |
| `ZeroRegions`、`OneResult`、`OneTypedResult`、`ZeroSuccessors`、`NOperands<2>::Impl`、`OpInvariants` | ODS 自动生成 | 结构约束 |

要点：

- trait 不是运行时数据，而是**类型层面的能力声明**
- 通用 pass 通过 trait/interface 查询 op 的性质，而不是针对每个 op 写特判：

  ```cpp
  op->hasTrait<mlir::OpTrait::Elementwise>();
  dyn_cast<mlir::MemoryEffectOpInterface>(op);
  mlir::isSpeculatable(op);
  ```

- 同一个 trait 可以同时被很多 op 使用
- 你不需要手写这个类；它由 `mlir-tblgen` 从 `.td` 生成

### Trait vs Interface vs Verifier vs Attribute

| 概念 | 作用 | 例子 |
|---|---|---|
| Trait | 编译期属性/能力标记，可能带生成代码或 `verifyTrait` | `Pure`、`SameOperandsAndResultType` |
| Interface | op 实现的一组 API 契约 | `MemoryEffectOpInterface`、`InferTypeOpInterface` |
| Verifier | 运行时（编译期校验 IR 时）检查具体规则 | `EvalOp::verify()` |
| Attribute | IR 实例上的数据 | `{tag = "x"}`、`coefficients` |

trait 常常会“蕴含”某些 interface。例如 `Pure` 会生成：

- `MemoryEffectOpInterface`：声明没有内存副作用
- `ConditionallySpeculatable` / `AlwaysSpeculatable`：声明可以安全地推测执行

## 3. `Pure`：最重要的 trait

`Pure` 表示：

> 这个 op 没有副作用，并且总是可以安全地推测执行（speculatable）。

生成代码里可以看到：

```cpp
::mlir::ConditionallySpeculatable::Trait
::mlir::OpTrait::AlwaysSpeculatableImplTrait
::mlir::MemoryEffectOpInterface::Trait
```

以及一个空的 `getEffects`：

```cpp
void AddOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
}
```

空 effects 的意思是“不读写任何内存”。

### `Pure` 让哪些优化生效

| 优化 | 为什么需要 `Pure` |
|---|---|
| **LICM**（循环不变量代码外提） | 只有无副作用、可推测执行的 op 才能安全地移出循环 |
| **control-flow-sink** | 只有无副作用的 op 才能安全地下沉到分支中 |
| **CSE**（公共子表达式消除） | 无副作用的重复计算可以合并 |
| **DCE**（死代码消除） | 无副作用的未使用 op 可以删除 |
| **SCCP**（稀疏条件常量传播） | 常量和纯操作可以被折叠/传播 |

### 实验 1：LICM

```bash
$TO tests/code_motion.mlir --loop-invariant-code-motion
```

关键变化：

```mlir
// 之前：poly.mul 在循环体内
%ret_val = affine.for %i = 0 to 100 iter_args(%sum_iter = %p0) -> !poly.poly<10> {
  %2 = poly.mul %p0, %p1 : !poly.poly<10>
  %sum_next = poly.add %sum_iter, %2 : !poly.poly<10>
  affine.yield %sum_next : !poly.poly<10>
}

// 之后：poly.mul 被提到循环外
%2 = poly.mul %0, %1 : !poly.poly<10>
%3 = affine.for %arg0 = 0 to 100 iter_args(%arg1 = %0) -> (!poly.poly<10>) {
  %4 = poly.add %arg1, %2 : !poly.poly<10>
  affine.yield %4 : !poly.poly<10>
}
```

为什么能提出来？

1. `%p0`、`%p1` 定义在循环外 → 循环不变量
2. `poly.mul` 有 `Pure` → 无副作用、可推测执行
3. LICM 因此可以安全地把它移到循环外

### 实验 2：control-flow-sink

```bash
$TO tests/control_flow_sink.mlir -control-flow-sink
```

结果：

```mlir
%0 = scf.if %arg0 -> (!poly.poly<10>) {
  %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %1 = poly.from_tensor %cst : tensor<3xi32> -> !poly.poly<10>
  %2 = poly.mul %1, %1 : !poly.poly<10>
  scf.yield %2 : !poly.poly<10>
} else {
  %cst = arith.constant dense<[9, 8, 16]> : tensor<3xi32>
  %1 = poly.from_tensor %cst : tensor<3xi32> -> !poly.poly<10>
  %2 = poly.mul %1, %1 : !poly.poly<10>
  scf.yield %2 : !poly.poly<10>
}
```

`poly.from_tensor` 原本定义在 `scf.if` 之前，被**下沉**到两个分支里。
这也是因为它是 `Pure`，下沉不会改变程序行为。

### 实验 3：CSE

```bash
$TO tests/cse.mlir -cse
```

两个相同的 `poly.mul %p0, %p0` 被合并成一个，后续 add 复用同一个结果。
`Pure` 让 CSE 能确认重复计算没有副作用。

### 实验 4：SCCP

```bash
$TO tests/sccp.mlir -pass-pipeline='builtin.module(func.func(sccp))'
```

输出：

```mlir
%0 = poly.constant dense<[2, 8, 20, 24, 18]> : tensor<5xi32> : !poly.poly<10>
%1 = poly.constant dense<[1, 4, 10, 12, 9]> : tensor<5xi32> : !poly.poly<10>
%2 = poly.constant dense<[1, 2, 3]> : tensor<3xi32> : !poly.poly<10>
return %1 : !poly.poly<10>
```

`poly.mul` 和 `poly.add` 被折叠成 `poly.constant`。注意 SCCP **不会删除死代码**：
`%0`（add 的结果）没有被返回，但仍然保留；这是测试注释特意说明的行为。

SCCP 依赖的是 `ConstantLike` + folder，而不仅是 `Pure`；folder 的细节是第 6 课。

## 4. 其他内置 trait

### `SameOperandsAndResultType`

```tablegen
class Poly_BinOp<...> : Op<..., [Pure, ElementwiseMappable, SameOperandsAndResultType]>
```

它要求所有操作数和结果类型完全一致：

```mlir
// 合法
%0 = poly.add %a, %b : !poly.poly<10>

// 非法
%0 = poly.add %a, %b : (!poly.poly<10>, !poly.poly<9>) -> !poly.poly<10>
```

### `AllTypesMatch<["point", "output"]>`

只要求指定的一组值类型相同：

```tablegen
def Poly_EvalOp : Op<..., [AllTypesMatch<["point", "output"]>, Has32BitArguments]>
```

`poly.eval` 的多项式类型可以和 point/output 不同：

```mlir
%r = poly.eval %p, %x : (!poly.poly<10>, i32) -> i32
```

但 `point` 和 `output` 必须都是 `i32`（或者都是 `complex<f64>`）。

### `ElementwiseMappable`

它是四个 trait 的组合：

```text
Elementwise + Scalarizable + Vectorizable + Tensorizable
```

含义：

- 对标量定义的操作可以逐元素应用到 vector/tensor
- 所有容器形状必须一致
- **不允许 broadcasting**

当前仓库同时有 `SameOperandsAndResultType`，所以 tensor + scalar 也会被拒绝。

### `ConstantLike`

要求：

- 0 个操作数
- 1 个结果

它标记“这个 op 产生常量”，配合 folder 和 `materializeConstant` 参与常量折叠
（第 6 课）。

## 5. 自定义 native trait：`Has32BitArguments`

### 5.1 ODS 声明

```tablegen
def Has32BitArguments : NativeOpTrait<"Has32BitArguments"> {
  let cppNamespace = "::mlir::tutorial::poly";
}
```

`NativeOpTrait` 表示这个 trait 的 C++ 实现在手写代码里，不是 ODS 自动生成的。

### 5.2 C++ 实现

`lib/Dialect/Poly/PolyTraits.h`：

```cpp
template <typename ConcreteType>
class Has32BitArguments
    : public OpTrait::TraitBase<ConcreteType, Has32BitArguments> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    for (auto type : op->getOperandTypes()) {
      // 跳过非整数类型（例如 poly、complex）
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

要点：

- `TraitBase<ConcreteType, Has32BitArguments>` 是 CRTP
- `verifyTrait` 会在 op 校验时被调用
- 遍历所有操作数，跳过非整数类型
- 整数必须是 32 位，否则报错

### 5.3 实验 5：trait verifier

```bash
$TO tests/poly_verifier.mlir 2>&1 | head -3
```

输出：

```text
error: 'poly.eval' op requires each numeric operand to be a 32-bit integer
```

`poly.eval` 的 point 是 `i64`，被 trait 拒绝。

注意 trait 与 op verifier 的分工：

- `Has32BitArguments`：检查所有整数类操作数都是 32 位
- `EvalOp::verify()`：进一步要求 point 是 signless i32 或 complex

两层都会执行；测试注释里也说明二者都可能触发。

## 6. 实验汇总

先设置路径：

```bash
cd /home/zhouzhou/npu/mlir-tutorial
BIN=$PWD/externals/llvm-project/build/bin
TO=$PWD/build/tools/tutorial-opt
```

上面实验 1–5 的完整命令：

```bash
# 1) LICM：poly.mul 外提
$TO tests/code_motion.mlir --loop-invariant-code-motion

# 2) sink：poly.from_tensor 下沉到分支
$TO tests/control_flow_sink.mlir -control-flow-sink

# 3) CSE：重复 poly.mul 合并
$TO tests/cse.mlir -cse

# 4) SCCP：poly.mul/add 变成 poly.constant
$TO tests/sccp.mlir -pass-pipeline='builtin.module(func.func(sccp))'

# 5) trait verifier：i64 被拒绝
$TO tests/poly_verifier.mlir 2>&1 | head -3
```

## 7. 源码位置

| 内容 | 文件 |
|---|---|
| trait 声明 | `lib/Dialect/Poly/PolyOps.td` |
| 自定义 trait 实现 | `lib/Dialect/Poly/PolyTraits.h` |
| 生成的 trait 基类 | `build/lib/Dialect/Poly/PolyOps.h.inc` |
| LICM 测试 | `tests/code_motion.mlir` |
| sink 测试 | `tests/control_flow_sink.mlir` |
| CSE 测试 | `tests/cse.mlir` |
| SCCP 测试 | `tests/sccp.mlir` |
| trait verifier 测试 | `tests/poly_verifier.mlir` |

## 8. 练习

### 练习 1：为什么能外提？

`poly.mul` 被 LICM 移出循环，依赖哪几个条件？如果去掉 `Pure` 会怎样？

### 练习 2：为什么能下沉？

`poly.from_tensor` 为什么能被 control-flow-sink 移到 `scf.if` 的两个分支里？

### 练习 3：trait 与 verifier 的分工

`Has32BitArguments` 和 `EvalOp::verify()` 各自检查什么？为什么 `complex<f64>`
能通过？

### 练习 4：两种类型约束

`SameOperandsAndResultType` 和 `AllTypesMatch<["point", "output"]>` 有什么区别？
各举一个例子。

### 练习 5：ElementwiseMappable

为什么 `poly.add` 能作用于 `tensor<2x!poly.poly<10>>`，却不能作用于
`tensor + scalar`？

## 9. 参考答案

<details>
<summary>练习 1</summary>

条件：操作数在循环外定义（循环不变量），并且 op 是 `Pure`（无副作用、可推测
执行）。去掉 `Pure` 后，编译器不能假设它没有副作用，LICM 会保守地不移动它。

</details>

<details>
<summary>练习 2</summary>

`poly.from_tensor` 是 `Pure`，下沉到分支不会改变语义；它的结果只在对应分支中被
使用，因此可以移动到使用点附近。

</details>

<details>
<summary>练习 3</summary>

`Has32BitArguments` 遍历所有操作数，跳过非整数类型，要求整数是 32 位；
`EvalOp::verify()` 再检查 point 必须是 signless i32 或 complex。`complex<f64>`
不是整数，trait 跳过它；verifier 的 `dyn_cast<ComplexType>` 接受它。

</details>

<details>
<summary>练习 4</summary>

`SameOperandsAndResultType` 要求所有操作数和结果类型完全一致，例如
`poly.add` 的两个 poly 和结果必须同类型。`AllTypesMatch` 只要求指定的一组值一致，
例如 `poly.eval` 只要求 point 和 output 一致，polynomial 可以是不同类型。

</details>

<details>
<summary>练习 5</summary>

`ElementwiseMappable` 允许容器类型，但要求容器形状一致且不允许 broadcasting；
`SameOperandsAndResultType` 进一步要求 tensor + scalar 类型完全一致，所以
tensor + scalar 被拒绝。

</details>

## 10. 验收清单

- [ ] 能说出 trait 是什么，以及它与 interface/verifier/attribute 的区别
- [ ] 能解释 `Pure` 生成了哪些 interface/trait
- [ ] 能解释 LICM、sink、CSE、DCE 为什么依赖 `Pure`
- [ ] 能区分 `SameOperandsAndResultType` 和 `AllTypesMatch`
- [ ] 能解释 `ElementwiseMappable` 的四个组成 trait
- [ ] 能读懂并实现 `Has32BitArguments` 这样的 native trait
- [ ] 能运行 LICM/sink/CSE/SCCP 实验并解释结果

## 下一课

**第 6 课：Folders and Constant Propagation**：[06-folders.md](06-folders.md)

- `hasFolder = 1` 与 `Op::fold`
- `OpFoldResult` 的三种返回值
- `ConstantLike` + folder + `materializeConstant` 的循环
- `poly.add/mul/from_tensor/constant` 的折叠
- SCCP 与 canonicalize/CSE 如何利用 folder

对应源码：`lib/Dialect/Poly/PolyOps.cpp`、`tests/poly_canonicalize.mlir`