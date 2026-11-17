# Dialect Conversion 的类型边界辅助函数

> 本文是第 9 课 [Dialect Conversion](09-dialect-conversion.md) 的补充。
>
> `ConvertAdd`、`ConvertMul` 这些 pattern 只负责单个 `poly.*` op。
> 但函数签名、`return`、`call` 和分支也会携带 `!poly.poly<N>` 类型，
> 它们需要单独处理，否则会留下无法消除的 `unrealized_conversion_cast`。

## 1. 问题：op 转换了，边界没转换

假设所有 `poly.*` op 都已经被转换：

```mlir
func.func @f(%a: !poly.poly<10>) -> !poly.poly<10> {
  %b = poly.add %a, %a : !poly.poly<10>
  return %b : !poly.poly<10>
}
```

`poly.add` 会变成：

```mlir
%t = arith.addi %a_tensor, %a_tensor : tensor<10xi32>
```

但函数本身仍然是：

```mlir
func.func @f(%a: !poly.poly<10>) -> !poly.poly<10>
```

于是：

- 函数参数类型还是 `!poly.poly<10>`
- 函数返回值类型还是 `!poly.poly<10>`
- `return` 的操作数类型也要跟着改

只转换 op 不转换这些“类型边界”，IR 仍然不合法。

---

## 2. 四个 helper 的分工

| helper | 处理的边界 |
|---|---|
| `populateFunctionOpInterfaceTypeConversionPattern` | 函数签名、入口 block 参数 |
| `populateReturnOpTypeConversionPattern` | `func.return` 的操作数 |
| `populateCallOpTypeConversionPattern` | `func.call` 的操作数和结果类型 |
| `populateBranchOpInterfaceTypeConversionPattern` | 分支传给后继 block 的参数 |

`PolyToStandard.cpp` 中的用法：

```cpp
populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
    patterns, typeConverter);

populateReturnOpTypeConversionPattern(patterns, typeConverter);
populateCallOpTypeConversionPattern(patterns, typeConverter);
populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
```

它们做的事情不是“再写一个普通 rewrite”，而是：

> 给函数、返回、调用、分支这些结构 op 添加 conversion pattern，
> 让它们也遵守同一个 `TypeConverter`。

---

## 3. `populateFunctionOpInterfaceTypeConversionPattern`

```cpp
populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
    patterns, typeConverter);
```

### 3.1 它把什么改掉

输入：

```mlir
func.func @f(%a: !poly.poly<10>) -> !poly.poly<10> {
  ...
}
```

输出：

```mlir
func.func @f(%a: tensor<10xi32>) -> tensor<10xi32> {
  ...
}
```

同时入口 block 的参数类型也会更新：

```text
%a: !poly.poly<10>  ->  %a: tensor<10xi32>
```

### 3.2 它不改什么

- 不转换函数体里的 `poly.add` / `poly.mul`
- 不负责 `return` 的操作数
- 不负责 `call`

它只处理函数“外壳”：

```text
FunctionType 的输入/输出
+ 入口 block 的参数类型
```

### 3.3 适用对象

这个 helper 适用于实现 `FunctionOpInterface`、并且用 `FunctionType`
表示签名的 op。

`func::FuncOp` 是最常见的例子。

---

## 4. `populateReturnOpTypeConversionPattern`

```cpp
populateReturnOpTypeConversionPattern(patterns, typeConverter);
```

### 4.1 它把什么改掉

```mlir
%p = poly.add ... : !poly.poly<10>
return %p : !poly.poly<10>
```

转换成：

```mlir
%t = arith.addi ... : tensor<10xi32>
return %t : tensor<10xi32>
```

它会用转换后的操作数重建 `func.return`。

### 4.2 为什么需要它

函数签名虽然已经被函数 pattern 改成返回 `tensor<10xi32>`，
但 `return` 是另一个 op，仍然可能持有旧的 `!poly.poly<10>` 操作数。

如果不处理 `return`，函数声明和实际返回值的类型会对不上。

---

## 5. `populateCallOpTypeConversionPattern`

```cpp
populateCallOpTypeConversionPattern(patterns, typeConverter);
```

### 5.1 它把什么改掉

假设：

```mlir
func.func @callee(%a: !poly.poly<10>) -> !poly.poly<10> {
  ...
}

func.func @caller(%a: !poly.poly<10>) {
  %r = call @callee(%a)
       : (!poly.poly<10>) -> !poly.poly<10>
}
```

函数签名转换后：

```mlir
func.func @callee(%a: tensor<10xi32>) -> tensor<10xi32>
```

调用点也必须改成：

```mlir
func.func @caller(%a: tensor<10xi32>) {
  %r = call @callee(%a)
       : (tensor<10xi32>) -> tensor<10xi32>
}
```

否则 caller 还在按旧签名调用 callee。

### 5.2 它处理什么

- 转换 `call` 的操作数类型
- 转换 `call` 的结果类型
- 重建 `func.call`

它不负责改被调函数的函数体；那部分由函数签名 pattern 和函数体内 op 的
conversion pattern 负责。

---

## 6. `populateBranchOpInterfaceTypeConversionPattern`

```cpp
populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
```

### 6.1 它处理什么

所有实现 `BranchOpInterface` 的分支 op，例如：

- `cf.br`
- `cf.cond_br`
- `cf.switch`

分支 op 会把值传给后继 block：

```mlir
cf.br ^bb1(%p : !poly.poly<10>)
...
^bb1(%arg: !poly.poly<10>):
```

如果 block 参数类型变成 tensor：

```mlir
^bb1(%arg: tensor<10xi32>):
```

那么分支传进去的值也要变成 tensor：

```mlir
cf.br ^bb1(%t : tensor<10xi32>)
```

这个 helper 会找到分支的后继操作数，替换成转换后的值。

### 6.2 为什么只处理“后继操作数”

分支 op 可能有一些操作数不是传给后继 block 的，例如条件、switch 的 flag。
不能把分支的所有操作数都统一替换，只处理
`getSuccessorOperands(succIdx)` 里的 forwarded operands。

---

## 7. 动态合法性判断

这些 helper 只添加 pattern，还需要告诉 `ConversionTarget`：

> 什么样的函数/return/call 已经合法？

### 7.1 `func.func`

```cpp
target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
  return typeConverter.isSignatureLegal(op.getFunctionType()) &&
         typeConverter.isLegal(&op.getBody());
});
```

`isSignatureLegal` 检查：

```text
函数参数类型 + 返回值类型
是否都已经不需要转换
```

`isLegal(&op.getBody())` 检查：

```text
函数 region 里 block 的参数类型是否合法
```

### 7.2 `TypeConverter::isLegal`

核心定义是：

```cpp
bool TypeConverter::isLegal(Type type) const {
  return convertType(type) == type;
}
```

也就是：

> 转换后还是自己，才算合法。

例如：

```text
!poly.poly<10>  -> tensor<10xi32>   不合法
tensor<10xi32>  -> tensor<10xi32>   合法
i32             -> i32              合法
```

相关重载：

| 方法 | 检查什么 |
|---|---|
| `isLegal(Type)` | 单个类型 |
| `isLegal(TypeRange)` | 一组类型 |
| `isLegal(Operation *)` | 操作数和结果类型 |
| `isLegal(Region *)` | region 内 block 参数类型 |
| `isSignatureLegal(FunctionType)` | 函数输入/输出类型 |

### 7.3 `func.return`

```cpp
target.addDynamicallyLegalOp<func::ReturnOp>(
    [&](func::ReturnOp op) {
      return typeConverter.isLegal(op);
    });
```

`isLegal(op)` 检查 return 的操作数/结果类型。

### 7.4 `func.call`

```cpp
target.addDynamicallyLegalOp<func::CallOp>(
    [&](func::CallOp op) {
      return typeConverter.isLegal(op);
    });
```

call 的操作数和结果类型都合法，才跳过转换。

---

## 8. `markUnknownOpDynamicallyLegal`

代码中还有：

```cpp
target.markUnknownOpDynamicallyLegal([&](Operation *op) {
  return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
         isLegalForBranchOpInterfaceTypeConversionPattern(op, typeConverter) ||
         isLegalForReturnOpTypeConversionPattern(op, typeConverter);
});
```

这是兜底规则：

- 对不是分支、不是 return-like 的 op：直接视为合法
- 对分支 op：检查它的后继操作数是否合法
- 对 return-like op：检查 return 类型是否合法

它避免把所有未知 op（例如 `tensor`、`scf`）都误判成非法。

---

## 9. 整体流程

```text
ConversionTarget：
  arith 合法
  poly 非法
  func 动态合法（签名/block 参数合法才合法）
  return/call 动态合法
        │
        ▼
applyPartialConversion
        │
        ├─ ConvertAdd/ConvertSub/... 转换 poly op
        │
        ├─ FunctionOpInterfaceSignatureConversion
        │    更新函数签名和入口 block 参数
        │
        ├─ ReturnOpTypeConversion
        │    更新 return 操作数
        │
        ├─ CallOpSignatureConversion
        │    更新 call 的操作数/结果类型
        │
        └─ BranchOpInterfaceTypeConversion
             更新分支传给后继 block 的值
```

最终：

```text
函数参数/返回值：tensor<Nxi32>
poly op：消失
return/call/branch：全部使用新类型
```

没有 unresolved 的 `unrealized_conversion_cast`，pass 成功。

---

## 10. 如果漏掉这些 helper 会怎样

### 10.1 漏掉函数签名转换

只转换 `poly.add`，函数参数仍是 `!poly.poly<10>`：

```text
error: failed to legalize unresolved materialization
       from '!poly.poly<10>' to 'tensor<10xi32>'
```

### 10.2 漏掉 return 转换

函数签名变了，但 `return` 还在返回 `!poly.poly<10>`，类型不一致。

### 10.3 漏掉 call 转换

callee 签名变了，caller 还在按旧签名传 poly 值。

### 10.4 漏掉 branch 转换

block 参数类型变了，但分支还在传旧类型，后继 block 的参数类型对不上。

---

## 11. 一个完整例子

输入：

```mlir
func.func @callee(%a: !poly.poly<10>) -> !poly.poly<10> {
  %b = poly.add %a, %a : !poly.poly<10>
  return %b : !poly.poly<10>
}

func.func @caller(%a: !poly.poly<10>) -> !poly.poly<10> {
  %r = call @callee(%a) : (!poly.poly<10>) -> !poly.poly<10>
  return %r : !poly.poly<10>
}
```

转换后：

```mlir
func.func @callee(%arg0: tensor<10xi32>) -> tensor<10xi32> {
  %0 = arith.addi %arg0, %arg0 : tensor<10xi32>
  return %0 : tensor<10xi32>
}

func.func @caller(%arg0: tensor<10xi32>) -> tensor<10xi32> {
  %0 = call @callee(%arg0) : (tensor<10xi32>) -> tensor<10xi32>
  return %0 : tensor<10xi32>
}
```

这个例子同时需要：

- `ConvertAdd`
- 函数签名转换
- `return` 转换
- `call` 转换

---

## 12. 常见坑

1. **只注册 pattern，不注册动态合法性**
   函数/return/call 可能被当作已经合法，或者无法正确进入转换流程。

2. **用 `op.getOperands()` 而不是 `adaptor`**
   在普通 op conversion 中，操作数必须使用 `adaptor` 的转换后版本。

3. **把分支的所有操作数都替换**
   只应替换 `getSuccessorOperands()` 中的 forwarded operands。

4. **忘了函数体 block 参数**
   函数签名改了，但入口 block 参数没改，仍然不合法。

5. **call 和 callee 签名不同步**
   函数转换和 call 转换必须使用同一个 `TypeConverter`。

6. **`isLegal` 不是“能不能转换”**
   它的意思是“转换后是否还是自己”；poly 类型通常不是合法类型。

---

## 13. 练习

### 练习 1：为什么 `func.func` 需要动态合法性？

如果只注册：

```cpp
target.addIllegalDialect<PolyDialect>();
```

而不注册 `func.func` 的动态合法性，会发生什么？

<details>
<summary>参考答案</summary>

`func.func` 本身不属于 `poly` dialect，所以不会被
`addIllegalDialect<PolyDialect>()` 标记为非法；但它的签名可能包含
`!poly.poly<N>`。需要动态合法性回调检查签名和 block 参数，决定是否需要
函数签名转换。

</details>

### 练习 2：漏掉 `populateReturnOpTypeConversionPattern` 会怎样？

<details>
<summary>参考答案</summary>

函数签名可能已经变成 `tensor<10xi32>`，但 `return` 仍返回
`!poly.poly<10>`，类型不一致，可能留下 unresolved materialization 或验证失败。

</details>

### 练习 3：分支为什么要用 `getSuccessorOperands`？

<details>
<summary>参考答案</summary>

分支 op 的操作数不一定全部传给后继 block；例如条件分支的条件、switch 的 flag
等不是后继参数。只应转换 `getSuccessorOperands(succIdx)` 中的 forwarded
operands，否则会错误替换非分支参数。

</details>

## 14. 参考

- 第 9 课：[Dialect Conversion](09-dialect-conversion.md)
- `lib/Conversion/PolyToStandard/PolyToStandard.cpp`
- `mlir/include/mlir/Transforms/DialectConversion.h`
- `mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h`
- `mlir/lib/Dialect/Func/Transforms/FuncConversions.cpp`
- MLIR 官方文档：Dialect Conversion
  <https://mlir.llvm.org/docs/DialectConversion/>