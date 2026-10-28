# 第 4 课：Defining a New Dialect

> 对应文章 5：[Defining a New Dialect](https://jeremykun.com/2023/08/21/mlir-defining-a-new-dialect/)。
>
> 本课从零理解 MLIR 的 Dialect、Type、Op 三层定义：如何用 ODS/TableGen 描述
> `!poly.poly<N>` 类型和 `poly.add/sub/mul/...` 操作，以及这些声明如何变成
> 可解析、可打印、可注册的 C++ 类。

## 1. 学习目标

- 理解 Dialect / Type / Op 的关系，以及为什么需要自定义 dialect
- 能读懂 `PolyDialect.td`、`PolyTypes.td`、`PolyOps.td`
- 理解 `mnemonic`、`cppNamespace`、`useDefaultTypePrinterParser`
- 能读懂 Type 的 `parameters` 与 `assemblyFormat`
- 能读懂 Op 的 arguments/results、type constraints、traits、assemblyFormat
- 知道 `mlir-tblgen` 的 `-gen-dialect-*`、`-gen-typedef-*`、`-gen-op-*` 分别生成什么
- 能运行 `poly_syntax.mlir`，并写出自己的 Poly IR

## 2. 什么是 Dialect

MLIR 的核心 IR 是统一的，但操作、类型和属性可以按领域扩展。一个 **Dialect**
就是一个命名空间，包含：

- **Operations**：`poly.add`、`poly.mul`、`poly.eval`……
- **Types**：`!poly.poly<10>`
- **Attributes**：本教程暂未定义专属 attribute
- 可选：interfaces、traits、canonicalization patterns 等

自定义 dialect 的意义：

- 用高层语义表达领域概念（多项式），而不是一开始就拆成 arith/memref
- 后续通过 pass / conversion 逐层 lower 到标准 dialect
- 让优化在高层语义上进行（例如多项式乘法折叠）

本教程的 `poly` dialect 就是这样一个“小型领域语言”。

## 3. Dialect 定义：`PolyDialect.td`

```tablegen
include "mlir/IR/OpBase.td"

def Poly_Dialect : Dialect {
  let name = "poly";
  let summary = "A dialect for polynomial math";
  let description = [{
    The poly dialect defines types and operations for single-variable
    polynomials over integers.
  }];

  let cppNamespace = "::mlir::tutorial::poly";

  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}
```

字段说明：

| 字段 | 作用 |
|---|---|
| `name = "poly"` | IR 中的命名空间：`poly.add`、`!poly.poly<10>` |
| `cppNamespace` | 生成代码的 C++ 命名空间 |
| `useDefaultTypePrinterParser` | 自动生成类型解析/打印，使用 Type 的 assemblyFormat |
| `hasConstantMaterializer` | 声明 dialect 实现 `materializeConstant`，用于从属性重建常量 |

### 生成代码

`mlir-tblgen` 会生成：

- `PolyDialect.h.inc`：`PolyDialect` 类声明、`getDialectNamespace()`
- `PolyDialect.cpp.inc`：构造函数、析构函数、TypeID

`PolyDialect.cpp` 里的初始化：

```cpp
void PolyDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Poly/PolyTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Poly/PolyOps.cpp.inc"
      >();
}
```

这告诉 MLIR：这个 dialect 拥有哪些类型和操作。

### 注册到工具

`tools/tutorial-opt.cpp`：

```cpp
registry.insert<mlir::tutorial::poly::PolyDialect>();
```

所以 `tutorial-opt --help` 的 Available Dialects 里会出现 `poly`。

## 4. 定义 Type：`PolyTypes.td`

### 4.1 类型基类

```tablegen
class Poly_Type<string name, string typeMnemonic> : TypeDef<Poly_Dialect, name> {
  let mnemonic = typeMnemonic;
}
```

- `TypeDef<Poly_Dialect, name>`：声明这是 `poly` dialect 的类型
- `mnemonic = "poly"`：IR 中写成 `!poly.poly<...>`
- 类名 `Polynomial` 会生成 C++ 类 `PolynomialType`

### 4.2 Polynomial 类型

```tablegen
def Polynomial : Poly_Type<"Polynomial", "poly"> {
  let summary = "A polynomial with u32 coefficients";
  let description = [{
    A type for polynomials with integer coefficients in a single-variable
    polynomial ring.
  }];

  let parameters = (ins "int":$degreeBound);
  let assemblyFormat = "`<` $degreeBound `>`";
}
```

**参数**：

- `parameters = (ins "int":$degreeBound)` 表示类型携带一个 `int` 参数
- IR 写法：`!poly.poly<10>`
- `degreeBound` 是多项式环的次数上界；本教程的乘法语义是模 `x^N = 1`
  （见 `MulOp::fold` 中的 `(i + j) % degree`）

**assemblyFormat**：

```tablegen
let assemblyFormat = "`<` $degreeBound `>`";
```

- 反引号里是字面量：`<`、`>`
- `$degreeBound` 引用参数
- 最终语法：`!poly.poly<10>`

因为 dialect 设置了 `useDefaultTypePrinterParser = 1`，MLIR 会自动生成
类型的 parser/printer，不需要手写 `parseType` / `printType`。

> 文章 5 展示了没有该选项时手写 parse/print 的版本；当前仓库使用自动生成，
> 更简洁，也是更常见的做法。

#### degreeBound 与模 x^N = 1

`degreeBound = N` 表示多项式所属的环是：

```text
Z[x] / (x^N - 1)
```

也就是“整数系数的多项式，并且规定 `x^N = 1`”。

含义：

- 系数下标 `i` 对应 `x^i`：`[c0, c1, c2, ...]` 表示
  `c0 + c1*x + c2*x^2 + ...`
- 因为 `x^N = 1`，所以 `x^N = x^0`、`x^{N+1} = x^1`……
- 指数只在 `0..N-1` 之间循环，所以叫“次数上界/模”

乘法时，两个系数 `a_i`、`b_j` 相乘得到 `x^{i+j}`，需要把指数对 N 取模：

```text
x^{i+j} = x^{(i+j) mod N}
```

这就是 `MulOp::fold` 里的：

```cpp
result[(i + j) % degree] += lhs[i] * rhs[j];
```

##### 例子：N = 3，p = 1 + x²

```text
p = [1, 0, 1]        // 1 + 0*x + 1*x^2
p * p = 1 + 2x^2 + x^4
```

在 N = 3 的环里：

```text
x^4 = x^{4 mod 3} = x^1 = x
```

所以：

```text
p * p = 1 + x + 2x^2 = [1, 1, 2]
```

用本项目的 IR 验证：

```mlir
%p = poly.constant dense<[1, 0, 1]> : tensor<3xi32> : !poly.poly<3>
%q = poly.mul %p, %p : !poly.poly<3>
```

`--canonicalize` 后得到：

```mlir
%0 = poly.constant dense<[1, 1, 2, 0, 0]> : tensor<5xi32> : !poly.poly<3>
```

前三个系数是 `[1, 1, 2]`；后面两个 0 是因为 fold 的结果 tensor 长度取
`lhs.size() + rhs.size() - 1 = 5`，而所有乘积项的指数都被折回到 0..2，索引 3、4
没有贡献。

##### 常见误解

1. **`degreeBound` 不等于某个多项式的实际次数**
   它定义环的模，实际多项式可以更短。例如 `[1, 0, 1]` 的实际次数是 2，但类型仍是
   `!poly.poly<3>`。

2. **只对指数取模，不对系数取模**
   本教程没有系数模数 `q`；系数是普通整数，乘法后可能变大（例如上面的 2）。

3. **`poly.eval` 不套用 `x^N = 1`**
   当前 `ConvertEval` 是对系数数组做普通多项式求值，不按环回绕；乘法的模运算和
   求值是两件事。

4. **不同 `degreeBound` 是不同类型**
   `!poly.poly<3>` 和 `!poly.poly<5>` 不能混用；`SameOperandsAndResultType` 会
   拒绝类型不一致的运算。

### 4.3 类型缺失参数会怎样

写 `!poly.poly`（不写 `<10>`）会解析失败：

```text
error: expected '<'
```

因为 assemblyFormat 要求 `<` 和 `degreeBound`。这说明 assemblyFormat 同时定义了
类型的**语法**和**合法性下限**。

## 5. 定义 Op：`PolyOps.td`

> 如果 `assemblyFormat` 到文本语法的转换总是看不明白，先读
> [如何阅读和编写 assemblyFormat](assembly-format.md)。

### 5.1 一个最简 op

文章 5 的第一版：

```tablegen
def Poly_AddOp : Op<Poly_Dialect, "add"> {
  let arguments = (ins Polynomial:$lhs, Polynomial:$rhs);
  let results = (outs Polynomial:$output);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";
}
```

- `Op<Poly_Dialect, "add">`：在 `poly` dialect 中定义 `poly.add`
- `arguments`：两个输入 `lhs`、`rhs`，类型都是 `Polynomial`
- `results`：一个输出 `output`
- `assemblyFormat`：定义 pretty 语法

对应的 IR：

```mlir
%0 = poly.add %arg0, %arg1 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
```

### 5.2 仓库中的 binop 基类

当前仓库把 add/sub/mul 的公共部分抽成 `Poly_BinOp`：

```tablegen
def PolyOrContainer : TypeOrValueSemanticsContainer<Polynomial, "poly-or-container">;

class Poly_BinOp<string mnemonic>
    : Op<Poly_Dialect, mnemonic,
         [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let arguments = (ins PolyOrContainer:$lhs, PolyOrContainer:$rhs);
  let results = (outs PolyOrContainer:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($output))";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Poly_AddOp : Poly_BinOp<"add"> { ... }
def Poly_SubOp : Poly_BinOp<"sub"> { ... }
def Poly_MulOp : Poly_BinOp<"mul"> { ... }
```

几个关键点：

| 元素 | 作用 |
|---|---|
| `PolyOrContainer` | 允许 `!poly.poly<N>`、`vector<...>`、`tensor<...>` |
| `Pure` | 没有副作用，可被 DCE/重排 |
| `ElementwiseMappable` | 支持对 tensor/vector 逐元素应用 |
| `SameOperandsAndResultType` | 操作数与结果类型必须一致 |
| `hasFolder` | 声明有 fold 方法（第 6 课） |
| `hasCanonicalizer` | 声明有 canonicalization patterns（第 8 课） |
| `qualified(type($output))` | 打印完整类型名，如 `!poly.poly<10>` |

因此 `poly.add` 既能处理标量 poly，也能处理：

```mlir
%8 = poly.add %7, %7 : tensor<2x!poly.poly<10>>
```

#### `poly.add` 的“实际加法”在哪里

ODS 里的 `def Poly_AddOp` 只描述**接口和语法**，不包含算术语义。真正的加法分布在
三个地方：

1. **常量折叠**：`lib/Dialect/Poly/PolyOps.cpp` 的 `AddOp::fold`，对系数属性
   逐元素相加（只对常量生效）
2. **dialect conversion**：`lib/Conversion/PolyToStandard/PolyToStandard.cpp` 的
   `ConvertAdd`，把 `poly.add` 降成 `arith.addi`
3. **后续 lowering**：`arith.addi` 再经过 tensor/linalg/memref/LLVM，最终变成
   `llvm.add`（真正的机器加法）

`ConvertAdd` 的核心：

```cpp
LogicalResult matchAndRewrite(AddOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  arith::AddIOp addOp = rewriter.create<arith::AddIOp>(
      op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
  rewriter.replaceOp(op.getOperation(), addOp);
  return success();
}
```

类型转换器把 `!poly.poly<N>` 变成 `tensor<Nxi32>`，所以 lowering 后是：

```mlir
%0 = arith.addi %arg0, %arg1 : tensor<3xi32>
```

`poly.add` 本身没有“执行函数”；MLIR 是 IR + 变换框架，不是高层 dialect 的解释器。

### 5.3 与 tensor 互转

```tablegen
def Poly_FromTensorOp : Op<Poly_Dialect, "from_tensor", [Pure]> {
  let arguments = (ins TensorOf<[AnyInteger]>:$input);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` qualified(type($output))";
  let hasFolder = 1;
}

def Poly_ToTensorOp : Op<Poly_Dialect, "to_tensor", [Pure]> {
  let arguments = (ins Polynomial:$input);
  let results = (outs TensorOf<[AnyInteger]>:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` type($output)";
}
```

它们把“系数 tensor”和“poly 值”互相转换，是后续 lowering 的桥梁：

```mlir
%3 = poly.from_tensor %cst : tensor<3xi32> -> !poly.poly<10>
%14 = poly.to_tensor %1 : !poly.poly<10> -> tensor<10xi32>
```

### 5.4 eval

`poly.eval` 表示多项式求值：给定多项式 `p` 和点 `x`，计算
`p(x) = c0 + c1*x + c2*x^2 + ...`，结果类型与 `x` 相同；它本身只是抽象 IR，
具体计算在 lowering 后完成。

```tablegen
def IntOrComplex : AnyTypeOf<[AnyInteger, AnyComplex]>;

def Poly_EvalOp : Op<Poly_Dialect, "eval",
    [AllTypesMatch<["point", "output"]>, Has32BitArguments]> {
  let arguments = (ins Polynomial:$polynomial, IntOrComplex:$point);
  let results = (outs IntOrComplex:$output);
  let assemblyFormat =
      "$polynomial `,` $point attr-dict `:` `(` qualified(type($polynomial)) `,` type($point) `)` `->` type($output)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}
```

- `IntOrComplex`：求值点可以是整数或复数
- `AllTypesMatch`：`point` 和 `output` 类型必须相同
- `Has32BitArguments`：自定义 trait，检查 32 位整数（第 5 课）
- `hasVerifier`：自定义校验（第 7 课）

IR：

```mlir
%6 = poly.eval %4, %5 : (!poly.poly<10>, i32) -> i32
%complex_eval = poly.eval %4, %z : (!poly.poly<10>, complex<f64>) -> complex<f64>
```

### 5.5 constant

```tablegen
def Poly_ConstantOp : Op<Poly_Dialect, "constant", [Pure, ConstantLike]> {
  let arguments = (ins AnyIntElementsAttr:$coefficients);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$coefficients attr-dict `:` qualified(type($output))";
  let hasFolder = 1;
}
```

- `ConstantLike`：让 MLIR 把它当作常量 op（用于常量传播、模式匹配）
- `AnyIntElementsAttr`：系数是整数元素的 DenseElementsAttr

IR：

```mlir
%10 = poly.constant dense<[2, 3, 4]> : tensor<3xi32> : !poly.poly<10>
```

## 6. 构建与生成

### CMake

```cmake
set(LLVM_TARGET_DEFINITIONS PolyOps.td)
mlir_tablegen(PolyOps.h.inc -gen-op-decls)
mlir_tablegen(PolyOps.cpp.inc -gen-op-defs)
mlir_tablegen(PolyTypes.h.inc -gen-typedef-decls -typedefs-dialect=poly)
mlir_tablegen(PolyTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=poly)
mlir_tablegen(PolyDialect.h.inc -gen-dialect-decls -dialect=poly)
mlir_tablegen(PolyDialect.cpp.inc -gen-dialect-defs -dialect=poly)
add_public_tablegen_target(MLIRPolyOpsIncGen)
add_dependencies(mlir-headers MLIRPolyOpsIncGen)
```

| 生成选项 | 产物 |
|---|---|
| `-gen-dialect-decls` | `PolyDialect.h.inc` |
| `-gen-dialect-defs` | `PolyDialect.cpp.inc` |
| `-gen-typedef-decls` | `PolyTypes.h.inc` |
| `-gen-typedef-defs` | `PolyTypes.cpp.inc` |
| `-gen-op-decls` | `PolyOps.h.inc` |
| `-gen-op-defs` | `PolyOps.cpp.inc` |

#### 生成文件在哪里

生成文件在 `build/` 构建树里，目录结构与源码镜像：

```text
build/lib/Dialect/Poly/
├── PolyDialect.h.inc
├── PolyDialect.cpp.inc
├── PolyTypes.h.inc
├── PolyTypes.cpp.inc
├── PolyOps.h.inc
├── PolyOps.cpp.inc
└── PolyCanonicalize.cpp.inc
```

其他生成文件：

```text
build/lib/Transform/Affine/Passes.h.inc
build/lib/Transform/Arith/Passes.h.inc
build/lib/Transform/Arith/MulToAddPdll.h.inc
build/lib/Transform/Noisy/Passes.h.inc
build/lib/Conversion/PolyToStandard/PolyToStandard.h.inc
```

查看全部：

```bash
find build/lib -name '*.inc' | sort
```

这些是构建产物，不要手改；修改 `.td` 后重新构建会自动重新生成。

注意这里只用 `PolyOps.td` 作为输入：它 include 了 `PolyDialect.td` 和
`PolyTypes.td`，所以 TableGen 一次能看到所有定义，再用不同 `-gen-*` 选项生成
不同产物。

库目标：

```cmake
add_mlir_dialect_library(MLIRPoly
    PolyDialect.cpp
    PolyOps.cpp
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/Dialect/Poly
    LINK_LIBS PUBLIC
    )
```

### Bazel

`lib/Dialect/Poly/BUILD` 用 `td_library` + 多个 `gentbl_cc_library` 完成同样的
生成，再把这些生成目标作为 `cc_library` 的依赖。

#### 手写文件与生成文件的分工

| 文件 | 来源 | 内容 |
|---|---|---|
| `PolyDialect.td` / `PolyTypes.td` / `PolyOps.td` | 手写 | ODS 声明 |
| `PolyDialect.h` / `PolyTypes.h` / `PolyOps.h` | 手写 | include 生成代码，暴露类声明 |
| `PolyDialect.cpp` | 手写 | dialect 初始化 + include 生成定义 |
| `PolyOps.cpp` | 手写 | fold / verify / canonicalization 等自定义行为 |
| `*.h.inc` / `*.cpp.inc` | `mlir-tblgen` 生成 | op/type/dialect 的样板代码 |
| `PolyCanonicalize.cpp.inc` | `mlir-tblgen` 生成 | `PolyPatterns.td` 的声明式 pattern |

几个容易混淆的点：

- `.h.inc` / `.cpp.inc` 在 `build/` 目录，是构建产物，**不要手改**
- `PolyOps.h` 用 `#define GET_OP_CLASSES` + include `PolyOps.h.inc` 来拿到
  op 类声明
- `PolyOps.cpp.inc`（op 定义、parser/printer、注册）是由
  `PolyDialect.cpp` include 的，不是 `PolyOps.cpp`
- `PolyOps.cpp` 只包含**手写行为**：fold、verifier、canonicalization pattern 注册

按文章顺序，这些手写文件的“写作时间”是：

1. 文章 5：创建 `.td`、`.h`、最小 `.cpp` 骨架
2. 文章 6：增加 `PolyTraits.h` 和 trait 声明
3. 文章 7：在 `PolyOps.cpp` 写 `fold`
4. 文章 8：在 `PolyOps.cpp` 写 `verify`
5. 文章 9：增加 `PolyPatterns.td`，在 `PolyOps.cpp` 注册 canonicalization patterns

当前仓库是系列完成后的最终状态，所以这些方法都已经存在。

#### 如何新增 `.h` / `.cpp` 文件

先明确一个关键点：

> `.h` / `.cpp` 通常是**每个 dialect 一套**，在创建 dialect 时写一次；
> 在同一个 dialect 里新增一个 op，通常只改 `.td`，不需要新增 `.h` / `.cpp`。

### 为什么新增 op 通常不用改 `.h` / `.cpp`

- `PolyOps.h` 已经通过 `GET_OP_CLASSES` include 了 `PolyOps.h.inc`，新 op 的类会
  自动出现在里面
- `PolyDialect.cpp` 里的 `GET_OP_LIST` 会自动把新 op 注册进 dialect
- 如果新 op 没有自定义行为，ODS 生成的 parser/printer/verifier 已经够用
- 例如新增 `poly.neg`，只改 `PolyOps.td` 并重建，就能解析和打印

需要改 `.h` / `.cpp` 的情况：

| 需求 | 改哪里 |
|---|---|
| 新增 op（无自定义行为） | 只改 `.td` |
| 新增 op + fold/verify/canonicalizer | `.td` 声明 + `PolyOps.cpp` 实现 |
| 新增 type | `PolyTypes.td`，`.h` 通常不用改 |
| 新增 dialect | 新增一套 `.td` / `.h` / `.cpp` + 构建文件 |
| 自定义 parser/printer（不用 assemblyFormat） | `.cpp` |
| 新增独立 `.cpp` 文件 | 还要加入 CMake/Bazel 的源文件列表 |

### 一个最小 dialect 的 `.h` / `.cpp` 模板

假设新 dialect 叫 `foo`，目录 `lib/Dialect/Foo/`。

#### `FooDialect.h`

```cpp
#ifndef LIB_DIALECT_FOO_FOODIALECT_H_
#define LIB_DIALECT_FOO_FOODIALECT_H_

#include "mlir/include/mlir/IR/DialectImplementation.h"

#include "lib/Dialect/Foo/FooDialect.h.inc"

#endif
```

#### `FooTypes.h`

```cpp
#ifndef LIB_DIALECT_FOO_FOOTYPES_H_
#define LIB_DIALECT_FOO_FOOTYPES_H_

#include "mlir/include/mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Foo/FooTypes.h.inc"

#endif
```

#### `FooOps.h`

```cpp
#ifndef LIB_DIALECT_FOO_FOOOPS_H_
#define LIB_DIALECT_FOO_FOOOPS_H_

#include "lib/Dialect/Foo/FooDialect.h"
#include "lib/Dialect/Foo/FooTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "lib/Dialect/Foo/FooOps.h.inc"

#endif
```

#### `FooDialect.cpp`

```cpp
#include "lib/Dialect/Foo/FooDialect.h"

#include "lib/Dialect/Foo/FooOps.h"
#include "lib/Dialect/Foo/FooTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"
#include "mlir/include/mlir/IR/Builders.h"

#include "lib/Dialect/Foo/FooDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Foo/FooTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/Foo/FooOps.cpp.inc"

namespace mlir {
namespace tutorial {
namespace foo {

void FooDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Foo/FooTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Foo/FooOps.cpp.inc"
      >();
}

} // namespace foo
} // namespace tutorial
} // namespace mlir
```

#### `FooOps.cpp`

```cpp
#include "lib/Dialect/Foo/FooOps.h"

// 在这里实现 fold / verify / getCanonicalizationPatterns 等
// ODS 无法自动生成的行为
```

如果使用声明式 canonicalization patterns，还要：

```cpp
#include "lib/Dialect/Foo/FooCanonicalize.cpp.inc"
```

### 这些宏分别在做什么

| 宏 | 出现位置 | 作用 |
|---|---|---|
| `GET_TYPEDEF_CLASSES` | `FooTypes.h` / `FooDialect.cpp` | 生成/声明 type 类 |
| `GET_TYPEDEF_LIST` | `FooDialect.cpp` 的 `addTypes<...>` | 展开类型注册列表 |
| `GET_OP_CLASSES` | `FooOps.h` / `FooDialect.cpp` | 生成/声明 op 类 |
| `GET_OP_LIST` | `FooDialect.cpp` 的 `addOperations<...>` | 展开 op 注册列表 |

同一个 `.inc` 可以在同一个 TU 里 include 多次（不同宏看到不同 section），
这是生成文件的设计方式。

### 构建文件

CMake 需要加一段 TableGen 规则（参考 `lib/Dialect/Poly/CMakeLists.txt`）：

```cmake
set(LLVM_TARGET_DEFINITIONS FooOps.td)
mlir_tablegen(FooOps.h.inc -gen-op-decls)
mlir_tablegen(FooOps.cpp.inc -gen-op-defs)
mlir_tablegen(FooTypes.h.inc -gen-typedef-decls -typedefs-dialect=foo)
mlir_tablegen(FooTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=foo)
mlir_tablegen(FooDialect.h.inc -gen-dialect-decls -dialect=foo)
mlir_tablegen(FooDialect.cpp.inc -gen-dialect-defs -dialect=foo)
add_public_tablegen_target(MLIRFooOpsIncGen)

add_mlir_dialect_library(MLIRFoo
    FooDialect.cpp
    FooOps.cpp
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/Dialect/Foo
    DEPENDS
    MLIRFooOpsIncGen
    LINK_LIBS PUBLIC
    )
```

Bazel 参照 `lib/Dialect/Poly/BUILD` 的 `td_library` + `gentbl_cc_library` +
`cc_library`。

最后在 `tools/tutorial-opt.cpp` 注册：

```cpp
registry.insert<mlir::tutorial::foo::FooDialect>();
```

### 新增 op 的完整清单

1. 在 `FooOps.td`（或 `PolyOps.td`）里加 `def`
2. 重建 `tutorial-opt`
3. 写一段 IR 验证解析/打印
4. 如果需要 fold/verify/canonicalization：
   - `.td` 加 `hasFolder` / `hasVerifier` / `hasCanonicalizer`
   - 在 `FooOps.cpp` 实现
5. 如果新增了独立 `.cpp` 文件：加入 CMake/Bazel
6. 如果新增了 dialect：注册到 `tutorial-opt`
7. 加 FileCheck 测试

以 Poly 为例，新增 `poly.neg` 时只需要改 `PolyOps.td`；
`PolyOps.h` 和 `PolyDialect.cpp` 会自动包含/注册它。只有当你需要为 `neg`
实现 lowering 时，才去改 `PolyToStandard.cpp`。

## 7. 实验

先设置路径：

```bash
cd /home/zhouzhou/npu/mlir-tutorial
BIN=$PWD/externals/llvm-project/build/bin
TO=$PWD/build/tools/tutorial-opt
mkdir -p build/scratch
```

### 实验 1：往返打印

```bash
$TO tests/poly_syntax.mlir
```

输出（节选）：

```mlir
func.func @test_type_syntax(%arg0: !poly.poly<10>) -> !poly.poly<10> {
  return %arg0 : !poly.poly<10>
}
func.func @test_op_syntax(%arg0: !poly.poly<10>, %arg1: !poly.poly<10>) -> !poly.poly<10> {
  %0 = poly.add %arg0, %arg1 : !poly.poly<10>
  %1 = poly.sub %arg0, %arg1 : !poly.poly<10>
  %2 = poly.mul %arg0, %arg1 : !poly.poly<10>
  ...
}
```

能被解析再打印，说明 dialect/type/op 的定义和 assemblyFormat 都正确。

### 实验 2：generic 形式

```bash
$TO tests/poly_syntax.mlir --mlir-print-op-generic | head -20
```

可以看到：

```mlir
%0 = "poly.add"(%arg0, %arg1) : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
```

pretty 语法只是 assemblyFormat；真正的 op 名、操作数、结果类型都在 generic 形式里。

### 实验 3：写自己的 Poly IR

创建 `build/scratch/poly_demo.mlir`：

```mlir
func.func @demo(%a: !poly.poly<5>, %b: !poly.poly<5>) -> !poly.poly<5> {
  %c = poly.constant dense<[1, 2, 3]> : tensor<3xi32> : !poly.poly<5>
  %s = poly.add %a, %b : !poly.poly<5>
  %m = poly.mul %s, %c : !poly.poly<5>
  return %m : !poly.poly<5>
}
```

运行：

```bash
$TO build/scratch/poly_demo.mlir
```

输出保持原样，说明 `!poly.poly<5>` 和三个 op 都能正常解析/打印。

### 实验 4：缺参数的类型

```bash
cat > build/scratch/poly_bad.mlir <<'EOF'
func.func @bad(%a: !poly.poly) -> !poly.poly {
  return %a : !poly.poly
}
EOF
$TO build/scratch/poly_bad.mlir 2>&1 | head -3
```

输出：

```text
build/scratch/poly_bad.mlir:1:30: error: expected '<'
```

因为 assemblyFormat 要求 `<degreeBound>`。

### 实验 5：查看 dialect 注册

```bash
$TO --help 2>&1 | grep -o 'poly' | head -1
```

`poly` 出现在 Available Dialects 中，说明 `registry.insert<PolyDialect>()` 生效。

### 实验 6：tensor of poly

`tests/poly_syntax.mlir` 中有：

```mlir
%7 = tensor.from_elements %arg0, %arg1 : tensor<2x!poly.poly<10>>
%8 = poly.add %7, %7 : tensor<2x!poly.poly<10>>
```

这是 `ElementwiseMappable` + `PolyOrContainer` 的效果：同一个 op 既能处理标量，
也能处理容器。

`tensor.from_elements` 把两个标量多项式打包成一个 rank-1 tensor：

```text
%7 = [p0, p1]
```

`poly.add %7, %7` 的语义是**逐元素**对两个多项式做加法：

```text
%8 = [p0 + p0, p1 + p1]
```

它不是把系数数组拼起来，也不是矩阵加法；每个元素本身是一个 `!poly.poly<10>`。

#### 为什么能这样写

1. `PolyOrContainer`：

```tablegen
def PolyOrContainer : TypeOrValueSemanticsContainer<Polynomial, "poly-or-container">;
```

它接受：

- 标量 `Polynomial`
- `Polynomial` 的 vector/tensor 容器

2. `ElementwiseMappable`：

```tablegen
[Pure, ElementwiseMappable, SameOperandsAndResultType]
```

`ElementwiseMappable` 是 `Elementwise + Scalarizable + Vectorizable + Tensorizable`
的组合 trait，表示：

- op 对标量定义
- 可以逐元素应用到 vector/tensor
- 所有容器形状必须一致
- **不允许 broadcasting**

3. `SameOperandsAndResultType`：两个操作数和结果必须是同一个类型。

#### generic 形式

```mlir
%10 = "poly.add"(%9, %9)
      : (tensor<2x!poly.poly<10>>, tensor<2x!poly.poly<10>>)
        -> tensor<2x!poly.poly<10>>
```

#### 不支持广播

`tensor + scalar` 会失败：

```mlir
%t = tensor.from_elements %p, %p : tensor<2x!poly.poly<10>>
%r = poly.add %t, %p : tensor<2x!poly.poly<10>>
```

报错（类型不一致）：

```text
error: use of value '%p' expects different type than prior uses:
       'tensor<2x!poly.poly<10>>' vs '!poly.poly<10>'
```

因为 `SameOperandsAndResultType` 要求类型完全一致，`ElementwiseMappable` 也明确
不允许 broadcasting。

#### 这个实验的定位

`tests/poly_syntax.mlir` 里的 `%8` 并没有被使用，它主要用于验证：

- op 的 ODS 约束允许 tensor of poly
- assemblyFormat 能打印完整的 tensor 类型
- ElementwiseMappable 的类型规则成立

真正把 tensor 级 op 拆成标量/循环，是后续 lowering 的工作。

### 实验 7：从零创建一个迷你 dialect（box）

作为本课的实战，仓库里新增了一个 `box` dialect。**完整的、可照着敲的分步教程
见 [mini-dialect-box.md](mini-dialect-box.md)**；下面是速览。

设计：

- 类型 `!box.box<T>`：包装任意一个 MLIR 类型
- `box.pack`：把 `T` 放入 box
- `box.unpack`：从 box 取出值
- verifier：检查 pack/unpack 的元素类型是否一致
- fold：`unpack(pack(x)) -> x`、`pack(unpack(b)) -> b`

文件：

```text
lib/Dialect/Box/
├── BoxDialect.td / .h / .cpp
├── BoxTypes.td / .h
├── BoxOps.td / .h / .cpp
├── CMakeLists.txt
└── BUILD

tests/box_syntax.mlir
tests/box_fold.mlir
tests/box_verifier.mlir
```

关键定义：

```tablegen
def Box : Box_Type<"Box", "box"> {
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

def Box_PackOp : Op<Box_Dialect, "pack", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs Box:$box);
  let assemblyFormat =
      "$input attr-dict `:` qualified(type($input)) `->` qualified(type($box))";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def Box_UnpackOp : Op<Box_Dialect, "unpack", [Pure]> {
  let arguments = (ins Box:$box);
  let results = (outs AnyType:$value);
  let assemblyFormat =
      "$box attr-dict `:` qualified(type($box)) `->` qualified(type($value))";
  let hasVerifier = 1;
  let hasFolder = 1;
}
```

运行：

```bash
$TO tests/box_syntax.mlir
$TO tests/box_fold.mlir --canonicalize
$TO tests/box_verifier.mlir -verify-diagnostics
```

结果：

- `box` 出现在 `tutorial-opt --help` 的 Available Dialects 中
- `box.pack` / `box.unpack` 能解析和打印
- `unpack(pack(x))` 在 `--canonicalize` 后折叠成 `x`
- 元素类型不一致时 verifier 报错：

```text
error: 'box.pack' op input type must match box element type
```

这个 dialect 完整走了一遍：`.td` 声明 → 生成 `.inc` → 手写 `.h/.cpp` →
CMake/Bazel 接入 → 注册到 `tutorial-opt` → 测试。

## 8. 源码位置

| 内容 | 文件 |
|---|---|
| Dialect 定义 | `lib/Dialect/Poly/PolyDialect.td` |
| Type 定义 | `lib/Dialect/Poly/PolyTypes.td` |
| Op 定义 | `lib/Dialect/Poly/PolyOps.td` |
| Dialect 初始化/常量物化 | `lib/Dialect/Poly/PolyDialect.cpp` |
| Op 实现（folder/verifier 等） | `lib/Dialect/Poly/PolyOps.cpp` |
| 生成规则 | `lib/Dialect/Poly/CMakeLists.txt`、`BUILD` |
| 注册 | `tools/tutorial-opt.cpp` |
| 测试 | `tests/poly_syntax.mlir` |

## 9. 练习

### 练习 1：`!poly.poly<10>` 的含义

`10` 代表什么？本教程中多项式乘法为什么会出现 `(i + j) % degree`？

### 练习 2：容器的来源

`poly.add` 为什么能接受 `tensor<2x!poly.poly<10>>`？是哪几个声明共同作用的结果？

### 练习 3：自动 parser/printer

`useDefaultTypePrinterParser = 1` 生成了什么？如果去掉它，需要自己实现哪些函数？

### 练习 4：常量物化

`hasConstantMaterializer = 1` 对应的实现在哪里？它把什么转换成了什么？

### 练习 5（可选）：新增 `poly.neg`

在 `PolyOps.td` 中加入一个取负 op：

```tablegen
def Poly_NegOp : Op<Poly_Dialect, "neg", [Pure, SameOperandsAndResultType]> {
  let summary = "Negation of a polynomial.";
  let arguments = (ins Polynomial:$input);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($output))";
}
```

重建 `tutorial-opt`，然后写一个使用 `poly.neg` 的 IR，确认能解析和打印。

## 10. 参考答案

<details>
<summary>练习 1</summary>

`10` 是 `degreeBound`，表示多项式环的次数上界。本教程定义乘法模 `x^10 = 1`，
所以两个系数下标 `i`、`j` 相乘后落在 `(i + j) % 10` 上（见 `MulOp::fold`）。

</details>

<details>
<summary>练习 2</summary>

`PolyOrContainer` 允许标量 poly、vector、tensor；`ElementwiseMappable` 表示
对容器逐元素应用；`SameOperandsAndResultType` 保证容器形状和元素类型一致。
三者共同作用，使 `poly.add` 能处理 `tensor<2x!poly.poly<10>>`。

</details>

<details>
<summary>练习 3</summary>

它会根据 Type 的 assemblyFormat 自动生成 dialect 的 `parseType()` 和
`printType()`。去掉后需要手写这两个函数（文章 5 的早期版本就是这样），
或者为每个类型手动实现解析/打印逻辑。

</details>

<details>
<summary>练习 4</summary>

实现在 `PolyDialect.cpp` 的 `PolyDialect::materializeConstant`。它把
`DenseIntElementsAttr` 转换成一个 `poly.constant` op；当框架需要把折叠得到的
属性重新变成 IR 值时使用。

</details>

<details>
<summary>练习 5</summary>

加入 .td 后，`mlir_tablegen` 会生成 `Poly_NegOp` 的类、parser/printer 和注册；
`PolyDialect::initialize` 中的 `GET_OP_LIST` 会自动包含它。重建后可以解析
`poly.neg %x : !poly.poly<10>`。

</details>

## 11. 验收清单

- [ ] 能说出 Dialect / Type / Op 各自的职责
- [ ] 能读懂 `PolyDialect.td` 的字段与生成代码
- [ ] 能解释 `!poly.poly<N>` 的语法和参数
- [ ] 能读懂一个 Op 的 arguments/results/assemblyFormat
- [ ] 能解释 `PolyOrContainer`、`ElementwiseMappable` 的作用
- [ ] 知道六种 `-gen-*` 产物分别是什么
- [ ] 能运行 `poly_syntax.mlir` 并写自己的 Poly IR
- [ ] 知道 `poly` dialect 如何注册到 `tutorial-opt`

## 下一课

**第 5 课：Using Traits**（文档待编写）

- `Pure`、`ElementwiseMappable`、`SameOperandsAndResultType`
- 自定义 trait `Has32BitArguments` 与 `verifyTrait`
- trait 如何让 LICM、control-flow-sink、SCCP 等通用优化生效
- 测试 `code_motion.mlir`、`control_flow_sink.mlir`、`sccp.mlir`

对应源码：`lib/Dialect/Poly/PolyTraits.h`、`PolyOps.td`