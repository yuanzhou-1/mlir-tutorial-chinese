# 从零创建一个迷你 Dialect：box

> 这是一份**可以照着敲**的完整教程。完成后你会得到一个可用的 `box` dialect：
>
> - 类型 `!box.box<T>`：包装任意 MLIR 类型
> - `box.pack %v : T -> !box.box<T>`
> - `box.unpack %b : !box.box<T> -> T`
> - verifier：检查 pack/unpack 的元素类型一致
> - fold：`unpack(pack(x)) -> x`、`pack(unpack(b)) -> b`
>
> 所有代码都已经在仓库中实现并验证通过。本文按“先写什么、为什么、怎么验证”
> 的顺序展开。

## 最短操作路径（TL;DR）

只想先跑通的话，按这个顺序：

1. **写文件**：在 `lib/Dialect/Box/` 下创建 10 个文件
   - 3 个 `.td`：`BoxDialect.td`、`BoxTypes.td`、`BoxOps.td`
   - 3 个 `.h`：`BoxDialect.h`、`BoxTypes.h`、`BoxOps.h`
   - 2 个 `.cpp`：`BoxDialect.cpp`、`BoxOps.cpp`
   - 2 个构建文件：`CMakeLists.txt`、`BUILD`
2. **改接入点**：
   - `lib/Dialect/CMakeLists.txt`
   - `tools/CMakeLists.txt`
   - `tools/BUILD`
   - `tools/tutorial-opt.cpp`
3. **构建**：
   ```bash
   ninja -C build -j 100 tutorial-opt
   ```
4. **验证**：
   ```bash
   $TO tests/box_syntax.mlir
   $TO tests/box_fold.mlir --canonicalize
   $TO tests/box_verifier.mlir -verify-diagnostics
   ```

下面的章节会逐文件解释这些内容。

### 本教程涉及的文件

| 类型 | 文件 |
|---|---|
| 新增（dialect） | `lib/Dialect/Box/BoxDialect.td`、`BoxTypes.td`、`BoxOps.td` |
| 新增（头文件） | `lib/Dialect/Box/BoxDialect.h`、`BoxTypes.h`、`BoxOps.h` |
| 新增（实现） | `lib/Dialect/Box/BoxDialect.cpp`、`BoxOps.cpp` |
| 新增（构建） | `lib/Dialect/Box/CMakeLists.txt`、`BUILD` |
| 新增（测试） | `tests/box_syntax.mlir`、`tests/box_fold.mlir`、`tests/box_verifier.mlir` |
| 修改（接入） | `lib/Dialect/CMakeLists.txt`、`tools/CMakeLists.txt`、`tools/BUILD`、`tools/tutorial-opt.cpp` |

## 目录

1. [先理解文件角色](#1-先理解文件角色)
2. [准备工作](#2-准备工作)
3. [写 Dialect 声明](#3-写-dialect-声明)
4. [写 Type 声明](#4-写-type-声明)
5. [写 Op 声明](#5-写-op-声明)
6. [写 `.h` 包装头](#6-写-h-包装头)
7. [写 `.cpp` 实现](#7-写-cpp-实现)
8. [接入构建系统](#8-接入构建系统)
9. [注册到 tutorial-opt](#9-注册到-tutorial-opt)
10. [验证解析和打印](#10-验证解析和打印)
11. [验证 verifier](#11-验证-verifier)
12. [验证 fold](#12-验证-fold)
13. [查看生成代码与宏展开](#13-查看生成代码与宏展开)
14. [注册原理](#14-注册原理)
15. [常见错误排查](#15-常见错误排查)
16. [如何继续扩展](#16-如何继续扩展)
17. [清理这个 dialect](#17-清理这个-dialect)
18. [验收清单](#18-验收清单)

## 1. 先理解文件角色

一个 dialect 需要两类文件：

```text
手写声明 + 手写实现
├── BoxDialect.td / BoxTypes.td / BoxOps.td   # ODS 声明（单一事实来源）
├── BoxDialect.h / BoxTypes.h / BoxOps.h      # 手写头文件，暴露生成的类
├── BoxDialect.cpp                            # 手写：dialect 初始化 + include 生成定义
└── BoxOps.cpp                                # 手写：verifier / fold 等自定义行为

构建时自动生成
└── build/lib/Dialect/Box/
    ├── BoxDialect.h.inc / BoxDialect.cpp.inc
    ├── BoxTypes.h.inc / BoxTypes.cpp.inc
    └── BoxOps.h.inc / BoxOps.cpp.inc
```

规则：

| 文件 | 谁写 | 作用 |
|---|---|---|
| `.td` | 你 | 声明 dialect/type/op |
| `.h` | 你 | include 生成的类声明，提供稳定入口 |
| `.cpp` | 你 | 写 ODS 无法自动生成的行为 |
| `.inc` | `mlir-tblgen` | 生成样板代码，**不要手改** |

生成的 `.inc` 不能独立编译，它需要手写的 `.h`/`.cpp` 在正确的宏下 include。

## 2. 准备工作

```bash
cd /home/zhouzhou/npu/mlir-tutorial

export TOP=$PWD
export BIN=$TOP/externals/llvm-project/build/bin
export TO=$TOP/build/tools/tutorial-opt

mkdir -p lib/Dialect/Box
```

确认环境已经构建过：

```bash
$TO --help | head -5
```

如果 `$TO` 不存在，先回到第 0 课完成构建。

## 3. 写 Dialect 声明

文件：`lib/Dialect/Box/BoxDialect.td`

```tablegen
#ifndef LIB_DIALECT_BOX_BOXDIALECT_TD_
#define LIB_DIALECT_BOX_BOXDIALECT_TD_

include "mlir/IR/OpBase.td"

def Box_Dialect : Dialect {
  let name = "box";
  let summary = "A tiny dialect for boxing values";
  let description = [{
    The box dialect wraps a value of any type in a !box.box<T> type.
  }];

  let cppNamespace = "::mlir::tutorial::box";

  let useDefaultTypePrinterParser = 1;
}

#endif  // LIB_DIALECT_BOX_BOXDIALECT_TD_
```

逐项解释：

| 字段 | 作用 |
|---|---|
| `name = "box"` | IR 中的前缀：`box.pack`、`!box.box<T>` |
| `cppNamespace` | 生成代码的 C++ 命名空间 |
| `useDefaultTypePrinterParser = 1` | 根据 Type 的 `assemblyFormat` 自动生成类型 parser/printer |
| `summary` / `description` | 文档元数据，不影响行为 |

生成产物（构建后）：

- `BoxDialect.h.inc`：`BoxDialect` 类声明
- `BoxDialect.cpp.inc`：构造函数、析构函数、TypeID

## 4. 写 Type 声明

文件：`lib/Dialect/Box/BoxTypes.td`

```tablegen
#ifndef LIB_DIALECT_BOX_BOXTYPES_TD_
#define LIB_DIALECT_BOX_BOXTYPES_TD_

include "BoxDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Box_Type<string name, string typeMnemonic> : TypeDef<Box_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Box : Box_Type<"Box", "box"> {
  let summary = "A boxed value";
  let description = [{
    A type that wraps a single value of any MLIR type.
  }];

  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

#endif  // LIB_DIALECT_BOX_BOXTYPES_TD_
```

关键点：

- `TypeDef<Box_Dialect, name>`：声明这是 `box` dialect 的类型
- `mnemonic = "box"`：IR 中写成 `!box.box<...>`
- `parameters = (ins "Type":$elementType)`：类型携带一个 **Type 参数**
  - 生成的 C++ 访问器是 `getElementType()`
  - 参数参与类型唯一化：`!box.box<i32>` 和 `!box.box<i64>` 是不同类型
- `assemblyFormat = "`<` $elementType `>`"`：定义语法
  - 字面量 `<`、`>`
  - `$elementType` 打印/解析内部的类型

生成的 C++ 类大致是：

```cpp
class BoxType : public Type::TypeBase<BoxType, Type, BoxTypeStorage> {
public:
  static BoxType get(MLIRContext *context, Type elementType);
  static constexpr StringLiteral getMnemonic() { return {"box"}; }
  Type getElementType() const;
};
```

## 5. 写 Op 声明

文件：`lib/Dialect/Box/BoxOps.td`

```tablegen
#ifndef LIB_DIALECT_BOX_BOXOPS_TD_
#define LIB_DIALECT_BOX_BOXOPS_TD_

include "BoxDialect.td"
include "BoxTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Box_PackOp : Op<Box_Dialect, "pack", [Pure]> {
  let summary = "Pack a value into a box";
  let description = [{
    Wraps a value of any type in a !box.box<T>.
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs Box:$box);
  let assemblyFormat =
      "$input attr-dict `:` qualified(type($input)) `->` qualified(type($box))";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def Box_UnpackOp : Op<Box_Dialect, "unpack", [Pure]> {
  let summary = "Unpack a boxed value";
  let description = [{
    Extracts the value stored in a !box.box<T>.
  }];

  let arguments = (ins Box:$box);
  let results = (outs AnyType:$value);
  let assemblyFormat =
      "$box attr-dict `:` qualified(type($box)) `->` qualified(type($value))";
  let hasVerifier = 1;
  let hasFolder = 1;
}

#endif  // LIB_DIALECT_BOX_BOXOPS_TD_
```

逐项解释：

### `Op<Box_Dialect, "pack", [Pure]>`

- 在 `box` dialect 中定义 `box.pack`
- `Pure`：无副作用，可参与 LICM/CSE/DCE

### `arguments` / `results`

- `AnyType:$input`：输入可以是任意一个 MLIR 类型
- `Box:$box`：结果必须是 `!box.box<T>`
- `Box_UnpackOp` 反过来：输入必须是 box，结果任意

### `assemblyFormat`

```tablegen
"$input attr-dict `:` qualified(type($input)) `->` qualified(type($box))"
```

生成的文本语法：

```mlir
%b = box.pack %v : i32 -> !box.box<i32>
```

- `$input`：SSA 操作数
- `attr-dict`：可选属性字典
- `qualified(type(...))`：打印带 dialect 前缀的完整类型
- 反引号内是字面量

> 不熟悉 `assemblyFormat` 的逐段翻译，见
> [如何阅读和编写 assemblyFormat](assembly-format.md)。

### `hasVerifier = 1`

声明 `PackOp::verify()` / `UnpackOp::verify()` 会在 `BoxOps.cpp` 中实现。

### `hasFolder = 1`

声明 `fold()` 会实现。没有它，即使写了 fold 方法也不会被调用。

## 6. 写 `.h` 包装头

### `lib/Dialect/Box/BoxDialect.h`

```cpp
#ifndef LIB_DIALECT_BOX_BOXDIALECT_H_
#define LIB_DIALECT_BOX_BOXDIALECT_H_

#include "mlir/include/mlir/IR/DialectImplementation.h"

#include "lib/Dialect/Box/BoxDialect.h.inc"

#endif
```

作用：把生成的 `BoxDialect` 类声明暴露给其他代码。

### `lib/Dialect/Box/BoxTypes.h`

```cpp
#ifndef LIB_DIALECT_BOX_BOXTYPES_H_
#define LIB_DIALECT_BOX_BOXTYPES_H_

#include "mlir/include/mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Box/BoxTypes.h.inc"

#endif
```

`GET_TYPEDEF_CLASSES` 让生成的 `.h.inc` 展开出 `BoxType` 类声明。

### `lib/Dialect/Box/BoxOps.h`

```cpp
#ifndef LIB_DIALECT_BOX_BOXOPS_H_
#define LIB_DIALECT_BOX_BOXOPS_H_

#include "lib/Dialect/Box/BoxDialect.h"
#include "lib/Dialect/Box/BoxTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "lib/Dialect/Box/BoxOps.h.inc"

#endif
```

注意这里必须包含：

- `Builders.h`：生成的 op 类里会用到 `OpBuilder` / `ImplicitLocOpBuilder`，
  只有前向声明会编译失败（我们在实践中就遇到了）
- `SideEffectInterfaces.h`：`Pure` 生成的接口需要它的完整定义

## 7. 写 `.cpp` 实现

### `lib/Dialect/Box/BoxDialect.cpp`

```cpp
#include "lib/Dialect/Box/BoxDialect.h"

#include "lib/Dialect/Box/BoxOps.h"
#include "lib/Dialect/Box/BoxTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"
#include "mlir/include/mlir/IR/Builders.h"

#include "lib/Dialect/Box/BoxDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Box/BoxTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/Box/BoxOps.cpp.inc"

namespace mlir {
namespace tutorial {
namespace box {

void BoxDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Box/BoxTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Box/BoxOps.cpp.inc"
      >();
}

} // namespace box
} // namespace tutorial
} // namespace mlir
```

解释：

- `BoxDialect.cpp.inc`：生成 dialect 构造函数/析构函数
- `GET_TYPEDEF_CLASSES` + `BoxTypes.cpp.inc`：生成类型的定义、parser/printer
- `GET_OP_CLASSES` + `BoxOps.cpp.inc`：生成 op 的定义、parser/printer、注册列表
- `initialize()` 里 `GET_TYPEDEF_LIST` / `GET_OP_LIST` 再 include 一次，
  展开成 `addTypes<BoxType>()` / `addOperations<PackOp, UnpackOp>()`
- `TypeSwitch.h`：生成的类型 printer 需要它；缺了会报 incomplete type

### `lib/Dialect/Box/BoxOps.cpp`

```cpp
#include "lib/Dialect/Box/BoxOps.h"

#include "mlir/include/mlir/IR/PatternMatch.h"

namespace mlir {
namespace tutorial {
namespace box {

LogicalResult PackOp::verify() {
  auto boxTy = cast<BoxType>(getBox().getType());
  if (boxTy.getElementType() != getInput().getType())
    return emitOpError("input type must match box element type");
  return success();
}

LogicalResult UnpackOp::verify() {
  auto boxTy = cast<BoxType>(getBox().getType());
  if (boxTy.getElementType() != getValue().getType())
    return emitOpError("result type must match box element type");
  return success();
}

OpFoldResult PackOp::fold(PackOp::FoldAdaptor) {
  // pack(unpack(b)) -> b
  if (auto unpack = getInput().getDefiningOp<UnpackOp>())
    return unpack.getBox();
  return {};
}

OpFoldResult UnpackOp::fold(UnpackOp::FoldAdaptor) {
  // unpack(pack(v)) -> v
  if (auto pack = getBox().getDefiningOp<PackOp>())
    return pack.getInput();
  return {};
}

} // namespace box
} // namespace tutorial
} // namespace mlir
```

#### verifier

`box.pack` 的 ODS 约束只保证结果是 `Box` 类型，但不保证 box 的元素类型和
`$input` 一致。所以 verifier 检查：

```text
!box.box<i64> 的结果 != i32 的输入  -> 报错
```

`cast<BoxType>` 是安全的，因为 ODS 的 `Box:$box` 已经保证它是 `BoxType`。

#### fold

`OpFoldResult` 可以返回：

- `Value`：用已有 SSA 值替换（这里就是这种情况）
- `Attribute`：常量属性
- 空：不能折叠

`unpack(pack(x))` 直接返回 `pack.getInput()`；`pack(unpack(b))` 返回
`unpack.getBox()`。运行 `--canonicalize` 时，folder 会反复应用这些规则。

## 8. 接入构建系统

### `lib/Dialect/Box/CMakeLists.txt`

```cmake
set(LLVM_TARGET_DEFINITIONS BoxOps.td)
mlir_tablegen(BoxOps.h.inc -gen-op-decls)
mlir_tablegen(BoxOps.cpp.inc -gen-op-defs)
mlir_tablegen(BoxTypes.h.inc -gen-typedef-decls -typedefs-dialect=box)
mlir_tablegen(BoxTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=box)
mlir_tablegen(BoxDialect.h.inc -gen-dialect-decls -dialect=box)
mlir_tablegen(BoxDialect.cpp.inc -gen-dialect-defs -dialect=box)
add_public_tablegen_target(MLIRBoxOpsIncGen)

add_mlir_doc(BoxDialect BoxDialect Box/ -gen-dialect-doc)

add_mlir_dialect_library(MLIRBox
    BoxDialect.cpp
    BoxOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/Dialect/Box

    DEPENDS
    MLIRBoxOpsIncGen

    LINK_LIBS PUBLIC
    )
```

作用：

- 六条 `mlir_tablegen` 生成 dialect/type/op 的 `.inc`
- `add_public_tablegen_target` 把生成目标导出
- `add_mlir_dialect_library(MLIRBox ...)` 编译手写的两个 `.cpp`，并依赖生成目标

### 让 CMake 找到它

`lib/Dialect/CMakeLists.txt`：

```cmake
add_subdirectory(Box)
add_subdirectory(Noisy)
add_subdirectory(Poly)
```

### 让 tutorial-opt 链接它

`tools/CMakeLists.txt` 的 `LIBS` 里加入：

```cmake
MLIRBox
```

### 重新构建

```bash
cd /home/zhouzhou/npu/mlir-tutorial
ninja -C build -j 100 tutorial-opt
```

CMake 会自动重新配置（因为 CMakeLists 变了）。如果报找不到目标，手动运行一次：

```bash
cmake -S . -B build
```

### Bazel（可选）

仓库同时支持 Bazel。对应文件：

- `lib/Dialect/Box/BUILD`：`td_library` + 三个 `gentbl_cc_library` + `cc_library`
- `tools/BUILD`：依赖 `//lib/Dialect/Box`

本环境没有安装 Bazel，Bazel 路径未实际验证；CMake 路径已验证。

## 9. 注册到 tutorial-opt

`tools/tutorial-opt.cpp`：

```cpp
#include "lib/Dialect/Box/BoxDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::tutorial::box::BoxDialect>();
  registry.insert<mlir::tutorial::poly::PolyDialect>();
  ...
}
```

注册后，`tutorial-opt` 才能解析 `box.*` 和 `!box.box<T>`。

验证：

```bash
$TO --help 2>&1 | grep -w box
```

`box` 会出现在 Available Dialects 中。

## 10. 验证解析和打印

创建 `tests/box_syntax.mlir`：

```mlir
// RUN: tutorial-opt %s | FileCheck %s

// CHECK-LABEL: func.func @roundtrip
// CHECK: box.pack
// CHECK: box.unpack
// CHECK: return
func.func @roundtrip(%x: i32) -> i32 {
  %b = box.pack %x : i32 -> !box.box<i32>
  %y = box.unpack %b : !box.box<i32> -> i32
  return %y : i32
}

// CHECK-LABEL: func.func @from_arg
// CHECK: box.unpack
// CHECK: return
func.func @from_arg(%b: !box.box<i32>) -> i32 {
  %v = box.unpack %b : !box.box<i32> -> i32
  return %v : i32
}

// CHECK-LABEL: func.func @box_of_poly
// CHECK: box.pack
// CHECK: box.unpack
func.func @box_of_poly(%p: !poly.poly<5>) -> !poly.poly<5> {
  %b = box.pack %p : !poly.poly<5> -> !box.box<!poly.poly<5>>
  %v = box.unpack %b : !box.box<!poly.poly<5>> -> !poly.poly<5>
  return %v : !poly.poly<5>
}
```

运行：

```bash
$TO tests/box_syntax.mlir
```

输出（节选）：

```mlir
func.func @roundtrip(%arg0: i32) -> i32 {
  %0 = box.pack %arg0 : i32 -> !box.box<i32>
  %1 = box.unpack %0 : !box.box<i32> -> i32
  return %1 : i32
}
func.func @box_of_poly(%arg0: !poly.poly<5>) -> !poly.poly<5> {
  %0 = box.pack %arg0 : !poly.poly<5> -> !box.box<!poly.poly<5>>
  %1 = box.unpack %0 : !box.box<!poly.poly<5>> -> !poly.poly<5>
  return %1 : !poly.poly<5>
}
```

手动执行 FileCheck：

```bash
$TO tests/box_syntax.mlir | $BIN/FileCheck tests/box_syntax.mlir
```

## 11. 验证 verifier

创建 `tests/box_verifier.mlir`：

```mlir
// RUN: tutorial-opt %s -verify-diagnostics

func.func @bad_pack(%x: i32) -> i32 {
  // expected-error @+1 {{input type must match box element type}}
  %b = box.pack %x : i32 -> !box.box<i64>
  return %x : i32
}
```

运行：

```bash
$TO tests/box_verifier.mlir -verify-diagnostics
```

退出码 0 表示测试通过（期望的错误确实出现了）。

如果想看原始错误：

```bash
$TO tests/box_verifier.mlir
```

```text
error: 'box.pack' op input type must match box element type
```

## 12. 验证 fold

创建 `tests/box_fold.mlir`：

```mlir
// RUN: tutorial-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @roundtrip
// CHECK-SAME: %[[X:.*]]: i32
// CHECK-NOT: box.pack
// CHECK-NOT: box.unpack
// CHECK: return %[[X]] : i32
func.func @roundtrip(%x: i32) -> i32 {
  %b = box.pack %x : i32 -> !box.box<i32>
  %y = box.unpack %b : !box.box<i32> -> i32
  return %y : i32
}

// CHECK-LABEL: func.func @box_of_poly
// CHECK-SAME: %[[P:.*]]: !poly.poly<5>
// CHECK-NOT: box.pack
// CHECK-NOT: box.unpack
// CHECK: return %[[P]] : !poly.poly<5>
func.func @box_of_poly(%p: !poly.poly<5>) -> !poly.poly<5> {
  %b = box.pack %p : !poly.poly<5> -> !box.box<!poly.poly<5>>
  %v = box.unpack %b : !box.box<!poly.poly<5>> -> !poly.poly<5>
  return %v : !poly.poly<5>
}
```

运行：

```bash
$TO tests/box_fold.mlir --canonicalize
```

输出：

```mlir
func.func @roundtrip(%arg0: i32) -> i32 {
  return %arg0 : i32
}
func.func @box_of_poly(%arg0: !poly.poly<5>) -> !poly.poly<5> {
  return %arg0 : !poly.poly<5>
}
```

`box.pack` / `box.unpack` 配对消失了，说明 fold 生效。

### 一次性验证全部测试

```bash
$TO tests/box_syntax.mlir | $BIN/FileCheck tests/box_syntax.mlir
$TO tests/box_fold.mlir --canonicalize | $BIN/FileCheck tests/box_fold.mlir
$TO tests/box_verifier.mlir -verify-diagnostics
```

三条命令都无输出、退出码为 0，即全部通过。

这些 `.mlir` 文件放在 `tests/` 下，lit 会自动收集（`glob_lit_tests()` /
`add_lit_testsuite`）；本课程用上面的手动命令执行。

## 13. 查看生成代码与宏展开

```bash
find build/lib/Dialect/Box -name '*.inc' | sort
```

输出：

```text
build/lib/Dialect/Box/BoxDialect.cpp.inc
build/lib/Dialect/Box/BoxDialect.h.inc
build/lib/Dialect/Box/BoxOps.cpp.inc
build/lib/Dialect/Box/BoxOps.h.inc
build/lib/Dialect/Box/BoxTypes.cpp.inc
build/lib/Dialect/Box/BoxTypes.h.inc
```

对应关系：

| `.td` | 生成文件 |
|---|---|
| `BoxDialect.td` | `BoxDialect.h.inc` / `BoxDialect.cpp.inc` |
| `BoxTypes.td` | `BoxTypes.h.inc` / `BoxTypes.cpp.inc` |
| `BoxOps.td` | `BoxOps.h.inc` / `BoxOps.cpp.inc` |

### `GET_TYPEDEF_LIST` / `GET_OP_LIST` 的展开

`BoxTypes.cpp.inc` 里有：

```cpp
#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST
::mlir::tutorial::box::BoxType
#endif
```

`BoxOps.cpp.inc` 里有：

```cpp
#ifdef GET_OP_LIST
#undef GET_OP_LIST
::mlir::tutorial::box::PackOp,
::mlir::tutorial::box::UnpackOp
#endif
```

所以：

```cpp
addTypes<
#define GET_TYPEDEF_LIST
#include "BoxTypes.cpp.inc"
      >();
```

预处理后就是：

```cpp
addTypes<::mlir::tutorial::box::BoxType>();
```

同理：

```cpp
addOperations<::mlir::tutorial::box::PackOp, ::mlir::tutorial::box::UnpackOp>();
```

解释：

- `#include` 是文本替换：`.inc` 的内容被粘贴到 `addTypes<...>` 里面
- `#ifdef GET_TYPEDEF_LIST` 决定只展开列表这一段
- `#undef` 防止宏泄漏，并允许同一个 `.inc` 在同一个文件里被多次 include
  （例如 `GET_TYPEDEF_CLASSES` 一次、`GET_TYPEDEF_LIST` 一次）
- 列表里的逗号由 TableGen 生成好，正好符合 C++ 模板参数语法
- 新增类型/op 后重新构建，TableGen 会自动把新类加进列表，不需要手改

查看实际 TableGen 命令：

```bash
ninja -C build -t commands lib/Dialect/Box/BoxOps.h.inc
```

这些 `.inc` 是构建产物，不要手改。

## 14. 注册原理

本节解释 `addTypes` / `addOperations` 在哪里定义、展开后做了什么。

### 函数定义

它们不是 TableGen 生成的，而是 `mlir::Dialect` 的成员函数模板，定义在：

```text
externals/llvm-project/mlir/include/mlir/IR/Dialect.h
```

核心代码：

```cpp
class Dialect {
public:
  template <typename... Args>
  void addOperations() {
    (void)std::initializer_list<int>{
        0, (RegisteredOperationName::insert<Args>(*this), 0)...};
  }

  template <typename... Args>
  void addTypes() {
    (void)std::initializer_list<int>{0, (addType<Args>(), 0)...};
  }

private:
  template <typename T>
  void addType() {
    addType(T::getTypeID(), AbstractType::get<T>(*this));
    detail::TypeUniquer::registerType<T>(context);
  }
};
```

生成的 `BoxDialect` 继承它：

```cpp
class BoxDialect : public ::mlir::Dialect {
  void initialize();
  ...
};
```

生成的构造函数会调用 `initialize()`：

```cpp
BoxDialect::BoxDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<BoxDialect>()) {
  initialize();
}
```

### 类型注册

所以 `addTypes<BoxType>()` 实际会展开为对：

```cpp
addType<BoxType>();
```

`addType<T>()` 做三件事：

1. `T::getTypeID()`：取生成的类型 ID
2. `AbstractType::get<T>(*this)`：用生成的 mnemonic/parser/printer 构造类型元数据
3. `TypeUniquer::registerType<T>(context)`：注册 type storage/uniquing

非模板的 `Dialect::addType` 实现在：

```text
externals/llvm-project/mlir/lib/IR/MLIRContext.cpp
```

### 操作注册

`addOperations<PackOp, UnpackOp>()` 类似地展开为：

```cpp
RegisteredOperationName::insert<PackOp>(*this);
RegisteredOperationName::insert<UnpackOp>(*this);
```

`RegisteredOperationName::insert<T>` 定义在：

```text
externals/llvm-project/mlir/include/mlir/IR/OperationSupport.h
```

它把 op 名字、parser/printer、verifier 等注册到 `MLIRContext`。

### 分工

分工可以总结为：

```text
MLIR 核心：addTypes / addOperations 模板
TableGen：生成 BoxType / PackOp / UnpackOp 类 + 类名列表
BoxDialect::initialize()：把两者接起来
```

## 15. 常见错误排查

| 错误 | 原因 | 修复 |
|---|---|---|
| `custom op 'box.pack' is unknown` | 忘记注册 dialect | 在 `tutorial-opt.cpp` 加 `registry.insert<BoxDialect>()` |
| `invalid use of incomplete type 'mlir::ImplicitLocOpBuilder'` | `BoxOps.h` 缺少 `Builders.h` | 加入 `#include "mlir/include/mlir/IR/Builders.h"` |
| `invalid use of incomplete type 'llvm::TypeSwitch'` | `BoxDialect.cpp` 缺少 `TypeSwitch.h` | 加入 `#include "llvm/include/llvm/ADT/TypeSwitch.h"` |
| `Cannot find target MLIRBox` | 忘记 `add_subdirectory(Box)` 或 `MLIRBox` 未加入 LIBS | 检查两处 CMake 修改 |
| 修改 `.td` 后没有新类 | 没有重新构建 | `ninja -C build tutorial-opt` |
| verifier 没触发 | 忘记 `hasVerifier = 1` | 在 `.td` 中加上 |
| fold 没生效 | 忘记 `hasFolder = 1`，或没跑 `--canonicalize` | 加上并运行 canonicalize |
| TableGen 报 `expected ...` | `.td` 语法错误 | 检查分号、`include`、include guard |
| `type constraint failed` | `Box` 结果类型不对 | 检查 assemblyFormat 和 `!box.box<...>` |

## 16. 如何继续扩展

### 新增一个 op

1. 在 `BoxOps.td` 加 `def`
2. 重建
3. 通常不需要改 `.h`/`.cpp`，因为 `GET_OP_CLASSES`/`GET_OP_LIST` 会自动包含
4. 如果需要 verifier/fold/canonicalizer，再加 `hasVerifier`/`hasFolder` 并在
   `BoxOps.cpp` 实现

例如加一个 `box.is_empty`（需要额外语义）或 `box.map` 时，流程完全一样。

### 加 canonicalization pattern

1. 新建 `BoxPatterns.td`，用 `Pat<...>` 写规则
2. CMake 增加 `mlir_tablegen(BoxCanonicalize.cpp.inc -gen-rewriters)`
3. `BoxOps.cpp` include `BoxCanonicalize.cpp.inc`
4. 在 op 的 `getCanonicalizationPatterns` 注册

### 加 lowering

`box.pack`/`box.unpack` 目前只有高层语义，没有 conversion。如果要 lower 到标准
dialect，需要新增一个 Conversion pass（例如把 `!box.box<T>` 转成 `T`，把
`pack`/`unpack` 消掉），类似第 9/10 课的 `PolyToStandard`。

## 17. 清理这个 dialect

如果只是练习、不想保留：

1. 删除 `lib/Dialect/Box/`
2. 删除 `tests/box_*.mlir`
3. 从 `lib/Dialect/CMakeLists.txt` 删除 `add_subdirectory(Box)`
4. 从 `tools/CMakeLists.txt` 删除 `MLIRBox`
5. 从 `tools/BUILD` 删除 `//lib/Dialect/Box`
6. 从 `tools/tutorial-opt.cpp` 删除 include 和 `registry.insert`
7. 重新构建

## 18. 验收清单

- [ ] 能说出 `.td` / `.h` / `.cpp` / `.inc` 四类文件的分工
- [ ] 能独立写出一个带 Type 参数的 TypeDef
- [ ] 能写出两个最简单 op 的 ODS 定义
- [ ] 知道 `GET_TYPEDEF_CLASSES` / `GET_OP_CLASSES` / `GET_TYPEDEF_LIST` /
      `GET_OP_LIST` 的作用
- [ ] 能完成 CMake 接入和 dialect 注册
- [ ] 能写 verifier 和 fold，并用 `--canonicalize` 验证
- [ ] 能用 FileCheck 写测试
- [ ] 遇到 incomplete type 错误时知道补哪个头文件

---

这就是一个完整、可运行的最小 dialect。照着这份教程，你可以把 `box` 替换成
任何领域概念（`pair`、`rational`、`matrix` 等），流程完全一致。

## 参考

- [第 4 课：Defining a New Dialect](04-new-dialect.md)
- [如何阅读和编写 assemblyFormat](assembly-format.md)
- [MLIR Defining Dialects: Operations](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
- [MLIR Defining Dialects: Attributes and Types](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/)