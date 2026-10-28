# 第 3 课：Using TableGen for Passes

> 对应文章：[Using TableGen for Passes](https://jeremykun.com/2023/08/10/mlir-using-tablegen-for-passes/)。
>
> 本课解释 `Passes.td` 到底做了什么：它不是运行时逻辑，而是**构建期的代码生成器**。
> 我们会看清 `mlir-tblgen` 生成的三部分代码、宏协议、注册流程，以及 CMake/Bazel
> 如何把它接进构建系统。

## 1. 学习目标

- 理解为什么 MLIR 用 TableGen 声明 pass，而不是手写全部样板代码
- 能读懂 `Passes.td` 的字段：summary、description、dependentDialects、作用对象
- 理解 `Passes.h.inc` 中三段生成代码分别是什么
- 理解 `GEN_PASS_DECL_*` / `GEN_PASS_DEF_*` / `GEN_PASS_REGISTRATION` 宏协议
- 能追出 `--affine-full-unroll` 从声明到命令行可用的完整链路
- 能手动运行 `mlir-tblgen` 并解释构建系统如何调用它
- （可选）动手新增一个 pass 声明与实现

## 2. 为什么需要 TableGen

如果全部手写，一个 pass 至少需要：

- 一个类，继承 `Pass` / `OperationPass<T>`
- `getArgument()`：命令行名字
- `getDescription()`：帮助信息
- `getDependentDialects()`：需要加载的 dialect
- `clonePass()`：pass manager 复制 pass 时使用
- 工厂函数 `createXxx()`
- 注册函数 `registerXxx()`
- 如果有选项，还要写解析、打印、复制逻辑

这些代码高度重复、容易写错，而且改一个字段要改多处。

TableGen 的做法是：**在 `.td` 文件里声明一次，构建时生成上述所有样板代码**。

```tablegen
def AffineFullUnroll : Pass<"affine-full-unroll"> {
  let summary = "Fully unroll all affine loops";
  let dependentDialects = ["mlir::affine::AffineDialect"];
}
```

一句话：**TableGen 不是解释器，也不是运行时；它是一个 C++ 代码生成器。**

## 3. 声明文件 `Passes.td`

仓库中的完整声明：

```tablegen
include "mlir/Pass/PassBase.td"

def AffineFullUnroll : Pass<"affine-full-unroll"> {
  let summary = "Fully unroll all affine loops";
  let description = [{
    Fully unroll all affine loops.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}

def AffineFullUnrollPatternRewrite : Pass<"affine-full-unroll-rewrite"> {
  let summary = "Fully unroll all affine loops using the pattern rewrite engine";
  let description = [{
    Fully unroll all affine loops using the pattern rewrite engine.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}
```

字段说明：

| 字段 | 作用 |
|---|---|
| `Pass<"name">` | 声明命令行名字 |
| `summary` | `--help` 中显示的一句话 |
| `description` | 更详细的说明（生成的文档使用） |
| `dependentDialects` | pass 运行前需要加载的 dialect |
| 可选的 `options` | pass 命令行选项（本仓库暂未使用） |
| 可选的 `statistics` | pass 统计信息（本仓库暂未使用） |
| 可选的 `constructor` | 自定义工厂逻辑（本仓库使用默认） |

### 作用对象：`Pass<"name">` vs `Pass<"name", "op">`

`Pass` 的第二个模板参数决定 pass 锚定在哪种 operation 上：

```tablegen
// 通用 pass：OperationPass<>，由 PassManager 决定锚点
def AffineFullUnroll : Pass<"affine-full-unroll"> { ... }

// 只作用于 func.func：OperationPass<func::FuncOp>
def SomeFuncPass : Pass<"some-func-pass", "func::FuncOp"> { ... }
```

本仓库的两个 unroll pass 都是通用形式，在 module 层级运行，然后自己 `walk` 到
`affine.for`。原文章的旧代码锚定在 `func::FuncOp` 上，这是另一种合理设计。

## 4. 生成物：`Passes.h.inc`

CMake 会用 `mlir-tblgen` 生成 `build/lib/Transform/Affine/Passes.h.inc`。
它虽然叫 `.inc`，本质是 C++ 头文件，包含三段内容。

### 4.1 工厂声明（`GEN_PASS_DECL_*`）

```cpp
#ifdef GEN_PASS_DECL_AFFINEFULLUNROLL
std::unique_ptr<::mlir::Pass> createAffineFullUnroll();
#undef GEN_PASS_DECL_AFFINEFULLUNROLL
#endif
```

作用：让其他文件能在**看不到 pass 类定义**的情况下创建 pass。

### 4.2 基类与工厂（`GEN_PASS_DEF_*`）

```cpp
#ifdef GEN_PASS_DEF_AFFINEFULLUNROLL

namespace impl {
template <typename DerivedT>
class AffineFullUnrollBase : public ::mlir::OperationPass<> {
public:
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("affine-full-unroll");
  }
  ::llvm::StringRef getDescription() const override {
    return "Fully unroll all affine loops";
  }

  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect>();
  }

  friend std::unique_ptr<::mlir::Pass> createAffineFullUnroll() {
    return std::make_unique<DerivedT>();
  }
};
} // namespace impl

std::unique_ptr<::mlir::Pass> createAffineFullUnroll() {
  return impl::createAffineFullUnroll();
}

#endif
```

#### 为什么工厂函数前面有 `friend`？

```cpp
friend std::unique_ptr<::mlir::Pass> createAffineFullUnroll() {
  return std::make_unique<DerivedT>();
}
```

`friend` 表示：这是一个**非成员函数**，但被允许访问本类的 private/protected
成员；同时它允许把这个函数的**定义直接写在类体内部**（普通非成员函数不能写在
类里）。

在这里：

- `createAffineFullUnroll()` 仍然是一个自由函数，位于外层的 `impl` 命名空间，
  不是类的成员函数，也没有 `this`
- 它利用 CRTP 的 `DerivedT`（即具体的 `AffineFullUnroll`）来构造 pass
- 把它定义在基类模板里，生成器就不需要为每个 pass 手写工厂实现
- 外层再提供一个同名包装函数，头文件只暴露这个包装的声明

调用链：

```text
mlir::tutorial::createAffineFullUnroll()       // 头文件声明的公开工厂
        └── impl::createAffineFullUnroll()     // friend 定义的内部工厂
                └── std::make_unique<DerivedT>()   // 具体 pass
```

生成的内容包括：

- `getArgumentName()` / `getArgument()`：命令行名字
- `getDescription()`：帮助信息
- `getPassName()` / `getName()`：pass 类名
- `clonePass()`：复制 pass
- `classof()`：支持 `isa` / `dyn_cast`
- `getDependentDialects()`：加载 dialect
- 工厂函数 `createAffineFullUnroll()`

#### `classof()` 与 `isa` / `dyn_cast`

LLVM/MLIR 通常不使用 C++ RTTI（`dynamic_cast`）做类型判断，而是用一套模板工具：

| 工具 | 作用 |
|---|---|
| `isa<T>(x)` | `x` 是不是 `T`？返回 bool |
| `dyn_cast<T>(x)` | 是则返回 `T*`，否则返回 `nullptr` |
| `cast<T>(x)` | 断言一定是 `T`，然后返回 `T*` |

这些工具要求目标类型 `T` 提供一个静态方法：

```cpp
static bool classof(const Base *obj);
```

生成代码里对应的实现是：

```cpp
static bool classof(const ::mlir::Pass *pass) {
  return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
}
```

意思是：**如果这个 `Pass` 的 TypeID 等于 `DerivedT` 的 TypeID，它就应当被当作
`DerivedT`**。

`TypeID` 是 MLIR 自己实现的轻量类型标识（不依赖 C++ RTTI）。有了 `classof`，
就可以写：

```cpp
void inspect(mlir::Pass *p) {
  if (mlir::isa<SomePass>(p)) {
    // p 实际上是 SomePass
  }
  if (auto *sp = mlir::dyn_cast<SomePass>(p)) {
    // 安全地使用 sp
  }
}
```

注意：本项目的具体 pass 类定义在 `.cpp` 里，头文件只暴露工厂函数，所以外部代码
通常不需要直接 `isa<AffineFullUnroll>`；`classof` 是生成的标准 pass 样板，
供包含 `GEN_PASS_DEF_*` 部分的代码使用。

这就是为什么实现文件只需要：

```cpp
struct AffineFullUnroll : impl::AffineFullUnrollBase<AffineFullUnroll> {
  using AffineFullUnrollBase::AffineFullUnrollBase;
  void runOnOperation() { ... }
};
```

其余样板全部由生成代码提供。

### 4.3 注册（`GEN_PASS_REGISTRATION`）

```cpp
#ifdef GEN_PASS_REGISTRATION

inline void registerAffineFullUnroll() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createAffineFullUnroll();
  });
}

inline void registerAffinePasses() {
  registerAffineFullUnroll();
  registerAffineFullUnrollPatternRewrite();
}

#endif
```

`registerPass` 把“创建 pass 的工厂”加入全局注册表。`MlirOptMain` 会遍历这个
注册表，为每个 pass 生成命令行选项。因此 `--affine-full-unroll` 才能被识别。

## 5. 宏协议：一个文件，三种视角

`Passes.h.inc` 被三个地方 include，每次定义不同的宏，看到的内容就不同：

| 文件 | 定义的宏 | 看到的内容 |
|---|---|---|
| `AffineFullUnroll.h` | `GEN_PASS_DECL_AFFINEFULLUNROLL` | 工厂声明 |
| `AffineFullUnroll.cpp` | `GEN_PASS_DEF_AFFINEFULLUNROLL` | 基类 + 工厂实现 |
| `Passes.h` | `GEN_PASS_REGISTRATION` | 注册函数 |

对应代码：

```cpp
// AffineFullUnroll.h
#define GEN_PASS_DECL_AFFINEFULLUNROLL
#include "lib/Transform/Affine/Passes.h.inc"

// AffineFullUnroll.cpp
#define GEN_PASS_DEF_AFFINEFULLUNROLL
#include "lib/Transform/Affine/Passes.h.inc"

// Passes.h
#define GEN_PASS_REGISTRATION
#include "lib/Transform/Affine/Passes.h.inc"
```

为什么要这样设计？

- **避免重复定义**：基类和工厂只在需要它们的 TU 中出现
- **隐藏实现**：头文件只暴露工厂声明，pass 类定义留在 `.cpp`
- **编译更快**：每个 TU 只解析自己需要的部分
- **可组合**：一个 `.td` 文件可以同时服务声明星、实现和注册

#### 什么是 TU（translation unit）

TU（**翻译单元**）是**一个 `.cpp` 文件经过预处理后交给编译器的完整代码**：

```text
AffineFullUnroll.cpp
  │ 预处理：展开 #include、替换宏、处理 #ifdef
  ▼
一个完整的 C++ 源文本（这就是一个 TU）
  │ 编译
  ▼
AffineFullUnroll.o
```

要点：

- 每个 `.cpp` 都是一个独立的 TU，编译器分别编译成 `.o`，最后由链接器合并
- 头文件本身**不单独编译**；它被哪个 `.cpp` include，就成为那个 TU 的一部分
- include guard 只能防止同一个 TU 内重复 include，不能防止多个 TU 各包含一次
- 非 inline 函数在整个程序中只能有一个定义（ODR，One Definition Rule）

所以“基类和工厂只在需要它们的 TU 中出现”的意思是：

- `AffineFullUnroll.cpp` 定义了 `GEN_PASS_DEF_*`，它的 TU 才会展开出基类和
  工厂实现
- 其他只 include `AffineFullUnroll.h`（`GEN_PASS_DECL_*`）的 TU 只看到工厂声明，
  不会看到定义
- 这样既避免多个 TU 重复定义同一个符号，也减少不必要的编译工作

## 6. 从声明到命令行的完整链路

```text
Passes.td
   │  mlir-tblgen -gen-pass-decls
   ▼
Passes.h.inc
   │
   ├── AffineFullUnroll.h  (GEN_PASS_DECL_*)   → createAffineFullUnroll() 声明
   ├── AffineFullUnroll.cpp (GEN_PASS_DEF_*)   → 基类 + 工厂实现
   └── Passes.h (GEN_PASS_REGISTRATION)        → registerAffinePasses()
                                                     │
                                                     ▼
                                        tutorial-opt.cpp 调用 registerAffinePasses()
                                                     │
                                                     ▼
                                        MlirOptMain 注册 --affine-full-unroll
                                                     │
                                                     ▼
                                        $TO input.mlir --affine-full-unroll
```

验证：

```bash
$TO --help 2>&1 | grep affine-full
```

输出：

```text
--affine-full-unroll          - Fully unroll all affine loops
--affine-full-unroll-rewrite  - Fully unroll all affine loops using the pattern rewrite engine
```

## 7. 构建系统如何生成它

### CMake

`lib/Transform/Affine/CMakeLists.txt`：

```cmake
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name=Affine)
add_public_tablegen_target(MLIRAffineFullUnrollPasses)
add_mlir_doc(Passes AffinePasses ./ -gen-pass-doc)
```

逐行解释：

| 语句 | 作用 |
|---|---|
| `set(LLVM_TARGET_DEFINITIONS Passes.td)` | 告诉 TableGen 输入文件 |
| `mlir_tablegen(Passes.h.inc -gen-pass-decls -name=Affine)` | 生成 pass 声明，组名 `Affine` |
| `add_public_tablegen_target(...)` | 导出生成目标，供其他 target 依赖 |
| `add_mlir_doc(... -gen-pass-doc)` | 生成 pass 文档 |

库目标通过 `DEPENDS MLIRAffineFullUnrollPasses` 保证“先跑 TableGen，再编译”。

### Bazel

```python
gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [(
        ["-gen-pass-decls", "-name=Affine"],
        "Passes.h.inc",
    )],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [...],
)
```

### 手动运行 mlir-tblgen

构建系统实际执行的命令（简化）：

```bash
mkdir -p build/scratch

$BIN/mlir-tblgen -gen-pass-decls -name Affine \
  -I$TOP/lib/Transform/Affine \
  -I$LLVM/llvm/include -I$LLVM/build/include \
  -I$LLVM/mlir/include -I$LLVM/build/tools/mlir/include \
  -I$TOP -I$LLVM -I$TOP/build \
  $TOP/lib/Transform/Affine/Passes.td \
  -o build/scratch/Passes.h.inc
```

我们实际运行过，生成的 `build/scratch/Passes.h.inc` 与构建目录里的文件完全一致：

```bash
diff -q build/scratch/Passes.h.inc build/lib/Transform/Affine/Passes.h.inc
# 无输出 = 完全相同
```

这说明 TableGen 是**确定性的纯代码生成**，没有隐藏魔法。

## 8. 实验

### 实验 1：观察生成文件的三段

```bash
INC=build/lib/Transform/Affine/Passes.h.inc

# 工厂声明
grep -n 'GEN_PASS_DECL_AFFINEFULLUNROLL' $INC

# 基类
grep -n 'class AffineFullUnrollBase' $INC

# 注册
grep -n 'registerAffinePasses' $INC
```

### 实验 2：手动生成并对比

按第 7 节的命令生成 `build/scratch/Passes.h.inc`，然后：

```bash
diff -u build/lib/Transform/Affine/Passes.h.inc build/scratch/Passes.h.inc | head
```

预期：没有差异。

### 实验 3：查看注册效果

```bash
$TO --help 2>&1 | grep -E 'affine-full-unroll|mul-to-add'
```

### 实验 4（可选）：新增一个自己的 pass

目标：新增 `--count-affine-for`，打印 IR 中 `affine.for` 的数量。

**第 1 步**：在 `lib/Transform/Affine/Passes.td` 末尾（`#endif` 之前）加入：

```tablegen
def CountAffineFor : Pass<"count-affine-for"> {
  let summary = "Count affine.for operations";
  let description = [{
    Count and print the number of affine.for operations in the module.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}
```

**第 2 步**：新增声明头文件 `lib/Transform/Affine/CountAffineFor.h`：

```cpp
#ifndef LIB_TRANSFORM_AFFINE_COUNTAFFINEFOR_H_
#define LIB_TRANSFORM_AFFINE_COUNTAFFINEFOR_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

#define GEN_PASS_DECL_COUNTAFFINEFOR
#include "lib/Transform/Affine/Passes.h.inc"

}  // namespace tutorial
}  // namespace mlir

#endif
```

然后把它加入 `lib/Transform/Affine/Passes.h`（放在注册 include 之前）：

```cpp
#include "lib/Transform/Affine/CountAffineFor.h"
```

**为什么必须加这一步**：`Passes.h` 定义 `GEN_PASS_REGISTRATION` 后会生成：

```cpp
inline void registerCountAffineFor() {
  ::mlir::registerPass([]() {
    return createCountAffineFor();   // 需要先看到声明
  });
}
```

如果没有 `GEN_PASS_DECL_COUNTAFFINEFOR` 的声明，编译器就会报：

```text
error: ‘createCountAffineFor’ was not declared in this scope
```

已有的两个 pass 之所以不需要额外操作，是因为 `AffineFullUnroll.h` 和
`AffineFullUnrollPatternRewrite.h` 已经分别做了 `GEN_PASS_DECL_*` 声明。

如果使用 Bazel 构建，还要把新头文件加入 `lib/Transform/Affine/BUILD` 的
`AffineFullUnroll` target：

```python
hdrs = [
    "AffineFullUnroll.h",
    "CountAffineFor.h",
    "Passes.h",
],
```

CMake 构建不需要修改 `CMakeLists.txt`。

**第 3 步**：在 `lib/Transform/Affine/AffineFullUnroll.cpp` 顶部把宏改成两个：

```cpp
#define GEN_PASS_DEF_AFFINEFULLUNROLL
#define GEN_PASS_DEF_COUNTAFFINEFOR
#include "lib/Transform/Affine/Passes.h.inc"
```

并在文件末尾加入实现：

```cpp
struct CountAffineFor : impl::CountAffineForBase<CountAffineFor> {
  using CountAffineForBase::CountAffineForBase;

  void runOnOperation() {
    int count = 0;
    getOperation()->walk([&](AffineForOp) { ++count; });
    llvm::outs() << "affine.for count: " << count << "\n";
  }
};
```

还需要包含：

```cpp
#include "llvm/Support/raw_ostream.h"
```

**第 4 步**：重建并运行：

```bash
ninja -C build -j 100 tutorial-opt
$TO --help 2>&1 | grep count-affine-for
$TO tests/affine_loop_unroll.mlir --count-affine-for | head -1
```

预期输出：

```text
affine.for count: 3
```

（第一个函数 1 个循环，第二个函数 2 个嵌套循环。这个 pass 只打印计数、
不修改 IR，所以完整输出中计数之后还会跟着原来的 module。）

**第 5 步（恢复）**：删除 `.td` 中的声明、`CountAffineFor.h`、`Passes.h` 中的
include，以及 `.cpp` 中的实现/宏，重新构建。

这个练习把本课的链路完整走了一遍：`.td` → 生成 base/注册 → 声明头文件 →
实现 → 重建 → 命令行可用。

## 9. 源码位置

| 内容 | 文件 |
|---|---|
| pass 声明 | `lib/Transform/Affine/Passes.td` |
| 生成的声明/基类/注册 | `build/lib/Transform/Affine/Passes.h.inc`（构建产物） |
| 工厂声明头 | `lib/Transform/Affine/AffineFullUnroll.h` |
| pass 实现 | `lib/Transform/Affine/AffineFullUnroll.cpp` |
| 注册入口 | `lib/Transform/Affine/Passes.h` |
| 调用注册 | `tools/tutorial-opt.cpp` |
| CMake 生成规则 | `lib/Transform/Affine/CMakeLists.txt` |
| Bazel 生成规则 | `lib/Transform/Affine/BUILD` |

## 10. 练习

### 练习 1：解释宏协议

为什么 `AffineFullUnroll.h` 和 `AffineFullUnroll.cpp` 都要 include 同一个
`Passes.h.inc`？它们看到的内容为什么不同？

### 练习 2：作用对象

把 `AffineFullUnroll` 的声明改成：

```tablegen
def AffineFullUnroll : Pass<"affine-full-unroll", "func::FuncOp"> { ... }
```

生成的基类会变成什么？当前实现还能编译吗？为什么？

### 练习 3：注册表

如果不调用 `registerAffinePasses()`，`--affine-full-unroll` 还能用吗？
`createAffineFullUnroll()` 还存在吗？

### 练习 4（可选）：完成实验 4

新增 `--count-affine-for`，重建并运行，观察输出。

## 11. 参考答案

<details>
<summary>练习 1</summary>

宏决定生成文件暴露哪一段：头文件定义 `GEN_PASS_DECL_*`，只看到工厂声明；
`.cpp` 定义 `GEN_PASS_DEF_*`，看到基类和工厂实现。这样避免重复定义，并隐藏实现。

</details>

<details>
<summary>练习 2</summary>

基类会从 `OperationPass<>` 变成 `OperationPass<func::FuncOp>`。这时
`getOperation()` 返回 `func::FuncOp`，而 `OperationPass` 的构造和
`FuncOp::getOperationName()` 都需要 `func::FuncOp` 的**完整定义**。因此必须在
包含生成代码（`GEN_PASS_DEF_*`）之前加入：

```cpp
#include "mlir/Dialect/Func/IR/FuncOps.h"
```

否则会报：

```text
error: invalid use of incomplete type ‘class mlir::func::FuncOp’
```

（`LoopUtils.h` 只做了 `class FuncOp;` 的前向声明，前向声明不足以用于继承、
静态成员调用和构造。）

包含之后可以编译，pass 会锚定到每个 `func.func` 上；`getOperation()` 变成
`func::FuncOp`，`walk` 只在该函数内遍历。行为上仍然能找到循环，测试通过。

</details>

<details>
<summary>练习 3</summary>

`--affine-full-unroll` 不再可用，因为注册表里没有这个 pass，`MlirOptMain` 不会
为它生成命令行选项。`createAffineFullUnroll()` 仍然存在（生成代码里有定义），
只是没有被注册。

</details>

<details>
<summary>练习 4</summary>

见实验 4 的步骤与预期输出。

</details>

## 12. 验收清单

- [ ] 能说出 TableGen 在构建期生成哪三类代码
- [ ] 能读懂 `Passes.td` 的字段与作用对象
- [ ] 能解释 `GEN_PASS_DECL_*`、`GEN_PASS_DEF_*`、`GEN_PASS_REGISTRATION`
- [ ] 能追出 `--affine-full-unroll` 从声明到命令行的链路
- [ ] 能手动运行 `mlir-tblgen` 并对比输出
- [ ] 能说出 CMake/Bazel 如何保证先生成再编译

## 下一课

**第 4 课：Defining a New Dialect**：[04-new-dialect.md](04-new-dialect.md)

- ODS 定义 Dialect、Type、Op
- `assemblyFormat` 与类型打印/解析
- `poly` 方言：`!poly.poly<N>`、`poly.add/sub/mul/eval/...`
- 测试 `tests/poly_syntax.mlir`

对应源码：`lib/Dialect/Poly/`