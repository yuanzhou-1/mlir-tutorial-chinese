# 第 1 课：IR 基础与运行/测试 Lowering

> 对应文章：[Running and Testing a Lowering](https://jeremykun.com/2023/08/10/mlir-running-and-testing-a-lowering/)。
>
> 本课是一条完整的 MLIR 开发闭环，分为两个部分：
>
> 1. **读懂 IR**：Operation / Region / Block / SSA，pretty 与 generic 语法
>    （第 2–5 节，实验 1–2）
> 2. **改写并验证 IR**：运行 pass、观察 IR 变化、用 FileCheck 固定行为、
>    用 `mlir-runner` 做功能测试（实验 3–4，第 7 节）
>
> 原文章以 `math.ctlz` lowering 为例展示第 2 部分；本课把读 IR 的基础和这套
> 测试流程放在一起，学完即可独立完成“写 IR → 跑 pass → 验证结果”的全过程。
>
> 完成标志：能读懂 generic 形式的 IR，能运行并验证一个 lowering，能解释 pass
> 前后 IR 为什么发生变化。

## 1. 学习目标

- 理解 MLIR 的三层核心：统一 IR 数据结构、Pass 框架、Dialect 机制
- 掌握 Operation 的组成：操作数、结果、属性、Region、类型
- 能区分 pretty 语法和 generic 语法，并知道前者只是后者的语法糖
- 能用 `--mlir-print-op-generic`、`--mlir-print-ir-after-all`、`--canonicalize` 观察 IR
- 能读懂 `// RUN:` 测试指令，用 FileCheck 验证 lowering
- 能用 `mlir-runner` 做一次端到端功能测试

## 2. MLIR 是什么

MLIR 不是一门语言，而是一套**构建编译器的可复用基础设施**：

1. **统一 IR 数据结构**：Operation / Region / Block / Value / Type / Attribute
2. **Pass 框架**：分析、变换、pipeline 管理、失败传播
3. **Dialect 机制**：任何人都能扩展 IR，加入自己的 op/type/attribute，而不用改 MLIR 本体

`arith.addi`、`func.func`、`affine.for`、`poly.mul` 都只是同一套数据结构里的
不同方言操作。MLIR 自带一批常用 dialect（arith/func/scf/affine/linalg/...），
本教程另外定义了 `poly` 和 `noisy`。

## 3. IR 的嵌套结构

```text
Module（也是一种 Operation）
└── func.func @foo(...)          ← Operation，带一个 Region
    └── Region
        └── Block (%arg0: i32)   ← 基本块，块参数是 SSA value
            ├── arith.constant   ← Operation（无 Region）
            ├── arith.addi
            └── affine.for       ← Operation，内部又有 Region
                └── Block ...
```

| 概念 | 含义 |
|---|---|
| **Operation（op）** | 唯一的指令单位，可带 0~N 个 Region、0~N 个结果 |
| **Region** | op 的“子程序体”，由若干 Block 组成 |
| **Block** | 基本块，有标签 `^bbN`、块参数和若干 op |
| **Value** | SSA 值，一个值只能定义一次，使用必须被定义支配 |
| **Type** | 值的类型，如 `i32`、`tensor<3xi32>`、`!poly.poly<10>` |
| **Attribute** | 编译期常量/元数据，如 `8 : i32`、`dense<[1,2,3]>`、affine map |

**核心认知**：MLIR 用 “Region + 块参数 + yield” 统一建模控制流和循环携带状态，
而不是像传统 IR 那样有单独的 phi 节点。这一点在 `affine.for` 上体现得最清楚。

## 4. Operation 的 generic 形式

任何 op 都可以打印成统一模板：

```text
%result = "dialect.op"(operands...) <{attributes}> (regions...) : (operand_types) -> (result_types)
```

例子：

```mlir
%c = "arith.constant"() <{value = 8 : i32}> : () -> i32
%r = "arith.muli"(%a, %c) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
"func.return"(%r) : (i32) -> ()
```

对照表：

| 元素 | `arith.muli` 中 |
|---|---|
| op 名 | `"arith.muli"` |
| 操作数 | `(%a, %c)` |
| 属性 | `<{overflowFlags = ...}>` |
| Region | 无 |
| 结果 | `%r`，类型在 `->` 后面 |
| 操作数类型 | `(i32, i32)` |

而 pretty 形式：

```mlir
%r = arith.muli %a, %c : i32
```

只是 `arith` dialect 为这个 op 定义的 `assemblyFormat`。**pretty 语法不是 IR 的
本质**；第 3 课我们会亲手为 `poly` 方言定义这种语法糖。

## 5. 示例逐行拆解

输入文件：`tests/affine_loop_unroll.mlir`（第一个函数）

```mlir
func.func @test_single_nested_loop(%buffer: memref<4xi32>) -> (i32) {
  %sum_0 = arith.constant 0 : i32
  %sum = affine.for %i = 0 to 4 iter_args(%sum_iter = %sum_0) -> i32 {
    %t = affine.load %buffer[%i] : memref<4xi32>
    %sum_next = arith.addi %sum_iter, %t : i32
    affine.yield %sum_next : i32
  }
  return %sum : i32
}
```

| 语法 | 含义 |
|---|---|
| `func.func @name(%arg: type) -> type` | func 方言的 op，`@name` 是 symbol，块参数是函数入参 |
| `%sum_0 = arith.constant 0 : i32` | 定义一个 SSA value |
| `affine.for ... { ... }` | 带 Region 的 op；Region 内有自己的 Block |
| `iter_args(%sum_iter = %sum_0)` | 循环携带变量（块参数），初值 `%sum_0` |
| `affine.yield %sum_next` | 把值从 Region“返回”给循环 op 的结果 |
| `return %sum : i32` | `func.return` 的简写 |

### MLIR 打印时会重命名 SSA 值

注意：我们写的是 `%buffer`、`%sum_0`，打印出来却变成 `%arg0`、`%c0_i32`。
**SSA 名字不是语义的一部分**，只是打印器为了可读性自动生成的。分析 IR 时不要依赖名字。

## 6. 实验：读 IR 与观察 pass

下面四个实验从“读”逐步过渡到“改”：实验 1–2 观察 IR 的两种打印形式，实验 3
运行第一个变换 pass，实验 4 用 PassManager 的调试输出观察每一步变化。

先设置路径：

```bash
cd /home/zhouzhou/npu/mlir-tutorial
BIN=$PWD/externals/llvm-project/build/bin
TO=$PWD/build/tools/tutorial-opt
mkdir -p build/scratch          # 本课程所有临时文件都放在项目内
```

### 实验 1：pretty 形式

```bash
$TO tests/affine_loop_unroll.mlir
```

输出（节选）：

```mlir
module {
  func.func @test_single_nested_loop(%arg0: memref<4xi32>) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg1 = 0 to 4 iter_args(%arg2 = %c0_i32) -> (i32) {
      %1 = affine.load %arg0[%arg1] : memref<4xi32>
      %2 = arith.addi %arg2, %1 : i32
      affine.yield %2 : i32
    }
    return %0 : i32
  }
  ...
}
```

### 实验 2：generic 形式

```bash
$TO tests/affine_loop_unroll.mlir --mlir-print-op-generic
```

`affine.for` 部分（节选）：

```mlir
// 文件开头的 affine map 定义
#map  = affine_map<(d0) -> (d0)>     // 恒等映射：load 的下标 = 归纳变量
#map1 = affine_map<() -> (0)>        // 无输入，常量 0：循环下界
#map2 = affine_map<() -> (4)>        // 无输入，常量 4：循环上界

%8 = "affine.for"(%7) <{lowerBoundMap = #map1, upperBoundMap = #map2,
                         step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 1>}> ({
^bb0(%arg6: index, %arg7: i32):
  %9 = "affine.load"(%arg5, %arg6) <{map = #map}> : (memref<4xi32>, index) -> i32
  %10 = "arith.addi"(%arg7, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  "affine.yield"(%10) : (i32) -> ()
}) : (i32) -> i32
```

#### 三类输入：操作数、属性、块参数

一个 op 从外部拿到的信息分三种，在 generic 形式里位置完全不同：

| 类别 | 在 generic 里的位置 | `affine.for` 里是什么 |
|---|---|---|
| **操作数（operands）** | op 名后面的 `(...)` | 循环携带变量的**初值** `%7`（打印后的 `%c0_i32`，原始源码中叫 `%sum_0`） |
| **属性（attributes）** | `<{...}>` 里 | 下界 map、上界 map、步长 |
| **Region 块参数** | Region 内入口块的 `^bb0(...)` | 归纳变量 + 当前迭代的 iter_arg |

三者的本质区别：

- **操作数是 SSA 值**，由 op 外面的 IR 产生，op 消费它们；
- **属性是编译期常量/元数据**，不是 SSA 值；
- **块参数由 op 自己产生**，供 Region 内部的指令使用。它们不是操作数。

#### 逐个对照

pretty 形式：

```mlir
%0 = affine.for %arg1 = 0 to 4 iter_args(%arg2 = %c0_i32) -> (i32) {
  %1 = affine.load %arg0[%arg1] : memref<4xi32>
  %2 = arith.addi %arg2, %1 : i32
  affine.yield %2 : i32
}
```

generic 形式（加上注释）：

```mlir
%8 = "affine.for"(%7)                 // %7 = iter_args 初值（%c0_i32），唯一的操作数
     <{lowerBoundMap = #map1,         // 下界 = affine_map<() -> (0)>，属性
       upperBoundMap = #map2,         // 上界 = affine_map<() -> (4)>，属性
       step = 1 : index,              // 步长，属性
       operandSegmentSizes = array<i32: 0, 0, 1>}>
                                      // [下界操作数个数, 上界操作数个数, 初值个数]
                                      // = 0, 0, 1：唯一的操作数 %7 属于初值组
     ({
     ^bb0(%arg6: index,               // 块参数 1 = 归纳变量（对应 %arg1）
          %arg7: i32):                // 块参数 2 = 当前 iter_arg（对应 %arg2）
       %9 = "affine.load"(%arg5, %arg6)    // %arg5 = 外层 memref（对应 %arg0）
       %10 = "arith.addi"(%arg7, %9)
       "affine.yield"(%10)            // 把“下一轮”的 iter_arg 值交还给 op
     }) : (i32) -> i32
     // (i32)  = 操作数 %7 的类型
     // -> i32 = op 结果的类型（必须与 affine.yield 的值类型一致）
```

#### 最容易卡住的三个点

1. **为什么边界不是操作数？**
   因为这里是常量 `0` 和 `4`，被编码成属性里的 affine map
   （`#map1 = affine_map<() -> (0)>`，输入个数为 0）。
   如果写成动态边界 `affine.for %i = 0 to %n`，`%n` 就会出现在操作数列表里，
   `operandSegmentSizes` 也会随之变化。

2. **`operandSegmentSizes = array<i32: 0, 0, 1>` 是什么？**
   `affine.for` 的操作数有三个可变长分组：下界操作数、上界操作数、初值。
   generic 形式把所有操作数平铺在一个括号里，需要这个属性告诉解析器每组各有几个。
   这里是 `0, 0, 1`：下界 0 个、上界 0 个、初值 1 个 → 唯一的 `%7` 是初值。

3. **归纳变量和 iter_arg 为什么是块参数，而不是操作数？**
   操作数是从 op **外部**传进来的；归纳变量和当前 iter_arg 是 op **每轮迭代时
   自己生成**、供 Region 内部使用的值，所以放在 Region 入口块的参数位置。
   `affine.yield` 把下一轮要用的 iter_arg 值交还给 op，循环结束时 op 的结果
   `%8` 就是最后一次 yield 的值。

#### 用迭代过程理解

对 `sum = Σ buffer[i] (i=0..3)`：

| 迭代 | 归纳变量 `%arg6` | iter_arg `%arg7` | yield `%10` |
|---|---|---|---|
| 0 | 0 | `%7`（初值 0） | `0 + buffer[0]` |
| 1 | 1 | `0 + buffer[0]` | `0 + buffer[0] + buffer[1]` |
| 2 | 2 | `... + buffer[1]` | `... + buffer[2]` |
| 3 | 3 | `... + buffer[2]` | `... + buffer[3]` |

循环结束后 `%8` = 最后一次 yield 的值 = 总和。

> SSA 名字不用纠结：pretty 里的 `%arg0/%arg1/%arg2` 在 generic 打印时被重新编号成了
> `%arg5/%arg6/%arg7`，**按角色对应，不要按名字对应**。

#### 动态边界会改变操作数分组

把上界改成动态值，可以直接验证“常量边界是属性、动态边界是操作数”这条规则：

```mlir
func.func @dyn(%n: index, %init: i32, %buf: memref<?xi32>) -> i32 {
  %r = affine.for %i = 0 to %n iter_args(%s = %init) -> i32 {
    %t = affine.load %buf[%i] : memref<?xi32>
    %s2 = arith.addi %s, %t : i32
    affine.yield %s2 : i32
  }
  return %r : i32
}
```

把上面的 IR 保存为 `build/scratch/dyn_for.mlir`，然后打印 generic 形式：

```bash
$TO build/scratch/dyn_for.mlir --mlir-print-op-generic
```

`affine.for` 部分：

```mlir
#map1 = affine_map<() -> (0)>
#map2 = affine_map<()[s0] -> (s0)>

%0 = "affine.for"(%arg0, %arg1)
     <{lowerBoundMap = #map1, upperBoundMap = #map2,
       operandSegmentSizes = array<i32: 0, 1, 1>, step = 1 : index}>
     ({ ... }) : (index, i32) -> i32
```

- `%arg0`（即 `%n`）成为**上界操作数**
- `%arg1`（即 `%init`）仍是初值操作数
- `operandSegmentSizes = [0, 1, 1]`：下界 0 个、上界 1 个、初值 1 个

### 实验 3：全展开循环

**这一步在做什么**：把循环“拆掉”——把循环体复制 N 份，并将归纳变量替换为每一轮的
常量。输入的 `0 to 4` 循环会被展开成 4 份连续的 load/add，循环 op 本身消失。

用语义等价的伪代码表示，目标是变成：

```mlir
// 展开后的逻辑（展示 iter_arg 如何串起来）
%0 = affine.load %buffer[0]        // i = 0
%1 = arith.addi %sum_0, %0         // iter_arg 传递
%2 = affine.load %buffer[1]        // i = 1
%3 = arith.addi %1, %2
%4 = affine.load %buffer[2]        // i = 2
%5 = arith.addi %3, %4
%6 = affine.load %buffer[3]        // i = 3
%7 = arith.addi %5, %6
return %7
```

**为什么要做**：

- 消除循环控制开销（比较、分支、跳转）
- 让每一轮的下标成为常量，暴露给后续的常量折叠、CSE、指令调度、向量化
- 代价是代码体积膨胀，所以通常只对**编译期已知且较小**的 trip count 做全展开

**前提**：`loopUnrollFull` 要求循环上下界在编译期可确定；动态边界会展开失败。

`--affine-full-unroll` 的手动实现输出（节选）：

```mlir
#map = affine_map<(d0) -> (d0 + 1)>
...
func.func @test_single_nested_loop(%arg0: memref<4xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = affine.load %arg0[%c0] : memref<4xi32>
  %1 = arith.addi %c0_i32, %0 : i32
  %2 = affine.apply #map(%c0)
  %3 = affine.load %arg0[%2] : memref<4xi32>
  %4 = arith.addi %1, %3 : i32
  ...
  return %10 : i32
}
```

`--affine-full-unroll-rewrite`（pattern rewrite 实现）输出（节选）：

```mlir
func.func @test_single_nested_loop(%arg0: memref<4xi32>) -> i32 {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %0 = affine.load %arg0[%c0] : memref<4xi32>
  %1 = affine.load %arg0[%c1] : memref<4xi32>
  ...
}
```

两者**语义相同，IR 形状不同**：

- 手动 pass 只做展开，索引计算保留为 `affine.apply #map`
- pattern rewrite 走的是 `applyPatternsAndFoldGreedily`，driver 在应用 pattern 时
  顺带做 fold，`affine.apply` 被折叠成常量

**给手动版补一次 canonicalize**：手动 pass 只负责展开，留下了 `affine.apply #map(%c0)`
和 `0 + x` 这类可以化简的表达式。`canonicalize` 是 MLIR 自带的清理 pass，会反复
应用各 op 的 fold 和 canonicalization pattern，把编译期能算出来的表达式折叠成
常量、删掉恒等运算。命令行上的多个 pass 从左到右执行，所以下面这条命令相当于
“先手动展开，再自动清理”：

```bash
$TO tests/affine_loop_unroll.mlir --affine-full-unroll --canonicalize
```

输出变成：

```mlir
func.func @test_single_nested_loop(%arg0: memref<4xi32>) -> i32 {
  %0 = affine.load %arg0[0] : memref<4xi32>
  %1 = affine.load %arg0[1] : memref<4xi32>
  %2 = arith.addi %0, %1 : i32
  %3 = affine.load %arg0[2] : memref<4xi32>
  %4 = arith.addi %2, %3 : i32
  %5 = affine.load %arg0[3] : memref<4xi32>
  %6 = arith.addi %4, %5 : i32
  return %6 : i32
}
```

### 实验 4：观察 pass 前后的完整 IR

```bash
$TO tests/affine_loop_unroll.mlir --affine-full-unroll --canonicalize --mlir-print-ir-after-all
```

**这一步在做什么**：它不是一个新的变换，而是 PassManager 提供的调试观测功能——
每执行完一个 pass，就把当时的 IR 完整打印出来，并附带头部标记：

```text
// -----// IR Dump After AffineFullUnroll (affine-full-unroll) //----- //
#map = affine_map<(d0) -> (d0 + 1)>
...
```

这是调试 pass 的第一神器：当一条 pipeline 有很多 pass 时，可以精确看到**是哪一个
pass 把 IR 改成了什么样**。

#### 输出流向

IR dump 默认写到 **stderr**，而 pipeline 的最终结果写到 **stdout**。所以：

```bash
# 只看最终结果（默认去掉了 dump）
$TO input.mlir --affine-full-unroll 2>/dev/null

# 只看 dump（丢弃最终结果）
$TO input.mlir --affine-full-unroll --mlir-print-ir-after-all 2>&1 >/dev/null

# 两者一起看
$TO input.mlir --affine-full-unroll --mlir-print-ir-after-all 2>&1
```

#### 多个 pass 会打印多次

```bash
$TO tests/affine_loop_unroll.mlir \
  --affine-full-unroll --canonicalize --mlir-print-ir-after-all \
  2>&1 >/dev/null | grep 'IR Dump'
```

输出：

```text
// -----// IR Dump After AffineFullUnroll (affine-full-unroll) //----- //
// -----// IR Dump After Canonicalizer (canonicalize) //----- //
```

加上 `--mlir-print-ir-before-all` 后，每个 pass 前后都会打印（共 4 段）。
这能清楚展示“同一条 IR 被一步步改成最终形态”的过程。

#### 常用变体

| 选项 | 作用 |
|---|---|
| `--mlir-print-ir-before-all` | 每个 pass 执行前打印 |
| `--mlir-print-ir-after-all` | 每个 pass 执行后打印 |
| `--mlir-print-ir-before=<pass>` | 只打印指定 pass 之前 |
| `--mlir-print-ir-after=<pass>` | 只打印指定 pass 之后 |
| `--mlir-print-ir-after-change` | 只在 IR 真正发生变化时打印 |
| `--mlir-print-ir-module-scope` | 即使 pass 运行在嵌套 op 上，也打印整个 module |
| `--mlir-print-ir-tree-dir=<dir>` | 每个阶段输出到单独文件，便于 diff |
| `--mlir-disable-threading` | 关闭多线程，保证 dump 顺序稳定、可复现 |

#### 实用建议

- IR 很大时不要用 `-all`，改用 `--mlir-print-ir-after=<具体 pass>`
- 想比较两个阶段，用 `--mlir-print-ir-tree-dir=build/scratch/ir-dumps` 然后 `diff`
- 结果不稳定或交错时，加 `--mlir-disable-threading`
- 这个功能来自 `PassManager::enableIRPrinting`，是 Pass 基础设施的一部分

## 7. 运行与测试 lowering

第 6 节的实验展示了如何运行和观察 pass。本节把这套做法系统化，以 MLIR 内置的
`math.ctlz` lowering 为例，完整走一遍 lowering 的开发闭环：运行 → 断言 → 功能验证。

### 7.1 运行一个 lowering

输入 `tests/ctlz_simple.mlir`：

```mlir
// RUN: mlir-opt %s --convert-math-to-funcs=convert-ctlz | FileCheck %s

func.func @main(%arg0: i32) -> i32 {
  // CHECK-NOT:  math.ctlz
  // CHECK:  call
  %0 = math.ctlz %arg0 : i32
  func.return %0 : i32
}
```

运行：

```bash
$BIN/mlir-opt tests/ctlz_simple.mlir --convert-math-to-funcs=convert-ctlz
```

输出（节选）：

```mlir
module {
  func.func @main(%arg0: i32) -> i32 {
    %0 = call @__mlir_math_ctlz_i32(%arg0) : (i32) -> i32
    return %0 : i32
  }
  func.func private @__mlir_math_ctlz_i32(%arg0: i32) -> i32
      attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    ...
    %1 = scf.if %0 -> (i32) {
      scf.yield %c32_i32 : i32
    } else {
      %2:2 = scf.for ... {
        ...
      }
      scf.yield %2#1 : i32
    }
    return %1 : i32
  }
}
```

`math.ctlz` 被替换成 `call`，并生成了一个 helper 函数（其中是 `scf.if`/`scf.for`
实现的循环）。**lowering 不一定是“一条 op 换一条 op”**，它可以生成新的函数和
控制流。

### 7.2 FileCheck：给 lowering 写断言

`// RUN:` 是给 lit 的指令，不是 MLIR 语法：

```text
// RUN: mlir-opt %s --convert-math-to-funcs=convert-ctlz | FileCheck %s
```

- `%s` 表示当前测试文件
- `FileCheck %s` 从 stdin 读 pass 输出，用源文件里的 `CHECK:` 注释做匹配
- 常用指令：
  - `CHECK`: 后面必须出现匹配行
  - `CHECK-NOT`: 两个 CHECK 之间不能出现
  - `CHECK-NEXT`: 必须是紧邻的下一行
  - `CHECK-SAME`: 必须和上一个匹配在同一行
  - `CHECK-LABEL`: 匹配函数等大块边界，方便定位
  - `%[[NAME:regex]]`: 捕获变量，后文用 `[[NAME]]` 引用

手动执行等价于 lit 的断言：

```bash
$BIN/mlir-opt tests/ctlz_simple.mlir --convert-math-to-funcs=convert-ctlz \
  | $BIN/FileCheck tests/ctlz_simple.mlir
# 无输出即通过（退出码 0）
```

如果断言失败，FileCheck 会打印 expected/actual 的差异。可以故意改坏一条 `CHECK:`
（比如改成 `CHECK: foo`）再运行，观察失败信息。

### 7.3 lit：批量组织测试

- lit 扫描测试文件，执行其中的每条 `// RUN:` 行，检查退出码
- `tests/lit.cfg.py` 配置测试格式（`ShTest`）、文件后缀、工具搜索路径
- 本项目在 Bazel 下通过 `bazel/lit.bzl` 的 `glob_lit_tests()` 生成测试目标
- 本环境尚未安装 lit，所以课程目前手动执行 RUN 命令。安装方式：

```bash
cd /home/zhouzhou/npu/mlir-tutorial
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# 注意：仓库自带的 lit.cfg.py 依赖 Bazel 的 RUNFILES_DIR，
# 脱离 Bazel 直接运行可能需要调整 PATH，因此本课程以手动 RUN 为准
```

### 7.4 功能测试：真正执行生成的代码

FileCheck 验证的是“IR 长什么样”；功能测试验证的是“程序跑起来结果对不对”。两者
互补：一个 lowering 可能生成语法正确但语义错误的 IR，只有真正执行才能发现。

`tests/ctlz_runner.mlir` 用 `mlir-runner` 完成这件事：先把 `math.ctlz` 一路
lowering 到 LLVM dialect，再 JIT 编译并执行入口函数，最后检查返回值。

#### 测试输入

文件里有两个入口函数，分别覆盖 i32 和 i64：

```mlir
func.func @test_7i32_to_29() -> i32 {
  %arg = arith.constant 7 : i32
  %0 = math.ctlz %arg : i32
  func.return %0 : i32
}

func.func @test_7i64_to_61() -> i64 {
  %arg = arith.constant 7 : i64
  %0 = math.ctlz %arg : i64
  func.return %0 : i64
}
```

`math.ctlz` 计算二进制前导零的个数：

- 7 的二进制是 `111`，在 32 位里前导零有 `32 - 3 = 29` 个
- 在 64 位里有 `64 - 3 = 61` 个

#### 第一步：lowering 到 LLVM dialect

RUN 行里的 pipeline 是：

```bash
$BIN/mlir-opt tests/ctlz_runner.mlir \
  -pass-pipeline="builtin.module( \
     convert-math-to-funcs{convert-ctlz}, \
     func.func(convert-scf-to-cf,convert-arith-to-llvm), \
     convert-func-to-llvm, \
     convert-cf-to-llvm, \
     reconcile-unrealized-casts)"
```

每个 pass 的职责：

| pass | 作用 |
|---|---|
| `convert-math-to-funcs{convert-ctlz}` | 把 `math.ctlz` 换成 `call` + 生成的 helper 函数 |
| `func.func(convert-scf-to-cf, convert-arith-to-llvm)` | 在每个函数内部，把 `scf` 结构化控制流降到 `cf`，把 `arith` 降到 LLVM dialect |
| `convert-func-to-llvm` | 函数签名、调用、返回 → `llvm.func` / `llvm.call` / `llvm.return` |
| `convert-cf-to-llvm` | 分支 → `llvm.br` / `llvm.cond_br` |
| `reconcile-unrealized-casts` | 清理 conversion 过程中产生的 `unrealized_conversion_cast` |

注意 pipeline 的嵌套写法：

- `builtin.module(...)`：这些 pass 在 module 层级运行
- `func.func(...)`：这两个 pass 只作用于每个 `func.func` 内部

lowering 后的 LLVM dialect（节选）：

```mlir
module {
  llvm.func @test_7i32_to_29() -> i32 {
    %0 = llvm.mlir.constant(7 : i32) : i32
    %1 = llvm.mlir.constant(29 : i32) : i32
    llvm.return %1 : i32
  }
  ...
}
```

这个测试里的 `ctlz(7)` 在编译期就被算成了常量 29。更一般的情况会保留对
`@__mlir_math_ctlz_i32` helper 函数的调用。

#### 第二步：用 mlir-runner 执行

把上一步的输出通过管道交给 `mlir-runner`：

```bash
$BIN/mlir-opt tests/ctlz_runner.mlir \
  -pass-pipeline="builtin.module( \
     convert-math-to-funcs{convert-ctlz}, \
     func.func(convert-scf-to-cf,convert-arith-to-llvm), \
     convert-func-to-llvm, \
     convert-cf-to-llvm, \
     reconcile-unrealized-casts)" \
  | $BIN/mlir-runner -e test_7i32_to_29 -entry-point-result=i32
```

输出：

```text
29
```

参数说明：

| 参数 | 作用 |
|---|---|
| `-e test_7i32_to_29` | 选择要执行的入口函数 |
| `-entry-point-result=i32` | 声明入口函数的返回类型，runner 会把结果打印出来 |
| `--shared-libs=...` | 需要动态库时链接额外库（本例不需要） |

`mlir-runner` 会把 LLVM dialect 模块翻译成 LLVM IR，JIT 编译成本机代码并执行。
因此它验证的是**整条 lowering pipeline 的语义**，而不只是文本形状。

#### 第二个测试：i64

同一个 pipeline，只换入口函数和返回类型：

```bash
$BIN/mlir-opt tests/ctlz_runner.mlir \
  -pass-pipeline="builtin.module( \
     convert-math-to-funcs{convert-ctlz}, \
     func.func(convert-scf-to-cf,convert-arith-to-llvm), \
     convert-func-to-llvm, \
     convert-cf-to-llvm, \
     reconcile-unrealized-casts)" \
  | $BIN/mlir-runner -e test_7i64_to_61 -entry-point-result=i64
# 61
```

两个测试覆盖不同位宽，可以防止 lowering 只对其中一种正确。

> 注意：这两个测试的输入都是常量，`math.ctlz` 会在编译期被折叠，因此它们主要
> 验证“lowering pipeline + runner”这条闭环；要测试 helper 函数的运行时行为，
> 需要让输入来自函数参数或外部数据。

#### 失败时怎么排查

1. **先看 lowering 输出**：去掉 `| mlir-runner ...`，直接观察 LLVM dialect IR
2. **看每一步**：在 `mlir-opt` 后加 `--mlir-print-ir-after-all`
3. **翻译成 LLVM IR**：
   ```bash
   ... | $BIN/mlir-translate --mlir-to-llvmir
   ```
   检查 `llvm.return` 的值和函数签名是否正确
4. **确认入口名**：`-e` 的名字必须和 `llvm.func` 的名字一致
5. **确认返回类型**：`-entry-point-result` 必须和函数返回类型一致，否则 runner 会报错

#### 与 mlir-cpu-runner 的关系

文章原文使用 `mlir-cpu-runner`。在当前 LLVM 版本中，其他 runner 的功能都已合并
进 `mlir-cpu-runner`，并统一改名为 **`mlir-runner`**（本环境没有构建
`mlir-cpu-runner`）。两者的用途相同：加载 LLVM dialect 模块并执行入口函数。

### 7.5 开发闭环

到这里，一个 lowering 的完整开发闭环已经打通：

```text
写/改 IR → 跑 pass → 看 IR（-print-ir-after-all）
        → 用 FileCheck 固定行为 → 用 mlir-runner 验证语义
```

第 2 课开始写自己的 pass 后，每改一次代码都重复这个循环。

## 8. 源码位置

| 内容 | 文件 |
|---|---|
| 手动展开 pass | `lib/Transform/Affine/AffineFullUnroll.cpp` |
| pattern rewrite 展开 pass | `lib/Transform/Affine/AffineFullUnrollPatternRewrite.cpp` |
| pass 的 TableGen 定义 | `lib/Transform/Affine/Passes.td` |
| 展开测试 | `tests/affine_loop_unroll.mlir` |
| ctlz 测试 | `tests/ctlz_simple.mlir`、`tests/ctlz.mlir` |
| ctlz 功能测试 | `tests/ctlz_runner.mlir` |
| lit 配置 | `tests/lit.cfg.py`、`bazel/lit.bzl` |

两个 pass 的关键代码（第 2 课详解）：

```cpp
// 手动：遍历 IR，直接展开
getOperation()->walk([&](AffineForOp op) { loopUnrollFull(op); });

// pattern：把展开封装成 RewritePattern，交给 greedy driver
applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
```

## 9. 练习

### 练习 1：canonicalize 前后对比

```bash
$TO tests/affine_loop_unroll.mlir --affine-full-unroll | grep -c affine.apply
$TO tests/affine_loop_unroll.mlir --affine-full-unroll --canonicalize | grep -c affine.apply
```

预期：前者非 0，后者为 0。解释原因。

### 练习 2：修改循环上界

复制测试文件，把第一个循环的上界从 `4` 改成 `5`：

```bash
sed 's/0 to 4/0 to 5/' tests/affine_loop_unroll.mlir > build/scratch/unroll5.mlir
$TO build/scratch/unroll5.mlir --affine-full-unroll --canonicalize
```

预测：展开后有几条 `affine.load`？索引分别是多少？

### 练习 3：自己写一段 IR

创建 `build/scratch/my_ir.mlir`：

```mlir
func.func @add_one_and_two() -> i32 {
  %a = arith.constant 1 : i32
  %b = arith.constant 2 : i32
  %c = arith.addi %a, %b : i32
  return %c : i32
}
```

然后：

```bash
$TO build/scratch/my_ir.mlir
$TO build/scratch/my_ir.mlir --mlir-print-op-generic
$TO build/scratch/my_ir.mlir --canonicalize
```

观察 `func.func` 在 generic 形式里为什么是 `: () -> ()`，而函数签名却在
`function_type` 属性里。

### 练习 4：用 FileCheck 验证 ctlz lowering

```bash
# 通过
$BIN/mlir-opt tests/ctlz_simple.mlir --convert-math-to-funcs=convert-ctlz \
  | $BIN/FileCheck tests/ctlz_simple.mlir

# 故意改坏断言后再运行，观察失败信息
cp tests/ctlz_simple.mlir build/scratch/ctlz_broken.mlir
sed -i 's/CHECK:  call/CHECK:  definitely_not_here/' build/scratch/ctlz_broken.mlir
$BIN/mlir-opt build/scratch/ctlz_broken.mlir --convert-math-to-funcs=convert-ctlz \
  | $BIN/FileCheck build/scratch/ctlz_broken.mlir
```

思考：FileCheck 默认从上到下顺序匹配，`CHECK-NOT` 的检查范围是什么？

### 练习 5：功能测试

1. 把 `test_7i32_to_29` 中的常量 `7` 改成 `1`，预测并验证返回值。
2. 用 `-entry-point-result=i64` 执行一个返回 `i32` 的函数，观察报错。
3. 去掉 pipeline 中的 `convert-math-to-funcs{convert-ctlz}`，为什么仍然能
   得到 29？提示：输入是常量。

## 10. 参考答案

<details>
<summary>练习 1</summary>

`--affine-full-unroll` 只展开循环，索引表达式保留为 `affine.apply #map`；
`--canonicalize` 会把作用在常量上的 affine.apply 折叠成常量索引，因此 `affine.apply`
数量变为 0。

</details>

<details>
<summary>练习 2</summary>

5 条 `affine.load`，索引分别为 0、1、2、3、4；随后是 5 个数相加。

</details>

<details>
<summary>练习 3</summary>

`func.func` 是一个“容器 op”：它的值语义输入/输出为空（`: () -> ()`），函数签名
`(i32) -> i32` 是元数据，存放在 `function_type` 属性中。真正的值输入/输出在它
内部的 Region/Block 参数和 `func.return` 上。

`--canonicalize` 后，`arith.addi` 会被常量折叠成 `arith.constant 3`，再经
canonicalize 把返回值直接替换为常量。

</details>

<details>
<summary>练习 4</summary>

第一条命令没有输出、退出码为 0，说明断言全部通过。第二条命令会报
`error: CHECK: expected string not found in input`，并给出匹配位置和实际输入。

`CHECK-NOT` 检查的范围是：从**上一条 CHECK 匹配行的末尾**到**下一条 CHECK 匹配
行的开头**之间。第一条 `CHECK-NOT` 则从输入开头检查到第一条 CHECK 的位置。
它是“在这段区间里不允许出现”，不是“全文不允许出现”。

</details>

<details>
<summary>练习 5</summary>

1. 返回值变为 `31`（`32 - 1 = 31`），`mlir-runner` 打印 `31`。
2. 报错 `Error: only single i64 function result supported`，退出码 1。
   `-entry-point-result` 必须与实际函数返回类型一致。
3. 输入是常量 7，`math.ctlz` 在转换过程中被**常量折叠**成 29，因此即使没有
   `convert-math-to-funcs`，最终 IR 也会直接返回 29。要真正执行 helper 函数，
   需要让值在运行时才确定（例如作为函数参数传入或从内存读取）。

</details>

## 11. 验收清单

- [ ] 能在 generic 形式中指出一个 op 的操作数、属性、Region、结果
- [ ] 能解释 `affine.for` 的 iter_args 对应 generic 中的什么
- [ ] 能说明 `affine.yield` 和 `func.return` 的共同角色
- [ ] 能解释为什么两个 unroll pass 的输出形状不同
- [ ] 能独立写出一个小 IR 并用 generic 形式阅读
- [ ] 能解释 `// RUN:`、`%s` 和 FileCheck 的关系
- [ ] 能手动运行一个 lit 测试并判断通过/失败
- [ ] 能说明 `mlir-runner` 功能测试验证了什么（i32 的 `ctlz(7) = 29`、i64 的 `ctlz(7) = 61`）
- [ ] 能说出 `ctlz_runner.mlir` pipeline 中每个 pass 的职责
- [ ] 能解释 `-e` 和 `-entry-point-result` 的作用

## 下一课

**第 2 课：Writing Our First Pass**：[02-first-pass.md](02-first-pass.md)

- 从零写一个 pass：`runOnOperation()` + `walk()`
- 把同一变换封装成 `OpRewritePattern`
- greedy rewrite driver 与 fold 如何配合
- 亲手新增/修改一个小 pass，并用 FileCheck 固定行为

对应源码：`lib/Transform/Affine/AffineFullUnroll.cpp`、
`AffineFullUnrollPatternRewrite.cpp`
一句话总结