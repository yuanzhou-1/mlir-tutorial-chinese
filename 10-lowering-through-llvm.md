# 第 10 课：Lowering through LLVM

> 对应文章：[Lowering through LLVM](https://jeremykun.com/2023/11/01/mlir-lowering-through-llvm/)。
>
> 上一课把 `poly` lower 到了 `tensor`/`scf`/`arith`。本课继续往下：
> `tensor` → `linalg` → `memref` → `scf`/`cf` → LLVM dialect，
> 最后翻译成 LLVM IR，编译成 x86 机器码并真正运行。

## 1. 学习目标

- 理解“exit dialect”和 `mlir-translate` 的角色
- 看懂 `--poly-to-llvm` 这个自定义 pipeline
- 理解每个阶段的顺序和原因：
  - linalg
  - bufferization
  - memref
  - scf/cf
  - LLVM dialect
- 能把 MLIR 导出成 LLVM IR、编译成对象文件、链接成可执行文件
- 能解释 `tests/poly_to_llvm.mlir` 和 `tests/poly_to_llvm_eval.mlir` 两个测试结果
- 能排查“pipeline 中某个 pass 失败”的问题

## 2. 从 MLIR 到机器码的路线

整体路线：

```text
poly dialect
    │  --poly-to-standard
    ▼
tensor + scf + arith
    │  linalg + bufferization
    ▼
memref + linalg + scf + arith
    │  linalg-to-loops + scf-to-cf
    ▼
memref + cf + arith
    │  cf-to-llvm + arith-to-llvm + func-to-llvm + memref-to-llvm
    ▼
LLVM dialect
    │  mlir-translate --mlir-to-llvmir
    ▼
LLVM IR
    │  llc
    ▼
x86 object file
    │  clang 链接
    ▼
可执行程序
```

这里的目标 dialect 是 LLVM dialect，所以叫 **exit dialect**：

> 从它开始，MLIR 之外的 LLVM 工具链可以接手。

`mlir-translate` 负责把 LLVM dialect 的 MLIR 模块翻译成 LLVM IR。
`llc` 再把 LLVM IR 编译成目标机器码。

---

## 3. `--poly-to-llvm` 是自定义 pipeline（`tools/tutorial-opt.cpp`）

`--poly-to-standard` 是单个 pass（由 `PolyToStandard.td` 生成）。
`--poly-to-llvm` 不是单个 pass，而是手写的 pipeline：

```cpp
void polyToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  // Poly -> tensor/scf/arith
  manager.addPass(mlir::tutorial::poly::createPolyToStandard());
  manager.addPass(mlir::createCanonicalizerPass());

  // tensor/arith -> linalg
  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // tensor -> memref
  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));

  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                       deallocationOptions);

  // linalg -> loops
  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  // 处理 memref.subview
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());

  // scf/cf -> LLVM
  manager.addPass(mlir::createSCFToControlFlowPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());

  // cleanup
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}
```

在 `tools/tutorial-opt.cpp` 里注册：

```cpp
mlir::PassPipelineRegistration<>(
    "poly-to-llvm",
    "Run passes to lower the poly dialect to LLVM",
    polyToLLVMPipelineBuilder);
```

所以：

| 选项 | 类型 |
|---|---|
| `--poly-to-standard` | 单个 pass |
| `--poly-to-llvm` | 多个 pass 组成的 pipeline |

---

## 4. 每个阶段在做什么

### 4.1 `poly-to-standard`

先复用第 9 课的 pass：

```text
!poly.poly<N>  -> tensor<Nxi32>
poly.add       -> arith.addi
poly.mul       -> scf.for + tensor.extract/insert
poly.eval      -> scf.for + Horner
...
```

这一步结束后，已经看不到 `poly.*`，但还有大量 `tensor`、`scf` 和
`arith`。

### 4.2 canonicalize

先做一次规范化，清理：

- 死代码
- 多余的常量
- 可以折叠的 op

这会让后续 lowering 更简单。

### 4.3 elementwise-to-linalg

把 `arith.addi`、`arith.subi` 等逐元素 tensor 运算变成 `linalg.generic`。

为什么需要？

因为后面的 linalg → loops 和 bufferization 更擅长处理 linalg 结构，
而不是零散的 arith tensor 运算。

### 4.4 tensor-to-linalg

处理 `tensor.pad` 等 tensor 结构：

```mlir
%padded = tensor.pad %cst low[0] high[7] { ... }
```

会被改写成 linalg 的 `fill` + `tensor.insert_slice` 等形式。

### 4.5 one-shot bufferize

这一步把 tensor 值语义变成 memref 指针语义：

```text
tensor<10xi32>  ->  memref<10xi32>
```

同时开启：

```cpp
bufferizationOptions.bufferizeFunctionBoundaries = true;
```

这样函数签名里的 tensor 也会变成 memref。

bufferization 之后，IR 里的 tensor op 会逐步消失，改成 memref/load/store。

### 4.6 buffer deallocation

`buildBufferDeallocationPipeline` 负责插入内存释放相关逻辑，避免 buffer 泄漏。

### 4.7 convert-linalg-to-loops

把 linalg op lower 成循环。

注意：linalg 的 loop lowering 通常要求 **buffer 语义**，所以必须放在
bufferization 之后。文章里也提到，在 tensor 语义上直接跑这个 pass 会失败，
因为 linalg op 还在等待 bufferize。

### 4.8 expand-strided-metadata

这一步主要为了 `memref.subview`：

```text
memref.subview -> 展开成 offset/size/stride 的 metadata 计算
```

后续 `finalize-memref-to-llvm` 需要这些元数据。

### 4.9 scf-to-cf

把结构化控制流：

```mlir
scf.for
scf.if
```

lower 成：

```mlir
cf.br
cf.cond_br
```

### 4.10 cf-to-llvm

把 `cf` 分支和 block 变成 LLVM dialect 的：

```mlir
llvm.br
llvm.cond_br
```

以及 LLVM 基本块。

### 4.11 arith-to-llvm

把算术 op 变成 LLVM dialect：

```mlir
arith.addi   -> llvm.add
arith.muli   -> llvm.mul
arith.cmpi   -> llvm.icmp
arith.remui  -> llvm.urem
```

注意顺序：`scf-to-cf` 可能生成新的 `arith.cmpi`，所以 `arith-to-llvm`
要放在它后面。

### 4.12 func-to-llvm

把：

```mlir
func.func
func.call
func.return
```

变成：

```mlir
llvm.func
llvm.call
llvm.return
```

### 4.13 finalize-memref-to-llvm

把 memref 的 load/store、dim、extract 等变成 LLVM pointer/load/store。

### 4.14 reconcile-unrealized-casts

清理前面各阶段可能留下的：

```mlir
builtin.unrealized_conversion_cast
```

如果还有无法消除的 cast，这一步会失败。

### 4.15 最后清理

- canonicalize
- SCCP
- CSE
- symbol DCE

把死代码、重复计算和未使用符号清理掉。

---

## 5. 运行 pipeline

```bash
cd /home/zhouzhou/npu/mlir-tutorial
export TOP=$PWD
export BIN=$TOP/externals/llvm-project/build/bin
export TO=$TOP/build/tools/tutorial-opt

$TO tests/poly_to_llvm_eval.mlir --poly-to-llvm
```

输出已经是 **LLVM dialect**，例如：

```mlir
module {
  llvm.mlir.global private constant @__constant_3xi32(dense<[2, 3, 4]> : tensor<3xi32>)
    {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<3 x i32>

  llvm.func @test_poly_fn(%arg0: i32) -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.addressof @__constant_3xi32 : !llvm.ptr
    ...
    llvm.br ^bb1(%4, %0 : i64, i32)
  ^bb1(%6: i64, %7: i32):
    %8 = llvm.icmp "slt" %6, %2 : i64
    llvm.cond_br %8, ^bb2, ^bb3
  ^bb2:
    %9 = llvm.sub %3, %6 : i64
    %10 = llvm.mul %arg0, %7 : i32
    %11 = llvm.getelementptr inbounds|nuw %5[%9] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %12 = llvm.load %11 : !llvm.ptr -> i32
    %13 = llvm.add %10, %12 : i32
    ...
  ^bb3:
    llvm.return %7 : i32
  }
}
```

这里 `scf.for` 已经变成了 LLVM 基本块 + `llvm.br` / `llvm.cond_br`，
arith 运算变成了 `llvm.mul` / `llvm.add` / `llvm.icmp`。

---

## 6. 翻译成 LLVM IR

`mlir-translate` 负责离开 MLIR：

```bash
$TO tests/poly_to_llvm_eval.mlir --poly-to-llvm \
  > build/scratch/lesson10/eval.mlir

$BIN/mlir-translate --mlir-to-llvmir \
  build/scratch/lesson10/eval.mlir \
  > build/scratch/lesson10/eval.ll
```

`eval.ll` 是真正的 LLVM IR：

```llvm
define i32 @test_poly_fn(i32 %0) {
entry:
  ...
  %cmp = icmp slt i64 %iv, 4
  br i1 %cmp, label %loop, label %exit
loop:
  %idx = sub i64 3, %iv
  %mul = mul i32 %0, %acc
  ...
}
```

注意：

- `mlir-opt` 仍然输出 MLIR
- `mlir-translate --mlir-to-llvmir` 才输出 LLVM IR

---

## 7. 编译成对象文件并链接

用 LLVM 工具把 LLVM IR 编译成 x86 对象：

```bash
$BIN/llc --relocation-model=pic -filetype=obj \
  build/scratch/lesson10/eval.ll \
  -o build/scratch/lesson10/eval.o
```

编译 C 的 main：

```bash
$BIN/clang -c tests/poly_to_llvm_main.c \
  -o build/scratch/lesson10/main.o
```

链接：

```bash
$BIN/clang build/scratch/lesson10/main.o \
    build/scratch/lesson10/eval.o \
    -o build/scratch/lesson10/eval_test
```

运行：

```bash
./build/scratch/lesson10/eval_test
```

输出：

```text
Result: 9
```

我们已经在这个 LLVM 构建里额外构建了 clang：

```text
externals/llvm-project/build/bin/clang
```

版本是 `clang version 22.0.0git`。上面的命令用 `$BIN/clang`，和测试文件里的
`clang` 等价；这两种路径都可以使用。

---

## 8. 两个测试分别验证什么

### 8.1 `poly_to_llvm_eval.mlir`

```mlir
%input = poly.constant dense<[2, 3, 4]>
         : tensor<3xi32> : !poly.poly<3>
%0 = poly.eval %input, %arg : (!poly.poly<3>, i32) -> i32
return %0 : i32
```

C main 用 `arg = 1` 调用，期望：

```text
2 + 3*1 + 4*1² = 9
```

输出：

```text
Result: 9
```

### 8.2 `poly_to_llvm.mlir`

输入：

```mlir
%tens = tensor.splat %arg : tensor<10xi32>
%input = poly.from_tensor %tens : tensor<10xi32> -> !poly.poly<10>
%0 = poly.constant dense<[2, 3, 4]> : tensor<3xi32> : !poly.poly<10>
%1 = poly.add %0, %input : !poly.poly<10>
%2 = poly.mul %1, %1 : !poly.poly<10>
%3 = poly.sub %2, %input : !poly.poly<10>
%4 = poly.eval %3, %arg : (!poly.poly<10>, i32) -> i32
```

在 `arg = 1` 时：

```text
input = 10 个 1，值为 10
%0 = 2 + 3 + 4 = 9
%1 = 10 + 9 = 19
%2 = 19² = 361
%3 = 361 - 10 = 351
```

所以测试期望：

```text
Result: 351
```

---

## 9. 源码位置

| 内容 | 文件 |
|---|---|
| pipeline 定义与注册 | `tools/tutorial-opt.cpp` |
| poly → tensor/scf/arith 各 pattern | `lib/Conversion/PolyToStandard/PolyToStandard.cpp` |
| poly → LLVM 相关 pattern | `lib/Conversion/PolyToLLVM/` |
| 本课测试 | `tests/poly_to_llvm.mlir`、`tests/poly_to_llvm_eval.mlir` |
| C main | `tests/poly_to_llvm_main.c` |

## 10. 为什么要分这么多阶段

一个 pipeline 不是“把所有 pass 随便加上去”，而是有依赖关系：

```text
poly-to-standard
  -> tensor 还在

convert-elementwise-to-linalg
convert-tensor-to-linalg
  -> tensor/arith 变成 linalg

one-shot-bufferize
  -> tensor -> memref

convert-linalg-to-loops
  -> 需要 buffer 语义

scf-to-cf
cf-to-llvm
arith-to-llvm
func-to-llvm
finalize-memref-to-llvm
  -> 逐步离开高层 dialect

reconcile-unrealized-casts
  -> 清理转换过程中的临时类型桥
```

如果顺序错了，常见结果：

- linalg-to-loops 报 “expected linalg op with buffer semantics”
- arith-to-llvm 之后又出现新的 arith op
- memref-to-llvm 之前还有 subview 元数据没展开
- 最后残留 unrealized_conversion_cast

---

## 11. 如何构建自己的 pipeline

文章建议的调试方法：

1. 从空 pipeline 开始
2. 找到当前 IR 里“最高层”的 op
3. 加一个能 lowering 它的 pass
4. 如果 pass 失败，找它依赖的前置 pass
5. 重复，直到输出 LLVM dialect
6. 再用 `mlir-translate` 和 `llc` 验证

本仓库的 `polyToLLVMPipelineBuilder` 就是按这个思路逐步搭出来的。

---

## 12. 常见问题

### 12.1 `--poly-to-llvm` 和 `--poly-to-standard` 的区别

- `--poly-to-standard`：只到 tensor/scf/arith
- `--poly-to-llvm`：继续到 LLVM dialect

`--poly-to-llvm` 是 `PassPipelineRegistration`，不是单个 TableGen pass。

### 12.2 为什么 `mlir-opt` 不能直接输出 LLVM IR

`mlir-opt` 处理的是 MLIR；LLVM IR 是另一种格式，需要
`mlir-translate --mlir-to-llvmir`。

### 12.3 为什么 linalg-to-loops 失败

linalg-to-loops 通常要求 linalg op 已经是 buffer 语义。先做
one-shot-bufferize，再 lower 到 loops。

### 12.4 为什么最后还有 `reconcile-unrealized-casts`

前面多个 conversion 可能产生临时类型 cast；这一步负责清理。如果还有无法消除的
cast，pipeline 会失败，说明某个类型边界没有转换。

### 12.5 `llc` / `clang` 找不到

本仓库的 LLVM 构建里有：

```text
externals/llvm-project/build/bin/llc
externals/llvm-project/build/bin/mlir-translate
```

本环境已经额外构建了 clang：

```text
externals/llvm-project/build/bin/clang
```

它也可以被 lit 的 `llvm_tools_dir` 找到。若换到别的环境，请确认 `clang` 在
`PATH` 中，或直接使用 `$BIN/clang`。

---

## 13. 练习

### 练习 1：解释 `351` 的计算

输入：

```mlir
%tens = tensor.splat %arg : tensor<10xi32>
%input = poly.from_tensor %tens : tensor<10xi32> -> !poly.poly<10>
%0 = poly.constant dense<[2, 3, 4]> : tensor<3xi32> : !poly.poly<10>
%1 = poly.add %0, %input
%2 = poly.mul %1, %1
%3 = poly.sub %2, %input
%4 = poly.eval %3, %arg
```

当 `arg = 1` 时，为什么结果是 `351`？

<details>
<summary>参考答案</summary>

`tensor.splat %arg` 产生 10 个 1，所以 `input` 是：

```text
1 + x + ... + x^9
```

在 `x = 1` 时值为 10。

`%0 = 2 + 3x + 4x²`，在 x=1 时为 9。

```text
%1 = 10 + 9 = 19
%2 = 19² = 361
%3 = 361 - 10 = 351
```

</details>

### 练习 2：为什么 bufferization 在 linalg-to-loops 之前？

<details>
<summary>参考答案</summary>

linalg-to-loops 的许多 lowering 需要 linalg op 具有 buffer 语义（memref）。
one-shot-bufferize 把 tensor 值转换成 memref，之后 linalg op 才能安全地
lower 成循环。

</details>

### 练习 3：为什么 arith-to-llvm 要放在 scf-to-cf 之后？

<details>
<summary>参考答案</summary>

scf-to-cf 在 lowering 结构控制流时可能生成新的 `arith.cmpi` 等 op。
如果先做 arith-to-llvm，这些新生成的 arith op 就没有被转换；所以要把
arith-to-llvm 放在后面。

</details>

### 练习 4：把 `--poly-to-llvm` 的输出接上 `mlir-translate`

写出从 `tests/poly_to_llvm_eval.mlir` 到 LLVM IR 的完整命令。

<details>
<summary>参考答案</summary>

```bash
$TO tests/poly_to_llvm_eval.mlir --poly-to-llvm \
  > build/scratch/lesson10/eval.mlir

$BIN/mlir-translate --mlir-to-llvmir \
  build/scratch/lesson10/eval.mlir \
  > build/scratch/lesson10/eval.ll
```

</details>

## 14. 验收清单

- [ ] 能解释 exit dialect 和 `mlir-translate`
- [ ] 能说出 `--poly-to-llvm` 是 pipeline 而不是单个 pass
- [ ] 能按顺序说出主要 lowering 阶段
- [ ] 能解释为什么 linalg-to-loops 在 bufferization 之后
- [ ] 能解释 `351` 和 `9` 两个结果
- [ ] 能把 MLIR LLVM dialect 导出成 LLVM IR
- [ ] 能用 llc 生成对象并用 clang 链接
- [ ] 能根据失败信息判断缺少哪个前置 pass

## 15. 下一课

**第 11 课：A Global Optimization and Dataflow Analysis**：[11-global-optimization-dataflow.md](11-global-optimization-dataflow.md)

- `Noisy` dialect
- 噪声模型
- `InferIntRangeInterface` 和整数范围分析
- 用 ILP 求解 `reduce_noise` 的插入位置
- 测试：`tests/noisy_*.mlir`

## 参考

- 本课对应文章：
  <https://jeremykun.com/2023/11/01/mlir-lowering-through-llvm/>
- MLIR 官方文档：Pass Management
  <https://mlir.llvm.org/docs/PassManagement/>
- MLIR 官方文档：Bufferization
  <https://mlir.llvm.org/docs/Bufferization/>
- 常用方言速查：`dialects/README.md`（原教程仓库）
- 本课测试：
  - `tests/poly_to_llvm.mlir`
  - `tests/poly_to_llvm_eval.mlir`
  - `tests/poly_to_llvm_main.c`
- 相关源码：
  - `tools/tutorial-opt.cpp`
  - `lib/Conversion/PolyToStandard/PolyToStandard.cpp`