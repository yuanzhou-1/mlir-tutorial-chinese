# FileCheck 使用指南：从 CHECK 到变量捕获

> 本文是课程参考文档，配合第 1 课（IR/测试）、第 6 课（fold）和第 7 课（verifier）。
>
> FileCheck 是 LLVM 自带的一个**文本匹配工具**，不是 MLIR 的一部分。它读取
> 测试文件里的 `CHECK:` 注释，再去检查某个命令的输出是否符合这些注释。

## 1. 学习目标

- 理解 FileCheck 的输入输出关系
- 掌握 `CHECK`、`CHECK-LABEL`、`CHECK-SAME`、`CHECK-NEXT`、`CHECK-NOT`
- 会写变量捕获 `[[NAME:regex]]` 和正则 `{{...}}`
- 会使用 `--check-prefix` 为同一文件写多组断言
- 理解 `RUN:`、`%s`、`%t` 和 lit 的关系
- 会检查 stdout 和 stderr
- 能看懂并在仓库测试中修改 FileCheck 断言

## 2. FileCheck 是什么

FileCheck 做一件事：

```text
读取“检查规则文件”        读取“待检查输出”
        │                       │
        └──────────┬────────────┘
                   ▼
          按顺序匹配，报告差异
```

典型用法：

```bash
some-command | FileCheck test.mlir
```

- `test.mlir` 是**规则文件**，里面写着 `CHECK:` 等注释
- `some-command` 的输出从 **stdin** 进入 FileCheck
- `%s` 在 lit 中表示“当前测试文件”

它和 `grep` 的区别：

- `grep` 只找一行
- FileCheck 按顺序检查多条规则
- 支持“下一行”“同一行”“不能出现”“变量捕获”等结构
- 失败时会给出非常详细的 expected/actual 对比

FileCheck **不理解 MLIR 语义**。它只看文本。

## 3. 最小例子

待检查输出 `build/scratch/filecheck_input.mlir`：

```mlir
func.func @demo(%arg0: i32) -> i32 {
  %0 = arith.constant 1 : i32
  %1 = arith.addi %arg0, %0 : i32
  return %1 : i32
}
```

规则文件 `build/scratch/filecheck_checks.mlir`：

```mlir
// CHECK-LABEL: func.func @demo
// CHECK-SAME: %[[ARG:.*]]: i32
// CHECK: %[[C:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[ARG]], %[[C]] : i32
// CHECK-NEXT: return %[[SUM]] : i32
// CHECK-NOT: arith.muli
```

运行：

```bash
$BIN/FileCheck build/scratch/filecheck_checks.mlir \
  < build/scratch/filecheck_input.mlir
```

无输出、退出码 0，表示通过。

这组规则在说：

1. 先找到 `func.func @demo`
2. 同一行后面要有 `%arg0: i32`，并把 `arg0` 捕获为 `ARG`
3. 后面某处要有 `arith.constant 1`
4. **下一行**必须紧跟 `arith.addi`，并且使用同一个 `ARG`
5. 再下一行必须 `return` 同一个 `SUM`
6. 在这些正匹配之间不能出现 `arith.muli`

## 4. 常用指令

### 4.1 `CHECK:`

```mlir
// CHECK: arith.addi
```

含义：从上一个匹配位置之后开始，找一行包含 `arith.addi`。

- 默认匹配是**子串匹配**，不要求整行相等
- 默认忽略空格和 tab 的差异
- 匹配成功后，位置移动到匹配文本的末尾
- 下一个普通 `CHECK` 可以继续在同一行匹配；要求换行用 `CHECK-NEXT`，
  要求同一行用 `CHECK-SAME`

如果写：

```mlir
// CHECK: foo
// CHECK: bar
```

FileCheck 先找 `foo`，再从 `foo` 所在行之后找 `bar`。顺序不能反。

### 4.2 `CHECK-LABEL:`

```mlir
// CHECK-LABEL: func.func @roundtrip
// CHECK: box.pack
// CHECK: box.unpack
// CHECK: return
```

`CHECK-LABEL` 用来划分**测试块**：

- 它本身也是一次正匹配
- 它之后到下一个 `CHECK-LABEL` 之间的检查，只能在这个块里找
- 防止 `@roundtrip` 的 `CHECK` 意外匹配到后面的 `@box_of_poly`
- 失败时更容易定位是哪个函数

注意：

- `CHECK-LABEL` 不能包含变量定义或变量引用
- 它通常匹配函数名、标签等唯一行

### 4.3 `CHECK-SAME:`

```mlir
// CHECK-LABEL: func.func @demo
// CHECK-SAME: %[[ARG:.*]]: i32
```

含义：`CHECK-SAME` 的匹配必须和**上一个匹配在同一行**。

它常用于“先匹配函数名，再匹配同一行的参数”：

```mlir
// CHECK-LABEL: func.func @roundtrip
// CHECK-SAME: %[[X:.*]]: i32
```

如果 `CHECK-SAME` 前一个匹配在上一行，而目标内容在下一行，就会失败。

`CHECK-SAME` 不能是文件里的第一条指令。

### 4.4 `CHECK-NEXT:`

```mlir
// CHECK: %[[C:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[ARG]], %[[C]] : i32
```

含义：匹配必须发生在**紧邻的下一行**。

- 中间不能插入其他行
- 不能有空白行
- 如果需要检查空行，用 `CHECK-EMPTY`

`CHECK-NEXT` 也要求前面有一个正匹配；不能是第一条。

### 4.5 `CHECK-NOT:`

```mlir
// CHECK-LABEL: func.func @roundtrip
// CHECK-SAME: %[[X:.*]]: i32
// CHECK-NOT: box.pack
// CHECK-NOT: box.unpack
// CHECK: return %[[X]] : i32
```

含义：在**上一个正匹配**和**下一个正匹配**之间，不能出现 `CHECK-NOT`
的字符串。

范围规则：

- 如果它前面没有正匹配，就从输入开头开始检查
- 如果它后面没有正匹配，就检查到输入结尾
- 多个 `CHECK-NOT` 可以连用

它特别适合验证“某个 op 被删除了”：

```mlir
// CHECK-NOT: poly.mul
// CHECK-NOT: poly.add
```

### 4.6 `CHECK-EMPTY:`

```mlir
// CHECK: foo
// CHECK-EMPTY:
// CHECK-NEXT: bar
```

含义：下一行必须是空行（连空格也没有）。

### 4.7 `CHECK-DAG:`

```mlir
// CHECK-DAG: poly.add
// CHECK-DAG: poly.mul
```

`CHECK-DAG` 允许这几条检查**以任意顺序**出现。它适合“这些 op 都存在，
但顺序不重要”的情况。

不要滥用 `CHECK-DAG`：如果顺序本身是语义的一部分，用普通 `CHECK`。

### 4.8 `CHECK-COUNT-N:`

```mlir
// CHECK-COUNT-2: noisy.reduce_noise
// CHECK-NOT: noisy.reduce_noise
```

意思是“正好出现 2 次”，后面的 `CHECK-NOT` 再保证没有多余出现。

### 4.9 常用命令行选项

| 选项 | 作用 |
|---|---|
| `--check-prefix=NAME` | 使用 `NAME:` 前缀，而不是默认的 `CHECK:` |
| `--check-prefixes=A,B` | 同时启用多个前缀 |
| `--match-full-lines` | 正匹配必须覆盖整行（`CHECK-NOT` 不受影响） |
| `--strict-whitespace` | 不忽略空格/tab 差异 |
| `--implicit-check-not=PATTERN` | 在所有正匹配之间隐式禁止 `PATTERN` |
| `--dump-input=fail` | 失败时输出带标注的输入（默认） |
| `--dump-input=always` | 总是输出带标注的输入 |

## 5. 变量捕获和正则

### 5.1 捕获变量

```mlir
// CHECK: %[[C:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[ARG]], %[[C]] : i32
```

语法：

```text
[[NAME:regex]]
```

- `NAME` 是变量名
- `regex` 是匹配模式
- 匹配到的文本存进 `NAME`
- 后面写 `[[NAME]]` 表示“必须和之前捕获的文本完全相同”

例如：

```mlir
// CHECK-SAME: %[[X:.*]]: i32
// CHECK: return %[[X]] : i32
```

如果第一行捕获的是 `arg0`，第二行就必须也是 `%arg0`，不能是 `%arg1`。

### 5.2 变量匹配的范围

在 `%[[X:.*]]` 里：

- `%` 是字面量，必须出现
- `[[X:.*]]` 捕获 `%` 后面的部分
- `.*` 可以匹配任意字符（包括空串）

如果要求至少一个字符，用：

```mlir
%[[X:.+]]
```

变量默认可以在后续 `CHECK` 中继续使用。启用 `--enable-var-scope` 时，
`CHECK-LABEL` 会清除局部变量；以 `$` 开头的变量是全局变量。

### 5.3 正则 `{{...}}`

FileCheck 的正则放在双大括号里：

```mlir
// CHECK: arith.constant {{[0-9]+}} : i32
```

这里的 `{{[0-9]+}}` 是正则，不是字面量。

也可以混合：

```mlir
// CHECK: %[[V:.*]] = arith.constant {{[0-9]+}} : i32
```

- `%[[V:.*]]`：捕获 SSA 名字
- `{{[0-9]+}}`：匹配一个或多个数字

正则使用 POSIX ERE 语法。常见写法：

```text
{{.*}}          任意字符
{{[0-9]+}}      一个或多个数字
{{%[a-z]+}}     百分号 + 小写字母
{{(foo|bar)}}   foo 或 bar
```

### 5.4 变量和正则需要区分

```mlir
// CHECK: %[[ARG:.*]]: i32
```

这里 `[[ARG:.*]]` 是 FileCheck 变量捕获。

```mlir
// CHECK: arith.constant {{.*}} : i32
```

这里 `{{.*}}` 是正则匹配。

两者不要混用。

### 5.5 `CHECK-LABEL` 不能用变量

```mlir
// 错误
// CHECK-LABEL: func.func @[[NAME:.*]]
```

`CHECK-LABEL` 只用于划分块，不允许变量定义或引用。

## 6. 多前缀：一个文件写多组断言

同一个测试文件可能要对不同 pass 的输出做检查。例如
`tests/poly_fold.mlir`：

```mlir
// RUN: tutorial-opt %s --canonicalize | FileCheck %s --check-prefix=CANON
// RUN: tutorial-opt -pass-pipeline="builtin.module(func.func(sccp))" %s | FileCheck %s --check-prefix=SCCP

// CANON-LABEL: func.func @add_constants
// CANON: poly.constant dense<[11, 22, 33]>
// CANON-NOT: poly.add

// SCCP-LABEL: func.func @add_constants
// SCCP: poly.constant dense<[11, 22, 33]>
// SCCP-NOT: poly.add
```

- 第一行 RUN 用 `CANON` 前缀
- 第二行 RUN 用 `SCCP` 前缀
- FileCheck 只处理被选中的前缀，不会把 `CANON` 和 `SCCP` 混在一起
- 同一个前缀里，`CHECK` 可以简写为 `PREFIX` 形式；例如 `CANON:` 对应默认的
  `CHECK:`，`CANON-LABEL` 对应 `CHECK-LABEL`

也可以一次指定多个前缀：

```bash
FileCheck %s --check-prefixes=CANON,SCCP
```

但通常一个 RUN 行只对应一种输出。

## 7. `RUN:`、`%s`、`%t` 和 lit

### 7.1 `RUN:` 不是 MLIR 语法

```mlir
// RUN: tutorial-opt %s --canonicalize | FileCheck %s
```

这是给 **lit** 的注释。lit 会：

1. 读取 `RUN:` 行
2. 把 `%s` 替换成当前测试文件路径
3. 执行整条 shell 命令
4. 检查命令退出码

这里的 `FileCheck %s` 不是 MLIR pass，而是一个独立程序。

### 7.2 `%s` 和 `%t`

| 占位符 | 含义 |
|---|---|
| `%s` | 当前测试文件的路径 |
| `%t` | lit 分配的临时文件路径 |
| `%t.something` | 基于 `%t` 的派生临时文件 |

例如 verifier 测试：

```mlir
// RUN: tutorial-opt %s 2>%t; FileCheck %s < %t
```

- `tutorial-opt %s` 运行测试文件
- `2>%t` 把 stderr 写入临时文件
- `FileCheck %s < %t` 用同一文件的 CHECK 规则检查 stderr

### 7.3 手动等价命令

没有 lit 时，可以直接执行 RUN 行：

```bash
$TO tests/poly_fold.mlir --canonicalize \
  | $BIN/FileCheck tests/poly_fold.mlir --check-prefix=CANON
```

无输出、退出码 0，表示通过。

## 8. stdout 和 stderr

很多 MLIR 工具的：

- 正常 IR 输出 → stdout
- 错误/诊断 → stderr

所以：

```bash
# 只检查正常 IR
$TO input.mlir | $BIN/FileCheck input.mlir

# 检查错误诊断
$TO input.mlir 2>&1 | $BIN/FileCheck input.mlir

# 写入临时文件再检查
$TO input.mlir 2>build/scratch/error.txt
$BIN/FileCheck input.mlir < build/scratch/error.txt
```

`tests/poly_verifier.mlir` 就是检查 stderr 的例子：

```mlir
// RUN: tutorial-opt %s 2>%t; FileCheck %s < %t

// CHECK: to be a 32-bit integer
```

如果只写 `| FileCheck`，错误信息会留在终端，FileCheck 读到的是空输入，测试
就会失败或误判。

### FileCheck vs `-verify-diagnostics`

| 工具 | 检查对象 | 适合 |
|---|---|---|
| FileCheck | 命令输出的文本 | 检查 IR、日志、任意输出 |
| `-verify-diagnostics` | 输入文件里的诊断注释 | 精确检查错误位置和消息 |

verifier 测试可以二选一：

```bash
# FileCheck 方式
$TO tests/poly_verifier.mlir 2>&1 | $BIN/FileCheck tests/poly_verifier.mlir

# 诊断注释方式
$TO build/scratch/poly_verifier_diag.mlir -verify-diagnostics
```

## 9. 仓库中的实际例子

### 9.1 简单正匹配：`tests/box_syntax.mlir`

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
```

它只要求这三个字符串按顺序出现，不要求具体 SSA 名字。

### 9.2 标签 + 变量 + NOT：`tests/box_fold.mlir`

```mlir
// CHECK-LABEL: func.func @roundtrip
// CHECK-SAME: %[[X:.*]]: i32
// CHECK-NOT: box.pack
// CHECK-NOT: box.unpack
// CHECK: return %[[X]] : i32
```

- `CHECK-LABEL` 锁定函数
- `CHECK-SAME` 捕获参数名
- `CHECK-NOT` 确认 pack/unpack 被删除
- 最后确认返回同一个参数

### 9.3 变量 + 同一行：`tests/poly_canonicalize.mlir`

```mlir
// CHECK-LABEL: func.func @test_difference_of_squares
// CHECK-SAME: %[[x:.+]]: !poly.poly<3>,
// CHECK-SAME: %[[y:.+]]: !poly.poly<3>
// CHECK: %[[sum:.+]] = poly.add %[[x]], %[[y]]
// CHECK: %[[diff:.+]] = poly.sub %[[x]], %[[y]]
// CHECK: %[[mul:.+]] = poly.mul %[[sum]], %[[diff]]
```

`CHECK-SAME` 让多个参数检查落在同一行函数签名上；变量保证后面的
`poly.add` 使用的是同一组 `x`/`y`。

### 9.4 多前缀：`tests/poly_fold.mlir`

前面 6 节已经展示。关键点：

```text
CANON-* 只给 --check-prefix=CANON 用
SCCP-*  只给 --check-prefix=SCCP 用
```

### 9.5 检查 stderr：`tests/poly_verifier.mlir`

```mlir
// RUN: tutorial-opt %s 2>%t; FileCheck %s < %t
// CHECK: to be a 32-bit integer
```

它验证的是 verifier 的诊断，不是 IR。

## 10. 调试 FileCheck 失败

FileCheck 失败时会输出：

```text
error: CHECK-NEXT: expected string not found in input
// CHECK-NEXT: ...
               ^
note: scanning from here
...
note: possible intended match here
...
Input was:
...
```

常用排查步骤：

1. 先看 `expected string not found` 是哪一条
2. 看 `scanning from here`：FileCheck 从哪一行开始找
3. 看 `possible intended match`：最接近的实际行
4. 检查是不是：
   - `CHECK` 写得太具体（SSA 名字变了）
   - `CHECK-NEXT` 中间多了空行
   - `CHECK-NOT` 的范围写错
   - 检查了 stdout，但错误在 stderr
   - 用了错误的前缀

`--dump-input=fail` 是默认值；可以显式指定：

```bash
FileCheck --dump-input=always test.mlir < output.txt
```

## 11. 常见错误

### 11.1 `CHECK-NEXT` 以为会跳过空行

不会。`CHECK-NEXT` 要求匹配紧邻的下一行，中间不能有空行或其他行；
需要检查空行时用 `CHECK-EMPTY`。

### 11.2 `CHECK-NOT` 没有后续正匹配

```mlir
// CHECK-NOT: foo
```

如果没有后面的 `CHECK`，它会检查到输入结尾。这有时是想要的，但也很容易
写出过强的断言。最好用正匹配把范围夹住。

### 11.3 `CHECK` 匹配到了后面的函数

用 `CHECK-LABEL` 分块。没有标签时，一个 `CHECK` 可能跨过当前函数匹配到后面
同名模式。

### 11.4 把正则写成了普通文本

```mlir
// CHECK: %{{[0-9]+}}
```

`{{...}}` 才是正则；`[[NAME:...]]` 是变量捕获。两者不要混。

### 11.5 变量捕获了空串

`.*` 可以匹配空串。如果需要至少一个字符，用 `.+`。

### 11.6 检查了 stdout，但输出在 stderr

诊断信息用 `2>&1` 或 `2>%t`。先确认命令输出到哪个流。

## 12. 练习

### 练习 1：给下面的输出写检查

输出：

```mlir
func.func @add(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  return %0 : i32
}
```

要求：

1. 用 `CHECK-LABEL` 锁定 `@add`
2. 捕获两个参数
3. 确认 `arith.addi` 使用了这两个参数
4. 确认下一行返回加法结果

<details>
<summary>参考答案</summary>

```mlir
// CHECK-LABEL: func.func @add
// CHECK-SAME: %[[A:.*]]: i32, %[[B:.*]]: i32
// CHECK: %[[SUM:.*]] = arith.addi %[[A]], %[[B]] : i32
// CHECK-NEXT: return %[[SUM]] : i32
```

</details>

### 练习 2：为什么这个测试会失败？

```mlir
// CHECK-LABEL: func.func @demo
// CHECK: arith.constant 1 : i32
// CHECK-NEXT: arith.addi
```

实际输出：

```mlir
func.func @demo(%arg0: i32) -> i32 {
  %0 = arith.constant 1 : i32
  // 这里是空行
  %1 = arith.addi %arg0, %0 : i32
  return %1 : i32
}
```

<details>
<summary>参考答案</summary>

`CHECK-NEXT` 要求 `arith.addi` 紧跟在常量行的下一行；中间有空行就会失败。
可以把空行删掉，或者改用普通 `CHECK:`（允许跨行），或者用
`CHECK-EMPTY` 明确检查空行。

</details>

### 练习 3：用多前缀区分两种输出

同一个文件：

```mlir
// RUN: my-tool --mode=fold %s | FileCheck %s --check-prefix=FOLD
// RUN: my-tool --mode=no-fold %s | FileCheck %s --check-prefix=NOFOLD
```

请写出两条检查：fold 模式下出现 `poly.constant`，no-fold 模式下保留
`poly.add`。

<details>
<summary>参考答案</summary>

```mlir
// FOLD: poly.constant
// FOLD-NOT: poly.add

// NOFOLD: poly.add
// NOFOLD-NOT: poly.constant
```

</details>

### 练习 4：检查 stderr

命令：

```bash
$TO bad.mlir
```

错误输出在 stderr，内容是：

```text
error: 'poly.eval' op requires each numeric operand to be a 32-bit integer
```

请写出 RUN 行和 CHECK。

<details>
<summary>参考答案</summary>

```mlir
// RUN: tutorial-opt %s 2>&1 | FileCheck %s
// CHECK: requires each numeric operand to be a 32-bit integer
```

也可以把 `2>&1` 换成 `2>%t; FileCheck %s < %t`。

</details>

## 13. 验收清单

- [ ] 能说清 FileCheck 的规则文件和待检查输出分别是什么
- [ ] 能使用 `CHECK`、`CHECK-LABEL`、`CHECK-SAME`、`CHECK-NEXT`
- [ ] 能使用 `CHECK-NOT` 并解释它的检查范围
- [ ] 能捕获变量并在后续检查中复用
- [ ] 能区分 `[[NAME:...]]` 和 `{{...}}`
- [ ] 能用 `--check-prefix` 写多组断言
- [ ] 能解释 `RUN:`、`%s`、`%t` 和 lit 的关系
- [ ] 能检查 stdout 和 stderr
- [ ] 能根据 FileCheck 失败信息定位问题

## 参考

- LLVM FileCheck 官方文档：
  <https://llvm.org/docs/CommandGuide/FileCheck.html>
- 第 1 课：[IR 基础与运行/测试 Lowering](01-ir-basics.md)
- 第 6 课：[Folders and Constant Propagation](06-folders.md)
- 第 7 课：[Verifiers](07-verifiers.md)
- 本仓库中的实际测试：
  - `tests/box_syntax.mlir`
  - `tests/box_fold.mlir`
  - `tests/poly_canonicalize.mlir`
  - `tests/poly_fold.mlir`
  - `tests/poly_verifier.mlir`