# 如何阅读和编写 assemblyFormat

> `assemblyFormat` 是 MLIR ODS 用来的“文本语法模板”。它看起来像正则表达式，
> 其实只是一个**从左到右的拼接模板**：固定标点 + 变量 + 指令。
>
> 本文给你一套逐段翻译的方法，并把课程里的所有例子拆开。

## 1. 一句话理解

```tablegen
let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($output))";
```

可以读成：

```text
打印/解析： 变量lhs  字面量","  变量rhs  属性字典  字面量":"  结果类型
```

对应 IR：

```mlir
%0 = poly.add %arg0, %arg1 : !poly.poly<10>
```

`assemblyFormat` **不包含**：

- op 的名字（`poly.add` 由 dialect + op mnemonic 决定）
- 结果的 SSA 名字（`%0 =` 由打印器加在外面）
- 大部分空格（parser 忽略空白，printer 自动加常规空格）

## 2. 只有三类东西

### 2.1 字面量（literal）

用反引号包起来，必须原样出现：

```tablegen
`:`  `,`  `->`  `(`  `)`  `<`  `>`  `{`  `}`  `=`
```

也可以放关键字：

```tablegen
`to`  `step`  `iter_args`
```

### 2.2 变量（variable）

以 `$` 开头，名字来自 ODS 的 arguments/results：

```tablegen
$lhs       // 操作数
$rhs       // 操作数
$output    // 结果（通常只用于 type($output)）
$coefficients  // 属性
$region    // region
```

不同类型的变量打印方式不同：

| 变量类型 | 打印成什么 |
|---|---|
| operand | SSA 名字，如 `%arg0` |
| result | 不作为 SSA 名字打印；通常只出现在 `type($result)` 里 |
| attribute | 属性值，如 `dense<[1, 2, 3]> : tensor<3xi32>` |
| region | region 的内容 |

### 2.3 指令（directive）

不带 `$` 的保留词：

```tablegen
attr-dict
operands
results
regions
successors
type($operand_or_result)
qualified(type(...))
functional-type($inputs, results)
```

它们会展开成一段固定的解析/打印逻辑。

## 3. 解码方法：五步

拿到一个 assemblyFormat，按这五步读：

1. **找 op 名**：由 `Op<Dialect, "mnemonic">` 决定
2. **把格式切成 token**：
   - 反引号内容 = 字面量
   - `$xxx` = 变量
   - 其他单词 = 指令
3. **给变量代入示例值**：
   - operand → `%arg0`
   - result 在 `type(...)` 里 → 结果类型
   - attribute → 属性文本
4. **按顺序拼接**
5. **补上 printer 自动加的东西**：
   - `%result =` 前缀
   - 常规空格和换行

下面用课程里的真实例子走一遍。

## 4. 例 1：`poly.add`

```tablegen
let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($output))";
```

切分：

| 片段 | 类型 | 含义 |
|---|---|---|
| `$lhs` | 变量 | 第一个操作数 |
| `` `,` `` | 字面量 | 逗号 |
| `$rhs` | 变量 | 第二个操作数 |
| `attr-dict` | 指令 | 可选属性字典 |
| `` `:` `` | 字面量 | 冒号 |
| `qualified(type($output))` | 指令 | 结果的完整类型 |

代入：

```text
$lhs                  -> %arg0
`, `                  -> ,
$rhs                  -> %arg1
attr-dict             -> （无额外属性，不打印）
`:`                   -> :
qualified(type(...))  -> !poly.poly<10>
```

拼接：

```mlir
%0 = poly.add %arg0, %arg1 : !poly.poly<10>
```

## 5. 例 2：`poly.eval`

```tablegen
let assemblyFormat =
    "$polynomial `,` $point attr-dict `:` `(` qualified(type($polynomial)) `,` type($point) `)` `->` type($output)";
```

切分：

| 片段 | 输出 |
|---|---|
| `$polynomial` | `%p` |
| `` `,` `` | `,` |
| `$point` | `%x` |
| `attr-dict` | 无 |
| `` `:` `` | `:` |
| `` `(` `` | `(` |
| `qualified(type($polynomial))` | `!poly.poly<10>` |
| `` `,` `` | `,` |
| `type($point)` | `i32` |
| `` `)` `` | `)` |
| `` `->` `` | `->` |
| `type($output)` | `i32` |

结果：

```mlir
%r = poly.eval %p, %x : (!poly.poly<10>, i32) -> i32
```

注意：

- `type($polynomial)` / `type($point)` 打印的是**操作数的类型**
- `type($output)` 打印的是**结果的类型**
- `$output` 本身不会打印成 `%y`；结果的 SSA 名字由外面的 `%r =` 决定

## 6. 例 3：`poly.constant` 为什么有两个冒号

```tablegen
let assemblyFormat = "$coefficients attr-dict `:` qualified(type($output))";
```

`$coefficients` 是**属性**，不是操作数。它打印 dense attribute 时会带上自己的类型：

```text
dense<[2, 3, 4]> : tensor<3xi32>
```

然后格式里的字面量 `:` 再打印一次，后面是结果类型：

```mlir
%c = poly.constant dense<[2, 3, 4]> : tensor<3xi32> : !poly.poly<10>
                   └── $coefficients ──┘   └─ `:` + 结果类型 ─┘
```

两个冒号：

1. 第一个来自属性自身的语法
2. 第二个来自 assemblyFormat 里的 `` `:` ``

## 7. 例 4：`box.pack` / `box.unpack`

```tablegen
// pack
"$input attr-dict `:` qualified(type($input)) `->` qualified(type($box))"

// unpack
"$box attr-dict `:` qualified(type($box)) `->` qualified(type($value))"
```

代入：

```mlir
%b = box.pack %v : i32 -> !box.box<i32>
%v = box.unpack %b : !box.box<i32> -> i32
```

注意这里没有 `attr-dict` 的属性时，`attr-dict` 不输出任何东西；`qualified` 强制
打印 dialect 前缀，所以是 `!box.box<i32>` 而不是省略形式。

## 8. 类型的 assemblyFormat

类型的格式只描述 `!dialect.mnemonic` 后面的部分：

```tablegen
// Poly
let parameters = (ins "int":$degreeBound);
let assemblyFormat = "`<` $degreeBound `>`";
```

`!poly.poly<10>`：

```text
!poly.poly          <  10  >
└─ 类型名 ─┘        └ 格式 ┘
```

```tablegen
// Box
let parameters = (ins "Type":$elementType);
let assemblyFormat = "`<` $elementType `>`";
```

`!box.box<i32>`：

```text
!box.box   <  i32  >
```

类型格式的 `$elementType` 是 Type 参数，打印时直接输出 `i32`、`!poly.poly<5>` 等。

## 9. 从目标 IR 反推 assemblyFormat

假设你想要的语法是：

```mlir
%r = my.op %a, %b : (i32, i32) -> i32 {tag = "x"}
```

步骤：

1. 去掉 op 名和结果前缀，留下：
   ```text
   %a, %b : (i32, i32) -> i32 {tag = "x"}
   ```
2. 把变化的 SSA 值换成变量：
   ```text
   $lhs, $rhs : (i32, i32) -> i32 {tag = "x"}
   ```
3. 把类型部分换成 `type` 指令：
   ```text
   $lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)
   ```
4. 把属性字典换成 `attr-dict`（位置与你想要的一致）：
   ```tablegen
   "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)"
   ```
5. 检查所有 operands/results 都被覆盖（或用了 `operands`/`results` 指令）。

## 10. 常用指令速查

| 指令 | 打印/解析什么 | 例子 |
|---|---|---|
| `attr-dict` | 属性字典 `{...}` | `{tag = "x"}` |
| `attr-dict-with-keyword` | `attributes {...}` | `attributes {tag = "x"}` |
| `operands` | 所有操作数（带类型） | `%a, %b : i32, i32` |
| `results` | 所有结果类型 | `i32, i64` |
| `regions` | 所有 region | `{ ... }` |
| `successors` | 所有后继块 | `^bb1, ^bb2` |
| `type($x)` | `$x` 的类型 | `i32` |
| `qualified(type($x))` | 带 dialect 前缀的类型 | `!poly.poly<10>` |
| `functional-type($in, results)` | 函数式类型 | `(i32, i32) -> i32` |
| `custom<Foo>(...)` | 调用你自己写的 parser/printer | 复杂语法 |

## 11. 常见坑

1. **结果变量不会打印 SSA 名字**
   `$output` 只用于 `type($output)`；结果是外层的 `%0 =` 生成的。

2. **属性和操作数不一样**
   `$coefficients` 会打印 `dense<...> : tensor<...>`，不是 `%coefficients`。

3. **忘记反引号**
   标点必须写成 `` `:` ``、`` `,` ``；直接写 `:` 会被当成指令/语法错误。

4. **忘记 `attr-dict`**
   额外属性就没有位置表示，解析/打印会丢信息。

5. **`type` vs `qualified(type)`**
   - `type` 可能省略 dialect 前缀
   - `qualified` 强制带前缀，自定义类型通常用它

6. **格式必须覆盖所有操作数/结果**
   如果逐个写了 `$a`、`$b`，就不用再写 `operands`；反过来如果用了
   `operands` 指令，就不要再单独写每个操作数。

7. **两个冒号的 constant**
   第一个是属性自己的类型，第二个是格式里的冒号。

8. **空格不是语义**
   parser 忽略空白；printer 自动加空格。不要把格式里的空格当成必须匹配的字符。

## 12. 练习

### 练习 1

```tablegen
let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
```

如果 `$input` 是 `%x : i32`，`$output` 是 `i64`，输出是什么？

### 练习 2

目标语法：

```mlir
%p = pair.make %a, %b : (i32, f64) -> !pair.pair<i32, f64>
```

请写出 assemblyFormat（假设字段名为 `first`、`second`、`result`）。

### 练习 3

为什么下面这个格式会丢掉属性？

```tablegen
let assemblyFormat = "$lhs `,` $rhs `:` type($output)";
```

## 13. 参考答案

<details>
<summary>练习 1</summary>

```mlir
%0 = my.op %x : i32 -> i64
```

（op 名由 `Op<...>` 决定；`%0 =` 由 printer 加。）

</details>

<details>
<summary>练习 2</summary>

```tablegen
"$first `,` $second attr-dict `:` `(` type($first) `,` type($second) `)` `->` qualified(type($result))"
```

</details>

<details>
<summary>练习 3</summary>

格式里没有 `attr-dict`，op 上的额外属性没有位置解析/打印，因此会丢失或被拒绝。
加上 `attr-dict` 即可。

</details>

## 14. 最终速查

```text
assemblyFormat
  = 字面量（反引号） + 变量（$xxx） + 指令（attr-dict/type/...）

读法：
  1. 找到 op 名
  2. 切 token
  3. 变量代入示例值
  4. 按顺序拼接
  5. 补上 %0 = 和空格

写法：
  1. 先写目标 IR
  2. 变化部分换成 $变量
  3. 固定标点加反引号
  4. 加 attr-dict
  5. 加 type/qualified(type)
  6. 确认覆盖所有 operands/results
```