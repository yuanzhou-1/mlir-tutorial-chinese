# Pat 与 Pattern 语法

> 本文单独讲 MLIR TableGen DRR 里的 `Pat` 和 `Pattern` 语法。
> 配合第 8 课 Canonicalizers and Declarative Rewrite Patterns 阅读。
>
> 目标：看到 `PolyPatterns.td` 时，能逐段说出它在匹配什么、创建什么、
> 约束是什么，以及为什么用 `Pat` 而不是 `Pattern`。

## 1. 一句话区别

```tablegen
Pat<...>       // 产生一个结果
Pattern<...>   // 产生一个或多个结果
```

`Pat` 只是 `Pattern` 的简化形式：

```tablegen
class Pat<dag pattern, dag result, ...>
    : Pattern<pattern, [result], ...>;
```

所以 `Pat` 等价于“结果列表只有一个元素”的 `Pattern`。

## 2. 最小例子

### 2.1 `Pat`

```tablegen
def LiftConjThroughEval : Pat<
  (Poly_EvalOp $f, (ConjOp $z, $fastmath)),
  (ConjOp (Poly_EvalOp $f, $z), $fastmath)
>;
```

读法：

```text
匹配：poly.eval(f, conj(z))
重写：conj(poly.eval(f, z))
```

### 2.2 `Pattern`

```tablegen
def DifferenceOfSquares : Pattern<
  (Poly_SubOp (Poly_MulOp:$lhs $x, $x), (Poly_MulOp:$rhs $y, $y)),
  [
    (Poly_AddOp:$sum $x, $y),
    (Poly_SubOp:$diff $x, $y),
    (Poly_MulOp:$res $sum, $diff),
  ],
  [(HasOneUse:$lhs), (HasOneUse:$rhs)]
>;
```

读法：

```text
匹配：x*x - y*y
重写：
  sum  = x + y
  diff = x - y
  res  = sum * diff
```

这里需要创建三个新 op，所以用 `Pattern`。

---

## 3. 两个类的完整参数

### 3.1 `Pat`

```tablegen
class Pat<dag pattern, dag result,
          list<dag> preds = [],
          list<dag> supplemental_results = [],
          dag benefitAdded = (addBenefit 0)>;
```

### 3.2 `Pattern`

```tablegen
class Pattern<dag source, list<dag> results,
              list<dag> preds = [],
              list<dag> supplemental_results = [],
              dag benefitAdded = (addBenefit 0)>;
```

参数对照：

| 位置 | `Pat` | `Pattern` | 作用 |
|---|---|---|---|
| 1 | `pattern` | `source` | 要匹配的 IR 子图 |
| 2 | `result` | `results`（列表） | 要生成的 IR |
| 3 | `preds` | `preds` | 额外约束 |
| 4 | `supplemental_results` | `supplemental_results` | 结果生成后的附加动作 |
| 5 | `benefitAdded` | `benefitAdded` | benefit 增量 |

最容易出错的是第 3 和第 5 个参数：

```tablegen
// 错误：把 benefit 放到了 constraints 的位置
Pattern<source, results, (addBenefit 1)>

// 正确：benefit 是第五个参数
Pattern<source, results, [constraints], [], (addBenefit 1)>
```

---

## 4. DAG 节点语法

### 4.1 基本形式

```tablegen
(OpDefName arg0, arg1, ...)
```

例子：

```tablegen
(Poly_EvalOp $f, (ConjOp $z))
```

含义：

```text
Poly_EvalOp
  ├── $f
  └── ConjOp
        └── $z
```

注意：这里写的是 **TableGen 的 op 定义名**：

```text
Poly_EvalOp
```

不是 IR 文本名：

```text
poly.eval
```

`poly.eval` 是 assemblyFormat 打印出来的名字；`Poly_EvalOp` 才是
`Op<...>` 的 TableGen record 名。

### 4.2 绑定值：`$x`

```tablegen
(Poly_EvalOp $f, ...)
```

`$f` 把匹配到的第一个操作数绑定到变量 `f`。后续可以在结果树或约束里使用：

```tablegen
(ConjOp (Poly_EvalOp $f, $z))
```

### 4.3 绑定整个 op：`Op:$name`

```tablegen
(Poly_MulOp:$lhs $x, $x)
```

- `$lhs`：绑定整个 `poly.mul` op（或其结果）
- `$x`：绑定它的操作数
- 第二个 `$x`：要求第二个操作数和第一个相同

所以它匹配 `x * x`，不匹配 `x * y`。

### 4.4 重复变量就是相等约束

```tablegen
(Poly_MulOp $x, $x)
```

不是“两个随便的操作数”，而是：

```text
operand0 == operand1
```

生成代码会检查：

```cpp
if (!(*x.begin() == *x0.begin()))
  return rewriter.notifyMatchFailure(...);
```

### 4.5 属性绑定

```tablegen
(ConjOp $z, $fastmath)
```

`$fastmath` 绑定 `complex.conj` 的 fastmath 属性，结果树里可以继续使用：

```tablegen
(ConjOp (Poly_EvalOp $f, $z), $fastmath)
```

### 4.6 嵌套

DAG 可以任意嵌套：

```tablegen
(Poly_SubOp
  (Poly_MulOp $x, $x),
  (Poly_MulOp $y, $y))
```

对应：

```text
poly.sub
  ├── poly.mul(x, x)
  └── poly.mul(y, y)
```

---

## 5. 结果树语法

### 5.1 `Pat` 的结果

```tablegen
Pat<
  (Poly_EvalOp $f, (ConjOp $z, $fastmath)),
  (ConjOp (Poly_EvalOp $f, $z), $fastmath)
>
```

第二个参数是**一棵结果树**，只能产生一个结果值。

### 5.2 `Pattern` 的结果列表

```tablegen
Pattern<
  (Poly_SubOp (Poly_MulOp:$lhs $x, $x), (Poly_MulOp:$rhs $y, $y)),
  [
    (Poly_AddOp:$sum $x, $y),
    (Poly_SubOp:$diff $x, $y),
    (Poly_MulOp:$res $sum, $diff),
  ],
  [(HasOneUse:$lhs), (HasOneUse:$rhs)]
>
```

第二个参数是**结果 op 列表**：

- `$sum`、`$diff` 是中间结果
- `$res` 是最终用来替换根 op 的结果
- 中间结果可以给后面的结果当操作数

对于单结果根 op：

```text
结果列表的最后一项  ->  替换根 op 的结果
前面的项            ->  辅助 op，保留在 IR 中
```

对于多结果根 op，列表会按顺序对应结果；这是高级用法，初学先掌握单结果即可。

### 5.3 用已有值替换

如果不需要创建新 op，可以用：

```tablegen
(replaceWithValue $x)
```

它表示“把匹配到的结果替换成已有 SSA 值 `$x`”。

### 5.4 指定 location / 结果类型

DRR 还支持：

```tablegen
(location $someOp)
(location "myLocation", $x, $y)
```

以及显式结果类型等指令。初学阶段先不用；知道它们存在即可。

---

## 6. 约束：`Constraint` 和 `CPred`

### 6.1 定义约束

```tablegen
def HasOneUse : Constraint<CPred<"$_self.hasOneUse()">, "has one use">;
```

- `CPred<"...">` 里是一段 C++ 表达式
- `$_self` 会被替换成绑定的实体
- 第二个字符串是错误/调试描述

### 6.2 使用约束

```tablegen
[(HasOneUse:$lhs), (HasOneUse:$rhs)]
```

含义：

- 对 `$lhs` 检查 `hasOneUse()`
- 对 `$rhs` 检查 `hasOneUse()`
- 两个条件都满足才允许重写

生成代码大致是：

```cpp
if (!lhs.getResult().hasOneUse())
  return rewriter.notifyMatchFailure(...);
if (!rhs.getResult().hasOneUse())
  return rewriter.notifyMatchFailure(...);
```

### 6.3 多个约束

约束列表是“与”关系：

```tablegen
[(HasOneUse:$lhs), (HasOneUse:$rhs)]
```

全部通过才匹配。

### 6.4 约束能引用什么

约束里可以引用源 pattern 中绑定的名字：

```tablegen
(Poly_MulOp:$lhs $x, $x)
```

这里的 `$lhs` 和 `$x` 都可以在约束列表中使用。

---

## 7. benefit

### 7.1 默认值

`Pattern` 的默认 benefit 是**源 pattern 中的 op 数量**：

- `LiftConjThroughEval`：`eval` + `conj` = 2
- `DifferenceOfSquares`：`sub` + 两个 `mul` = 3

生成的构造器里可以看到：

```cpp
DifferenceOfSquares(MLIRContext *context)
    : RewritePattern("poly.sub", 3, context, {...}) {}
```

第二个参数就是 benefit。

### 7.2 增加 benefit

第五个参数：

```tablegen
def MyPattern : Pattern<
  source,
  results,
  constraints,
  [],              // supplemental results
  (addBenefit 1)   // benefit += 1
>;
```

benefit 越高，greedy driver 越优先尝试这个 pattern。

不要为了“让它先跑”随意加很大的 benefit；优先顺序会影响 IR 的规范化方向。

---

## 8. 生成代码的结构

以 `DifferenceOfSquares` 为例，`-gen-rewriters` 生成的 C++ 大致是：

```cpp
struct DifferenceOfSquares : public RewritePattern {
  DifferenceOfSquares(MLIRContext *context)
      : RewritePattern("poly.sub", 3, context,
                       {"poly.add", "poly.mul", "poly.sub"}) {}

  LogicalResult matchAndRewrite(Operation *op0,
                                PatternRewriter &rewriter) const override {
    // 1. 匹配根 op：poly.sub
    auto castedOp0 = dyn_cast<SubOp>(op0);

    // 2. 匹配操作数：两个 poly.mul
    auto castedOp1 = dyn_cast<MulOp>(op1);
    auto castedOp2 = dyn_cast<MulOp>(op2);

    // 3. 约束：每个 mul 的结果只能有一个使用者
    if (!lhs.getResult().hasOneUse()) return failure();
    if (!rhs.getResult().hasOneUse()) return failure();

    // 4. 约束：x*x、y*y
    if (x != x0) return failure();
    if (y != y0) return failure();

    // 5. 创建结果 op
    auto sum  = rewriter.create<AddOp>(..., x, y);
    auto diff = rewriter.create<SubOp>(..., x, y);
    auto res  = rewriter.create<MulOp>(..., sum, diff);

    // 6. 替换根 op
    rewriter.replaceOp(op0, res);
    return success();
  }
};
```

这份代码可以帮助你排查：

- pattern 是否匹配到了根 op
- 操作数是否匹配
- 哪个约束失败
- 新 op 的类型/操作数是否正确

---

## 9. 完整对照

### 9.1 `Pat`：单结果

```tablegen
def LiftConjThroughEval : Pat<
  (Poly_EvalOp $f, (ConjOp $z, $fastmath)),
  (ConjOp (Poly_EvalOp $f, $z), $fastmath)
>;
```

```text
源：eval(f, conj(z))
结果：conj(eval(f, z))
结果数量：1
```

### 9.2 `Pattern`：多结果

```tablegen
def DifferenceOfSquares : Pattern<
  (Poly_SubOp (Poly_MulOp:$lhs $x, $x), (Poly_MulOp:$rhs $y, $y)),
  [
    (Poly_AddOp:$sum $x, $y),
    (Poly_SubOp:$diff $x, $y),
    (Poly_MulOp:$res $sum, $diff),
  ],
  [(HasOneUse:$lhs), (HasOneUse:$rhs)]
>;
```

```text
源：sub(mul(x,x), mul(y,y))
结果：add(x,y), sub(x,y), mul(add,sub)
结果数量：3（最后一个替换根 op）
约束：两个平方都只能有一个使用者
```

---

## 10. 常见语法错误

### 10.1 用了 IR 名字而不是 TableGen 名字

```tablegen
// 错误
(poly.eval $f, ...)

// 正确
(Poly_EvalOp $f, ...)
```

### 10.2 `Pat` 写了多个结果

```tablegen
// 错误：Pat 只接受一个 result DAG
Pat<source, [(OpA ...), (OpB ...)]>

// 正确：多个结果用 Pattern
Pattern<source, [(OpA ...), (OpB ...)]>
```

### 10.3 忘了列表方括号

```tablegen
// 错误
Pattern<source, (OpA ...), [(HasOneUse:$x)]>

// 正确
Pattern<source, [(OpA ...)], [(HasOneUse:$x)]>
```

### 10.4 benefit 位置放错

```tablegen
// 错误：第三个参数是约束，不是 benefit
Pattern<source, results, (addBenefit 1)>

// 正确：第五个参数
Pattern<source, results, constraints, [], (addBenefit 1)>
```

### 10.5 约束引用了未绑定的名字

```tablegen
// 错误：$lhs 没有在 source 里绑定
[(HasOneUse:$lhs)]
```

要先在源 pattern 里写：

```tablegen
(Poly_MulOp:$lhs $x, $x)
```

### 10.6 重复变量不是“任意两个”

```tablegen
(Poly_MulOp $x, $x)
```

这是相等约束，不是“两个独立的操作数”。

### 10.7 pattern 不收敛

不要写：

```text
A -> B
B -> A
```

canonicalizer 会反复运行，可能陷入循环。

---

## 11. 练习

### 练习 1：判断用 `Pat` 还是 `Pattern`

1. `f(conj(z)) -> conj(f(z))`
2. `x*x - y*y -> (x+y)(x-y)`

<details>
<summary>参考答案</summary>

1. 只产生一个 `conj`，用 `Pat`。
2. 要产生 `add`、`sub`、`mul` 三个 op，用 `Pattern`。

</details>

### 练习 2：标出绑定和约束

```tablegen
def DifferenceOfSquares : Pattern<
  (Poly_SubOp (Poly_MulOp:$lhs $x, $x), (Poly_MulOp:$rhs $y, $y)),
  [
    (Poly_AddOp:$sum $x, $y),
    (Poly_SubOp:$diff $x, $y),
    (Poly_MulOp:$res $sum, $diff),
  ],
  [(HasOneUse:$lhs), (HasOneUse:$rhs)]
>;
```

回答：

1. `$lhs`、`$rhs` 绑定什么？
2. `$x`、`$y` 绑定什么？
3. 哪两个约束在起作用？
4. 哪个结果替换根 op？

<details>
<summary>参考答案</summary>

1. `$lhs` 绑定第一个 `poly.mul`，`$rhs` 绑定第二个 `poly.mul`。
2. `$x`、`$y` 分别绑定两个 `mul` 的操作数。
3. `HasOneUse:$lhs` 和 `HasOneUse:$rhs`。
4. `$res`（最后的 `poly.mul`）替换根 `poly.sub`。

</details>

### 练习 3：修复 pattern

```tablegen
def Bad : Pattern<
  (Poly_SubOp (Poly_MulOp:$lhs $x, $x), (Poly_MulOp:$rhs $y, $y)),
  [(Poly_AddOp $x, $y)],
  (addBenefit 1),
  [(HasOneUse:$lhs), (HasOneUse:$rhs)]
>;
```

这个定义哪里错了？怎么改？

<details>
<summary>参考答案</summary>

第三个参数应该是约束列表，第四个参数是 supplemental results，第五个才是
benefit。应改为：

```tablegen
def Bad : Pattern<
  (Poly_SubOp (Poly_MulOp:$lhs $x, $x), (Poly_MulOp:$rhs $y, $y)),
  [(Poly_AddOp $x, $y)],
  [(HasOneUse:$lhs), (HasOneUse:$rhs)],
  [],
  (addBenefit 1)
>;
```

</details>

### 练习 4：写一个 `Pat`

匹配：

```text
poly.eval(f, conj(z))
```

重写为：

```text
conj(poly.eval(f, z))
```

并保留 fastmath 属性。

<details>
<summary>参考答案</summary>

```tablegen
def LiftConjThroughEval : Pat<
  (Poly_EvalOp $f, (ConjOp $z, $fastmath)),
  (ConjOp (Poly_EvalOp $f, $z), $fastmath)
>;
```

</details>

## 12. 语法速查

```tablegen
// 单结果
def MyPat : Pat<
  (OpA $x, (OpB $y)),          // source
  (OpC $x, $y),                // result
  [constraints],               // optional
  [],                          // supplemental results
  (addBenefit 0)               // optional
>;

// 多结果
def MyPattern : Pattern<
  (OpA (OpB:$lhs $x, $x)),     // source
  [                            // results
    (OpC:$tmp $x, $x),
    (OpD:$res $tmp, $x),
  ],
  [(HasOneUse:$lhs)],          // constraints
  [],
  (addBenefit 0)
>;
```

常用符号：

| 语法 | 含义 |
|---|---|
| `(Op ...)` | DAG 节点 |
| `$x` | 绑定/引用值 |
| `Op:$name` | 绑定整个 op/结果 |
| `[(...)]` | 约束列表 |
| `[...]` | 结果列表 |
| `(addBenefit N)` | benefit 增量 |
| `(replaceWithValue $x)` | 用已有值替换 |
| `CPred<"...">` | C++ 谓词 |
| `Constraint<...>` | 命名约束 |

## 参考

- MLIR 官方文档：Declarative Rewrite Rules
  <https://mlir.llvm.org/docs/DeclarativeRewrites/>
- 第 8 课：Canonicalizers and Declarative Rewrite Patterns（文档待编写）
- 本仓库示例：
  - `lib/Dialect/Poly/PolyPatterns.td`
  - `build/lib/Dialect/Poly/PolyCanonicalize.cpp.inc`