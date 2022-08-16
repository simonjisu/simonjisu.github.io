---
layout: post
title: "Introduction of Algorithms & Data Structures"
date: "2020-04-20 14:19:38 +0900"
categories: algorithms
author: "Soo"
comments: true
toc: true
---

# Algorithms and Data Structures

## 알고리즘(Algorithm)이란?

[위키백과](https://en.wikipedia.org/wiki/Algorithm)에 따르면 **알고리즘(Algorithm)**은 수학과 컴퓨터 과학분야에서 잘 정의(well-defined)되어진 컴퓨터로 실행 가능한 유한한 명령 시퀀스(sequence)다.

다음과 같은 특징을 가진다고 볼 수 있다.

* 하나의 값 혹은 여러 값의 집합을 **입력(input)**으로 받고, 궁극적으로 하나의 값 혹은 여러 값의 집합을 **출력(output)**으로 뱉어낸다.
* 모든 입력에 대해서 정확한 출력을 뱉어낼 경우, "알고리즘이 정확하다"라고 말 할 수 있다.

또한 알고리즘은 잘 정의된 문제를 해결하기 위한 도구(tool)로 볼 수 있다. 단, "잘 정의된 문제"이라는 문구에는 일반적인(general) 입력-출력 관계가 정의되어야 한다. 잘 정의만 되면, 알고리즘은 곧 문제를 해결하는 일련의 과정을 서술한다고 볼 수 있다.

## 자료구조(Data Structures)란?

[위키백과](https://en.wikipedia.org/wiki/Data_structure)에 따르면 **자료구조(Data Structure)**는 컴퓨터 과학에서 효율적인 접근 및 수정을 가능케 하는 자료의 조직, 관리, 저장을 의미한다.

자료구조에는 여러 종류가 있으며, 이러한 각각의 자료구조는 각자의 연산 및 목적에 맞추어져 있다. 향후 다뤄볼 자료구조들을 나열해보았다(더 추가할 예정).

* 배열(array)
* 해시 테이블(hash table)
* B-트리(B-Tree)

# 알고리즘의 분석

효율적인 알고리즘이란 무엇인지 알려면 "효율"을 정의해야 될 것이다. 그렇다면 알고리즘을 측정하고 분석 해야하는데, 알고리즘을 분석한다는 것은 곧 이에 소요되는 자원(resources)을 예측한다는 것이다. 대부분의 경우 **실행시간(running time)**과 **메모리 공간(memmory space)**를 측정한다. 따라서, 한 문제 대해서 여러 알고리즘을 실행하여 사용되는 자원들을 비교하여 제일 적은 자원을 소모하는 것이 곧 효율적인(effective) 알고리즘이다.

## 실행시간(Running time)

다른 말로 **알고리즘 복잡도(algorithm complexity)**이라 하는데, 알고리즘의 실행 횟수(primitive operations or steps)을 뜻하며, 입력 크기가 커질 수록 실행시간도 커진다. 사실 여기에는 실행 횟수만이 복잡도에 비례한다는 강력한 가정이 들어간다. 여담으로 실제 실행시간은 컴퓨터 CPU의 cache의 접근 속도, cache에 사용했던 메모리의 존재 여부등이 관여를 한다. 

실행시간을 측정했을 때, 가장 빨리된 경우를 best case, 최악의 경우를 worst case라고 하는데, 보통 worst case를 기준으로 복잡도를 측정하고 비교한다.

또한, 실행시간은 횟수에 관련있다고 했기에 수식으로 $T(N)$으로 표기되며, $N$은 입력의 크기다.

## Order of Growth Classification

증가 기준(Order-of-Growth)은 자료 개수의 증가에 따라 소요시간이 변하는 정도를 나타내며, 실제 걸린 시간을 무시하고 표기하자는 것이다. 예를 들어, 어떤 알고리즘의 실행시간이 $T(N) = C\times N^2 + D\times N + E$ 정도 걸린다면 뭉뚱그려서 $C\times N^2$ 정도 시간이 걸린다 라고 말할 수 있다($C$는 알수 없는 반복에 걸리는 시간을 말한다). 이렇게 상대적으로 큰 값을 취하여 시간을 근사하는 방법을 **점근법(asymptote)**이라고 한다.

점근 표기법(Asymptotic Notation)은 알고리즘의 복잡도를 단순화할 때나 무한급수의 뒷부분을 간소화할 때 쓰이며 $\Theta, O, \Omega$ 등이 있다. 보통 "빅오" 라고 많이 들어봤을 것이다.

Order of Growth의 분류로 다음과 같은 표를 그릴 수 있다.

{% include image.html id="1xH_B7ndU6XNlZY1lUnIom_q2Tdv7hNi2" desc="[출처] Robert-Sedgewick 교수의 Algoritms 강의" width="100%" height="auto" %}

## Asymptotic Notation

평소에 Big O Notation이라는 말을 많이 들어보는데, 무슨 뜻인지 잘 이해가 안됐었다. 여기서 Big O를 포함하여 3가지 점근 표기법을 정확한 알아본다.

알아보기 전에 중요한 가정이 있는데 아주 작은 입력크기 $N$에 대해서는 이러한 점근법이 작동하지 않는다. 즉, 우리가 말하는 "효율"이 좋다는 언제까지나 아주 큰 입력 $N$에 대해서 적용되는 말이다.

### $\Theta$ Notation

$\text{Big-}\Theta$: $g(n)$와 양의 상수$c_1, c_2, n_0$가 주어졌을 때, 모든 $n_0$보다 크거나 같은 $n$에 대해서, $0 \leq c_1 g(n) \leq f(n) \leq c_2 g(n)$ 식을 만족하는 $f(n)$을 $\Theta \big( g(n) \big)$ 로 표기한다.

위 정의는 간단히 말해, "$n_0$보다 큰 $n$크기의 입력에 대해서, 함수 $f(n)$의 값이 $(c_1 g(n), c_2 g(n))$ 구간에 존재한다"라고 추정하는 것이다. 즉, 아무리 커봐야 $c_2 g(n)$과 같거나 작을 것이고, 작아봐야 $c_1 g(n)$보다 같거나 클 것이다. 그림으로 표시하면 다음과 같다.

{% include image.html id="1OM8KllT_GcAt-wB7FtzulsoN3e13v0w3" desc="Big Theta Notation" width="100%" height="auto" %}

### $O$ Notation

$\text{Big-}O$: $g(n)$와 양의 상수$c, n_0$가 주어졌을 때, 모든 $n_0$보다 크거나 같은 $n$에 대해서, $0 \leq f(n) \leq c g(n)$ 식을 만족하는 $f(n)$을 $O \big( g(n) \big)$ 로 표기한다.

위 정의는 간단히 말해, "$n_0$보다 큰 $n$크기의 입력에 대해서, 함수 $f(n)$의 값이 $(0, c g(n))$ 구간에 존재한다"라고 추정하는 것이다. 즉, 아무리 커봐야 $c g(n)$과 같거나 작을 것이다. 그림으로 표시하면 다음과 같다.

{% include image.html id="1cPpJQJqrJ-orjTNDQCLWYQiCib_5kF1m" desc="Big O Notation" width="100%" height="auto" %}

### $\Omega$ Notation

$\text{Big-}\Omega$: $g(n)$와 양의 상수$c, n_0$가 주어졌을 때, 모든 $n_0$보다 크거나 같은 $n$에 대해서, $0 \leq c g(n) \leq f(n)$ 식을 만족하는 $f(n)$을 $\Omega \big( g(n) \big)$ 로 표기한다.

위 정의는 간단히 말해, "$n_0$보다 큰 $n$크기의 입력에 대해서, 함수 $f(n)$의 값이 $(c g(n), +\infty)$ 구간에 존재한다"라고 추정하는 것이다. 즉, 아무리 작아도 $c g(n)$보다는 클 것이다. 그림으로 표시하면 다음과 같다.

{% include image.html id="1D03BHXbVlkYqf_EIgCcQ-TTT-fkC38IH" desc="Big Omega Notation" width="100%" height="auto" %}

### 예시

$T(n) = 5n^2 + 12n + 4$만큼 실행시간이 걸리는 알고리즘이 있을 때, 각 표기법으로 표현해보자. 실제로는 근사 값이지만 보통 등호(=)를 사용한다(이번 예시에서는 n이 엄청 크다는 가정을 한다).

{% include image.html id="1cNJKtz7dJmN03nsc_6R3ca5vOZuvMw6I" desc="Big Theta Notation Example" width="100%" height="auto" %}

|표기 가능여부|이유|
|--|--|
|$T(n) \neq \Theta(n)$|$f(n)$값(파란선)이 $c_1g(n), c_2g(n)$사이에 들어와야 하는데 그렇지 않다.|
|$T(n) = \Theta(n^2)$| $f(n)$값(파란선)이 $c_1g(n), c_2g(n)$사이에 들어간다. |
|$T(n) \neq \Theta(n^3)$|$f(n)$값(파란선)이 $c_1g(n), c_2g(n)$사이에 들어와야 하는데 그렇지 않다.|

{% include image.html id="1Zx82-4p5KNf1UmVejUe4ibuv5ZdUpCZn" desc="Big O Notation Example" width="100%" height="auto" %}

|표기 가능여부|이유|
|--|--|
|$T(n) \neq O(n)$|$f(n)$값(파란선)이 $cg(n)$보다 작아야 하는데 그렇지 않다.|
|$T(n) = O(n^2)$| $f(n)$값(파란선)이 $cg(n)$보다 작다. |
|$T(n) = O(n^3)$| $f(n)$값(파란선)이 $cg(n)$보다 작다. |

{% include image.html id="1k4DPvz1TYW_Kp8CdbTuNzT6XCXR6TbCS" desc="Big Omega Notation Example" width="100%" height="auto" %}

|표기 가능여부|이유|
|--|--|
|$T(n) \neq \Omega (n)$| $f(n)$값(파란선)이 $cg(n)$보다 크다. |
|$T(n) = \Omega (n^2)$| $f(n)$값(파란선)이 $cg(n)$보다 커야 하는데 그렇지 않다. |
|$T(n) = \Omega (n^3)$| $f(n)$값(파란선)이 $cg(n)$보다 커야 하는데 그렇지 않다. |

그래프 관련 코드를 첨부한다.

```python
import numpy as np
import matplotlib.pyplot as plt

def T(n):
    return 5*(n**2) + 12*n + 4

def g1(n):
    return n

def g2(n):
    return n**2

def g3(n):
    return n**3

def BigTheta(g_fn, n, c_1, c_2):
    return c_1*g_fn(n), c_2*g_fn(n)

def BigO(g_fn, n, c):
    return c*g_fn(n)

def BigOmega(g_fn, n, c):
    return c*g_fn(n)

def draw(typ, g_fns, N, c_upper, c_lower, ylim=35000, **kwargs):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    n = np.linspace(0, N)
    f_values = T(n)
    ax.plot(n, f_values, label="$f(n)$", c=colors[0])
    for i, g_fn in enumerate(g_fns, 1):
        plot_bignotation(ax=ax, typ=typ, g_fn=g_fn, n=n, 
                         c_1=c_upper, c_2=c_lower, color=colors[i])
    ax.set_xlabel("$n$", fontsize = 16)
    ax.legend(fontsize=14)
    ax.set_ylim(0, ylim)
    ax.set_title(typ, fontsize=20)
    plt.show()
    if kwargs.get("return_fig") == True:
        return fig
    
def plot_bignotation(ax, typ, g_fn, n, c_1, c_2, color=None):
    typ_fn_dict = {"Big-Theta": BigTheta,"Big-O": BigO, "Big-Omega": BigOmega}
    g_dict = {g1: "n", g2: "n^2", g3: "n^3"}
    fn = typ_fn_dict.get(typ)
    g_str = g_dict.get(g_fn)
    if typ == "Big-Theta":
        upper, lower = fn(g_fn, n, c_1, c_2)
        ax.plot(n, lower, c=color, label=f"$c_1*g(n): g={g_str}, c_1={c_1}$")
        ax.plot(n, upper, c=color, label=f"$c_2*g(n): g={g_str}, c_2={c_2}$")
    elif typ == "Big-O":
        upper = fn(g_fn, n, c_1)
        ax.plot(n, upper, c=color, label=f"$c*g(n): g={g_str}, c={c_1}$")
    elif typ == "Big-Omega":
        lower = fn(g_fn, n, c_1)
        ax.plot(n, lower, c=color, label=f"$c*g(n): g={g_str}, c={c_2}$")
        
notations = ["Big-Theta", "Big-Omega", "Big-O"]
ylims = [50000, 50000, 50000]
g_fns = [g1, g2, g3]
N = 500
c_upper = 6
c_lower = 4
for typ, ylim in zip(notations, ylims): 
    draw(typ, g_fns, N, c_upper, c_lower, ylim)
```

# References:

* 본 글은 기본적으로 서울대학교 이재진 교수님의 강의를 듣고 제가 공부한 것을 정리한 글입니다.
* [Asymptotic Analysis](https://www.programiz.com/dsa/asymptotic-notations)
* [점근적 표기법 - 데이터 구조](https://www.scaler.com/topics/asymptotic-notations/)
