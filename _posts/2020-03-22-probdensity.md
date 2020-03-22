---
layout: post
title: "Probability Density and Transformation"
date: "2020-03-12 14:19:38 +0900"
categories: math
author: "Soo"
comments: true
toc: true
---

# Probability Density Function

실수(real-valued) 확률변수 $X$가 $(x, x+ \delta x)$구간의 값을 가지고, 해당 구간의 확률이 $f_X(x)\delta x$($\delta x \rightarrow 0$일 경우)로 정의 된다면, $f_X(x)$를 $X$의 **확률 밀도함수(probability density function)**라고 한다.

$$p_X(X \in (x, x+\delta x)) = \int_{x}^{x+\delta x} f_X(x) dx$$

확률 밀도함수는 다음 두 조건을 만족해야한다.

1. $f(x) \geq 0$
2. $\int_{-\infty}^{\infty} f(x) dx = 1$

---

# Transformation of Random Variable

확률변수의 **변환(Transformation)**이란 기존의 확률변수$X$를 새로운 확률변수$Y$로 변환 하는 것이다. 비선형 변환시 단순 함수와 다르게 변환되는데 어떻게 변화하는지 살펴보기로 한다. 여기서 이야기하는 변환은 다음 조건을 만족해야한다.

* 변환 함수 $g: Y \rightarrow X$는 전단사(bijective) 혹은 일대일 대응(one-to-one)이어야 한다. 일대일 대응이란, 모든 정의역$Y$에 존재하는 원소 $y$는 치역$X$에 대응하는 값이 유일하다(unique). 이를 다른 말로 하면, "$g$ 함수는 역을 가질 수 있다(invertible)"라고 한다 $g^{-1}: X \rightarrow Y$.

예를 들어, 확률변수 $X$에 해당하는 확률 밀도함수는 $f_X(x)$, 확률변수 $Y$에 해당하는 확률 밀도함수는 $f_Y(y)$인 경우에서 $x=g(y)$인 비선형 변환이 있다고 가정해본다. 그렇다면 두 확률 밀도함수는 정말 다른 것일까? 확률 밀도함수의 최댓값도 변수의 선택에 종속되어 변화했을까($\hat{x}=g(\hat{y})$의 관계를 유지하는지 아니면 변화했는지)? 이를 알아보기 위해 변화된 확률 분포를 분해해본다.

확률변수 $X$의 가측 부분집합(measurable subset)을 $\mathcal{X}_0 \subset \mathcal{X}$, 확률변수 $Y$의 정의역에 해당하는 가측 부분집합을 $\mathcal{Y}_0 \subset \mathcal{Y}$라고 정의한다. 변환식 $x = g(y)$을 $y$에 관해 미분하면, $dx = g'(y)dy$를 얻을 수 있으며, $X$의 확률 분포$p_X(x)$는 다음과 같이 변환할 수 있다.

* 여기서 가측 부분집합은 쉽게 얘기해서 정의된 범위라고 생각할 수 있다

$$\begin{aligned}
p_X(x) = \int_{\mathcal{X}_0} f_X(x) dx &= \int_{\mathcal{Y}_0} f_X(x) \vert \dfrac{dx}{dy} \vert dy \\
&= \int_{\mathcal{Y}_0} f_X(g(y)) \vert g'(y) \vert dy \\
&= \int_{\mathcal{Y}_0} f_Y(y) dy \\
&= p_Y(y)
\end{aligned}$$

위 수식에서 확률변수$Y$에 대한 확률 밀도함수는 $f_Y(y) = f_X(g(y)) \vert g'(y) \vert$로 변화했는데, 이는 $X$에 대한 확률 밀도함수에 Jacobian Factor $\dfrac{dx}{dy}= g'(y)$를 곱한 값이 된다. 즉, Jacobian Factor로 인해서 확률 밀도함수$f_Y(y)$의 값이 $f_X(g(y))$로부터 약간 변화한다는 것을 의미한다. 

## Example of Transformation

과연 다른지 $x = g(y) = \ln(y) - \ln(1-y) + 5$ 라는 변환으로 $\hat{x}=g(\hat{y})$ 관계($y$의 최댓값 위치가 변환된 최댓값을 결정)를 유지하고 있는지 아닌지 살펴본다. $g$의 역함수는 $g^{-1}(x) = \dfrac{1}{1 + \exp(-x + 5)}$인 sigmoid 함수가 된다. 즉, $y$의 정의역은 0과 1 사이의 실수, $x$는 $-\infty$와 $\infty$의 실수 값을 취할 수 있다. 또한 함수 $g$의 미분값은 $\dfrac{dx}{dy}=\dfrac{1}{y - y^2}$ 다.

```python
import numpy as np

def g(y):
    """x = g(y)"""
    return np.log(y) - np.log(1-y) + 5

def g_inv(x):
    """y = g^{-1}(x)"""
    return 1 / (1 + np.exp(-x + 5))

def dxdy(y):
    return 1 / (y - y**2)
```

확률변수 $X$가 평균이 6, 표준편차가 1인 가우시안 분포를 따른다고 가정하고($X \sim \mathcal{N}(6, 1)$), 5만개의 샘플을 추출하고, 역함수를 이용해 샘플링된 확률변수 $Y$의 값을 구한다. 샘플링된 분포 이외에 실제 분포를 그리기 위한 작업도 진행한다. 

```python
np.random.seed(88)
N = 50000
mu = 6.0
sigma = 1.0

sampled_x = np.random.normal(loc=mu, scale=sigma, size=(N,))
sampled_y = g_inv(sampled_x)

x = np.linspace(0, 10, N)  # 5만개의 균일된 간격인 x 값
y = g_inv(x)

px = gaussian(x, mu, sigma)
py = gaussian(g(y), mu, sigma)
py_real = px * np.abs(dxdy(y))
```

관련 분포를 그리면 다음 그림과 같다(관련 코드는 [링크](https://gist.github.com/simonjisu/57c6e2b89b4c9457541809ec5b5f51c9)에서 확인 할 수 있다). 각 선의 의미는 다음과 같다.

* <span style="color:#d40000">빨강</span>: 확률변수 $X$의 실제 분포
* <span style="color:#002ed4">파랑</span>: 확률변수 $Y$의 실제 분포
* <span style="color:#e3a205">노랑</span>: $y=g^{-1}(x)$로 변환된 확률변수 $X$의 분포

또한, 오른쪽 밑의 파란 막대 그래프가 샘플링된 확률변수 $X$의 분포, 왼쪽 파란 막대 그래프 부분이 $y=g^{-1}(x)$로 변환된 확률변수 $Y$의 분포다.

{% include image.html id="1c83fpP9BQb7DjtK0EmTUKSPcdt6gLCA5" desc="" width="100%" height="auto" %}

이 그래프에서 명백한 것은 $X$분포(<span style="color:#d40000">빨강</span>)의 최대값 $\hat{x}$과 실제 $Y$분포(<span style="color:#002ed4">파랑</span>)의 최대값 $\hat{y}$은 단순 $x=g(y)$(혹은 $y=g^{-1}(x)$)의 관계를 가지지 않는다. 즉, $X$분포와 $Y$분포는 서로 다른 특성을 가지며, 확률 밀도가 변수의 변환으로 인해서 바뀌었다고 할수 있다.

---

# Determinent of Jacobian

위에서 이야기한 $\vert \dfrac{dx}{dy} \vert$인 Jacobian Factor 란 무엇일까? 야코비 행렬식(Jacobian Determinant)을 기하학적으로 풀면 좌표계가 변환할 때  변환된 면적의 너비로 풀이할 수 있다. 

$$det \Big( \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix} \Big) = 6$$

{% include image.html id="1o-CffunWblVIBmwU0xJSROesmlrEMTBf" desc="" width="100%" height="auto" %}

위 행렬식값인 6의 의미는 단위 벡터 기저(basis)에서 새로운 기저로 변환했을 때 면적이 1(노란색 부분)에서 6(초록색 부분)만큼 바뀐 다는 뜻이다. 

* 단위 벡터 기저: $ \Big( \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \end{bmatrix} \Big)$

즉, 확률 변수의 변환 예제에서 작은 구간의 확률값 $f_X(x) dx$에 해당하는 면적에  $\dfrac{dx}{dy}=g'(y)=\dfrac{1}{y-y^2}$값을 곱한 만큼 바뀐다는 뜻이다. 

---

# Reference

* [prml-solution 1.4](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-web-sol-2009-09-08.pdf)
* [prml-errata](https://yousuketakada.github.io/prml_errata/prml_errata.pdf)