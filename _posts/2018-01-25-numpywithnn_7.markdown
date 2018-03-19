---
layout: post
title: "NUMPY with NN - 7: Batch Normalization"
categories: "DeepLearning"
author: "Soo"
date: "2018-01-25 12:54:15 +0900"
comments: true
---
# Numpy로 짜보는 Neural Network Basic - 7

---
## 학습관련 기술 Part 3

### 배치 정규화 (Batch Normalization)
배치 정규화란 미니배치 단위로 선형합인 **$a$** 값을 정규화하는 것이다. 즉, 미니배치에 한해서 데이터 분포가 평균이 0 분산이 1이 되도록 한다. 이는 데이터 분포가 덜 치우치게 하는 효과가 있어서 가중치 초기화 값의 영향을 덜 받게한다. 또한, 학습속도를 증가시키고 regularizer 역할을 하여 Overfitting을 방지함으로 Dropout의 필요성을 줄인다. ~~자세한 내용은 논문을 참고하자!~~

Paper: [<span style="color: #7d7ee8">Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift</span>](https://arxiv.org/abs/1502.03167)

기본적인 아이디어는 아래와 같다. $D$ 차원의 미니배치 데이터 $x = (x^{(1)}, \cdots, x^{(k)}, \cdots, x^{(D)})$에 대해서 각각의 평균과 분산을 구한 후, 정규화를 통해 새로운 $x^{(k)}$ ($\hat{x}^{(k)}$) 를 구한 후에 Scaling($\gamma$) 과 Shifting($\beta$)을 거쳐 새로운 $y$ 를 기존의 선형합성 곱인 $a$ 를 대신해 활성화 함수에 넣는 것이다.

<img src="/assets/ML/nn/6/batch_norm_idea.png" alt="Drawing" style="width=500px"/>

따라서, 하나의 Hidden Layer 는 $Affine \rightarrow BatchNorm \rightarrow Activation$ 으로 구성된다.

### 배치 정규화의 BackPropogation 이해하기

<img src="/assets/ML/nn/NN_batchnorm.png" alt="Drawing"/>

#### Forward:

 x 부터 out 까지 차근차근 진행해보자. 헷갈리지 말아야할 점은 위에 공식에서 $i$ 는 batch를 iteration 한 것이라는 점이다.

```
## Forward Process
# step-1: mu (D,)
mu = x.mean(axis=0)
# step-2: xmu (N, D)
xmu = x - mu
# step-3: sq (N, D)
sq = xmu**2
# step-4: var (D,)
var = np.mean(sq, axis=0)
# step-5: std (D,)
std = np.sqrt(var + 1e-6)
# step-6: invstd (D,)
invstd = 1.0 / std
# step-7: xhat (N, D)
xhat = xmu * invstd
# step-8: scale (N, D)
scale = gamma * xhat
# step-9: out (N, D)
out = scale + beta
```

#### Backward:

우리의 목표는 $\dfrac{\partial L}{\partial x}, \dfrac{\partial L}{\partial \gamma}, \dfrac{\partial L}{\partial \beta}$ 를 구해서, $\dfrac{\partial L}{\partial x}$ 는 Affine Layer로 역전파 시키고 $\gamma, \beta$ 는 학습 시키는 것이다.

**Step-9:**

Forward : $out(scale, \beta) = scale + \beta$

* 더하기 노드의 역전파는 그대로 흘러간다.

$$\begin{cases} dscale = \dfrac{\partial L}{\partial scale} = \dfrac{\partial L}{\partial out} \dfrac{\partial out}{\partial scale} = 1 * dout \\
\\
d\beta = \dfrac{\partial L}{\partial \beta} = \dfrac{\partial L}{\partial out} \dfrac{\partial out}{\partial \beta} = 1 * \sum_i^N dout \end{cases}$$


**Step-8:**

Forward : $scale(\gamma, \hat{x}_i) = \gamma \ * \ \hat{x}_i$

* 곱의 노드의 역전파는 들어왔던 신호를 역으로 곱해서 흘려 보낸다.

$$\begin{cases}
d\hat{x}_i = \dfrac{\partial L}{\partial \hat{x}_i} = \dfrac{\partial L}{\partial scale} \dfrac{\partial scale}{\partial \hat{x}_i} = 1 * \sum_i^N dout \\
\\
d\gamma = \dfrac{\partial L}{\partial \gamma} = \dfrac{\partial L}{\partial scale} \dfrac{\partial scale}{\partial \gamma} = \sum_i^N dout \ * \ \hat{x}_i
\end{cases}$$

**Step-7:**

Forward : $\hat{x}_i(xmu, invstd) = xmu \ * \ invstd$

* $xmu$는 윗쪽(step-7 $\rightarrow$ step-2)과 아래쪽(step-3 $\rightarrow$ step-2) 으로 두 번 돌아가기 때문에 첨자를 단다.

$$\begin{cases}
dxmu_1= \dfrac{\partial L}{\partial xmu_1} = \dfrac{\partial L}{\partial \hat{x}_i} \dfrac{\partial \hat{x}_i}{\partial xmu_1} = d\hat{x}_i \ * \ invstd \\
\\
dinvstd = \dfrac{\partial L}{\partial \hat{x}_i} = \dfrac{\partial L}{\partial \hat{x}_i} \dfrac{\partial \hat{x}_i}{\partial invstd} = d\hat{x}_i \ * \ xmu
\end{cases}$$

**Step-6:**

Forward : $invstd(\sigma) = \dfrac{1}{\sigma}$

* $f(x) = \dfrac{1}{x}$ 의 미분은 $f'(x) = -\dfrac{1}{x^2} = -f(x)^2$ 이기 때문에 아래와 같다.

$$d\sigma = \dfrac{\partial L}{\partial \sigma} = \dfrac{\partial L}{\partial invstd} \dfrac{\partial invstd}{\partial \sigma} = dinvstd \ * \ (-invstd^2)$$

**Step-5:**

Forward : $\sigma(var) = \sqrt{var + \epsilon}$

* $f(x) = \sqrt{x + \epsilon}$ 의 미분은 $f'(x) = -\dfrac{1}{2}(x+\epsilon)^{-\frac{1}{2}}$ 이기 때문에 아래와 같다.

$$dvar = \dfrac{\partial L}{\partial var} = \dfrac{\partial L}{\partial \sigma} \dfrac{\partial \sigma}{\partial var} = d\sigma \ * \ (-\dfrac{1}{2}(var+\epsilon)^{-\frac{1}{2}})$$

**Step-4:**

Forward : $var(sq) = \dfrac{1}{N} \sum_i^N sq$

* $f(x) = \dfrac{1}{N} \sum_i^N x_i$ 의 미분은 $f'(x) = \dfrac{1}{N} \sum_i^N 1$ 이기 때문에 아래와 같다. 단, x의 형상(shape)이 같아야한다.

$$dsq = \dfrac{\partial L}{\partial sq} = \dfrac{\partial L}{\partial var} \dfrac{\partial var}{\partial sq} = \dfrac{1}{N} dvar \ * \ \begin{bmatrix} 1 & \cdots & 1 \\ \vdots & \ddots & \vdots \\ 1 & \cdots & 1 \end{bmatrix}_{(N, D)} = \dfrac{1}{N} dvar \ * \ ones(N, D)$$

**Step-3:**

Forward : $sq = xmu^2$

* $f(x) = x^2$ 의 미분은 $f'(x) = 2x$ 이기 때문에 아래와 같다.

$$dxmu_2 = \dfrac{\partial L}{\partial xmu_2} = \dfrac{\partial L}{\partial sq} \dfrac{\partial sq}{\partial xmu_2} = dsq \ * \ 2 \ xmu$$

**Step-2:**

Forward : $xmu = x_i - \mu$

* $dxmu = dxmu_1 + dxmu_2$ 로 정의 된다. 곱의 미분 법칙 생각해보면 된다. $h(x) = f(x) g(x)$ 를 $x$ 에 대해서 미분하면 $f'(x)g(x) + f(x)g'(x)$ 기 때문이다. <br>
또한 이것도 덧셈과 마찬가지로 그대로 흘러 보내는다 밑에 쪽은 -1 을 곱해서 흘려 보낸다.

$$\begin{cases}
dx_1= \dfrac{\partial L}{\partial x_1} = \dfrac{\partial L}{\partial xmu} \dfrac{\partial xmu}{\partial x_1} = dmu \ * \ 1 \\
\\
d\mu = \dfrac{\partial L}{\partial \mu} = \dfrac{\partial L}{\partial xmu} \dfrac{\partial xmu}{\partial \mu} = \sum_i^N dxmu \ * \ (-1)
\end{cases}$$

**Step-1:**

Forward : $\mu = \dfrac{1}{N} \sum_i^N x_i$

* step-4에서 설명했다.

$$dx_2 = \dfrac{\partial L}{\partial x_2} = \dfrac{\partial L}{\partial \mu} \dfrac{\partial \mu}{\partial x_2} = \dfrac{1}{N} d\mu \ * \ \begin{bmatrix} 1 & \cdots & 1 \\ \vdots & \ddots & \vdots \\ 1 & \cdots & 1 \end{bmatrix}_{(N, D)} = \dfrac{1}{N} d\mu \ * \ ones(N, D)$$

**Step-0:**
* 최종적으로 구하는 $dx = \dfrac{\partial L}{\partial x} = dx_1 + dx_2$ 로 정의 된다.

```
## Backward Process
# step-9: out = scale + beta
dbeta = dout.sum(axis=0)
dscale = dout
# step-8: scale = gamma * xhat
dgamma = np.sum(xhat * dout, axis=0)
dxhat = gamma * dscale
# step-7: xhat = xmu * invstd
dxmu1 = dxhat * invstd
dinvstd = np.sum(dxhat * xmu, axis=0)
# step-6: invstd = 1 / std
dstd = dinvstd * (-invstd**2)
# step-5: std = np.sqrt(var + 1e-6)
dvar = -0.5 * dstd * (1 / np.sqrt(var + 1e-6))
# step-4: var = sum(sq)
dsq = (1.0 / batch_size) * np.ones(input_shape) * dvar
# step-3: sq = xmu**2
dxmu2 = dsq * 2 * xmu
# step-2: xmu = x - mu
dxmu = dxmu1 + dxmu2
dmu = -1 * np.sum(dxmu, axis=0)
dx1 = dxmu * 1
# step-1: mu = mean(x)
dx2 = (1.0 / batch_size) * np.ones(input_shape) * dmu
# step-0:
dx = dx1 + dx2
```

#### 실제 구현
그러나 실제 구현 시에는 training 과 testing을 나눠서 아래와 같이 진행된다.

<img src="/assets/ML/nn/6/batch_norm_al.png" alt="Drawing" style="width=500px"/>

---
### 첨부: Backpropogation 전체 미분 수학식

* 수식의 이해는 이분의 블로그에서 많은 참조를 했다. Blog: [[<span style="color: #7d7ee8">Clement Thorey</span>](http://cthorey.github.io/backpropagation/)]

$$\begin{aligned}
Y &= \gamma \hat{X} + \beta \\
\hat{X} &= (X - \mu)(\sigma^2+\epsilon)^{-1/2}
\end{aligned}$$

**size:**

$$\begin{aligned}
Y, \hat{X}, X &= (N, D) \\
\mu, \sigma, \gamma, \beta &= (D,)
\end{aligned}$$

<br>

$N$은 미니 배치 싸이즈고, $D$는 데이터의 차원 수다.

Matrix 로 정의한 수식을 다시 원소별로 표기를 정의 해보자. 매트릭스 $Y, X, \hat{X}$ 와 벡터 $\gamma, \beta$ 그리고 위에 수식은 아래와 같이 다시 정의 해볼 수 있다. (왜 매트릭스와 벡터인지는 Forward 과정에 나와있다. 각 차원별로 평균과 분산을 구하는걸 잊지말자)

$$\begin{aligned}
y_{kl} &= \gamma_l \hat{x}_{kl} + \beta_l \\
\hat{x}_{kl} &= (x_{kl} - \mu_l)(\sigma_l^2+\epsilon)^{-1/2}
\end{aligned}$$

$$where\quad \mu_l = \dfrac{1}{N} \sum_{p=1}^{N} x_{pl} , \quad \sigma_l^2 = \dfrac{1}{N} \sum_{p=1}^{N} (x_{pl}-\mu_l)^2 $$

$$with\quad k = [1, \cdots, N] \ ,\  l = [1, \cdots, D]$$

<br>

이제 우리고 구하려고 하는 미분 값들$(\dfrac{\partial L}{\partial x}, \dfrac{\partial L}{\partial \gamma}, \dfrac{\partial L}{\partial \beta})$을 하나씩 구해보자.

#### $x_{ij}$ 에 대한 미분

$$\begin{aligned}\dfrac{\partial L}{\partial x_{ij}}
&= \sum_{k,l} \dfrac{\partial L}{\partial y_{kl}} \dfrac{\partial y_{kl}}{\partial x_{ij}} \\
&= \sum_{k,l} \dfrac{\partial L}{\partial y_{kl}} \dfrac{\partial y_{kl}}{\partial \hat{x}_{kl}} \dfrac{\partial \hat{x}_{kl}}{\partial {x}_{ij}} \\
&= \sum_{k,l} \dfrac{\partial L}{\partial y_{kl}} \cdot \gamma_l \cdot \dfrac{\partial \hat{x}_{kl}}{\partial {x}_{ij}} \end{aligned}$$

<br>

$$\dfrac{\partial \hat{x}_{kl}}{\partial {x}_{ij}} = \dfrac{\partial f}{\partial {x}_{ij}} g + f \dfrac{\partial g}{\partial {x}_{ij}} \quad where \quad \begin{cases} f = (x_{kl} - \mu_l) \\ g = (\sigma_l^2+\epsilon)^{-1/2} \end{cases}$$ 에 대한 미분을 구해보자.

* 우선 분자 $f = (x_{kl} - \mu_l)$ 에 대한 미분을 하면 아래와 같다.

$$\dfrac{\partial f}{\partial {x}_{ij}} = \delta_{ik} \delta_{jl} - \frac{1}{N} \delta_{jl}$$

<br>

$$\delta_{m,n} = \begin{cases} 1 \quad where \quad m = n \\ 0 \quad otherwise \end{cases} $$

<br>

$\delta_{m,n}$ 은 앞첨자 $m$ 이 뒷첨자 $n$과 같다면 1이 된다는 뜻이다.

즉, 여기서 $i$ 가 $[1 \cdots k \cdots D]$ 까지, $j$ 가 $[1 \cdots l \cdots D]$ 까지 iteration 할 것인데, 오직 $i=k, j=l$ 일때만 앞 항인 $\delta_{il} \delta_{jl} = 1$ 이 될 것이고, $j=l$ 일때만 뒷항인 $\frac{1}{N} \delta_{jl} = \frac{1}{N}$ 이 될 것이다.

* 분모 $g = (\sigma_l^2+\epsilon)^{-1/2}$ 에 대한 미분은 아래와 같다.

$$\dfrac{\partial g}{\partial {x}_{ij}} = -\dfrac{1}{2}(\sigma_l^2 + \epsilon)^{-3/2} \dfrac{\partial \sigma_l^2}{\partial x_{ij}}$$

<br>

$$\begin{aligned} where \quad \sigma_l^2
&= \dfrac{1}{N} \sum_{p=1}^{N} (x_{pl}-\mu_l)^2 \\
\dfrac{\partial \sigma_l^2}{\partial x_{ij}}
&= \dfrac{1}{N} \sum_{p=1}^{N} 2(x_{pl}-\mu_l)(\delta_{ip} \delta_{jl} - \frac{1}{N} \delta_{jl}) \\
&= \dfrac{2}{N} (x_{il}-\mu_l) \delta_{jl} - \dfrac{2}{N^2} \sum_{p=1}^N (x_{pl}-\mu_l) \delta_{jl} \\
& = \dfrac{2}{N} (x_{il}-\mu_l) \delta_{jl} - \dfrac{2}{N} \delta_{jl} (\dfrac{1}{N}  \sum_{p=1}^N  (x_{pl}-\mu_l)) \cdots (1) \\
& = \dfrac{2}{N} (x_{il}-\mu_l) \delta_{jl}
\end{aligned}$$

<br>

(1) 번 식을 잠깐 이야기 하면 $\dfrac{1}{N} \sum_{p=1}^N  (x_{pl}-\mu_l) = 0$ 인것은 어떤 값들을 평균을 빼고 다시 평균 시키면 0이 된다.

$e.g)\quad \frac{(1-2)+(2-2)+(3-2)}{3}=0$

이제 드디어 $$\dfrac{\hat{x}_{kl}}{\partial {x}_{ij}}$$ 에 대해 구할수 있다. 곱의 미분 법칙을 사용하면 아래와 같이 전개 된다.

$$\begin{aligned} \dfrac{\hat{x}_{kl}}{\partial {x}_{ij}}
&= (\delta_{ik} \delta_{jl} - \frac{1}{N} \delta_{jl})(\sigma_l^2+\epsilon)^{-1/2}  -\dfrac{1}{N} (x_{kl} - \mu_l)(\sigma_l^2 + \epsilon)^{-3/2} (x_{il}-\mu_l) \delta_{jl} \\
\end{aligned}$$

최종적으로 우리의 목적 $$\dfrac{\partial L}{\partial x_{ij}}$$ 를 구해보자.

<br>

$$\begin{aligned}\dfrac{\partial L}{\partial x_{ij}}
&= \sum_{k,l} \dfrac{\partial L}{\partial y_{kl}} \cdot \gamma_l \cdot [(\delta_{ik} \delta_{jl} - \frac{1}{N} \delta_{jl})(\sigma_l^2+\epsilon)^{-1/2} - \dfrac{1}{N} (x_{kl} - \mu_l)(\sigma_l^2 + \epsilon)^{-3/2} (x_{il}-\mu_l) \delta_{jl}] \\
&= \sum_{k,l} \dfrac{\partial L}{\partial y_{kl}} \gamma_l [(\delta_{ik} \delta_{jl} - \frac{1}{N} \delta_{jl})(\sigma_l^2+\epsilon)^{-1/2}] - \sum_{k,l} \dfrac{\partial L}{\partial y_{kl}} \gamma_l [\dfrac{1}{N} (x_{kl} - \mu_l)(\sigma_l^2 + \epsilon)^{-3/2} (x_{il}-\mu_l) \delta_{jl}] \\
&= \sum_{k,l} \dfrac{\partial L}{\partial y_{kl}} \gamma_l (\delta_{ik} \delta_{jl})(\sigma_l^2+\epsilon)^{-1/2} - \frac{1}{N} \sum_{k,l} \dfrac{\partial L}{\partial y_{kl}} \gamma_l \delta_{jl}(\sigma_l^2+\epsilon)^{-1/2} - \sum_{k,l} \dfrac{\partial L}{\partial y_{kl}} \gamma_l [\dfrac{1}{N} (x_{kl} - \mu_l)(\sigma_l^2 + \epsilon)^{-3/2} (x_{il}-\mu_l) \delta_{jl}] \\
&= \dfrac{\partial L}{\partial y_{ij}} \gamma_l \delta_{ii} \delta_{jj} (\sigma_l^2+\epsilon)^{-1/2} - \frac{1}{N} \sum_k \dfrac{\partial L}{\partial y_{kj}} \gamma_l \delta_{jj}(\sigma_j^2+\epsilon)^{-1/2} - \dfrac{1}{N} \sum_{k} \dfrac{\partial L}{\partial y_{kj}} \gamma_l [ (x_{kj} - \mu_j)(\sigma_j^2 + \epsilon)^{-3/2} (x_{ij}-\mu_j) \delta_{jj}] \cdots (2) \\
&= \dfrac{\partial L}{\partial y_{ij}} \gamma_l (\sigma_l^2+\epsilon)^{-1/2} - \frac{1}{N} \sum_k \dfrac{\partial L}{\partial y_{kj}} \gamma_l (\sigma_j^2+\epsilon)^{-1/2} - \dfrac{1}{N} \sum_{k} \dfrac{\partial L}{\partial y_{kj}} \gamma_l (x_{kj} - \mu_j)(\sigma_j^2 + \epsilon)^{-3/2} (x_{ij}-\mu_j) \\
&= \dfrac{1}{N} \gamma_l (\sigma_l^2+\epsilon)^{-1/2} [N \dfrac{\partial L}{\partial y_{ij}} - \sum_k \dfrac{\partial L}{\partial y_{kj}} - (x_{ij}-\mu_j)(\sigma_j^2 + \epsilon)^{-1} \sum_{k} \dfrac{\partial L}{\partial y_{kj}}(x_{kj} - \mu_j)]
\end{aligned}$$

<br>

(2) 번 식으로 도출 되는 과정을 잘 살펴보면, 각 항마다 곱으로 구성되어 있다. 첫번째 항은 $\sum_{k, l}$ 에서 오직 $k=i, l=j$ 일때 남아 있고 나머지는 전부다 0 이고, 두번째 항은 오직 $l=j$ 일때 남아있고 나머지는 전부다 0 이다. 그리고 마지막도 마친가지로 $l=j$ 일때만 남아있는다.

#### $\gamma_j$ 에 대한 미분

위에 까지 이해했으면 $\gamma_l$ 에 대한 미분은 간단하다.

$$\begin{aligned}\dfrac{\partial L}{\partial \gamma_j}
&= \sum_{k,l} \dfrac{\partial L}{\partial y_{kl}} \dfrac{\partial y_{kl}}{\partial \gamma_j} \\
&= \sum_{k,l} \dfrac{\partial L}{\partial y_{kl}} \hat{x}_{kl} \delta_{jl} \\
&= \sum_k \dfrac{\partial L}{\partial y_{kj}} \hat{x}_{kj} \\
&= \sum_k \dfrac{\partial L}{\partial y_{kj}} (x_{kj} - \mu_j)(\sigma_j^2+\epsilon)^{-1/2}
\end{aligned}$$

#### $\beta_j$ 에 대한 미분

$$\begin{aligned}\dfrac{\partial L}{\partial \beta_j}
&= \sum_{k,l} \dfrac{\partial L}{\partial y_{kl}} \dfrac{\partial y_{kl}}{\partial \gamma_j} \\
&= \sum_{k,l} \dfrac{\partial L}{\partial y_{kl}} \delta_{jl} \\
&= \sum_k \dfrac{\partial L}{\partial y_{kj}}
\end{aligned}$$

<br>

여기서 우리는 왜 위에 step-9, 8 코드 구현에서 dgamma와 dbeta를 summation 하는지 알 수 있다.

다음 마지막 시간에는 모든걸 종합해서 학습하는 과정을 코드로 살펴보자.
