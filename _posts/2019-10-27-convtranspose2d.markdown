---
layout: post
title: "[PyTorch] ConvTranspose2d 와 Conv2d 의 관계"
date: "2019-10-27 21:56:38 +0900"
categories: deeplearning
author: "Soo"
comments: true
toc: true
---

최근 XAI 에 관련된 공부를 하면서 비전쪽의 많은 논문을 살펴보고 있다. "Visualizing and Understanding Convolutional Networks (2013)" 논문에서는 이미지 처리에서는 CNN 알고리즘이 제일 좋지만, 그 이유에 대해서 탐구를 시도한 논문이다. 도대체 Convolution의 필터가 어떤 역할을 하는지, 이들이 어떤 부분을 살펴보는 지를 확인한다. 오늘은 이 논문에서 제안하는 Deconvolutional layers(정확히는 Fractionally-strided convolution 이지만 차후에 언급한다)의 실체를 낱낱이 살펴보도록 한다. 

* link: [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)

# Convolution layer

Deconvolutional layer을 알아보기 전에 합성곱 연산(Convolutional Operation)에 대해 알아볼 필요가 있다. 합성곱 연산은 필터가 조금씩 이동하면서 이미지의 일부와 필터간 연산을 통해 진행된다.

<img src="https://drive.google.com/uc?id=17x4ZQ_r0FTa_mlDFiIWvMJcg22vRrBd6">

일반적으로 CNN 알고리즘은 `Convolution` - `Activation` - `Maxpooling` 과정을 거친다. 예를 들어, 위 그림과 같이 4x4 이미지는 3x3 필터를 통해 2x2 의 선형변환 값을 갖는다(padding=0, stride=1 인 경우). 그리고 활성화 함수를 통과한 뒤에 Maxpool 과정을 거친다. 이때 `Convolution` - `Activation`을 거쳤을 때 나오는 텐서를 Activation Map이라고 하고, `Pooling` 과정을 거쳤을 때 나오는 텐서를 Pooled Map 이라고 한다.

<img src="https://drive.google.com/uc?id=1Y4kIqXn7vUYQgoZWDdprrO-SP9a-Qogs">

이 논문에서는 그 과정을 역으로 한번 해보는 것을 제안했다. 위 그림처럼 마지막 Pooled Maps 에서 풀링된 위치를 기억했다가(Max Locations "Switches" 부분), 이 위치를 기반으로 역으로 Unpooled Maps 를 재구축한다(이 부분에 관심있는 분들은 이 논문을 한번 살펴보는 것을 추천드린다). 이번 글에서는 그 다음 스텝인 Convolution layer 에서 역으로 돌아가는 방법에 대해서 설명하려고 한다.

먼저 이미지의 크기를 $N$, 필터(커널)의 크기를 $K$, 패딩의 크기를 $P$, 스트라이드를 $S$ 라고 정의하고, 여러 변수를 정의 한다.

$$\begin{aligned} \text{input image size} &= N \times N =4 \times 4 \\ \text{filter size} &= K\times K = 3 \times 3 \\ \text{padding} &= P = 0 \\ \text{stride} &= S = 1 \\ \text{output image size} &= (\dfrac{N+2P-K}{S}+1, \dfrac{N+2P-K}{S}+1) \\&= 2 \times 2 \\ \text{input image} &: X^{(l)} = \begin{bmatrix} x_{11} & x_{12} & x_{13} &x_{14}\\  x_{21} & x_{22} & x_{23} &x_{24}\\ x_{31} & x_{32} & x_{33} &x_{34}\\ x_{41} & x_{42} & x_{43} &x_{44}\end{bmatrix}^{(l)} \\ \text{output image} &: x^{(l+1)} =\begin{bmatrix} x_{11} & x_{12}\\ x_{21} & x_{22} \end{bmatrix}^{(l+1)} \\ \text{filter} &: W = \begin{bmatrix} w_{11} & w_{12} & w_{13}\\  w_{21} & w_{22} & w_{23}\\ w_{31} & w_{32} & w_{33}\end{bmatrix} \end{aligned}$$

이제 수식으로 합성곱 연산을 정의한다. 

$$\begin{aligned} x_{pq}^{(l+1)} &= \sum_{p=i}^{K+i-1} \sum_{q=j}^{K+j-1} w_{pq} x_{pq}^{(l)} \quad \text{for }i, j \in (1, 2, \cdots,  N-K+1)\end{aligned}$$

위 수식으로는 어려워 보이지만 아래와 같은 연산을 `*` 라고 하면 결과는 2x2 행렬이 출력되며 다음과 같다.

$$\begin{aligned} X^{(l+1)} &= X^{(l)}*W\\&=\begin{bmatrix}w_{11} x^{(l)}_{11} + w_{12} x^{(l)}_{12} + w_{13} x^{(l)}_{13} + w_{21} x^{(l)}_{21} + w_{22} x^{(l)}_{22} + w_{23} x^{(l)}_{23} + w_{31} x^{(l)}_{31} + w_{32} x^{(l)}_{32} + w_{33} x^{(l)}_{33} & w_{11} x^{(l)}_{12} + w_{12} x^{(l)}_{13} + w_{13} x^{(l)}_{14} + w_{21} x^{(l)}_{22} + w_{22} x^{(l)}_{23} + w_{23} x^{(l)}_{24} + w_{31} x^{(l)}_{32} + w_{32} x^{(l)}_{33} + w_{33} x^{(l)}_{34}\\ w_{11} x^{(l)}_{21} + w_{12} x^{(l)}_{22} + w_{13} x^{(l)}_{23} + w_{21} x^{(l)}_{31} + w_{22} x^{(l)}_{32} + w_{23} x^{(l)}_{33} + w_{31} x^{(l)}_{41} + w_{32} x^{(l)}_{42} + w_{33} x^{(l)}_{43} & w_{11} x^{(l)}_{22} + w_{12} x^{(l)}_{23} + w_{13} x^{(l)}_{24} + w_{21} x^{(l)}_{32} + w_{22} x^{(l)}_{33} + w_{23} x^{(l)}_{34} + w_{31} x^{(l)}_{42} + w_{32} x^{(l)}_{43} + w_{33} x^{(l)}_{44}\end{bmatrix} \end{aligned}$$

파이토치에서 Convolution Layer 는 `Conv2d` 로 구현되어 있다.

* link: [torch.nn.Conv2d - PyTorch master documentation](https://pytorch.org/docs/stable/nn.html#conv2d)

# Deconvolution Layer? Transposed Convolution Layer!

저자는 이미 2011 년도에 Deconvolution Layer 를 제안했다. 

- link: [Adaptive Deconvolutional Networks for Mid and High Level Feature Learning](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.849.3679&rep=rep1&type=pdf)

간단하게 생각해보면 다음 그림과 같이 필터를 이동시키면서 원래 4x4 이미지(초록색)를 복원하면 될것 같다. 이 과정이 맞는지 이후에 살펴볼 것이다.

<img src="https://drive.google.com/uc?id=1R-C4g1zSpculTzC8w00IrM9CNM0vifN_">

흥미로운 것은 파이토치에서는 `ConvTranspose2d` 라고 구현이 되어 있다. 그리고 다음과 같은 설명이 덧붙여져 있다. 

> This module can be seen as the gradient of `Conv2d` with respect to its input. It is also known as a fractionally-strided convolution or a deconvolution (although it is not an actual deconvolution operation).

- link: [torch.nn.ConvTranspose2d - PyTorch master documentation](https://pytorch.org/docs/stable/nn.html#convtranspose2d)

왜 이름이 Deconvolution 이 아닐까? 읽어보면 이 연산은 `Conv2d`의 출력을 입력에 대해 미분을 연산하는 것과 같다고 한다. 미분을 한번 구해보고 이를 C 라고 하자.

$$\begin{aligned} C = \dfrac{\partial X^{(l+1)}}{\partial X^{(l)}}  &= \dfrac{\partial Vec(X^{(l+1)})}{\partial Vec(X^{(l)})} \\ &= \begin{bmatrix}  \dfrac{\partial x_{11}^{(l+1)}}{\partial x_{11}^{(l)}} & \dfrac{\partial x_{12}^{(l+1)}}{\partial x_{11}^{(l)}} & \dfrac{\partial x_{13}^{(l+1)}}{\partial x_{11}^{(l)}} & \dfrac{\partial x_{14}^{(l+1)}}{\partial x_{11}^{(l)}} \\ \vdots & \vdots & \vdots & \vdots \\ \dfrac{\partial x_{11}^{(l+1)}}{\partial x_{44}^{(l)}} & \dfrac{\partial x_{12}^{(l+1)}}{\partial x_{44}^{(l)}} & \dfrac{\partial x_{13}^{(l+1)}}{\partial x_{44}^{(l)}} & \dfrac{\partial x_{14}^{(l+1)}}{\partial x_{44}^{(l)}} \end{bmatrix} \\ & = \begin{bmatrix}w_{11} & 0 & 0 & 0\\w_{12} & w_{11} & 0 & 0\\w_{13} & w_{12} & 0 & 0\\0 & w_{13} & 0 & 0\\w_{21} & 0 & w_{11} & 0\\w_{22} & w_{21} & w_{12} & w_{11}\\w_{23} & w_{22} & w_{13} & w_{12}\\0 & w_{23} & 0 & w_{13}\\w_{31} & 0 & w_{21} & 0\\w_{32} & w_{31} & w_{22} & w_{21}\\w_{33} & w_{32} & w_{23} & w_{22}\\0 & w_{33} & 0 & w_{23}\\0 & 0 & w_{31} & 0\\0 & 0 & w_{32} & w_{31}\\0 & 0 & w_{33} & w_{32}\\0 & 0 & 0 & w_{33}\end{bmatrix} \end{aligned}$$

이 미분 과정을 파이썬의 `sympy` 패키지로 쉽게 만들 수 있다(Jupyter Notebook 에서 사용하길 권장). 다음 코드에서 C 매트릭스를 살펴보면 위와 같은 결과를 얻을 수 있다. 

```python
import numpy as np
from sympy import Symbol, MatrixSymbol, Matrix

# 노트북에서 수학식의 LaTeX 표현 사용
sympy.init_printing(use_latex='mathjax')

def convolution(x, w, K, N):
    res = []
    for i in range(N-K+1):
        for j in range(N-K+1):
            a = sum(Matrix(x)[i:(K+i), j:(K+j)].multiply_elementwise(Matrix(w)))
            res.append(a)
    return Matrix(res).reshape(N-K+1, N-K+1)

x_input = MatrixSymbol("x^{(l)}", 4, 4)
x_output = MatrixSymbol("x^{(l+1)}", 2, 2)
w = MatrixSymbol("w", 3, 3)

# Convolution Output
output = convolution(x_input, w, K=3, N=4)

# Calculate derivatives & get matrix C
C = output.reshape(1, 4).diff(Matrix(x_input).reshape(16, 1)).reshape(16, 4)
```

이 C 행렬은 재밌는 특징을 가진다. 매트릭스 형태의 입력 데이터를 한줄로 핀 후에 매트릭스 연산을 하고, 다시 형태를 변환 시켜주면 Convolution 의 출력값이 나온다. 정말 맞는지 살펴보기 위해서 다음 코드를 실행해보자.

```python
# forward (1, 16) x (16, 4) = (1, 4) = (2, 2)
(Matrix(x_input).reshape(1, 16) @ C).reshape(2, 2)
```
반대로 출력값을 한줄로 피고 C의 전치행렬과 곱한 후 다시 형태를 변환 시켜주면 입력과 다른 행렬이 나온다.

```python
# backward
(Matrix(x_output).reshape(1, 4) @ C.transpose()).reshape(4, 4)
```
실행하면 다음과 같은 행렬이 나오는데 이 연산 과정을 Deconvolution 연산, 정확히는 **Fractionally-strided convolution** 혹은 **Transpose convolution** 이라고 한다. 어떻게 계산된 것이며 어떤 뜻일까?

$$\begin{bmatrix}w_{11} x^{(l+1)}_{11} & w_{11} x^{(l+1)}_{12} + w_{12} x^{(l+1)}_{11} & w_{12} x^{(l+1)}_{12} + w_{13} x^{(l+1)}_{11} & w_{13} x^{(l+1)}_{12}\\w_{11} x^{(l+1)}_{21} + w_{21} x^{(l+1)}_{11} & w_{11} x^{(l+1)}_{22} + w_{12} x^{(l+1)}_{21} + w_{21} x^{(l+1)}_{12} + w_{22} x^{(l+1)}_{11} & w_{12} x^{(l+1)}_{22} + w_{13} x^{(l+1)}_{21} + w_{22} x^{(l+1)}_{12} + w_{23} x^{(l+1)}_{11} & w_{13} x^{(l+1)}_{22} + w_{23} x^{(l+1)}_{12}\\w_{21} x^{(l+1)}_{21} + w_{31} x^{(l+1)}_{11} & w_{21} x^{(l+1)}_{22} + w_{22} x^{(l+1)}_{21} + w_{31} x^{(l+1)}_{12} + w_{32} x^{(l+1)}_{11} & w_{22} x^{(l+1)}_{22} + w_{23} x^{(l+1)}_{21} + w_{32} x^{(l+1)}_{12} + w_{33} x^{(l+1)}_{11} & w_{23} x^{(l+1)}_{22} + w_{33} x^{(l+1)}_{12}\\w_{31} x^{(l+1)}_{21} & w_{31} x^{(l+1)}_{22} + w_{32} x^{(l+1)}_{21} & w_{32} x^{(l+1)}_{22} + w_{33} x^{(l+1)}_{21} & w_{33} x^{(l+1)}_{22}\end{bmatrix}$$

출력을 계산하는 Convolution 연산 과정에서 **"입력 픽셀"** 에서 **"출력 픽셀"** 과 연결된 가중치를 생각하면 편하다. 다음 그림을 살펴보면, 필터(노란색)가 지나가면서, **"입력 픽셀"** ($x_{12}^{(l)}$)과 **"출력 픽셀"** ($x_{11}^{(l+1)}$, $x_{12}^{(l+1)}$)사이에 연결된 두 개의 가중치 ($w_{11}$, $w_{12}$)를 통해 연산이 된다. 위 행렬에서 1행 2열에 있는 원소 값과 연관이 있는 것을 확인 할 수 있는데, fractionally-strided convolution 연산은 바로 **"출력 픽셀"** 에서 **"입력 픽셀"** 로 방향을 바꿔 연산하면 된다. 

<img src="https://drive.google.com/uc?id=1acZ6YvrW6xooXJd6nYpFYhm-eDHSe2f1">

Fractionally-strided convolution 연산은 다음 그림과 같다. "fractionally" 의 단어 뜻 처럼 필터가 출력 이미지의 일부분을 걸치면서 이동(stride)하면서 연산된다. 또한 가중치도 기존의 형태와 달리 약간의 변형(transpose)이 된다(정확한 전치행렬은 아니다). 그렇다면 "출력 픽셀"과 연관이 없는 부분은? 0으로 곱해져서 더해진다!

<img src="https://drive.google.com/uc?id=1WumIP2aCDNJ4cCWQW_2_Q0e1LCx_WkdQ">

따라서 다시 정리하면 Convolution Layer의 출력을 입력에 대한 미분을 구해서(C 행렬), 이를 한줄로 편 출력과 곱한 후에 형태를 입력 이미지로 변환해주는 것이 Convolution 의 반대 연산인 Fractionally-strided convolution 이다. 수식으로 다음과 같이 정리 할 수 있다.

$$X^{(l)} = [Vec\big(X^{(l+1)}\big)C^T]^{(N)}$$

- $^{(N)}$은 Vector Transpose이며, 이는 다음 노트북을 살펴보자. [Vector Transpose](https://nbviewer.jupyter.org/github/simonjisu/pytorch_tutorials/blob/master/00_Basic_Utils/04_Backpropagation_Matrix_diff.ipynb)

 

# Additional Reference

[A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)