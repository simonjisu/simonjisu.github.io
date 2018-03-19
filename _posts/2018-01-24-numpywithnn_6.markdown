---
layout: post
title: "NUMPY with NN - 6: Weight Initialization"
categories: "DeepLearning"
author: "Soo"
date: "2018-01-24 12:31:37 +0900"
comments: true
---
# Numpy로 짜보는 Neural Network Basic - 6

---
## 학습관련 기술 Part 2

### 가중치 초기값 설정(Weight Initialization)
이전에 활성화 함수가 왜 중요한지 이야기 했었다. 다시 한 번 이야기 하면, 비 선형 활성화 함수를 사용해서 선형으로만 표현할 수 없는 값을 표현할 수 있게 되며, 그로 인해 은닉층을 쌓는 의미가 생긴다. 이런 비 선형 함수의 가중치 미분을 구해서 학습하고자 하는 파라미터를 업데이트 하게 된다.

[[<span style="color: #7d7ee8">Numpy로 짜보는 Neural Network Basic - 2</span>](https://simonjisu.github.io/datascience/2017/12/08/numpywithnn_2.html)] 참고

그렇다면 Sigmoid를 예를 들어서 이야기 해보자, 아래 그림은 Sigmoid 함수와 미분의 그래프다.

$$\begin{aligned} \sigma(a) & = \dfrac{1}{1+e^{-a}} \\ \sigma'(a) &= \sigma(a)(1 - \sigma(a))\end{aligned}$$

<img src="/assets/ML/nn/6/sigmoid_prime.png" alt="Drawing"/>

선형결합을 통해 구해진 값 a은 뉴런을 거쳐서 Sigmoid로 활성화 된 함수는 대부분 $[0, 1]$ 사이의 값을 가지게 될 것이다. 선형결합을 통해 구해진 값이 조금만 커져도 (약 $[-5, 5]$ 이외 값) Gradient 값이 0으로 되는 경우가 많아진다. 이를 **Gradient Vanishing Problem**, 즉 가중치 0에 가까워져 업데이트 안되는 현상을 말한다. 따라서 가중치 초기 값이 엄청 작게 설정 했다고 해도 dot product해진 값이 커지면 이런 현상이 일어날 수가 있다.


#### Mnist 데이터로 살펴보기

실제로 활성화 함수 값이 어떤 분포인지 mnist 데이터로 살펴보자. 내가 만든 뉴럴 네트워크의 구조는 아래와 같다.

$$\begin{aligned} Input_{(784)}
&\rightarrow [Affine1 \rightarrow Activation1]_{hidden1: (100)} \\
&\rightarrow [Affine2 \rightarrow Activation2]_{hidden2: (50)} \\
&\rightarrow [Affine3 \rightarrow SoftmaxwithLoss]_{output:(10)_{}}
\end{aligned}$$

Hidden Node 는 각 100개와 50개로 설정하고 Activation Fucntion은 simgoid로 했다.
아래는 200 epoch까지 중간에 Activation 값들의 분포를 찍어본 것이다. 가중치 초기 값은 1을 곱한 것으로써 랜덤 Initialization 되었다고 생각하면 된다.(= $W$에다 1을 곱했다.)

<video controls="controls" autoplay loop muted markdown="1"> <source type="video/mp4" src="/assets/ML/nn/6/sig_act1.mp4" markdown="1"> </source> </video>

대부분의 값이 0과 1로 이루어져 있다는 것을 알 수 있다. 이는 즉 대부분의 값이 미분을 했을 때 0이될 가능성이 높다는 뜻이다.

<video controls="controls" autoplay loop muted markdown="1"> <source type="video/mp4" src="/assets/ML/nn/6/sig_back1.mp4" markdown="1"> </source> </video>

앞쪽의 Layer로 backward 할 수록 가중치 미분 값이 0에 가까워지는 것을 볼 수 있다.

우리는 [<span style="color: #7d7ee8">4편</span>](https://simonjisu.github.io/datascience/2017/12/15/numpywithnn_4.html)에서 가중치 값을 0.01 초기화 시켰더니 학습이 전혀 안된 모습을 볼 수 있었다. 그렇다면 0.01로 가중치 값을 초기화 하면 어떻게 될까?

<video controls="controls" autoplay loop muted markdown="1"> <source type="video/mp4" src="/assets/ML/nn/6/sig_act2.mp4" markdown="1"> </source> </video>

가중치 활성화 값이 0.5로 치우쳐져 두 레이어의 분포가 거의 비슷해졌다. 다수의 뉴런이 같은 값을 출력하고 있다는 뜻으로 우리가 비선형함수를 써서 예측 불가능하게 만드려고 한 노력을 물거품으로 만들어 버렸다. 즉, 위에서 이야기 했던 층을 여러게 쌓은 의미가 없어진다. 이를 **"표현력이 제한된다"** 라고 말한다.

<video controls="controls" autoplay loop muted markdown="1"> <source type="video/mp4" src="/assets/ML/nn/6/sig_back2.mp4" markdown="1"> </source> </video>

가중치의 미분 값들이 대부분 0 근처에 있는 것을 확인 할 수가 있다.

따라서 초기 값을 잘 설정해주어야 하는데, Sigmoid 함수에 대해서 자주 사용하는 Xavier 초기 값이 있다.

Paper: [<span style="color: #7d7ee8">Understanding the difficulty of training deep feedforward neural networks</span>](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

$$ W \sim Uniform(n_{in}, n_{out})$$

$$ Var(W)=\dfrac{1}{n_{in}} $$

즉 가중치 초기화를 할때 각 파라미터 $W$ 에 대하여 $\dfrac{1}{\sqrt{n_{in}}}$ 를 곱해주는 것이다.

아래는 Xavier 가중치 초기값을 설정했을 때의 Activation 분포다. 두 층이 전혀 다른 분포가 되어있는 것을 볼 수가 있다.

<video controls="controls" autoplay loop muted markdown="1"> <source type="video/mp4" src="/assets/ML/nn/6/sig_act3.mp4" markdown="1"> </source> </video>

<video controls="controls" autoplay loop muted markdown="1"> <source type="video/mp4" src="/assets/ML/nn/6/sig_back3.mp4" markdown="1"> </source> </video>

각각의 가중치 초기화 값에 대한 학습 결과를 살펴보자. 총 10000번의 epochs를 돌린 결과다.

| 학습결과 |
|:-:|
|<img src="/assets/ML/nn/6/sig1.png" alt="Drawing"/> |
| $w_{std} = 1$: 처음에 빠른것 같지만 나중에 굉장히 천천히 학습 되는 것을 확인 할 수 있다. <br> 또한 매번 실행시 학습 속도가 다르다. |
| <img src="/assets/ML/nn/6/sig2.png" alt="Drawing"/>   |
| $w_{std} = 0.01$: 학습이 전혀 안되는 것을 확인 할 수 있다.  |
|  <img src="/assets/ML/nn/6/sig3.png" alt="Drawing"/> |
| $w_{std} = \dfrac{1}{\sqrt{n}}$: test 성적이 조금 더 좋아 졌다. |

#### 활성화 함수를 바꿔보자: ReLu
ReLu의 미분 값은 $x > 0$ 에서 $1$ 이고 나머지는 $0$ 이다. ReLu는 **He** 라는 초기값을 설정하게 된다.

$$ W \sim Uniform(n_{in}, n_{out})$$

$$ Var(W)=\dfrac{2}{n_{in}} $$

즉, 가중치 $W$ 에 $\sqrt{\dfrac{2}{n_{in}}}$ 을 곱하게 된다. 아래 동영상을 보면 0보다 큰 부분에서 꾸준히 활성화 되는 모습을 볼 수 있다.

<video controls="controls" autoplay loop muted markdown="1"> <source type="video/mp4" src="/assets/ML/nn/6/relu_act3.mp4" markdown="1"> </source> </video>

<video controls="controls" autoplay loop muted markdown="1"> <source type="video/mp4" src="/assets/ML/nn/6/relu_back3.mp4" markdown="1"> </source> </video>

다음 시간에는 배치 정규화에 대해서 알아보자.
