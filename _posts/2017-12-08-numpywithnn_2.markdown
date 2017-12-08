---
layout: post
title: "NUMPY with NN - 2"
categories: "DataScience"
author: "Soo"
date: "2017-12-08 11:52:21 +0900"
---
# Numpy로 짜보는 Neural Network Basic - 2
---
저번 시간에는 Neural Network의 기본이 되었던 XOR 문제를 풀어보았다. 그 과정에서 Feedforward 퍼셉트론이라는 개념이 등장했고 오늘은 Feedforward 과정이 어떻게 진행되는지 알아보자.

## 활성화 함수(Activation Function)

<img src="/assets/ML/nn/NN.jpg" alt="Drawing" style="width: 350px;"/>

$$
a = b + w_1x_1 + w_2x_2\\
y = h(a)
$$


단일 퍼셉트론의 과정을 담은 그림과 수식이다. 각 뉴런(동그라미들)에서 다음 층의 뉴런(오른쪽 큰 동그라미)로 진행하는데 우선 각 $x$ 를 가중합 $a$ 를 구하고, 그 합을 다시 어떤 함수 $h$ 를 거쳐 Output인 $y$ 가 나오게 된다.

저번 시간에 이야기 했던 AND를 적용해보면 $h$ 는 0과 1을 반환하는 아래와 같은 함수일 것이다.

$$y = h(a) =
  \begin{cases}
  0\ \ (a = b + w_1x_1 + w_2x_2 \leq 0) \\
  1\ \ (a = b + w_1x_1 + w_2x_2 > 0) \\
  \end{cases}$$

이런 $h$ 함수를 **활성화 함수** 라고 부르며, 보통 비선형 함수를 쓴다.

### 왜 비선형을 쓰는가?
우리는 활성화 함수를 예측 불가능한 함수로 만들어야 하기 때문인데 비선형함수가 적합하기 때문이다. 선형함수의 특징은 더 해도 선형이라는 것인데, 예를 들어

$$y=3x$$

라는 선형 함수와 간단한 산수인 $1+5=6$ 이란 것을 생각해보자. 좌변에 $1$과 $5$를 선형함수에 넣어서 더한 값인 $3 + 15 = 18$ 이란 값을 우리는 우변의 $6$ 을 선형함수에 넣었을 때 값이랑 같다는 것을 충분히 알 수 있다. 이를 "**예측** 할 수 있다." 라고 이야기 한다. 이처럼 두 개의 선형함수를 더하면 선형이 된다는 것이다.

하지만 어떤 비선형 함수

$$y=3x^2$$
로 아까의 과정을 똑같이 해보자, 좌변의 값을 넣어서 더하면 $3 + 75 = 78$ 인데, 우변의 값을 넣으면 $108$ 이 된다. 따라서 비선형 함수는 더해도 같은 비선형이 아니며 다른 값이 나오기 때문에 예측 불가능하다 라고 말 할 수 있다.

비선형 함수를 통과함으로서 뉴런이 **활성화** 된다라고 이야기 한다.

또 한 가지 이유를 들자면, 선형일 경우에 여러 층을 쌓는 이유가 없어진다. 만약에 $h=c\cdot a$ 가 선형인 함수 였다면 여러 층을 거치게 되면 $h(h(h(a))) = c\cdot c\cdot c\cdot a$ 인데 이는 결국 $b\cdot a=c^3\cdot a$ 라는 선형함수로 바꿀 수 있어서 여전히 예측 가능하기 때문이다.

더 자세한 비선형함수와 선형함수의 차이는 링크의 블로그를 참조해보자 [[<span style="color: #7d7ee8">링크</span>](http://sdolnote.tistory.com/entry/LinearityNonlinearityFunction)]

### 자주 쓰는 활성화 함수들

#### 계단 함수(Step Function)

$$h(x) =
  \begin{cases}
  1\ \ (x > 0) \\
  0\ \ (x  \leq 0) \\
  \end{cases}$$

    def step_function(x):
        y = x > 0
        return y.astype(np.int)

<img src="/assets/ML/nn/step.png" alt="Drawing" style="width: 350px;"/>

#### 시그모이드 함수(Sigmoid Function)

$$h(x) = \frac{1}{1+exp(-x)}$$

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

<img src="/assets/ML/nn/sigmoid.png" alt="Drawing" style="width: 350px;"/>

#### ReLu 함수(ReLu Function)

$$h(x) =
  \begin{cases}
  x\ \ (x > 0) \\
  0\ \ (x \leq 0) \\
  \end{cases}$$

    def ReLu(x):
        return np.maximum(0, x)

<img src="/assets/ML/nn/relu.png" alt="Drawing" style="width: 350px;"/>

## Feedforward 과정

<img src="/assets/ML/nn/NN_2.jpg" alt="Drawing" style="width: 500px;"/>

(사진출처: 밑바닥부터 시작하는 딥러닝)

위와 같은 Neural Network의 과정을 한번 살펴 보자, 여러개의 Perceptron을 쌓으면 이런 모양이 나오는 것을 알 수 있다. 여기서 제일 왼쪽에 있는 $x_1, x_2$ 2개의 뉴런을 한 층으로 보며, 이를 입력층(Input Layer)라고 한다. (1 이란 뉴런은 매번 뉴런을 거칠 때마다 더 해주는 숫자기 때문에 앞으로도 한번만 쓰도록 한다.) 마찬가지로 중간에 두 개의 층을 은닉층(Hidden Layer)이라고 하며 마지막을 출력층(Output Layer)이라고 한다. 보통 입력층은 갯수로 안세며 위 그림은 총 3층인 Neural Network 라고 할 수 있다.

단계별로 살펴보자.

### Input $\rightarrow$ Hidden 1

Input에서 Hidden1 층으로 가는 과정을 행렬로 표시해볼 것이다.

가중치 $w$ 의 표기법은 $w_{오른쪽\ 뉴런위치,\ 왼쪽\ 뉴런위치}^{몇번째\ 층}$ 로써, 첫번째 층에서 입력층 $x_1$ 뉴런에서 히든층1 $a_2^{(1)}$ 방향인 가중치는 $w_{21}^{(1)}$ 이라고 표기합니다.

따라서 각각의 가중치 합을 구하면,

$A = \begin{bmatrix}
    a_1^{(1)} \newline
    a_2^{(1)} \newline
    a_3^{(1)}
    \end{bmatrix} =
    \begin{bmatrix}
    w_{11}^{(1)}x_1 + w_{12}^{(1)}x_2 + b_1^{(1)} \newline
    w_{21}^{(1)}x_1 + w_{22}^{(1)}x_2 + b_2^{(1)} \newline
    w_{31}^{(1)}x_1 + w_{32}^{(1)}x_2 + b_3^{(1)}
    \end{bmatrix}$

가 되는데, 이를 다시 간단하게 쓰면

$X =
  \begin{bmatrix}
  x_1 \newline
  x_2
  \end{bmatrix}$

$W^{(1)} =
      \begin{bmatrix}
      w_{11}^{(1)} & w_{12}^{(1)} \newline
      w_{21}^{(1)} & w_{22}^{(1)} \newline
      w_{31}^{(1)} & w_{32}^{(1)}
      \end{bmatrix}$

$B^{(1)} =
  \begin{bmatrix}
  b_1^{(1)} \newline
  b_2^{(1)} \newline
  b_3^{(1)}
  \end{bmatrix}$

$A = W^{(1)} \cdot X + B^{(1)}$ 가 되며, 형태는 $(3, 1) = (3, 2) \times (2, 1) + (3, 1)$ 로 된다. 이는 간단한 내적 연산으로 구할 수 있게 된다.

가중치의 합을 구하면 이제 비활성함수에 대입해서 뉴런을 활성화 시킨다.

$Z^{(1)} =
    \begin{bmatrix}
    z_1^{(1)} \newline
    z_2^{(1)} \newline
    z_3^{(1)}
    \end{bmatrix} =
    \begin{bmatrix}
    h(a_1^{(1)}) \newline
    h(a_2^{(1)}) \newline
    h(a_3^{(1)})
    \end{bmatrix}$

이렇게 나온 $Z^{(1)}$ 값들은 다음 층에서 입력으로 쓰이게 된다.

아래 코드의 Shape도 같이 잘 살펴보자.

    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.2],
                   [0.3, 0.4],
                   [0.5, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])

    print('X:', X.shape)
    print('W1:', W1.shape)
    print('B1:', B1.shape)
    # Input -> Hidden 1
    print('=================')
    print('Input -> Hidden1')
    print('=================')
    # linear sum
    A1 = np.dot(W1, X) + B1
    print('A1:', A1.shape)
    print(A1)
    # activation
    Z1 = sigmoid(A1)
    print('Z1:', Z1.shape)
    print(Z1)

>W1: (3, 2)
>
>B1: (3,)
>
>=================
>
>Input -> Hidden1
>
>=================
>
>A1: (3,)
>
>[ 0.3  0.7  1.1]
>
>Z1: (3,)
>
>[ 0.57444252  0.66818777  0.75026011]

### Hidden 1 $\rightarrow$ Hidden 2

    W2 = np.array([[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]])
    B2 = np.array([0.1, 0.2])

    print('Z1:', Z1.shape)
    print('W2:', W2.shape)
    print('B2:', B2.shape)
    # Hidden 1 -> Hidden 2
    print('=================')
    print('Hidden 1 -> Hidden 2')
    print('=================')
    # linear sum
    A2 = np.dot(W2, Z1) + B2
    print('A2:', A2.shape)
    print(A2)
    # activation
    Z2 = sigmoid(A2)
    print('Z2:', Z2.shape)
    print(Z2)

>Z1: (3,)
>
>W2: (2, 3)
>
>B2: (2,)
>
>=================
>
>Hidden 1 -> Hidden 2
>
>=================
>A2: (2,)
>
>[ 0.51615984  1.21402696]
>
>Z2: (2,)
>
>[ 0.62624937  0.7710107 ]

### Hidden 2 $\rightarrow$ Output

마지막 출력 층에서는 이전 층에 출력된 $Z$ 값들을 그대로 가져올 수 있다.

    def identity_function(x):
        return x

    W3 = np.array([[0.1, 0.2],
                   [0.3, 0.4]])
    B3 = np.array([0.1, 0.2])

    A3 = np.dot(W3, Z2) + B3
    Y = identity_function(A3)
    print(Y)

> [ 0.31682708  0.69627909]

혹은 Softmax라는 함수를 써서 각 Output의 확률로서 나타낼 수 있다. 보통을 이걸 쓴다.

#### Softmax

$$y_k = \frac{exp(a_k)}{\sum_{i=1}^{n}{exp(a_i)}}$$

    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

얼핏 잘 만든 것 같지만 컴퓨터에서 아주 큰 수를 계산시 Overflow문제가 발생한다. 오버플로우란, 사용 가능한 하드웨어(즉, 32bit 단위 워드의 하드웨어, 레지스터 등)로 연산 결과를 표현할 수 없을 때 오버플로우가 발생한다고 한다. (오버플로우 개념 출처: [[<span style="color: #7d7ee8">링크</span>](https://m.blog.naver.com/PostView.nhn?blogId=osw5144&logNo=120206206420&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F)])

간단히 예를 들어보면 아래의 코드를 실행해보면 금방 알 수 있다.

    a = np.array([1010, 1000, 990])
    softmax(a)

> /Users/user/anaconda/lib/python3.6/site-packages/ipykernel/\_\_main__.py:2: RuntimeWarning: overflow encountered in exp
> from ipykernel import kernelapp as app
>
> /Users/user/anaconda/lib/python3.6/site-packages/ipykernel/\_\_main__.py:2: RuntimeWarning: invalid value encountered in true_divide
  from ipykernel import kernelapp as app
>
> array([ nan,  nan,  nan])

경고가 뜨면서 NaN 값들만 나온다. 이를 방지하기 위해서 입력 신호 중 최대값을 이용하는게 일반적이다. 아래는 분모, 분자 변수에 어떤 상수 C'를 더해도 결국엔 Softmax가 되는 것을 증명 한 식이다.

$$
y_k = \frac{exp(a_k)}{\sum_{i=1}^{n}{exp(a_i)}}
\\ = \frac{Cexp(a_k)}{C\sum_{i=1}^{n}{exp(a_i)}}
\\ = \frac{exp(a_k+\log{C})}{\sum_{i=1}^{n}{exp(a_i+\log{C})}}
\\ = \frac{exp(a_k+C^{'})}{\sum_{i=1}^{n}{exp(a_i+C^{'})}}
$$

    c = np.max(a)
    print(a - c)
    print(softmax(a-c))

> [  0 -10 -20]
>
> [  9.99954600e-01   4.53978686e-05   2.06106005e-09]

이번에는 경고 없이 실행이 잘 된다. 이제 최종 Softmax는 아래와 같다. 이를 출력층에 적용하면 y값에 대한 확률을 볼 수 있다. 이를 0과 1사이의 값으로 만드는 이유가 있는데 향후 학습시에 필요하기 때문이다. (네트워크 학습에서 설명)

    def softmax(a):
        c = np.max(a)
        return np.exp(a - c) / np.sum(np.exp(a - c))

## Feedforward 실습

여태 보았던 3층 Neural Network를 만들어 보자, 어려운 것은 없고 아까 만들었던 것을 나열해보면 쉽다.

실습할 데이터는 mnist 데이터 이며, 아래 링크로 받을 수 있다.

<밑바닥 부터 시작하는 딥러닝> 책의 Github: [[<span style="color: #7d7ee8">링크</span>](
https://github.com/WegraLee/deep-learning-from-scratch)]

입력층에는 784 개의 뉴런, 은닉층1에는 50개, 은닉층2에는 100개, 마지막 층에는 10개의 뉴런으로 구성되어 있는 네트워크다. 활성화 함수는 sigmoid를 쓰고, 마지막에 Softmax로 확률을 구했다. 실행단계에서 batch라는 것이 있는데, 한번에 많은 양의 데이터를 계산하면 느리니, 조금씩 데이터를 사용해서 계산하는 방법이라고 생각하면 된다.

    # 네트워크 만들기
    from dataset.mnist import load_mnist
    import numpy as np

    class NN(object):
        def __init__(self):
            # W1(50, 784) X(784, batch_size)
            # W2(100, 50) Z1(50, batch_size)
            # W3(10, 100) Z2(100, batch_size)
            # B1(50, batch_size)
            # B2(100, batch_size)
            # B3(10, batch_size)
            self.W = {'W1': np.random.normal(size=(50, 784)),  
                      'W2': np.random.normal(size=(100, 50)),  
                      'W3': np.random.normal(size=(10, 100)),}  
            self.B = {'B1': np.random.normal(size=(50, batch_size)),  
                      'B2': np.random.normal(size=(100, batch_size)),  
                      'B3': np.random.normal(size=(10, batch_size)),}  

        def get_data(self):
            (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
                                                              normalize=True,
                                                              one_hot_label=False)
            return x_train, t_train, x_test, t_test

        def predict(self, X):
            W1, W2, W3 = self.W['W1'], self.W['W2'], self.W['W3']
            B1, B2, B3 = self.B['B1'], self.B['B2'], self.B['B3']
            # Input -> Hidden 1
            A1 = np.dot(W1, X) + B1
            Z1 = sigmoid(A1)
            # Hidden 1 -> Hidden 2
            A2 = np.dot(W2, Z1) + B2
            Z2 = sigmoid(A2)
            # Hidden 2 -> Output
            A3 = np.dot(W3, Z2) + B3
            Y = softmax(A3)

            return Y

    # 실행 단계
    model_mnist = NN()
    x_train, t_train, x_test, t_test = model_mnist.get_data()
    acc_count = 0
    batch_size = 100
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size].T  # (784, 100)
        y_batch = model_mnist.predict(x_batch) # (10, 100)
        p = np.argmax(y_batch, axis=0)
        acc_count += np.sum(p == t_train[i:i+batch_size])
    print("accuracy:", acc_count / len(x_train))

> accuracy: 0.0857833333333

정확도란 데이터 중에서 얼만큼 라벨 맞췃는지 측정하는 것인데, 당연하지만 결과가 아주 형편이 없다. 이제 네트워크를 학습시키면서 이를 향상 시킬 것이니까 너무 걱정하지 말자.
