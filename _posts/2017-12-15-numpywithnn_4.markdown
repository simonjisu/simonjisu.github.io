---
layout: post
title: "[deeplearning from scratch]-4: Backpropagation"
categories: deeplearning
author: "Soo"
date: "2017-12-15 12:21:48 +0900"
comments: true
toc: true
---
# Numpy로 짜보는 Neural Network Basic - 4
---

## 오차역전파(Backpropagation)

### 연쇄법칙의 원리
합성 함수의 미분은 합성 함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다.

$$\begin{cases} z = t^2 \\ t = x + y \end{cases}$$

위 식의 미분을 나타내면
$$ \frac{\partial{z}}{\partial{x}} = \frac{\partial{z}}{\partial{t}} \cdot \frac{\partial{t}}{\partial{x}} $$

따라서 $z$, $t$ 식을 미분하게 되면

$$\frac{\partial{z}}{\partial{t}} = 2t$$
$$\frac{\partial{t}}{\partial{x}}=1 $$

$$\therefore\  \frac{\partial{z}}{\partial{x}} = \frac{\partial{z}}{\partial{t}} \cdot \frac{\partial{t}}{\partial{x}} = 2t \cdot 1 = 2(x+y)$$

우리의 목적은 $L$ 에 대해서 $W$ 를 미분하여 조금씩 업데이트 하는 것임으로 아래와 같다고 할 수 있다.
$$\frac{\partial{L}}{\partial{W}} = \frac{\partial{L}}{\partial{Y}} \cdot \frac{\partial{Y}}{\partial{W}}$$

### 덧셈노드와 곱셈노드의 역전파

<img src="/assets/ML/nn/NN_add.png" alt="Drawing" style="width: 400px;"/>

(그림출처: ratsgo님의 블로그[[링크](https://ratsgo.github.io/deep%20learning/2017/05/14/backprop/)])

$$\begin{cases} L(z) \\ z = x + y \end{cases}$$

각각 미분하게 되면

$$\begin{cases}
    \dfrac{\partial{L}}{\partial{z}} \\
    \dfrac{\partial{z}}{\partial{x}} =
    \dfrac{\partial{z}}{\partial{y}} = 1
  \end{cases}$$

따라서 $L$ 을 각각 $x$ 와 $y$ 로 미분하려면

$$\begin{cases}
    \dfrac{\partial{L}}{\partial{x}} = \dfrac{\partial{L}}{\partial{z}} \cdot \dfrac{\partial{z}}{\partial{x}} = \dfrac{\partial{L}}{\partial{z}} \cdot 1 \\
    \dfrac{\partial{L}}{\partial{y}} = \dfrac{\partial{L}}{\partial{z}} \cdot \dfrac{\partial{z}}{\partial{y}} = \dfrac{\partial{L}}{\partial{z}} \cdot 1
  \end{cases}$$

따라서 **덧셈** 노드는 들어온 신호($\frac{\partial{L}}{\partial{z}}$)를 **그대로** 보낸다.


<img src="/assets/ML/nn/NN_multiply.png" alt="Drawing" style="width: 400px;"/>

(그림출처: ratsgo님의 블로그[[링크](https://ratsgo.github.io/deep%20learning/2017/05/14/backprop/)])

$$\begin{cases} L(z) \\ z = x \times y \end{cases}$$

각각 미분하게 되면

$$\begin{cases}
    \dfrac{\partial{L}}{\partial{z}} \\
    \dfrac{\partial{z}}{\partial{x}} = y \\
    \dfrac{\partial{z}}{\partial{y}} = x
  \end{cases}$$

  따라서 $L$ 을 각각 $x$ 와 $y$ 로 미분하려면

$$\begin{cases}
    \dfrac{\partial{L}}{\partial{x}} = \dfrac{\partial{L}}{\partial{z}} \cdot \dfrac{\partial{z}}{\partial{x}} = \dfrac{\partial{L}}{\partial{z}} \cdot y \\
    \dfrac{\partial{L}}{\partial{y}} = \dfrac{\partial{L}}{\partial{z}} \cdot \dfrac{\partial{z}}{\partial{y}} = \dfrac{\partial{L}}{\partial{z}} \cdot x
  \end{cases}$$

따라서 **곱셈** 노드는 들어온 신호에 서로 바뀐 입력신호 값을 **곱해서** 하류로 보낸다.

## Sigmoid 계층의 순전파와 역전파

$$ y = \frac{1}{1+\exp(-x)}$$

### Forward

<img src="/assets/ML/nn/NN_sigmoid_forward.png" alt="Drawing" style="width: 600px;"/>

### Backward

<img src="/assets/ML/nn/NN_sigmoid_back.png" alt="Drawing" style="width: 600px;"/>

<img src="/assets/ML/nn/NN_sigmoid_back2.png" alt="Drawing" style="width: 600px;"/>

> #### 역전파 1단계 ( / )
>
> "/" 연산은 입력변수 x를 $\dfrac{1}{x}$ 로 바꿔준다. 즉 $f_1(x) = \dfrac{1}{x}$ 가 된다.
>
> 미분을 하게 되면 $\dfrac{\partial{f_1}}{\partial{x}} = -\dfrac{1}{x^2} = -y^2$가 되서 입력신호를 하류로 보낸다.
>
> #### 역전파 2단계 ( + )
>
> "+" 연산은 신호를 그대로 하류로 흘러 보낸다
>
> #### 역전파 3단계 (exp)
>
> "exp"연산은 $f_2(x) = exp(x)$ 이며, 미분도 $\dfrac{\partial{f_2}}{\partial{x}} = exp(x)$ 로 그대로 곱해서 하류로 보낸다.
>
> #### 역전파 4단계 ( x )
>
> "$\times$"연산은 서로 바뀐 입력신호의 값을 곱해서 보낸다.

따라서, 최종적으로 시그모이드의 역전파 출력값은 아래와 같다.

<img src="/assets/ML/nn/NN_sigmoid_last.png" alt="Drawing" style="width: 400px;"/>

$$\begin{aligned}
\dfrac{\partial{L}}{\partial{y}}y^{2}\exp(-x)
&= \dfrac{\partial{L}}{\partial{y}} \dfrac{1}{[1+\exp(-x)]^2}\exp(-x) \\
&= \dfrac{\partial{L}}{\partial{y}} \dfrac{1}{1+\exp(-x)} \dfrac{\exp(-x)}{1+\exp(-x)} \\
&= \dfrac{\partial{L}}{\partial{y}}y(1-y) \\
\end{aligned}$$

이것을 코드로 구현하게 되면
```python
class Sigmoid(object):
    def __init__(self):
        self.out = None  # 역전파시 곱해야 하기 때문에 저장해둔다

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
```

## Affine 계층과 Affine Transform
기하학에서 신경망 순전파 때 수행하는 행렬의 내적을 Affine Transform이라 하며, Affine 계층은 어파인 변환을 수행 처리하는 계층이다.

위키백과[[링크](https://ko.wikipedia.org/wiki/%EC%95%84%ED%95%80_%EB%B3%80%ED%99%98)]

### Forward

$A = X \cdot W + B$

### Backward

$\begin{cases}
    \dfrac{\partial{L}}{\partial{X}} = \dfrac{\partial{L}}{\partial{A}} \cdot \dfrac{\partial{A}}{\partial{X}} = \dfrac{\partial{L}}{\partial{A}} \cdot W^T \\
    \dfrac{\partial{L}}{\partial{W}} = \dfrac{\partial{L}}{\partial{A}} \cdot \dfrac{\partial{A}}{\partial{W}} = X^T \cdot \dfrac{\partial{L}}{\partial{A}} \\
    \dfrac{\partial{L}}{\partial{B}} = 1
  \end{cases}$

```python
class Affine(object):
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(self.b, axis=0)

        return dx
```

## Backpropogation 사용한 학습구현

```python
import collections
from layers import *

class TwoLayer(object):
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(hidden_size)

        # 계층 생성
        self.layers = collections.OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLu1'] = ReLu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x: 입력 데이터, t: 정답 데이터
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = np.sum(y == t) / float(x.shape[0])
        return acc

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        # save
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
```

TwoLayer Neural Network 를 생성하는 객체는 따로 파일에 저장하고 불러내는 것이 좋다. OrderedDict은 dictionary 형태로 입력 순서를 기억해주는 좋은 함수다.

```python
from dataset.mnist import load_mnist
from two_layer_nn import TwoLayer

# data_loading
(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

#highper parameter
epoch_num = 10000
train_size = x_train.shape[0]
batch_size = 100
alpha = 0.01  # learning rate
epsilon = 1e-6

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

start = time.time()
nn = TwoLayer(input_size=784, hidden_size=100, output_size=10, weight_init_std=0.01)
for epoch in range(epoch_num):
    # get mini batch:
    batch_mask = np.random.choice(train_size, batch_size) # shuffle 효과
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    # gradient 계산
    grad = nn.gradient(x_batch, y_batch)

    # update
    for key in ['W1', 'b1', 'W2', 'b2']:
        nn.params[key] = nn.params[key] - alpha * grad[key]

    # record
    loss = nn.loss(x_batch, y_batch)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산
    if epoch % iter_per_epoch == 0:
        train_acc = nn.accuracy(x_train, y_train)
        test_acc = nn.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('# {0} | trian acc: {1:.5f} | test acc: {2:.5f}'.format(epoch, train_acc, test_acc))

end = time.time()
print('total time:', (end - start))

# 결과
# 0 | trian acc: 0.10775 | test acc: 0.10700
# 600 | trian acc: 0.10775 | test acc: 0.10700
# 1200 | trian acc: 0.10775 | test acc: 0.10700
# 1800 | trian acc: 0.10775 | test acc: 0.10700
# 2400 | trian acc: 0.10775 | test acc: 0.10700
# 3000 | trian acc: 0.10775 | test acc: 0.10700
# 3600 | trian acc: 0.10775 | test acc: 0.10700
# 4200 | trian acc: 0.10775 | test acc: 0.10700
# 4800 | trian acc: 0.10775 | test acc: 0.10700
# 5400 | trian acc: 0.10775 | test acc: 0.10700
# 6000 | trian acc: 0.10775 | test acc: 0.10700
# 6600 | trian acc: 0.10775 | test acc: 0.10700
# 7200 | trian acc: 0.10775 | test acc: 0.10700
# 7800 | trian acc: 0.10775 | test acc: 0.10700
# 8400 | trian acc: 0.10775 | test acc: 0.10700
# 9000 | trian acc: 0.10775 | test acc: 0.10700
# 9600 | trian acc: 0.10775 | test acc: 0.10700
# total time: 61.07117795944214
```

학습은 전혀 안되지만 수치 미분보다 더 빠르게 진행된다는 것을 알 수 있다.

왜 학습이 안됐을까에 대해서는 담은 시간에 이야기 하겠다.
