---
layout: post
title: "[deeplearning from scratch]-3: Loss Function"
categories: deeplearning
author: "Soo"
date: "2017-12-10 14:06:55 +0900"
comments: true
toc: true
---
# Numpy로 짜보는 Neural Network Basic - 3
---

저번 시간에는 Feedforward 과정을 보았는데, 정확도가 8.578% 밖에 안됐다. 이제 Neural Network가 데이터로부터 어떻게 학습하여 정확도를 올리는지 보자.

## 손실 함수(Loss Function)
왜 우리의 목표인 정확도를 안쓰고 손실 함수라는 매개변수를 설정하는 걸까?

그 이유는 먼저 밝히면 신경망 학습에 미분이 사용되기 때문이다. 최적의 가중치(그리고 편향)을 탐색할 때 손실 함수의 값을 가능한 작게하는 가중치 값을 찾는데, 이때 가중치의 미분을 계산하고, 그 미분 값을 단서로 가중치를 서서히 갱신하는 과정을 거친다. 그러나 손실함수에 정확도를 쓰면 가중치의 미분이 대부분의 장소에서 0이 되기 때문에 가중치 값을 갱신할 수가 없다.

mnist 데이터의 경우 최종 출력층에 나온 $y$ 값은 Softmax에 의해 $(10 \times 1)$ 행렬의 확률로 출력되고, 그에 응답하는 정답 $t$ 는 one-hot encoded된 행렬이다.
```python
y = np.array([0.05, 0.01, 0.7, 0.14, 0.05, 0.0, 0.05, 0.0, 0.0, 0.0])
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
```

### 평균 제곱 오차(MSE)

$$E=\frac{1}{2}\sum_{k}{(y_k - t_k)^2}$$
```python
def mean_squared_error(y, t):
    return (1/2) * np.sum((y - t) ** 2)

mean_squared_error(y, t)
```
> 0.05860000000000002

### 교차 엔트로피 오차(Cross Entropy)

$$E=-\sum_{k}{t_k\log{y_k}}$$
```python
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

cross_entropy_error(y, t)
```
> 0.51082545709933802

여기서 delta라는 작은 값을 더해준 이유는 y값이 0이면 $\log 0= -\inf$가 되서 미분 계산이 불가능하기 때문이다.

### 미니 배치 학습

$$E=-\frac{1}{N}\sum_{n}{\sum_{k}{t_k\log{y_k}}}$$

엄청나게 많은 양의 데이터를 사용하는데 오차를 한번에 계산하려면 오랜 시간이 든다. 따라서 작은 양의 데이터를 사용해 조금씩 오차의 합을 구한다음에 그것의 평균을 내면 전체의 근사치로 사용할 수 있다.
```python
def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t] + delta)) / batch_size
```
## 미분

목적을 정했으니 이제 학습에 들어가면된다. 손실함수를 가중치에 대한 미분을 구해야 한다.

### 수치 미분과 중심 차분법

수치 미분이란 변화율이라고 볼 수 있다. $x$에서 $h$만큼 변했을 때 $f(x)$의 변화량을 나타낸 것이다.

$$\frac{df(x)}{dx} = \lim_{h\rightarrow0}{\frac{f(x+h) - f(x)}{h}}$$

그러나 $f(x+h) - f(x)$ 는 굉장히 작은 수라 컴퓨터로 구현시 Underflow문제에 봉착하게 된다.
```python
np.float32(1e-50)
```
> 0.0

따라서 수치 미분에서 $h$는 되도록 너무 작은 값은 못쓴다.

**중심 차분법** 을 이용하면 미분은 아래와 같다.

$$\frac{df(x)}{dx} = \lim_{h\rightarrow0}{\frac{f(x+h) - f(x-h)}{2h}}$$
```python
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2*h)
```
예시 함수 $y = 0.01 x^2 + 0.1 x$ 의 수치 미분을 보자
```python
def f1(x):
    return 0.01 * x**2 + 0.1 * x

print(numerical_diff(f1, 5))
print(numerical_diff(f1, 10))
```
> 0.1999999999990898
>
> 0.2999999999986347

정확하게 0.2와 0.3이 나오지 않는 이유는 이진수 부동소수점 방식[[링크](https://ko.wikipedia.org/wiki/%EB%B6%80%EB%8F%99%EC%86%8C%EC%88%98%EC%A0%90)]의 정확도 문제니까 round 함수를 사용해 반올림하여 사용해야한다.

<img src="/assets/ML/nn/numerical_diff.png" alt="Drawing" style="width: 500px;"/>

2차원 이상의 데이터는 어떻게 짜야할까? 아래의 코드를 참조하자
```python
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad
```
* np.nditer: iterator 객체를 만들어 준다. 행마다 원소가 iterate 된다.

### Gradient Descent

경사 하강법이란 현 위치에서 기울어진 방향으로 일정 거리를 이동하고, 또 그 위치에서 기울기를 구해서 그 방향으로 계속 나아가는 방법이다. 이렇게 해서 손실함수를 점점 작게 만들어 손실함수의 최저점으로 이끌고 간다(가능하다면).

$$w_{new} = w_{old} - \eta \cdot \frac{\partial f}{\partial w_{old}}$$

$\eta$ 는 학습률(learning rate)라고 하며 갱신하는 양을 나타낸다.

아래 그림은 주변이 높고 중앙이 낮은 모양(그릇을 생각하자)을 3차원에서 2차원으로 그렸을 때, $(4, 5)$ 점에서 시작해서 경사 하강법으로 최저점을 찾는 과정이다. 함수는 $f(x) = x^2\ , x\in \mathbb{R}^3$ 다.

<img src="/assets/ML/nn/GDanimation.gif" alt="Drawing" style="width: 500px;"/>



## 학습 알고리즘

### 간단한 NN 으로 가중치의 미분 구해보기
```python
class simpleNet(object):
    def __init__(self):
        # Input size = 2
        # Output size = 3
        self.W = np.random.normal(size=(2,3))

    def predict(self, x):
        a = np.dot(x, self.W)
        y = softmax(a)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])
nn = simpleNet()

f = lambda w: nn.loss(x, t)
dW = numerical_gradient(f, nn.W)
print(dW)
```
> [[ 0.05244267  0.24743359 -0.29987626]
>
>  [ 0.07866401  0.37115039 -0.44981439]]

### 확률적 경사 하강법(SGD)
아래 방법은 데이터를 무작위로 가져와서 학습하는 것이기 때문에 확률적 경사 하강법(Stochastic Gradient Descent)이라고도 한다.


* 1단계: 미니배치

  훈련 데이터 중 일부를 무작위로 가져온 데이터를 미니 배치라고 하며, 미니 배치의 손실 함수 값을 줄이는 것이 목표다.

* 2단계: 기울기 산출

  미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구한다. 기울기는 손실 함수의 값을 가장 작게 만든다.

* 3단계: 매개변수(가중치) 갱신

  가중치 매개변수를 기울기 방향으로 아주 조금 갱신한다.

* 4단계: 반복

  1~3 단계를 반복한다.

### 2층 Neural Network 실습

<img src="/assets/ML/nn/NN_3.png" alt="Drawing" style="width: 350px;"/>

* Input Size: m = 3
* Hidden Size: h = 4
* Output Size: o = 3
$$ X_{(batch,\ m)} \cdot W1_{(m,\ h)} + B1_{(batch,\ h)} \rightarrow A1_{(batch,\ h)} $$
$$ sigmoid(A1_{(batch,\ h)}) \rightarrow Z1_{(batch,\ h)}$$
$$ Z1_{(batch,\ h)} \cdot W1_{(h,\ o)} + B1_{(batch,\ o)} \rightarrow A2_{(batch,\ o)} $$
$$ \sigma(A2_{(batch,\ o)}) \rightarrow Y_{(batch,\ o)}$$

이것을 구현해보자. 수치로 구현한 2층 Neural Network 코드는 [[여기](https://github.com/WegraLee/deep-learning-from-scratch/blob/master/ch04/two_layer_net.py)]서 가져왔다.

```python
(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

#highper parameter
epoch_num = 1
train_size = x_train.shape[0]
batch_size = 100
alpha = 0.1  # learning rate
epsilon = 1e-6

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
nn = TwoLayer(input_size=784, hidden_size=100, output_size=10)

start = time.time()
for epoch in range(epoch_num):
    # get mini batch:
    batch_mask = np.random.choice(train_size, batch_size) # shuffle 효과
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    # gradient 계산
    grad = nn.num_gradient(x_batch, y_batch)

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
        print('trian acc: {0:.5f} | test acc: {1:.5f}'.format(train_acc, test_acc))

    # stop point
    if epoch > 10:
        stop_point = np.sum(np.diff(np.array(train_loss_list[i-11:])) < epsilon)
        if stop_point == 10:
            print(epoch)
            break

end = time.time()
print('total time:', (end - start))
```
> trian acc: 0.10442 | test acc: 0.10280
>
> total time: 175.5657160282135


2단계에서 수치미분을 구현하기는 쉬우나 업데이트 하는데 시간이 너무 오래 걸린다. 1 Epoch만 돌렸는데 175초 걸렸다.

따라서 가중치 매개변수의 기울기를 효율적으로 계산하는 **오차역전파** 방법으로 업데이트 해야한다. 이건 다음 글에서 계속 진행하겠다.
