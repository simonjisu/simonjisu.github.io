---
layout: post
title: "NUMPY with NN - 5"
categories: "DataScience"
author: "Soo"
date: "2018-01-13 14:17:06 +0900"
---
# Numpy로 짜보는 Neural Network Basic - 5
---
## 학습관련 기술 Part 1

### Optimizer

손실 함수 값을 가능한 낮게 만들어 매개변수 최적값을 찾는 과정을 **최적화** 라고 한다. 여기서 몇가지 방법을 한번 살펴본다.

#### SGD(확률적 경사 하강법)
$W \leftarrow W - \eta \dfrac{\partial L}{\partial W}$

$\eta$ 는 학습률로 얼만큼 가중치를 업데이트 할지 정하는 하이퍼파라미터다. 즉 우리가 미리 정해줘야하는 변수다. 그러나 SGD 알고리즘에서는 이 변수에 따라서 학습되는 모양이 다르다.

    class SGD(object):
        def __init__(self, lr=0.01):
            self.lr = lr

        def update(self, params, grads):
            for key in params.keys():
                params[key] -= self.lr * grads[key]

**장점**:
* 일부 데이터로 업데이트를 해서 진동이 심할 수도 있지만, 전체 데이터의 Gradient를 구하는 것보다 빠르다

**단점**:
* learning rate에 따라서 global min을 찾지 못하고 local min에 갇힐 가능서 존재
* Oscilation(발진 현상): 해에 접근 할 수록 수렴 속도($\dfrac{\partial L}{\partial W}$)가 느려짐, 따라서 협곡 같은 모양에서 헤매는 경우 존재한다. 그렇다고 lr을 너무 높히면 발산 할 수도 있음(loss값이 커지는 현상)

아래와 같은 함수의 최적값을 찾아보자.

$f(x, y) = \dfrac{1}{20} x^2 + y^2$

    def f(x, y):
        return np.array((1/20)*(x**2) + (y**2))

$f$ 를 미분하면 아래와 같다.

$\dfrac{\partial f}{\partial x}, \dfrac{\partial f}{\partial y} = \dfrac{x}{10}, 2y$

    def f_prime(x, y, grads=None):
        if grads is None:
            grads = {}

        grads['x'] = (1/10)*x
        grads['y'] = 2*y
        return grads

시작은 **(-7, 2)** 점부터 시작한다고 하면 아래처럼 그림으로 표현할 수 있다.

<img src="/assets/ML/nn/fgraph.png" alt="Drawing" style="width: 400px;"/>

이 함수의 최저점은 (0, 0) 점으로 볼 수 있다.

이제 learning rate 를 0.1 과 0.9로 각각 정해서 SGD를 적요해보자. 총 30 epoch동안 Gradient를 구하고 이를 조금씩 업데이트 하는 방식을 취했다.

|Video|Graph|
|:-:|:-:|
|<video controls="controls" style="width: 400px;" autoplay loop muted markdown="1"> <source type="video/mp4" src="/assets/ML/nn/SGD_0.1.mp4"  markdown="1"> </source>| <img src="/assets/ML/nn/SGD_0.1.png" alt="Drawing" style="width: 400px;"/> |

learning rate 가 0.1 일때 학습이 조금씩 진행 되는 것을 볼 수 있다. 그러나 epoch 횟수가 너무 적어 최적의 값까지 도달을 못했다.

|Video|Graph|
|:-:|:-:|
|<video controls="controls" style="width: 400px;" autoplay loop muted markdown="1"> <source type="video/mp4" src="/assets/ML/nn/SGD_0.9.mp4"  markdown="1"> </source>| <img src="/assets/ML/nn/SGD_0.9.png" alt="Drawing" style="width: 400px;"/> |

learning rate 가 0.9 일때 학습이 크게 진행 되는 것을 볼 수 있다. 그러나 변동이 심해서 크게 흔들리면서 최저점으로 가는 모습을 볼 수 있다.

학습률이 다르다는 것은 한 번 나아갈때 폭의 길이를 보면 그 차이를 알 수 가 있다.

#### Momentum
$v \leftarrow \gamma v - \eta \dfrac{\partial L}{\partial W}$

$W \leftarrow W + v$

모멘텀 방식은 gradient 방향에 일종의 관성을 더해줘서 기존의 이동 방향에 힘들 실어줘 더 이동할 수 있게 만들어준다. $v$ 의 초기값은 0으로 설정하고 진행한다. 따라서 첫 step이 후 기존에 이동했던 방향을 저장해둔 $v$ 가 추가로 저장 되어 다음 step에 더해져 조금 더 움직이게 된다.

|Video|Graph|
|:-:|:-:|
|<video controls="controls" style="width: 400px;" autoplay loop muted markdown="1"> <source type="video/mp4" src="/assets/ML/nn/Momentum.mp4"  markdown="1"> </source>| <img src="/assets/ML/nn/Momentum.png" alt="Drawing" style="width: 400px;"/> |

learning rate가 0.1 일때 SGD보다 더 많이 가는 것을 알 수 있다.

|Video|Graph|
|:-:|:-:|
|<video controls="controls" style="width: 400px;" autoplay loop muted markdown="1"> <source type="video/mp4" src="/assets/ML/nn/Momentum_0.9.mp4"  markdown="1"> </source>| <img src="/assets/ML/nn/Momentum_0.9.png" alt="Drawing" style="width: 400px;"/> |

learning rate가 0.9 일때 주변을 헤매면서 가는 모습을 볼 수 있다. 하이퍼파라미터를 잘 조정해야 학습이 빠르게 진행 된 다는 것을 알 수 있다.

#### Adagrad
$h \leftarrow h + \dfrac{\partial L}{\partial W} \odot \dfrac{\partial L}{\partial W}$

$W \leftarrow W - \eta \dfrac{1}{\sqrt{h +\epsilon}} \dfrac{\partial L}{\partial W}$

학습률($\eta$)에 대한 고민이 많이지자 이를 해결해보기 위해 나온 알고리즘이 AdaGrad 다.

학습률을 처음에 크게 했다 나중에 차차 줄여가는 **학습률 감소(learning rate decay)** 기술이 이 알고리즘의 특징이다. 각각의 매개변수에 맞춤형 학습률 값을 맞춰 줄 수가 있다.

$\odot$ 는 여기서 dot product가 아닌 element-wise multiplication를 말한다. 수식을 보면 gradient를 제곱하여 h에 저장한다. 업데이트시 여태까지 저장해온 gradient 제곱 값을 분모로 두게 된다. 따라서 시간이 지날 수록 gradient 누적 값이 큰 것은 learning rate 가 반대로 작아지게 된서 학습률이 조정 된다. 이를 적응적으로(adaptive) 학습률을 조정한다고 한다.

    class Adagrad(object):
        def __init__(self, lr=0.01):
            self.lr = lr
            self.h = None
            self.epsilon = 1e-6  # 0으로 나누눈 것을 방지

        def update(self, params, grads):
            if self.h is None:
                self.h = {}
                for key, val in params.items():
                    self.h[key] = np.zeros_like(val)

            for key in params.keys():
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / np.sqrt(self.h[key] + self.epsilon)

|Video|Graph|
|:-:|:-:|
|<video controls="controls" style="width: 400px;" autoplay loop muted markdown="1"> <source type="video/mp4" src="/assets/ML/nn/Adagrad.mp4"  markdown="1"> </source>| <img src="/assets/ML/nn/Adagrad.png" alt="Drawing" style="width: 400px;"/> |

학습률을 1.5로 크게 주었는데도 차차 감소하면서 학습되는 과정을 볼 수 가 있다.

그러나 이렇게 좋아보이는 방법도 **단점** 이 있다.

과거의 기울기 값들을 전부 누적해서 더하기 때문에 h 값이 많이 커지게 되면 학습률 부분($\dfrac{1}{\sqrt{h +\epsilon}}$)이 0에 가까워져 더 이상 학습이 진행이 안되는 상황이 발생할 수 있다.

이를 개선하기 위해서 RMSProp과 Adadelta라는 방법이 있다. (코드는 기본 알고리즘 원리만 구현해놨다. 구체적으로 효율적인 학습을 위해서 조금씩 변형이 가해진다. 논문 참조 할 것, ~~아직 이해중~~)

RMSProp:

    class RMSProp(object):
        def __init__(self, lr=0.01, gamma=0.9):
        """G는 이동평균의 개념으로 과거 1보다 작은 gamma값을 곱해서 서서히 잊게 하고 새로운 값을 조금씩 더 해준다."""
            self.lr = lr
            self.gamma = gamma  # decay term
            self.G = None
            self.epsilon = 1e-6  # 0으로 나누눈 것을 방지

        def update(self, params, grads):
            if self.G is None:
                self.G = {}
                for key, val in params.items():
                    self.G[key] = np.zeros_like(val)

            for key in params.keys():
                self.G[key] += self.gamma * self.G[key] + (1 - self.gamma) * (grads[key] * grads[key])
                params[key] -= self.lr * grads[key] / np.sqrt(self.G[key] + self.epsilon)

AdaDelta:

    class AdaDelta(object):
        def __init__(self, gamma=0.9):
            """
            https://arxiv.org/pdf/1212.5701
            """
            self.gamma = gamma  # decay term
            self.G = None  # accumulated gradients
            self.s = None  # accumulated updates
            self.del_W = None
            self.epsilon = 1e-6  # 0으로 나누눈 것을 방지
            self.iter = 0

        def update(self, params, grads):
            if (self.G is None) | (self.s is None) | (self.del_W is None):
                # Initialize accumulation variables
                self.G = {}
                self.s = {}  
                self.del_W = {}
                for key, val in params.items():
                    self.G[key] = np.zeros_like(val)
                    self.s[key] = np.zeros_like(val)
                    self.del_W[key] = np.zeros_like(val)

            for key in params.keys():
                self.G[key] += self.gamma * self.G[key] + (1 - self.gamma) * (grads[key] * grads[key])
                self.del_W[key] = -(np.sqrt(self.s[key] + self.epsilon) / np.sqrt(self.G[key] + self.epsilon)) * grads[key]
                self.s[key] += self.gamma * self.s[key] + (1 - self.gamma) * self.del_W[key]**2
                params[key] += self.del_W[key]

#### Adam(Adaptive Moment Estimation)
**Adam** (Adaptive Moment Estimation)은 RMSProp과 Momentum 방식을 합친 것 같은 알고리즘이다.

<img src="/assets/ML/nn/Algorithm_Adam.png" alt="Drawing" style="width: 800px;"/>

출처: [<span style="color: #7d7ee8">https://arxiv.org/abs/1412.6980v8</span>](https://arxiv.org/abs/1412.6980v8)

* $m_t$: the exponential moving averages of the gradient (Momentum쪽)
* $v_t$: the squared gradient (RMSProp쪽)
* $\beta_1$: the exponential decay rates for $m_t$, 보통 0.9 취함
* $\beta_2$: the exponential decay rates for $v_t$, 보통 0.999 취함

알고리즘 그대로 짜는게 아니라 조금더 효율적인 계산을 하기 위해서 아래와 같은 내용을 이해하고 보정해줘야 한다...(자세한 건 논문에 더 있음)

The moving averages themselves are estimates of the 1st moment (the mean) and the 2nd raw moment (the uncentered variance) of the gradient.
However, these moving averages are initialized as (vectors of) 0’s, leading to moment estimates that are biased towards zero, especially
during the initial timesteps, and especially when the decay rates are small (i.e. the $\beta$s are close to 1).

자세한 설명은 추후 보정하겠다. ~~아직 이해중이다~~

아래는 다른 사람의 코드를 따왔다. 출처: [<span style="color: #7d7ee8">https://github.com/WegraLee/deep-learning-from-scratch/</span>](https://github.com/WegraLee/deep-learning-from-scratch/blob/master/common/optimizer.py)

    class Adam(object):
        """Adam (http://arxiv.org/abs/1412.6980v8)"""

        def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.iter = 0
            self.m = None
            self.v = None
            self.history_ = {'x':[], 'y':[]}

        def update(self, params, grads):
            if self.m is None:
                self.m, self.v = {}, {}
                for key, val in params.items():
                    self.m[key] = np.zeros_like(val)
                    self.v[key] = np.zeros_like(val)

            self.iter += 1
            lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

            for key in params.keys():
                # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
                # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
                self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
                self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

                params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

                # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
                # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
                # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

|Video|Graph|
|:-:|:-:|
|<video controls="controls" style="width: 400px;" autoplay loop muted markdown="1"> <source type="video/mp4" src="/assets/ML/nn/Adam.mp4"  markdown="1"> </source>| <img src="/assets/ML/nn/Adam.png" alt="Drawing" style="width: 400px;"/> |

논문 결론 부에는 Adam 알고리즘이 큰 데이터 셋이나 고차원 파라미터 공간을 학습하는데 효율적이다라고 이야기 하고 있다.

(논문을 보려면 시계열 지식이 필요하다. ㅠㅠ)

다음 시간에는 가중치 초기화와 배치 노말라이제이션에 대에서 이야기 해보도록 하겠다.
