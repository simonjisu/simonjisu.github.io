---
layout: post
title: "BasicNN - 1"
categories: "DataScience"
author: "Soo"
date: "2017-12-07 17:54:18 +0900"
---
# Numpy로 짜보는 Neural Network Basic - 1
---
Neural Network의 역사: [링크](http://solarisailab.com/archives/1206) 참고

Neural Network를 알려먼 퍼셉트론이란 개념을 우선 이야기 해보자
## 퍼셉트론(Perceptron)
퍼셉트론(perceptron)은 인공신경망의 한 종류로서, 1957년에 코넬 항공 연구소(Cornell Aeronautical Lab)의 프랑크 로젠블라트 (Frank Rosenblatt)에 의해 고안되었다. 이것은 가장 간단한 형태의 피드포워드(Feedforward) 네트워크 - 선형분류기- 으로도 볼 수 있다.

퍼셉트론이 동작하는 방식은 다음과 같다. 각 노드의 가중치와 입력치를 곱한 것을 모두 합한 값이 활성함수에 의해 판단되는데, 그 값이 임계치(보통 0)보다 크면 뉴런이 활성화되고 결과값으로 1을 출력한다. 뉴런이 활성화되지 않으면 결과값으로 -1을 출력한다.

(출처: 위키백과 [링크](https://ko.wikipedia.org/wiki/%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0))

퍼셉트론을 이야기 하면 XOR문제를 빠트릴 수가 없는데, 그 이유는 XOR문제를 푸는데 다층 퍼셉트론이 사용되며, 이게 Neural Network의 모태이기 되기 때문이다.

### AND, OR, NAND
XOR문제를 이야기 하기 전에 AND, OR, NAND에 대해 이야기 해야한다. 어떤 이산 변수 $x_1$과 $x_2$가 있다고 생각해보자. 어떤 선형식을 통해서 아래와 같은 표를 분류하고 싶다.

| $x_1$ | $x_2$ | $y$ |
|:--:|:--:|:--:|
|0|0|0|
|1|0|0|
|0|1|0|
|1|1|1|

어떤 선형식을 세워야 $x_1$과 $x_2$를 넣었을 때 y값이 나올까? 아래 식을 한번 보자.

$$
y =
\begin{cases}
  0\ \ (b + w_1x_1 + w_2x_2 \leq 0) \\
  1\ \ (b + w_1x_1 + w_2x_2 > 0) \\
\end{cases}
$$

만약에 $b$ 가 임의의 음수이고, $w_1$, $w_2$ 값이 $b$ 보다 작거나 같은 임의의 양수면 이 식은 항상 성립한다. 각종 변수가 이를 만족 할 때 **AND 게이트** 라고 부르며 코드로 이렇게 짜볼 수 있다.

    def AND(x1, x2):
      x = np.array([x1, x2])
      w = np.array([0.5, 0.5])
      b = -0.7
      tmp = np.sum(x*w) + b
      if tmp <= 0:
          return 0
      else:
          return 1

확인해보면

    xx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for x in xx:
      print('AND({0},{1}) : {2}'.format(x[0], x[1], AND(x[0], x[1])))

>AND(0,0) : 0
>
>AND(0,1) : 0
>
>AND(1,0) : 0
>
>AND(1,1) : 1

위에 있는 식을를 그림으로 표현하면 아래와 같은 그림이다. ($b$는 빠졌지만 위에다가 같이 더해준다.)

<img src="/assets/ML/perceptron/perceptron_1.png" alt="Drawing" style="width: 400px;"/>

이제 OR 게이트와, NAND게이트도 한 번 생각해보자.

**NAND 게이트** 는 아래와 같은 표로 나타낼 수 있다.

| $x_1$ | $x_2$ | $y$ |
|:--:|:--:|:--:|
|0|0|1|
|1|0|1|
|0|1|1|
|1|1|0|

    def NAND(x1, x2):
        x = np.array([x1, x2])
        w = np.array([-0.5, -0.5])
        b = 0.7
        tmp = np.sum(x*w) + b
        if tmp <= 0:
            return 0
        else:
            return 1

결과 값을 측정해보면

    xx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for x in xx:
        print('NAND({0},{1}) : {2}'.format(x[0], x[1], NAND(x[0], x[1])))

>NAND(0,0) : 1
>
>NAND(0,1) : 1
>
>NAND(1,0) : 1
>
>NAND(1,1) : 0

마찬가지로 **OR 게이트** 는 아래와 같은 표로 나타낼 수 있다.

| $x_1$ | $x_2$ | $y$ |
|:--:|:--:|:--:|
|0|0|0|
|1|0|1|
|0|1|1|
|1|1|1|

    def OR(x1, x2):
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.2
        tmp = np.sum(x*w) + b
        if tmp <= 0:
            return 0
        else:
            return 1

결과 값을 확인해보면

    xx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for x in xx:
        print('OR({0},{1}) : {2}'.format(x[0], x[1], OR(x[0], x[1])))

> OR(0,0) : 0
>
>OR(0,1) : 1
>
>OR(1,0) : 1
>
>OR(1,1) : 1

이제 정리해서 각각의 게이트가 제대로 선형으로 분리 되었는지 확인해보자. x축과 y축은 각각 $x_1$ 과 $x_2$를 나타내며, 빨간 동그라미는 1, X로 표시된 곳은 0이라는 뜻이다. 우리의 예시 $(b=-0.7\ w_1, w_2=0.5)$ AND 게이트에서 $x_1$이 0일때 절편인 $x_2$가 1.2가 되니까 맞게 분류하는 것을 볼 수 있다.

<img src="/assets/ML/perceptron/perceptron_2.png" alt="Drawing" style="width: 800px;"/>

### XOR 문제
XOR 문제란 어떤 선형식으로 이산 변수 $x_1$과 $x_2$에 대해서 둘중에 하나라도 1이 되면 결과 값으로 1를 반환하, 둘다 0이거나 1이면 0을 반환하는 식을 찾는 것이다.

표로 그려보면 아래와 같은 문제다.

| $x_1$ | $x_2$ | $y$ |
|:--:|:--:|:--:|
|0|0|0|
|1|0|1|
|0|1|1|
|1|1|0|

그림으로 보면 아래와 같은 문제를 푸는 것이다.

<img src="/assets/ML/perceptron/perceptron_3.png" alt="Drawing" style="width: 400px;"/>

아까와 같은 방법으로 단일 선으로 이 문제를 풀수 있을까?

절대 풀 수 없다. 그래서 나타난 것이 게이트를 겹쳐서 올리는 것이다. 아래 그림을 보자.

<img src="/assets/ML/perceptron/perceptron_4.png" alt="Drawing" style="width: 400px;"/>

이런 구조로 게이트를 짜면 어떻게 될까? 예를 들어 $x_1, x_2 = 0$ 이라고 가정해보자. $x_1$과 $x_2$가 NAND 게이트를 거치면 1, OR 게이트를 거치면 0이 나온다. 1과 0이 AND 게이트를 거치면 0이 된다! 마찬가지로 해보면 XOR 문제를 풀 수가 있다! 이처럼 게이트 층을 두개 만들어 XOR문제를 풀었으며, 이것이 **다층 퍼셉트론** 의 기원이라고 말 할 수 있다.

표로 그려보면 아래와 같다. $s_1$ 과 $s_2$는 각각 NAND 게이트와 OR 게이트를 뜻하면 최종단에 AND 게이트를 거쳐 y 값을 구할 수 있다.

| $x_1$ | $x_2$ | $s_1$ | $s_2$ | $y$ |
|:--:|:--:|:--:|:--:|:--:|:--:|
|0|0|1|0|0|
|1|0|1|1|1|
|0|1|1|1|1|
|1|1|0|1|0|

코드로 구현하느 것은 아까 만든 코드를 나열하면 된다.

    def XOR(x1, x2):
        s1 = NAND(x1, x2)
        s2 = OR(x1, x2)
        y = AND(s1, s2)
        return y

확인해보면

    xx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for x in xx:
        print('XOR({0},{1}) : {2}'.format(x[0], x[1], XOR(x[0], x[1])))

>XOR(0,0) : 0
>
>XOR(0,1) : 1
>
>XOR(1,0) : 1
>
>XOR(1,1) : 0

Rosenblatt이 제시한 정확한 단일 퍼셉트로은 아래의 그림과 같다. Input과 Weight를 곱해서 더한 다음에 Activation function을 적용해서 그 값이 0보다 크면 1 작으면 -1를 반환하는 Feedforward 선형 분류기의 구조다.

<img src="/assets/ML/perceptron/perceptron_5.png" alt="Drawing" style="width: 600px;"/>

이는 XOR 문제를 해결 할 수 없었고, Multi-Layer Perceptrons이 1986년에 등장 했는데 중간에 Hidden 층을 더 쌓으면서 XOR 문제를 해결하였다.

다음 시간에는 Feedforward과정을 자세히 알아본다.
