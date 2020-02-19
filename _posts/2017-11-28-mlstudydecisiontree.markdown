---
layout: post
title: "ML-DecisionTree"
date: "2017-11-28 14:48:12 +0900"
comments: true
toc: true
---
# [ML] DecisionTree

의사결정나무의 소개는 [링크](https://ko.wikipedia.org/wiki/%EA%B2%B0%EC%A0%95_%ED%8A%B8%EB%A6%AC_%ED%95%99%EC%8A%B5%EB%B2%95)로 대체하고 어떻게 진행되는지만 알아보자

* 의사결정나무의 장점: 어떤 변수가 분류에 영향을 끼쳤는지 사람이 보기 쉽게 되어있다. 사람 입장에서 해석에 용이하다.
* 의사결정나무의 단점: 오버피팅이 심하다. 따라서 보통 랜덤 포레스트라는 앙상블 방법론을 쓴다. 초기 단계에서 랜덤 요소를 넣어서 오버피팅을 방지한다.

의사 결정나무모델은 분류를 할때 책상정리에 비유한다. 더러운 책상에 있는 물건을 어떤 기준에 따라서 하나씩 정리하는 거다.

<img src="/assets/ML/DecisionTree_Desk.jpeg" alt="Drawing" style="width: 400px;"/>

(사진출처: 네이버 블로그)

그렇다면 데이터를 분류하는 기준은 도대체 무엇인가?

# 엔트로피(Entropy)와 정보획득(Information Gain)
데이터를 분류하는 기준은 아래와 같다.

1. 어떤 기준으로 분류 후에 histogram으로부터 조건부 엔트로피를 계산 함
2. 이전 entropy와 새로구한 조건부 엔트로피의 차이(:=Infomation Gain)이 최대 인 것을 best feature로 선택한다

[엔트로피](https://ko.wikipedia.org/wiki/%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC)를 이해하자면 정리안된 책상의 상태를 생각하면 편할 것이다. 엔트로피가 높으면 책상이 굉장히 정리가 안된 상태(혼돈의 상태)고, 엔트로피가 적을 수록 점점 정리되어가는 책상을 생각하면 된다.

엔트로피의 계산 방식은 아래와 같다.
> 엔트로피: $H[Y] = -\sum_{k=1}^K p(y_k) \log_2 p(y_k)$
>
> 조건부 엔트로피: $H[Y \mid X] = - \sum_i \sum_j \,p(x_i, y_j) \log_2 p(y_j \mid x_i)$

* $y_k$: $k$ 카테고리에 속하는 $y$ 의 갯수
* $p(y_k)$: 변수 $y$ 가 카테고리 $k$ 에 속할 확률

본격적인 계산을 위해 엔트로피를 아래와 같은 데이터가 있다고 가정하고 단계별로 진행 해보자.

| f1 | f2 | f3 | y |
|:-:|:-:|:-:|:-:|
|0|1|1| yes|
|0|1|0| no|
|1|1|1| no|
|0|1|0| no|
|0|0|1| no|
|1|0|1| yes|

## 계산과정
### [1 단계] base entropy: 분류가 안되었을 때의 entropy
기초 엔트로피(base entropy), 즉 분류가 되기전의 상태를 계산해야 분류후에 엔트로피의 차이를 구할 수 있다.

기초 엔트로피를 $E_{base}$ 라고 하면,

|y=yes|y=no|total|
|:-:|:-:|:-:|
|2|4|6|

class의 히스토그램을 그리고 갯수를 세어본다. 지금의 class는 $(yes, no)$ 2개로 $K=2$ 가 되고, 히스토그램에 따라서 계산하면 기초 엔트로피를 구할 수 있다.
> $E_{base} = -[\ P(y_{=yes})\log{P(y_{=yes})} + P(y_{=no})\log{P(y_{=no})}\ ]\\
= -(\frac{2}{6}\log{\frac{2}{6}}+\frac{4}{6}\log{\frac{4}{6}}) = 0.9182$

### [2 단계] feature별로 조건부 엔트로피를 구하고 Infomation Gain구함
Information Gain 은 이전 단계 엔트로피에서 각 feature의 엔트로피를 빼면 구할 수 있다. 즉, 각 feature가 기준이 되어서 엔트로피를 제일 작게 만드는, 혹은 Information Gain을 제일 크게 만드는 쪽으로 분류를 진행하는 것이다.
#### feature 1  

|f1|y=yes|y=no|total|
|:-:|:-:|:-:|:-:|
|x=1|1|1|2|
|x=0|1|3|4|
|total|2|4|6|

>$E_1 = -[\ P(y_{=yes},x_{=1})\log{P(y_{=yes}|x_{=1})} + P(y_{=no},x_{=1})\log{P(y_{=no}|x_{=1})} +P(y_{=yes},x_{=0})\log{P(y_{=yes}|x_{=0})} + P(y_{=no},x_{=0})\log{P(y_{=no}|x_{=0})}\ ]$
>$= -[\ \frac{1}{6}\log{\frac{1}{2}}+\frac{1}{6}\log{\frac{1}{2}}+\frac{1}{6}\log{\frac{1}{4}}+\frac{3}{6}\log{\frac{3}{4}}\ ] = 0.8741$
>
>$IG_1 = E_{base} - E_1 = 0.0441$

마찬가지로 feature2 와 feature3도 똑같이 구할 수 있다.
#### feature 2

|f2|y=yes|y=no|total|
|:-:|:-:|:-:|:-:|
|x=1|1|3|4|
|x=0|1|1|2|
|total|2|4|6|

> $E_2 = -[\ \frac{1}{6}\log{\frac{1}{4}} + \frac{3}{6}\log{\frac{3}{4}} + \frac{1}{6}\log{\frac{1}{2}} + \frac{1}{6}\log{\frac{1}{2}}] = 0.8741$
>
> $IG_2 = E_{base} - E_2 = 0.0441$

#### feature 3

|f3|y=yes|y=no|total|
|:-:|:-:|:-:|:-:|
|x=1|2|0|2|
|x=0|2|2|4|
|total|4|2|6|

> $E_3 = -[\ \frac{2}{6}\log{\frac{2}{2}} + \frac{0}{6}\log{\frac{0}{2}} + \frac{2}{6}\log{\frac{2}{4}} + \frac{2}{6}\log{\frac{2}{4}}] = 0.6666$
>
>$IG_3 = E_{base} - E_3=0.2516$

### [3단계] 결과 및 선택:

| f|Entropy|IG|
|:-:|:-:|:-:|
|base |0.9182| -  |
|f1 |0.8742|0.0441|
|f2 |0.8742|0.0441|
|f3 |0.6667|0.2516|

결과에 따라 첫번째 기준으로 엔트로피가 가장 많이 줄고, IG가 가장 높은 feature3를 선택하게 된다.

따라서 feature3 기준으로 feature값이 1인경우 y=yes, 0인 경우 y=no로 나눠지게 된다.

| f1 | f2 | f3 | y |
|:-:|:-:|:-:|:-:|
|0|1|<span style="color: #7d7ee8">1</span>| <span style="color: #7d7ee8">yes</span>|
|0|1|<span style="color: #e87d7d">0</span>| <span style="color: #e87d7d">no</span>|
|1|1|1| no|
|0|1|<span style="color: #e87d7d">0</span>| <span style="color: #e87d7d">no</span>|
|0|0|1| no|
|1|0|<span style="color: #7d7ee8">1</span>| <span style="color: #7d7ee8">yes</span>|

남은 데이터는 아래와 같다.

| f1 | f2 | y |
|:-:|:-:|:-:|
|1|1|no|
|0|0|no|

이제 다시 위에 과정을 반복하게 된다.
> $E_{base2} = -(0 + 1 \cdot \log{1}) =0$

이 예제에서는 더이상 나눌 엔트로피가 없기 때문에 사실상 어떤 기준으로 선택해도 no가 나오지만 컴퓨터는 계산시 둘중 아무거나 기준으로 결과를 낼 것이다.

데이터를 없에지 않는 방법도 존재한다. feature3를 선택하고 남은 feature들 중에서 다시 선택하는 방법이다.

## 코드
모든 코드는 Github[<span style="color: #7d7ee8">링크</span>](https://github.com/simonjisu/ML/tree/master/DecisionTreeModel), DecisionTree.py에 공개되어 있다.
