---
layout: post
title: "Bidirectional LSTM + self Attention Model"
date: "2018-04-04 01:01:13 +0900"
categories: "DataScience"
author: "Soo"
comments: true
---
# Naver Sentiment Movie Corpus Classification

네이버 영화 감성분류 with Bidirectional LSTM + Self Attention

## 목표

* 영화 리뷰를 통해 긍정인지 부정인지 분류하는 문제 (Many-to-One)
* 사용한 모델: Bidirectional LSTM with Self Attention Model
* 이번 글은 논문과 제가 분석한 모델의 중요 요소를 곁들여 쓴 글입니다.
* GitHub Code Link: [<span style="color: #7d7ee8">nsmc_study</span>](https://github.com/simonjisu/nsmc_study)
* 그림이나 글은 퍼가셔도 좋지만, 출처 좀 남겨주세요~

Reference Paper: [<span style="color: #7d7ee8">A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING</span>](https://arxiv.org/pdf/1703.03130.pdf)

## 모델 핵심 부분 설명

그림과 수식을 함께 보면 이해하기 쉽다

<ul id="light-slider1">
  <li><img src="/assets/ML/nsmc/Self_Attention0.png"></li>
  <li><img src="/assets/ML/nsmc/Self_Attention1.png"></li>
  <li><img src="/assets/ML/nsmc/Self_Attention2.png"></li>
  <li><img src="/assets/ML/nsmc/Self_Attention3.png"></li>
  <li><img src="/assets/ML/nsmc/Self_Attention4.png"></li>
  <li><img src="/assets/ML/nsmc/Self_Attention5.png"></li>
  <li><img src="/assets/ML/nsmc/Self_Attention6.png"></li>
</ul>

어떤 $n$ 개의 토근으로 이루어진 하나의 문장이 있다고 생각해보자.

$$S = (w_1, w_2, \cdots, w_n)\qquad\qquad (1)$$

여기서 $w_i$ 는 one-hot 인코딩된 단어가 아닌, $d$ 차원에 임베딩된 문장에서 $i$ 번째 단어다.

따라서 $S$ 는 단어 벡터들을 concat 한 $n \times d$ 형태를 가지는 매트릭스다.

문장 $S$ 는 각기 다른 문장과는 독립적이다. (하나의 문장이 하나의 평점과 세트로 생각하면 된다.) 하나의 문장에서 단어들 간의 관계를 알기 위해서 우리는 bidirectional LSTM 으로 하나의 문장을 처리하게 된다.

$$\begin{aligned}
\overrightarrow{h_t} &= \overrightarrow{LSTM}(w_t, \overrightarrow{h_{t-1}})\qquad\qquad (2) \\
\overleftarrow{h_t} &= \overleftarrow{LSTM}(w_t, \overleftarrow{h_{t-1}})\qquad\qquad (3)
\end{aligned}$$

그후 우리는 각각의 $\overrightarrow{h_t}$ 과 $\overleftarrow{h_t}$ 를 concatenate 하여 하나의 히든 state $h_t$ 를 얻게 된다. 각 unidirectional LSTM(한 쪽 방향 LSTM)의 히든 유닛 크기를 $u$ 라고 하자. 조금 간단하게 표현하기 위해서 모든 $n$ 개의 $h_t$ 들을 $H$ 라고 하며, $n \times 2u$ 의 크기를 가진다.

$$H = (h_1, h_2, \cdots, h_n) \qquad\qquad (4) $$

우리의 목적은 길이가 변화하는 문장을 어떤 **고정된 크기** 의 임베딩으로 인코딩 하는 것이다. 이 목적을 달성하기 위해서 $H$ 와 attention 매커니즘이 요구되는 일종의 선형결합을 선택하게 된다. 즉, 아래와 같은 식과 $H$ 를 토대로, 어떤 벡터 $a$ 를 얻게 된다.

$$a = softmax(w_{s2} \tanh (W_{s1}H^T)) \qquad\qquad (5)$$

여기서 $W_{s1}$ 는 $d_a \times 2u$ 형태를 가진 매트릭스, 그리고 $w_{s2}$ 는 $d_a$ 사이즈를 가진 벡터다. $d_a$ 는 하이퍼파라미터(hyperparameter)로 우리가 정할 수 잇는 변수다. $H$ 의 크기도 $n \times u$ 이기 때문에, 벡터 $a$ 는 $n$ 의 크기를 가진다. 또한 $softmax()$ 함수는 모든 weight들의 합을 1로 만들어 준다.

그후 우리는 LSTM 의 히든상태들의 집합인 $H$ 를 주어진 $a$ 로 곱해서 한 문장을 임베딩한 벡터 $m$ 을 얻을 수 있다.

이 벡터 $m$ 은 학습시 한 문장에서 어떤 단어를 중심적으로 보았는지 알 수 있다. 예를 들어 어떤 연관된 단어나 구문 등등.

문장과 단어의 관계로 추가 설명하자면 아래와 같다.

각 단어를 input으로 받은 hidden 상태의 노드들은 단어를 통과해서 각 단어의 숨겨진 특성을 대표하고 있다. 학습 시 Task 에 따라 다르겠지만, 분류라고 가정한다면 분류에 도움이 되는 히든 상태는 높은 값을 가지게 될 것이며, 이를 어떤 선형 변환 과정을 거쳐 softmax 취한다는 것은 한 문장에서 분류에 도움이 된 근거 단어 혹은 중요 단어의 확률을 구한다는 것이 된다. (그래서 attention 이라고 하는 것 같다.) 따라서 이는 한 문장에서 **의미적인(semantic)** 부분을 나타내고 있다고 할 수 있다.

이 확률 $a$ 를 기존의 hidden 상태와 곱해서 의미부분을 조금더 강조하게 되는 벡터 $m$ 을 구했다고 보면 된다.

<ul id="light-slider2">
  <li><img src="/assets/ML/nsmc/Self_Attention7.png"></li>
  <li><img src="/assets/ML/nsmc/Self_Attention8.png"></li>
  <li><img src="/assets/ML/nsmc/Self_Attention9.png"></li>
</ul>

하지만 한 문장 내에서 중요한 부분 혹은 의미가 있는 부분은 여러군데 일 수가 있다. (여러 의미가 하나의 문장을 구성한다.) 특히 긴 문장일 수록 그렇다. 예를 들어 "아이언맨과 캡틴아메리카" 면 "과"로 이어진, "아이언맨", "캡틴아메리카" 두 단어는 중요한 의미가 있는 단어 일 수 있다. 따라서 한 문장에서 의미가 있는 부분을 나타내려면 $m$ 이란 벡터를 여러 번 수행해서 문장의 다른 부분까지 커버해야 한다. 이는 우리가 **attention** 을 **여러번(hops)** 하게 되는 이유다.

따라서, 문장에서 우리가 정하는 어떤 수 $r$ 번의 다른 부분을 추출 해낸다고 하면, 기존의 $w_{s2}$ 는 $r \times d_a$ 크기를 가진 $W_{s2}$ 라는 매트릭스로 확장된다. 이에따라 기존에 $a$ 벡터도 $r$ 번을 수행해 concatenate 한 $r \times n$ 크기의 매트릭스 $A$ 가 된다.  

$$A=softmax(W_{s2}tanh(W_{s1}H^T))  \qquad\qquad (6)$$

여기서 $softmax()$ 는 input $W_{s2}tanh(W_{s1}H^T)$ 의 2번째 차원을 기준으로 softmax 하게 된다. (즉, 각 row 별로 softmax 해줌)

사실 $(6)$ 번 수식은 bias 가 없는 2-Layers MLP 로 간주할 수도 있다.

위에 식에 따라 임베딩된 벡터 $m$ 도 $r \times 2u$ 크기의 매트릭스 $M$ 로 확장된다. 가중치를 담은 매트릭스 $A(r \times n)$ 와 LSTM 의 히든 상태들인 $H(n \times 2u)$를 곱해서 새로운 임베딩 매트릭스 $M$ 을 얻을 수 있다.

$$M=AH  \qquad\qquad (7)$$

마지막으로 $M$을 Fully Connected MLP 에 넣어서 하고 싶은 분류를 하면 된다.

### Penalization Term

임베딩된 매트릭스 $M$ 은 $r$ hops 동안 계속해서 같은 유사도 벡터 $a$ 를 곱하게 되면 **중복 문제(redundancy problems)** 가 생길 수 있다. 즉, 같은 단어 혹은 구문만 계속해서 attention 하게 되는 문제다.

<img src="/assets/ML/nsmc/Penalty.png">

* 그림: 왼쪽(a)은 패널티를 안준 것, 오른쪽(b) 는 준것

따라서, $r$ hops 동안 weight 벡터들의 합을 다양성을 높히는 일종의 패널티를 줘야한다.

제일 좋은 방법은 $r$ hops 안에 있는 아무 두 벡터 간의 **[<span style="color: #7d7ee8">쿨백-라이블러 발산 (Kullback–Leibler divergence)</span>](https://ko.wikipedia.org/wiki/%EC%BF%A8%EB%B0%B1-%EB%9D%BC%EC%9D%B4%EB%B8%94%EB%9F%AC_%EB%B0%9C%EC%82%B0)** 함수를 쓰는 것이다. 매트릭스 $A$ 의 각각의 행(row) 벡터들이 하나의 의미(semantic)를 가지는 단어 혹은 구문이 될 확률분포이기 때문에, 다양한 분포에서 나오는 것은 우리의 목적이 된다. (문장은 여러 단어/구문으로 구성되어 있기때문) 그러므로 KL divergence 값을 **최대** 로 만들면 중복 문제는 해결된다.

그러나 논문에서는 위와 같은 경우에 불안정(unstable) 한다는 것을 알아냈다. 논문 저자들은 어림짐작해 보았을 때, KL divergence 를 최대화 할때(보통의 경우 KLD를 최소화 하는 것을 한다.), 매트릭스 $A$ 구하는 단계에서 softmax 시 많은 값들이 0 이거나 아주 작은 값이라서 불안정한 학습을 야기했을 가능성이 있다는 것이다.

따라서, 논문에서는 매트릭스의 **[<span style="color: #7d7ee8">Frobenius norm</span>](http://mathworld.wolfram.com/FrobeniusNorm.html)** 을 쓰게 되는데 아래와 같다. ($Norm_2$와 비슷해 보이지만 다르다)

$$P ={ {\|AA^T - I\|}_F}^2$$

이 패널티 값과 기존의 Loss 와 같이 최소화 하는 방향으로 간다. 이 패널티의 뜻은 무엇일까?

두 개의 다른 유사도 벡터의 합 $a^{i}$ 과 $a^{j}$ 를 생각해보자. Softmax 로 인해서 모든 $a$ 값들의 합은 1이 될 것이다. 따라서 이들을 일종의 이산 확률분포 (discrete probability distribution)에서 나오는 확률질량 함수로 간주할 수 있다.

매트릭스 $AA^T$ 중, 모든 비대각 $a_{ij}\ (i \neq j)$ 원소에 대해서, 원소의 곱(elementary product)은 아래 두개의 분포를 가지고 있다.

$$0< a_{ij} = \sum_{k=1}^{n} a_k^i a_k^j <1$$

여기서 $a_k^i, a_k^j$ 는 각각 $a^i, a^j$ 의 k 번째 원소다. 제일 극단적인 경우를 생각해보면, $a^i$ 와 $a^j$ 가 일치하지 않다면 (혹은 다른 분포를 나타내고 있다면) 0 이 되고, 완전이 일치해서 같은 단어 혹은 구문을 이야기 하고 있다면 (혹은 같은 분포를 나타내고 있다면) 1 에서 최대값을 가지게 될 것이다.

따라서, $AA^T$ 의 대각 행렬(같은 단어 혹은 구문)을 대략 1 이 되게 강제한다. $I$ (Identity) 매트릭스를 빼줌으로써 달성하는데, 이는 자기 자신을 제외한 각기 다른 $a^i$ 간 원소들의 합인 $a_{ij}$ 들이 0 으로 최소화되게 만들어 버린다. 즉, 최대한 $a^i$ 간의 분포가 일치하지 않게 만드려고 노력하는 것이다. 이렇게 함으로써 $r$ 번의 hops 마다 각각 다른 단어에 집중하게 만드는 효과를 낼 수 있어서, 중복문제를 해결 할 수가 있다.

## 네이버 영화 리뷰 테스트 결과 및 시각화
총 150000 개의 Train Set과 50000 개의 Test Set 으로 진행했고, 모델에서는 hyperparameter가 많기 때문에 몇 가지 실험을 진행 했다.

간단한 실험을 위해서 사전에 단어들을 word2vec 으로 학습시키지 않고, mecab 으로 tokenizing 만해서 임베딩 시켰다. (실험을 안해봐서 사실 크게 상관있나 모르겠다. 나중에 여러가지로 실험해볼 예정)

내가 주로 건드린건 LSTM 에서의 **hidden layer의 갯수** 와 hops 인 **$r$** 을 바꾸어 보았다.

### model 1: 1 개의 Hidden Layer 와 5번의 hops

<img src="/assets/ML/nsmc/model_1.png">

### model 2: 1 개의 Hidden Layer 와 20번의 hops

<img src="/assets/ML/nsmc/model_2.png">

hops 가 많아지면 긍정/부정을 판단하게 되는 근거도 많아지고, 모델의 정확도도 향상되는 것을 2번에서 볼 수 있다.

### model 3: 3 개의 Hidden Layer 와 5번의 hops

<img src="/assets/ML/nsmc/model_3.png">

3번째 모델은 조금 이상하다고 느껴진 것이 있다. 그림을 보면 기계가 문장의 앞뒤만 보고 리뷰가 긍정인지 부정인지 판단했다는 것이다. 그림만 보면 과최적화된 느낌? 정확히 각 층의 layer 값을 보지는 못했지만, 층이 깊어 질 수록 기계가 이전 단계의 layer 에서 추출한 특징들로 학습해서 긍부정을 판단 했을 가능성이 있다. 점수는 높게 나왔으나 사람이 판단하기에는 부적절한 모델

## 향후 해볼 수 있는 과제들
* 전처리 단계에서 임베딩시 다양한 임베딩을 해볼 수 있을 것 같다. 예를 들어 word2vec으로 미리 선학습 후에 만든다던지, 아니면 N-hot 인코딩 (단어 원형 + 품사 + 어미) 등등 시도해볼 수 있는 것은 많다.
* LSTM Cell 로 구현
* 이와 연관은 좀 덜하지만, CNN으로 분류하는 것과 비교해 성능이 더 잘나올지? **김윤** 님의 논문 참고 : [<span style="color: #7d7ee8">링크 </span>](http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf)

---
공부에 도움 주신 분들 및 공부에 도움 되었던 싸이트:
* 김성동님: https://github.com/DSKSD
* 같은 논문을 Tensorflow로 구현하신 flrngel님: https://github.com/flrngel/Self-Attentive-tensorflow

감사합니다.
