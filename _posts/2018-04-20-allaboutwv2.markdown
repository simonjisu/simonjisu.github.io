---
layout: post
title: "All about Word Vectors: Word2Vec"
date: "2018-04-20 10:19:06 +0900"
categories: nlp
author: "Soo"
comments: true
toc: true
---

본 포스팅은 [CS224n](http://web.stanford.edu/class/cs224n/) Lecture 2 강의내용을 기반으로 강의 내용 이해를 돕고자 작성 됐습니다.

자연어 처리 공부를 해보신 분이라면 한번쯤 접한 그림이 있을 것이다.

<img src="/assets/ML/nlp/L2_linear-relationships.png">

> "king" - "man" + "woman" = ?

느낌상 "왕"에서 "남자"라는 속성을 빼주고, "여자"의 속성을 더해주면?

"queen" 이 나와야할 것 같다. Word Representation은 이런 것을 가능하게 했다.

이번 시간에는 **Word2vec** 에 대해서 알아보려고 한다.

# Word2Vec

Word2Vec은 두 가지 알고리즘이 있다.

> 1. Skip-grams(SG)
>     * target 단어를 기반으로 context 단어들을 예측한다. (position independent)
> 2. Continuous Bag of Words (CBOW)
>     * context 단어들 집합(bag-of-words context)으로부터 target 단어를 예측한다.

<ul id="light-slider1">
  <li><img src="/assets/ML/nlp/L2_skipgram1.png"></li>
  <li><img src="/assets/ML/nlp/L2_skipgram2.png"></li>
  <li><img src="/assets/ML/nlp/L2_cbow1.png"></li>
  <li><img src="/assets/ML/nlp/L2_cbow2.png"></li>
</ul>

그리고 몇 가지 효율적인 훈련 방법들이 있다.

> Two (moderately efficient) training methods (vs Naive Softmax)
> 1. Hierarchical softmax
> 2. Negative sampling

출처: [CS224n Lecture 2](http://web.stanford.edu/class/cs224n/syllabus.html)

이번 포스팅에서는 Skip-gram 과 Negative Sampling을 메인으로 소개하겠다.

---

# Skip-gram model with Naive Softmax
Paper: [Distributed Representations of Words and Phrases
and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf) (Mikolov et al. 2013)

<br>

## Embedding Look up

모델 설명에 들어가기 앞서 **Embedding Look up** 이란 것을 알아보자. 이 용어는 이제 여기저기서 많이 나올텐데 알아두면 좋다.

우리가 하고 싶은 것은 엄청나게 차원이 큰 one-hot vector 를 고정된 작은 차원으로 넣고 싶은 것이다. 어떻게 하면 단어들을 **2-dimension matrix** 로 표현 할 수 있을까?

아래 그림의 예를 보자. 8차원 one-hot vector를 3차원으로 만들고 싶다. 그렇다면 $3\times 8$ 행렬을 만들어서 각 column vector 가 하나의 3차원 단어를 표현하면 2-D Matrix 가 되지 않는가? 이 Matrix를 **Embedding Matrix** 라고 부르기로 하자

<ul id="light-slider2">
  <li><img src="/assets/ML/nlp/L2_embedlookup1.png"></li>
  <li><img src="/assets/ML/nlp/L2_embedlookup2.png"></li>
</ul>

그렇다면 어떻게 각 단어와 이 Embedding Matrix 를 매칭 시킬수 있을까? 여기서 **내적** 을 활용하게 된다.

<ul id="light-slider3">
  <li><img src="/assets/ML/nlp/L2_embedlookup3.png"></li>
  <li><img src="/assets/ML/nlp/L2_embedlookup4.png"></li>
  <li><img src="/assets/ML/nlp/L2_embedlookup5.png"></li>
</ul>

그런데 자세히 보니, one-hot vector의 숫자 $1$ 이 위치한 index 가 Embedding Matrix 의 column vector 의 index 와 같다. 따라서 중복되지 않는 단어사전을 만들고, 각 단어에 대해 index를 메긴 다음, 찾고 싶은 단어를 Embedding Matrix 에서 column vector index 만 **조회(Look up)** 하면 되는 것이다.

<ul id="light-slider4">
  <li><img src="/assets/ML/nlp/L2_embedlookup6.png"></li>
  <li><img src="/assets/ML/nlp/L2_embedlookup7.png"></li>
</ul>

**코드 예시:**
```
import numpy as np
sentence = "I am going to watch Avengers Infinity War".split()
embedding_matrix = np.array([[1,2,5,1,9,10,3,4], [5,1,4,1,8,1,2,5], [7,8,1,4,1,6,2,1]])
vocab = {w: i for i, w in enumerate(sentence)}
word = "I"
print(embedding_matrix)
print("="*30)
print("Word:", word)
print("Index:", vocab[word])
print("Vector:", embedding_matrix[:, vocab.get(word)])
```
> [[ 1  2  5  1  9 10  3  4]
>
>  [ 5  1  4  1  8  1  2  5]
>
>  [ 7  8  1  4  1  6  2  1]]
>
> ==============================
>
> Word: I
>
> Index: 0
>
> Vector: [1 5 7]

이해가 됐으면 이제 모델로 들어가보자.

<br>

<img src="/assets/ML/nlp/L2_model_train.png">

## 요약

Skip-gram 모델을 한 마디로 설명하자면, 문장의 모든 단어가 한번 씩 중심단어 $c$ 가 되어, $c$ 주변 문맥 단어 $o$ 가 나올 확률을 최대화 하는 것이다.

## 목적

각 중심단어 $c$ 에 대해서 아래의 **가능도/우도 (Likelihood)** 를 구해본다.

$$L(\theta) = \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} p(w_{t+j} | w_t; \theta) \quad \cdots\cdots \quad (1)$$

수식을 말로 풀어보자. 각 포지션 $(\prod_{t=1}^{T})$ 의 중심단어 $c$ = $w_t$ 에 대해서, $w_t$ 가 주어졌을 때 다른 문맥단어 $o$ = $w_{t+j}$ 가 나오는 확률 $\big( p (w_{t+j} \vert w_t; \theta) \big)$ 을 가능하게 만드는 $\theta$ 를 구하는 것이다. 단 $j$ 는 윈도우 크기 $m$ 을 넘지 않으며, $0$ 이 될 수 없다.

따라서 **Likelihood** 를 **최대화** 하는 것이 우리의 목적이 되겠다.

그러나 여기서는 우리가 좋아하는 Gradient Descent 를 사용하기 위해서 이 식을 **Negative Log Likelihood** 로 변형해서 쓰기로한다.

$$\min J(\theta) = -\dfrac{1}{T} \sum_{t=1}^T \sum_{-m \leq j \leq m,\ j \neq 0} \log p(w_{t+j} | w_t) \quad \cdots\cdots \quad (2)$$

* $(1)$ 식과 $(2)$ 식이 왜 동등한지는 밑에 **<span style="color: #e87d7d">참고 1</span>** 을 확인하길 바란다.

<br>

그렇다면 단어가 등장할 확률 $p(w_{t+j} \vert w_t)$ 는 어떻게 구할 것인가?

**Softmax** 라는 input 값을 0과 1 사이로 만들어 주는 친근한 함수가 있다.

$$p(o|c) = \dfrac{\exp(u_o^T V_c)}{\sum_{w=1}^V \exp(u_w^T V_c)} \quad \cdots\cdots \quad (3)$$

따라서 모델에 있는 모든 파라미터를 $\theta \in \Bbb{R}^{2dV}$ 로 두고, $(2)$ 식을 최적화 한다.

* 왜 $\theta \in \Bbb{R}^{2dV}$ 인가?
Center Word 의 Embedding Matrix $W$ Context Words 의 Embedding Matrix $W'$ 두개를 학습 시켜야하기 때문이다.
* **<span style="color: #e87d7d">주의 )</span>** $W'$ 는 $W$ 의 전치 행렬이 아니라 완전히 새로운 Embedding Matrix 다.

<br>

## Update

Gradient를 통해서 각 파라미터들을 업데이트 하게 된다. $(3)$ 식의 $\log$ 를 취하게 되면 아래와 같다.

$$f = \log \dfrac{\exp(u_o^T V_c)}{\sum_{w=1}^V \exp(u_w^T V_c)}$$

이제 $f$ 의 Gradient 를 구해보자.

$$\begin{aligned} \dfrac{\partial f}{\partial V_c}
&= \dfrac{\partial }{\partial V_c} \big(\log(\exp(u_o^T V_c)) - \log(\sum_{w=1}^V \exp(u_w^T V_c))\big) \\
&= u_o - \dfrac{1}{\sum_{w=1}^V \exp(u_w^T V_c)}(\sum_{x=1}^V \exp(u_x^T V_c) u_x ) \\
&= u_o - \sum_{x=1}^V \dfrac{\exp(u_x^T V_c)}{\sum_{w=1}^V \exp(u_w^T V_c)} u_x \\
&= u_o - \sum_{x=1}^V P(x | c) u_x
\end{aligned}$$

* $u_o$ : observed word, output context word
* $P(x\vert c)$: probs context word $x$ given center word $c$  
* $P(x\vert c)u_x$: Expectation of all the context words: likelihood occurance probs $\times$ context vector

흥미로운 점: **미분 값** 은 관측된 context word 벡터 $u_o$ 에서 center word $c$ 가 주어졌을 때 나올 수 있는 모든 단어의 기대치를 빼준 다는 것이다.

---

## 참고 1: Why MLE is equivalent to minimize NLL?

**Likelihood** 의 정의:

$$L(\theta|x_1,\cdots,x_n) = f(x_1, \cdots, x_n|\theta) = \prod_{i=1}^n f(x_i|\theta)$$

log를 취하게 되면 아래와 같다.

$$\log L(\theta|x_1,\cdots,x_n) =  \sum_{i=1}^n log f(x_i|\theta)$$

**MLE(maximum likelihood estimator)** 의 정의:

$$\hat{\theta}_{MLE} = \underset{\theta}{\arg \max} \sum_{i=1}^n \log f(x_i|\theta)$$

$$\underset{x}{\arg \max} (x) = \underset{x}{\arg \min}(-x)$$

때문에 우리는 아래의 식을 얻을 수 있다.

$$\hat{\theta}_{MLE} = \underset{\theta}{\arg \max} \sum_{i=1}^n \log f(x_i|\theta) = \underset{\theta}{\arg \min} -\sum_{i=1}^n \log f(x_i|\theta)$$

* 왜 log 로 바꾸는 것인가?
    1. 컴퓨터 연산시 곱하기 보다 더하기를 쓰면 **복잡도** 가 훨씬 줄어들어 계산이 빠르다. ($O(n) \rightarrow O(1)$)
    2. **언더플로우** 를 방지할수 있다. 언더플로우란 1보다 작은 수를 계속곱하면 0에 가까워져 컴퓨터에서 0 으로 표시되는 현상을 말한다.
    3. 자연로그함수는 **단조증가함수(monotonic increase function)** 라서 대소관계가 바뀌지 않는다. 예를 들자면, $5 < 10 \Longleftrightarrow log(5) < log(10)$ 의 관계가 바뀌지 않는 다는 것. 따라서 언제든지 지수를 취해서 다시 원래의 값으로 복귀 가능.

* 참고
  - [why minimize negative log likelihood](https://quantivity.wordpress.com/2011/05/23/why-minimize-negative-log-likelihood/)
  - [(ratsgo 님) 손실함수](https://ratsgo.github.io/deep%20learning/2017/09/24/loss/)

---

<br>

다음 시간에는 **Naive Softmax** 로 훈련 시켰을 때의 단점과 이를 보완 해준 **<span style="color: #e87d7d">Negative Sampling</span>** 에 대해서 알아보자.
