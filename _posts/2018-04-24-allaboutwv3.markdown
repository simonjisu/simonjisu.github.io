---
layout: post
title: "All about Word Vectors: Negative Sampling"
date: "2018-04-24 16:14:13 +0900"
categories: "NLP"
author: "Soo"
comments: true
---
# All about Word Vectors: Negative Sampling

---
본 포스팅은 [CS224n](http://web.stanford.edu/class/cs224n/) Lecture 3 강의내용을 기반으로 강의 내용 이해를 돕고자 작성 됐습니다.

<img src="/assets/ML/nlp/L2_model_train.png">

## Navie Softmax 의 단점

Navie Softmax 를 최종단에 출력으로 두고 Backpropagation 할때는 큰 단점이 있다.

사실 Softmax가 그리 값싼 연산은 아니다. 우리가 학습하고 싶은 단어 벡터 1000개가 있다고 가정해보자. 그렇다면 매 window size=2 마다, 다시 말해 총 업데이트 할 5개의 단어 (중심단어 1 + 주변 단어 2 x 2) 를 위해서, $W, W'$ 안에 파라미터를 업데이트 해야하는데, 그 갯수가 최소 $(2 \times d \times 1000)$ 만큼된다.

$$\triangledown_\theta J_t(\theta) \in \Bbb{R}^{2dV}$$

많은 양의 단어에 비해 업데이트 하는 파라미터수는 적기 때문에 gradient matrix $\triangledown_\theta J_t(\theta)$ 가 굉장히 sparse 해질 수 있다 (0이 많다는 소리). Adam 같은 알고리즘은 sparse 한 matrix 에 취약하다.

[Numpy with NN: Optimizer 편 참고](https://simonjisu.github.io/deeplearning/2018/01/13/numpywithnn_5.html)

그래서 **"window에 실제로 등장하는 단어들만 업데이트 하면 좋지 않을까?"** 라는 생각을 하게 된다.

## Negative Sampling

> paper 1: [Distributed representaions of Words and Phrases and their Compositionality (Mikolov et al. 2013)](https://arxiv.org/abs/1310.4546)
>
> paper 2: [word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method](https://arxiv.org/abs/1402.3722)

요약하면 아래와 같은 목적함수를 최대화 하는 것이다.

$$\begin{aligned}
J(\theta) &= \dfrac{1}{T}\sum_{t=1}^{T} J_t(\theta)\\
J_t(\theta) &= \underbrace{\log \sigma(u_o^T v_c)}_{(1)} + \underbrace{\sum_{i=1}^{k} \mathbb{E}_{j \backsim P(w)} [\log \sigma(-u_j^T v_c)]}_{(2)}
\end{aligned}$$

* $T$: total num of words
* $\sigma$: sigmoid function
* $P(w) = {U(w)^{3/4}} / {Z}$: unigram distribution U(w) raised to the 3/4 power
    * The power makes less frequent words be sampled more often

말로 풀어보자면, 모든 단어 $T$ 에 대해서 중심단어 $c$ 와 그 주변단어 $o$ 가 같이 나올 확률 **[수식 (1)]** 을 최대화 하고, 그 주변단어가 아닌 집합에서 sampling 하여 나온 $k$ 개의 단어의 확률 **[수식 (2)]** 을 최소화 시키는 것이다. (음수가 붙기 때문에 최소하하게 되면 최대화가 된다.)

---

### 상세 논문 설명

논문 기준으로 위에 **<span style="color: #e87d7d">표기법</span>** 이 조금 다르다.
* 여기서 **$w$ = center word, $c$ = context** 다.

출발점은 아래와 같다.

> $(w, c)$ 세트가 정말로 corpus data로 부터 왔는가?

라고 생각하고 아래와 같은 **정의** 를 하게 된다.

* $P(D = 1 \vert w, c)$ : $(w, c)$ 가 corpus data로 부터 왔을 확률
* $P(D = 0 \vert w, c) = 1 - P(D = 1 \vert w, c)$ : $(w, c)$ 가 corpus data로부터 오지 않았을 확률

따라서, 우리의 목적은 확률 $P(D = 1\vert\ w, c)$ 를 최대화하는 parameter $\theta$를 찾는 것이기 때문에 아래와 같은 목적함수를 세울 수 있다.

$$\begin{aligned} &\arg \underset{\theta}{\max} \underset{(w,c) \in D}{\prod} P(D=1\vert\ w,c;\theta) \\
= &\arg \underset{\theta}{\max} \log \underset{(w,c) \in D}{\prod} P(D=1\vert\ w,c;\theta) \\
= &\arg \underset{\theta}{\max} \underset{(w,c) \in D}{\sum} \log P(D=1\vert\ w,c;\theta)
\end{aligned}$$

파라미터 $\theta$ 는 단어들의 벡터라고 생각할 수 있다. 즉, 위의 식을 만족하는 어떤 최적의 단어 벡터를 찾는것이다.

또한, 확률 $P(D=1\vert\ w,c;\theta)$ 은 sigmoid로 아래와 같이 정의 할 수 있다.

$$P(D=1\vert\ w,c;\theta) = \dfrac{1}{1+e^{-v_c v_w}}$$

따라서 우리의 목적함수는 아래와 같이 다시 고쳐 쓸수 있다.

$$\arg \underset{\theta}{\max} \underset{(w,c) \in D}{\sum} \log \dfrac{1}{1+e^{-v_c v_w} }$$

그러나 우리의 목적 함수는 매 $(w, c)$ 세트마다 $P(D=1\vert\ w,c;\theta)=1$ 를 만족하는 trivial solution이 존재한다. $v_c = v_w$ 이며, $\forall v_c,\ v_w$ 에 대해 $v_c \cdot v_w = K$ 를 만족하는 $\theta$ (보통 $K$ 가 40이 넘어가면 위 방정식의 값이 0에 가까워짐) 는 모든 값을 똑같이 0으로 만들어 버리기 때문에, 같은 값을 갖지 못하게 하는 매커니즘이 필요하다. ($\theta$ 에 뭘 넣어도 0이 되면 최대값을 찾는 의미가 없어진다, 자세한건 밑에 <span style="color: #e87d7d">참고 1</span> 를 참조) 여기서 "같은 값을 같는다" 라는 말은 단어 벡터가 같은 값을 갖는 것이다.

따라서, 하나의 방법으로 랜덤 $(w, c)$ 조합을 생성하는 집합 $D'$를 만들어 corpus data 로부터 올 확률 $P(D=1\vert \ w,c;\theta)$ 를 낮게 강제하는 것이다. 즉, $D'$ 에서 생성된 $(w, c)$ 조합은 **corpus data 로부터 오지 않게** 하는 확률 $P(D=0\vert\ w,c;\theta)$ 을 최대화 하는 것.

$$\begin{aligned}
& \arg \underset{\theta}{\max} \underset{(w,c) \in D}{\prod} P(D=1\vert\ w,c;\theta) \underset{(w,c) \in D'}{\prod} P(D=0\vert\ w,c;\theta) \\
&= \arg \underset{\theta}{\max} \underset{(w,c) \in D}{\prod} P(D=1\vert\ w,c;\theta) \underset{(w,c) \in D'}{\prod} \big(1- P(D=1\vert\ w,c;\theta) \big) \\
&= \arg \underset{\theta}{\max} \underset{(w,c) \in D}{\sum} \log P(D=1\vert\ w,c;\theta) + \underset{(w,c) \in D'}{\sum} \log \big(1- P(D=1\vert\ w,c;\theta) \big) \\
&= \arg \underset{\theta}{\max} \underset{(w,c) \in D}{\sum} \log \dfrac{1}{1+e^{-v_c v_w} } + \underset{(w,c) \in D'}{\sum} \log \big(1- \dfrac{1}{1+e^{-v_c v_w} } \big) \\
&= \arg \underset{\theta}{\max} \underset{(w,c) \in D}{\sum} \log \dfrac{1}{1+e^{-v_c v_w} } + \underset{(w,c) \in D'}{\sum} \log \dfrac{1}{1+e^{v_c v_w} }
\end{aligned}$$

$\sigma(x) = \dfrac{1}{1+e^{-x} }$ 시그모이드 함수로 정의 하면, 아래와 같이 정리 할 수 있다.

$$\begin{aligned}
& \arg \underset{\theta}{\max} \underset{(w,c) \in D}{\sum} \log \dfrac{1}{1+e^{-v_c v_w} } + \underset{(w,c) \in D'}{\sum} \log \dfrac{1}{1+e^{v_c v_w} } \\
&= \arg \underset{\theta}{\max} \underset{(w,c) \in D}{\sum} \log \sigma(v_c v_w) + \underset{(w,c) \in D'}{\sum} \log \sigma(- v_c v_w) \quad \cdots (3)
\end{aligned}$$

이는 <span style="color: #e87d7d">paper 1</span> 의 (4) 번 식과 같아지는다.

$$\log \sigma(u_c^T v_w) + \sum_{i=1}^{k} \mathbb{E}_{j \backsim P(w)} [\log \sigma(-u_j^T v_w)]$$

다른 점이라면, 우리가 만든 (3)식에서는 전체 corpus ($D \cup D'$) 을 포함하지만, Mikolov 논문의 식은 $D$ 에 속하는 $(w, c)$ 조합 하나와 $k$ 개의 다른 $(w, c_j)$ 의 조합을 들었다는 것이다. 구체적으로, $k$ 번의 negative sampling 에서 Mikolov 는 $D'$ 를 $k \times D$ 보다 크게 설정했고, k개의 샘플 $(w, c_1), (w, c_2), \cdots, (w, c_k)$ 에 대해서 $c_j$ 는 **unigram distribution** 에 **3/4** 승으로 부터 도출된다. 이는 아래의 분포에서 $(w, c)$ 조합을 추출 하는 것과 같다.

$$p_{words}(w) = \dfrac{p_{contexts} (c)^{3/4} }{Z}$$

* $p_{words}(w)$, $p_{contexts} (c)$ 는 각각 words and contexts 의 unigram distribution 이다.
* $Z$ 는 normalization constant

Unigram distribution 은 단어가 등장하는 비율에 비례하게 확률을 설정하는 분포다. 예를 들어 "I have a pen. I have an apple. I have a pineapple." 라는 문장이 있다면, 아래와 같은 분포를 만들 수 있다.

|I|have|a|pen|an|apple|pineapple|.|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|3/15|3/15|2/15|1/15|1/15|1/15|1/15|3/15|

여기서 3/4 승을 해주면, 가끔 등장하는 단어는 확률을 높혀주는 효과가 있다. 물론 자주 나오는 단어의 확률도 올라가지만 가끔 등장하는 단어의 상승폭 보다 적다.

| | a | $a^{\frac{3}{4} }$|
|-|-|-|
| apple |$\frac{1}{15}=0.067$   | ${\frac{1}{15} }^{\frac{3}{4} }=0.131$  |
| have  |$\frac{3}{15}=0.020$   | ${\frac{3}{15} }^{\frac{3}{4} }=0.299$   |

Mikolov 논문에서는 context는 하나의 단어이기 때문에 $p_{words}(w)$ 는 아래와 동일하다.

$$p_{words}(w) = p_{contexts} (c) = \dfrac{count(x)}{ \vert text \vert }$$

---

### 참고 1. Trivial Solution

$$\begin{aligned} L(\theta;w,c) &= \underset{(w,c) \in D}{\sum} \log \dfrac{1}{1+e^{-v_c v_w} } \\
&= \underset{(w,c) \in D}{\sum} \log(1) - \log(1+e^{-v_c v_w}) \\
&= \underset{(w,c) \in D}{\sum} - \log(1+e^{-v_c v_w})
\end{aligned}$$

같은 두 벡터의 내적을 하게 되면 값은 최대가 된다. $\cos$ 값이 1이 되기 때문이다. (여기서는 최대 값이 중요한건 아니지만 값이 커진다는데 의의가 있다.)
$$a\cdot a=\vert a \vert \vert a \vert \cos \theta $$

```
a = np.array([1,2,3,4,5,6,7])
b = np.array([.1,.2,.3,.4,.5,.6,.7])
print(np.dot(a, a))
print(np.dot(a, b))
```
> 140
>
> 14.0

즉, $v_c = v_w$ 이며, $\forall v_c,\ v_w$ 에 대해 $v_c \cdot v_w = K$ 를 만족하는 모든 값들이 $e^{-v_c v_w}$ 를 0으로 만든다면, $L(\theta; w, c)$ 값은 0이 될것이다. 이때 $v_c, v_w$ 가 무수히 많은 해가 존재하는데 이것을 **trivial solution** 이라고 한다.

---

<br>

다음 시간에는 말뭉치의 공기정보(co-occurance)를 고려해 단어를 벡터화 시킨 **GloVe** 에 대해 알아보자.
