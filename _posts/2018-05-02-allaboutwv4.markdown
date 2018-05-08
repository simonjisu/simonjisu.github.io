---
layout: post
title: "All about Word Vectors: GloVe"
date: "2018-05-02 23:22:05 +0900"
categories: "NLP"
author: "Soo"
comments: true
---
# All about Word Vectors: GloVe

---
본 포스팅은 [CS224n](http://web.stanford.edu/class/cs224n/) Lecture 3 강의내용을 기반으로 강의 내용 이해를 돕고자 작성 됐습니다.

## Co-occurrence

공기(Co-occurrence) 란 무엇인가? 두 개 이상의 어휘가 일정한 범위(range) 혹은 거리(distance) 내에서 함께 출현하는 현상을 말한다. 여기서 어휘는 단어 뿐만 아니라 형태소, 합성어 등의 단위로 의미를 부여할 수 있는 언어 단위다. 그렇다면 왜 **공기 관계** 를 살피는 것일까?

공기 관계를 통해 문서나 문장으로 부터 **추상화된 정보** 를 얻기 위해서다. 이는 자연어처리의 가정을 생각해보면 이해할 수 있을 것이다.

> 비슷한 맥락에 등장하는 단어들은 유사한 의미를 지니는 경향이 있다.

때문에, 두 단어가 같이 등장한 횟수가 많아지면 **유사한 의미** 를 가졌다고 볼 수도 있다는 것이다. 이런 유사한 의미를 추상화된 정보로 볼 수 있다.

공기(Co-occurrence) 정보를 수집하는 방법은 두 가지다.

1. window 기반: 품사와 의미(semantic) 정보를 캡쳐할 수 있다.
2. word-document co-occurrence matrix 기반: 조금 더 일반적인 토픽을 추출 할 수 있고, 이는 Latent Semantic Analysis 와 연결 된다.

## Example: Window based co-occurrence matrix

> I like deep learning.
>
> I like NLP.
>
> I enjoy flying.

위 3 문장을 사용해서, window size = 1 로 지정하는 co-occurrence matrix 를 만들어보자. 무슨 뜻인지는 아래 코드를 실행한 표를 살펴보자.

단, 한 단어에 대해서 좌측에서 등장했는지 우측에서 등장했는 지는 상관없다(이는 co-occurrence matrix 가 대각을 기준으로 대칭하는 결과를 불러옴)

```
import pandas as pd
import numpy as np
from collections import deque, Counter
from itertools import islice
from scipy.sparse import coo_matrix

flatten = lambda t: [tuple(j) for i in t for j in i]
window = 1

def get_cooccur_list(sentence, window):
    s_len = len(sentence)
    ngram_list = [deque(islice(sentence, i), window+1) for i in range(s_len+1)][2:]
    return ngram_list

sentences = ['I like deep learning .', 'I like NLP .', 'I enjoy flying .']
tokens = [s.split() for s in sentences]
vocab = list(set([w for s in tokens for w in s]))
# print(vocab)
vocab = ['I', 'like', 'enjoy', 'deep', 'learning', 'NLP', 'flying', '.'] # 표와 같은 모양을 만들어주기 위해 다시 지정
vocab2idx = {w: i for i, w in enumerate(vocab)}
tokens_idx = [[vocab2idx.get(w) for w in s] for s in tokens]
co_occurs = [get_cooccur_list(s, window) for s in tokens_idx]

d = Counter()
d.update(flatten(co_occurs))
row, col, data = list(zip(*[[r, c, v] for (r, c), v in d.items()]))
temp = coo_matrix((data, (row, col)), shape=(len(vocab), len(vocab))).toarray()
co_mat = temp.T + temp

pd.DataFrame(co_mat, index=vocab, columns=vocab)
```

코드를 실행하면 아래와 같은 표가 나온다. window size 가 1이니까 "I" 주변 한칸에 동시 등장 단어는 "like" 가 2번 "enjoy" 가 1번이다.

|counts| I | like | enjoy | deep | learning | NLP | flying | . |
|--|--|--|--|--|--|--|--|--|
| I| 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
|like| 2 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
|enjoy| 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
|deep| 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
|learning| 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
|NLP| 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
|flying| 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
|.| 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 |

co-occurrence matrix와 같은 단어 벡터는 어떤 문제점이 있을까?

첫째로, 단어가 많아지면 벡터가 엄청 길어진다(데이터 차원이 커진)는 것이다. 이에 따른 많은 저장 비용이 들어갈 것이다. 둘째로, sparsity issues가 있을 수 있다(models are less robust).


> 그렇다면 꼭 하나의 단어로 해야만 하는가? 문서 전체의 단어의 공기 정보를 추출 하는 것은 안되는가?

이와 같은 생각이 GloVe 를 탄생시켰다.

## GloVe

Paper: [GloVe: Global Vectors for Word Representation](https://www.aclweb.org/anthology/D14-1162)

GloVe 의 학습방법은 아래와 같다.

$$J(\theta) = \dfrac{1}{2} \sum_{i,j=1}^{W} f(P_{ij})(u_i^T v_j - \log P_{ij})^2$$

논문해설을 통해서 자세히 보자.

---
### 논문 해설

**GloVe:** Global Vectors라고 명칭을 지은 이유는 모델에서 직접적으로 문서 전체의 코퍼스 통계량을 포착할 수있기 때문이다. (the global corpus statistics are captured directly by the model)

그전에 notation 을 정의해보자.

Define notation:
* $X$: 단어간의 공기 매트릭스 (matrix of word-word co-occurrence counts)
* $X_{ij}$: 단어 $j$ 와 문맥 단어 $i$ 가 같이 등장한 횟수 (the number of times that word $j$ occurs in the context word $i$)
* $X_i = \sum_k X_{ik}$: 어떤 단어든 문맥 단어 $i$ 와 등장한 횟수 (the number of times any word appears in the context of word $i$)
* $P_{ij} = P(j | i) = X_{ij} / X_i$: 단어 $j$ 와 문맥 단어 $i$ 동시 등장할 확률, 문맥 단어 $i$ 가 주어졌을 때 $j$ 가 등장할 확률 (probability that word $j$ appear in the context word $i$)

아래의 예시를 보자.

|Probability and Ratio|$k=solid$|$k=gas$|$k=water$|$k=fashion$|
|:-|:-:|:-:|:-:|:-:|
|$P(k\vert ice)$   | 0.00019  | 0.000066  | 0.003  | 0.000017  |
|$P(k\vert steam)$   | 0.000022  | 0.00078  |  0.0022 | 0.000018  |
|$P(k\vert ice)/P(k\vert steam)$   | 8.9  | 0.085  | 1.36  | 0.96  |

위의 표에 따르면 $i=ice, j=steam$ 일때 $solid$ 와 동시 등장 확률이 높은 단어는 $ice$ 다. 직관적으로 생각해도, 단단한 $ice$ 가 $solid$ 와 연관될 확률이 더 높다. 따러서, 우리는 $P(k\vert ice)/P(k\vert steam)$ 를 구해서, 연관이 있는 단어일 경우 이 비율이 크게 높으며, 아니면 그 반대다. **(엄청 크거나 혹은 엄청 작거나)**

이처럼 직접적으로 단어간의 동시등장 확률을 비교하는 것보다. 확률간의 비율을 구하는 것이 **연관성이 없는** 단어(water & fashion)들로 부터 관련된 단어(solid & gas)를 구별하기 좋으며, **관련성 있는** 단어(solid & gas)들을 차별화 하기에도 좋다.

따라서 **<span style="color: #e87d7d">동시 등장 확률의 비율(ratios of co-occurrence probabilities)</span>** 을 모델이 학습하게 하는 것이 바람직 해보인다.

중요한 것은 이 비율은 3개의 단어 $i, j, k$ 와 연관이 있다. 따라서 아래의 함수를 구성할 수가 있다.

$$F(w_i, w_j, \tilde{w}_k) = \dfrac{P_{ik} }{P_{jk} } \cdots (1)$$

* $w \in \Bbb{R}^d$: word vectors
* $\tilde{w} \in \Bbb{R}^d$: separate context word vectors

$(1)$ 식과 같이, 단어 벡터 공간에서 $w_i, w_j, \tilde{w}_k$ 를 input으로 넣었을 때, $\dfrac{P_{ik} }{P_{jk} }$ 비율을 나타내는 하는 선형구조인 함수를 구하는 것이 목적이다. 그리고 $F$ 를 아래와 같이 변형시켜 본다.

$$F((w_i - w_j)^T \tilde{w}_k) = \dfrac{P_{ik} }{P_{jk} } \cdots (2)$$

이로써 선형적인 관계를 포착하고, 양변 모두 스칼라 값으로 정해진 함수가 만들어 졌다. 그러나 단어 $i, j$ 와 $k$ 동시 등장 비율의 **임의적인 차별화** 를 위해서 어떤 조건들을 만족해야한다. 그 조건들이란 단어 벡터 $w$ 와 문맥 단어 벡터 $\tilde{w}$ 간 서로 자유롭게 교환 될 수가 있어야 한다. 즉, 단어간 공기 매트릭스 $X$의 대칭(symmetric) 특성을 보존해야 한다.

**임의적인 차별화** 가 무슨 말이냐면, 단어 $k$ 와 $i, j$ 단어 간의 비율을 확인 할때, $i$ 와 $k, j$ (혹은 $j$ 와 $i, k$) 의 관계도 확인 할 수 있어야 된다는 말이다.

대칭(symmetric) 을 만족하려면 2 단계로 진행 된다. 우선, 두 그룹 $(\Bbb{R}, +)$ 과 $(\Bbb{R}_{>0}, \times)$ 에 대해서 함수 $F$ 가 **homomorphism** 이어야 한다. (homomorphism 해설: 밑에 <span style="color: #e87d7d">참고 1</span>를 보라), 예를 들어 아래와 같다.

$$F(w_i, w_j, \tilde{w}_k) = \dfrac{F(w_i^T \tilde{w}_k) }{F(w_j^T \tilde{w}_k) } \cdots (3)$$

$(2)$ 식에 의해서, 아래와 같이 풀 수 있다.

$$F(w_i^T \tilde{w}_k) = P_{ik} = \dfrac{X_{ik} }{X_i} \cdots (4)$$

$(3)$ 식에 만족하는 해답은 $F = \exp$ 임으로, 아래의 식을 도출 해낼 수 있다.

$$w_i^T \tilde{w}_k = \log(P_{ik}) = \log(X_{ik}) - \log(X_i) \cdots (5)$$

다음으로, $(5)$ 식은 $\log(X_i)$ 만 아니였다면 대칭이었을 것이다. $\log(P_{ik})=\log(P_{ki})$ 를 만족하는지 한번 보자.

$$\begin{aligned}
\log(P_{ik}) &= \log(X_{ik}) - \log(X_i) \\
\log(P_{ki}) &= \log(X_{ki}) - \log(X_k)
\end{aligned}$$

당연하게도, $\log(X_i) \neq \log(X_k)$ 이기 때문에 $\log(P_{ik}) \neq \log(P_{ki})$ 이다.

하지만 $\log(X_i)$ 부분은 $k$ 에 대해서 독립적(independent) 이기 때문에, $w_i$ 의 bias $b_i$ 항으로 들어갈 수 있다. 그리고 대칭성을 유지하기 위해서 $\tilde{w}_k$ 의 bias $\tilde{b}_k$ 항도 더해준다.

$$w_i^T \tilde{w}_k + b_i + \tilde{b}_k = \log(X_{ik}) \cdots (6)$$

$(6)$ 식이 우리의 제일 간단한 선형관계인 solution 이라고 해도 되지만 이는 문제가 좀 있다. $X_{ik} = 0$ 에서 명확하게 정의 되지 않는다. 이를 해결하기 위해서 $X_{ik}+1$ 하는 방법도 있지만, sparsity issue 를 벗어나기 힘들다. 그리고 하나의 큰 약점이 있다면 거의 등장하지 않는 단어들에게 동시 등장 비율이 모두 같을 수 있다는 점이다. 이게 왜 문제가 되냐면, co-occurrence 가 적을 수록 많이 등장하는 단어들 보다 정보 함량이 적고 데이터도 noisy 하기 때문이다.

연구팀은 새로운 weighted least squares regression model 을 제시하여 문제를 풀고자 했다.

$$J(\theta) = \dfrac{1}{2} \sum_{i,j=1}^{W} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2 \cdots (7)$$

가중치 함수 (Weighting function) $f(X_{ij})$ 는 아래의 특성을 따라야 한다.

1. $f(0) = 0$. 만약 $f$ 가 연속함수(continuous function) 라면, $x \rightarrow 0$ 으로 갈때 $\lim_{x\rightarrow 0} f(x) log^2x$ 도 빠르게 수렴한다. 단, 유한한 값이여야 한다.
2. $f(x)$ 는 감소함수가 되면 안된다. (non-decreasing) 이유는 동시 등장이 희박한 단어들의 가중치가 많아져서는 안되기 때문이다.
3. $f(x)$ 는 큰 $x$ 값에 대해서 상대적으로 작은 값이어야 한다. 그 이유는 공기 횟수가 큰 단어들의 가중치가 너무 높게 설정하지 않기 위해서다.

연구팀은 이에 적합한 함수를 찾았다.

$$f(x) =
\begin{cases} (x/x_{max})^{\alpha} \quad if\ x < x_{max} \\
1 \quad otherwise
\end{cases}$$

<img src="/assets/ML/nlp/L3_weight_f.png">

그림: $f(x)$ with $\alpha = 3/4$ 일때 좋은 성과를 얻었다. 재밌는 것은 Mikolov 논문에서 나온 unigram distribution 에 3/4 승을 해주는 것과 같다는 것을 발견했다.

조금더 general 한 weighting function 은 아래와 같다. (자세한건 논문 3.1 참고)

$$\hat{J} = \sum_{i,j} f(X_{ij})(w_i^T \tilde{w}_j - \log X_{ij})^2$$


---
### 참고 1: Homomorphism

혹시나 틀렸으면 댓글로 이야기 해주세요.

우선 Group [(위키 링크)](https://en.wikipedia.org/wiki/Group_(mathematics) ) 이란 것을 알아야한다. 내가 이해한 바로는 **Group $(G, * )$** 이란, 집합 $G$ 와 연산 $* $ 로 구성되어 있다. 이 연산을 "the Group Law of $G$" 라고 부른다. 집합 $G$ 에 속한 원소 $a, b$ 의 연산을 $a * b$ 라고 표현한다. 또한 아래의 조건들을 만족해야한다.
1. Closure: $* $ 연산은 $G$ 에 대해 닫혀 있어야한다. 즉, $a * b$ 연산도 집합 $G$ 에 속해야한다.
2. Associativity: 교환 법칙이 성립해야한다. $(a * b) * c = a * (b * c)$
3. Identity element: 항등원이 존재해야 한다. $a * e = a = e * a$
4. Inverse element: 역원이 존재해야 한다. $a * x = e = x * a$

Group 를 이해 했으면 이제 Homomorphism 을 이해해보자.

**정의:**

두 그룹 <span style="color: #15b23c">$(G, * )$</span> 과 <span style="color: #9013b2">$(H, @)$</span> 가 있으면, 모든 <span style="color: #15b23c">$x, y \in G$</span> 에 대해서 $f:$ <span style="color: #15b23c">$G$</span> $\rightarrow$ <span style="color: #9013b2">$H$</span>, <span style="color: #15b23c">$f(x * y)$</span> $=$ <span style="color: #9013b2">$f(x) @ f(y)$</span> 를 만족하는 map 을 말한다.

**예시)**
두 그룹 <span style="color: #15b23c">$(\Bbb{R}, + )$</span> 와 <span style="color: #9013b2">$(\Bbb{R}_{>0}, \times )$</span> 사이에 어떤 map $f:$ <span style="color: #15b23c">$\Bbb{R}$</span> $\rightarrow$ <span style="color: #9013b2">$\Bbb{R}_{>0}$</span>, $f(x)=e^x$ 가 있다면, $f$ 가 Homomorphism 인지를 밝혀라.

> for any <span style="color: #15b23c">$x, y \in \Bbb{R}$</span>,
>
> <span style="color: #15b23c">$f(x + y)$</span> $=$ <span style="color: #9013b2">$e^{x+y}$</span> $=$ <span style="color: #9013b2">$e^x \times e^y$</span> $=$ <span style="color: #9013b2">$f(x) \times f(y)$</span> 임으로
>
> Homomorphism 을 만족한다.

---
word2vec 과 glove 관련 포스팅은 **"All about word vectors"** 시리즈로 마치겠다. 기회가 되면 gensim 의 사용법과, 데이터 차원 축소와 시각화 방법인 t-SNE 을 포스팅 하도록 하겠다.
