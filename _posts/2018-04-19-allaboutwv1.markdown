---
layout: post
title: "All about Word Vectors: Intro"
date: "2018-04-19 16:41:36 +0900"
categories: nlp
author: "Soo"
comments: true
---
# All about Word Vectors: Intro

---

본 포스팅은 [CS224n](http://web.stanford.edu/class/cs224n/) Lecture 2 강의내용을 기반으로 강의 내용 이해를 돕고자 작성 됐습니다.

## 자연어 처리 (Natural Language Processing)
이야기를 하기 앞서서, "언어"를 살펴보자. [위키백과](https://ko.wikipedia.org/wiki/%EC%96%B8%EC%96%B4) 에 따르면 아래와 같다.

> 언어(言語)에 대한 정의는 여러가지 시도가 있었다. 아래는 그러한 예의 일부이다.
> * 사람들이 자신의 머리 속에 있는 생각을 다른 사람에게 나타내는 체계.
> * 사물, 행동, 생각, 그리고 상태를 나타내는 체계.
> * 사람들이 자신이 가지고 있는 생각을 다른 사람들에게 전달하는 데 사용하는 방법.
> * 사람들 사이에 공유되는 의미들의 체계.
> * 문법적으로 맞는 말의 집합(절대적이 아님).
> * 언어 공동체 내에서 이해될 수 있는 말의 집합.

위의 예시를 추려내보면 어떤 추상적인 내용을 사람들간의 공통된 약속으로 규정했다는 것이다. 기계한테 어떻게 언어를 처리하도록 알려줘야하나? **자연어 처리** 는 생각보다 오래된 역사를 가지고 있었다.

1950년도 이전 부터 자연어를 처리하려는 시도가 꽤 많았던 모양이다. 1954년 조지 타운 실험은 60 개 이상의 러시아어 문장을 영어로 완전 자동 번역하는 작업을 진행했다. 그는 3-5년 안으로 해결 가능하다고 주장했지만 1966 년 ALPAC 보고서에 따르면 실제로 진전이 엄청느려서 연구 자금이 크게 줄었다고 한다. 그리고 최초의 통계 기계 번역 시스템이 개발 된 1980 년대 말까지 기계 번역에 대한 연구는 거의 이루어지지 않았다고 한다. (지금은 너두나두 번역기 만들 수 있지만...)

또한, 1980년대까지 대부분의 자연어 처리 시스템은 손으로 쓴 복잡한 규칙 세트를 기반으로 했다. 그러나 점차 통계 기반의 자연어 처리 기법이 복잡한 자연어를 모델링 하는데 부상했다. (Reference: [NLP wikipedia](https://en.wikipedia.org/wiki/Natural-language_processing))

또한, 자연어 처리의 기본 가정을 항상 염두하고 공부해야 할 것이다. 좋은 소개글을 링크로 걸어 두었으니 참고하길 바란다.

참고: ratsgo 님의 블로그 - [idea of statistical semantics](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/10/frequency/)

<br>

## 단어의 표현(Word Representation)

어떻게 하면 단어의 "의미"를 표현할 수 있을까?

가장 간단한 방법은 단어를 종류별로 분류(taxonomy) 하는 것이다.

영어에는 유명한 **WordNet** 이라는 프로젝트가 있다. 이는 1985년부터 심리학 교수인 조지 A. 밀러가 지도하는 프린스턴 대학의 인지 과학 연구소에 의해 만들어졌고 유지되고 있다. 기본적으로 상위어(hypernyms) 밑에 동의어(synonym) 세트를 여러개 구성하는 것이다.

좋긴한데 몇 가지 단점이 있다.

첫째로, 단어간의 미묘한 차이, 뉘앙스(nuances)를 표현 할 수가 수 없다. 아래의 예를 보자.
```
from nltk.corpus import wordnet as wn
for synset in wn.synsets("adept"):
    print("({})".format(synset.pos()) + ", ".join([l.name() for l in synset.lemmas()]))
```
> (n) ace, adept, champion, sensation, maven, mavin, virtuoso, genius, hotshot, star, superstar, whiz, whizz, wizard, wiz
> (s) adept, expert, good, practiced, proficient, skillful, skilful

* "I'm good at deep learning" VS "I'm expert at deep learning" 이 두 문장은 확연히 다른 느낌의 문장이다. 잘하는 것과 전문가의 차이는 사람이 느끼기엔 다르다.

둘째로, 업데이트 비용이 많이 든다. 새로운 단어가 계속 나오면 업데이트 해줘야한다, 즉 구축비용이 쎄다는 것이다.

셋째로, 사람마다 주관적이기 때문에 명쾌한 기준이 없다.

마지막으로, 유사도 계산이 어렵다는 점이다. 즉, 같은 상위어에 속해 있는 하위어는 비슷한 것은 알겠는데, 정량적으로 이를 계산할 방법이 없다는 것이다.

### Bag of words representation

또다른 방법으로 discrete 된 심볼로 단어를 표현했는데 이를 **one-hot representation** 라고 하며, 아래와 같이 표현했다.

$$word = [0, 0, 0, 1, 0, 0, 0]$$

이러한 방법론을 **Bag of words representation** 이라 한다. 그러나 이는 두 가지 단점이 있다.

첫째로, 차원(Dimensionality)의 문제. 단어가 많아 질 수록 벡터가 엄청 길어진다.

둘째로, 제한적 표현(Localist representation)의 문제. 즉, 단어의 내적의미를 포함하지 않고, 각 단어들이 독립적이다. 예를 들면, "hotel" 과 "motel" 의 유사성을 계산하려고 하면, 0 이 나올 수 밖에 없다.

$$\begin{aligned}
motel &= \begin{bmatrix} 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix} \\
hotel &= \begin{bmatrix} 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \end{bmatrix} \\
\end{aligned}$$

$$hotel \cdot motel^T = 0$$

<br>

## 분포 유사성 기반 표현 (Distributional similarity based representations)

연구자들은 one-hot vector 와 다른 어떤 유사도를 계산할 수 있는 벡터를 만들고 싶어했다. 따라서 유사도의 정보를 어디서 얻을 수 있을까를 찾기 시작했다. 그리고 어떤 핵심 아이디어를 생각해냈다.

> 어떤 단어의 "의미"는 그 단어 근처에 자주 출현하는 단어로부터 얻을 수 있다.

<img src="/assets/ML/nlp/L2_context.png">

출처: [CS224n Lecture 2](http://web.stanford.edu/class/cs224n/syllabus.html)

그들은 주변 단어의 정보로 어떤 단어의 의미를 규정하는 시도를 하였고, 이는 modern statistical NLP 에서 많은 각광을 받기 시작했다. 그리고 어떤 단어 $w$ 에 대해서 주변에 나타나는 단어의 집합을 **맥락/문맥(context)** 이라고 했다.

### Word Vectors

이전에 0과 1로 채워진 one-hot vector 와 달리 문맥에서 비슷한 단어들을 잘 예측 될 수 있게 단어 타입 별로 촘촘한 벡터(dense vector)를 만든다. 핵심 아이디어는 아래와 같다.

> Idea:
> * We have a large corpus of text
> * Every word in a fixed vocabulary is represented by a vector
> * Go through each **position** $t$ in the text, which has a **center word** $c$ and **context ("outside") words** $o$
> * Use the similarity of the word vectors for $c$ and $o$ to calculate the probability of $o$ given $c$ (or vice versa)
> * Keep adjusting the word vectors to maximize this probability

출처: [CS224n Lecture 2](http://web.stanford.edu/class/cs224n/syllabus.html)

요약하면 방대한 텍스트 데이터를 기반으로, 중심단어 $c$ 가 주어졌을 때, 그 주변단어 $o$ 가 나올 확률 분포를 최대화 하는 것을 구하는 것이다.

* Word vectors 는 때때로 Word Embeddings, Word Representation 이라고 불린다.

이렇게 해서 나온 알고리즘이 <span style="color: #e87d7d">"Word2Vec"</span> 이며, 여기서 잠깐 끊고 다음 글에서 소개하도록 한다.
