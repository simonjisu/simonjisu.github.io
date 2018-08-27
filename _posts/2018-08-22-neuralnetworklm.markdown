---
layout: post
title: "A Neural Probabilistic Language Model"
date: "2018-08-22 23:26:14 +0900"
categories: "NLP"
author: "Soo"
comments: true
---

# [PAPER] A Neural Probabilistic Language Model
---

클릭하면 링크를 따라갑니다.

**paper: [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) - Yoshua Bengio, 2003**

* [Slide Share](http://bit.ly/2OkYFkY)
* [Code Repo](http://bit.ly/2PsEPpg)
* [Notebook](https://nbviewer.jupyter.org/github/simonjisu/deepnlp_study/blob/master/notebook/01_NNLM.ipynb)

---

## 메인 아이디어

기존의 통계 기반의 Language Modeling 은 N-Gram 을 기반으로, 이전 토큰의 나오는 단어를 기반으로 다음 단어의 확률을 극대화 작업이었다. 처음부터 끝까지 보지 않고 N-Gram으로 잘라서 예측하게된 이유는 예측해야할 단어와 아주 오래된 단어간의 상관관계가 적다(혹은 분포가 다르다)라고 생각했기 때문이다.

$$P(w_t \vert w_1, w_2, \cdots, w_{t-1}) \approx P(w_t \vert t_{t-(n-1)}, \cdots t_{t-1})$$

확률을 정의 할때는 Counting 을 사용했다. 예를 들어, "New York" 뒤에 "University" 가 나올  확률을 예측한다고 해보자. 데이터에서 "New York University" 가 등장한 횟수를 세고, "New York" 뒤에 올 수있는 다른 모든 단어의 등장 횟수를 다 Count 한다.

$$P(University \vert New, York) = \dfrac{Count(New, York, University)}{\sum_w Count(New, York, w)}$$ 

해당 방법은 간단하나 문제점이 있다.

* 첫째, 훈련데이터에서 보지 못한 새로운 단어 조합이 등장하게 되면 확률이 0 이 된다. 
* 둘째, N-Gram 의 N 은 작은 수로 적을 수 밖에 없다. 1만개의 단어가 있으면 1-Gram 은 1만, 2-Gram 은 약 5천만 ($C_{10000}^{2}$), 3-Gram 은 약 1700억 ($C_{10000}^{3}$) 이 된다. 즉, N 이 커질수록 계산을 하기 위한 더 많은 컴퓨터 자원이 필요하다. (논문이 나온 2003년때 쯤에는 요즘같이 계산을 하기 위한 GPU 도 없었을 것이다.)

따라서 논문에는 위 두가지 문제점을 해결하기 위해 N 을 더 늘리고, 새로 등장한 단어에 대해서도 예측가능한 모델을 만들고자 했다.

Yoshua Bengio 교수님이 제안한 모델의 특징은 3 가지로 요약 할 수 있다.

> 1. 단어를 m 차원에 벡터와 연관 짓는다.
> 2. m 차원 벡터로 표현된 단어들의 조건부 확률을 표현한다.
> 3. 조건부 확률과 m 차원의 벡터를 동시에 학습한다.

아래에서 더 풀어서 설명한다.

---

## 모델 설명

<img src="https://dl.dropbox.com/s/8thdipjnc7bl95f/0826_nnlm.png">

두 가지 단계로 모델이 구성되 있다.

### 1 단계: Distributed feature vectors

> 각 단어를 $C$ 행렬을 통해 $m$ 차원 벡터로 표현한다. 

단어를 $m$ 차원 실수 벡터로 연관 지어야 한다. 최근에는 이 방법을 임베딩 (Embedding)이라고 하는데 논문에서는 분산된 특징 벡터 (Distributed feature vectors) 라고 했다.  

$$C(i) \in \Bbb{R}^m$$

$C$ 행렬의 $i$ 번째 행을, $i$ 번째 단어의 벡터라고 규정 지었으며, $C$ 의 형태는 $\vert V \vert \times m$ 다.

### 2 단계: Probability functions

> $m$ 차원으로 표현된 벡터를 2 층의 신경망을 사용해서 조건부 확률을 구성한다.

우선 임베딩된 $m$ 차원의 벡터들을 concatenate 하여 하나의 벡터로 만든다. 이를 context 라고 한다. 

$$x = \big( C(w_{t-n+1}), \cdots, C(w_{t-2}), C(w_{t-1}) \big)$$

그 후 2층의 신경망에 통과시켜 Softmax 로 최종적인 확률을 구한다.

$$y = U \tanh(d + Hx)$$

$$P(w_i = i \vert context) = \dfrac{\exp(y_{w_t})}{\sum_i \exp(y_i)}$$

### 기타: direct connection

논문에서는 실험적으로 선형적인 관계식을 하나 더 넣어서 context 와 y 사이의 선형관계를 알아내고자 했다.

$$y = \underbrace{b + Wx}_{\text{direct connection}} +U \tanh(d + Hx)$$

---

## 실험결과

### Perplexity

Test Measurement 로 **Perplexity** 를 선택했다. 정의는 아래와 같다.

> A measurement of how well a probability distribution or probability model (q) predicts a sample

$$PP = \exp(-\dfrac{1}{N} \sum_{i=1}^N \log_e q(x_i))$$

자세히 보면 지수안에 엔트로피 함수가 들어가 있는 것을 볼 수 있다. 

해당 수식을 말로 풀어보면, 모든 테스트 세트에서 확률 모델 $q$ 의 불확실 정도가 어떻게 되는지를 측정한다. 즉, 이 값이 높을 수록 모델이 예측을 잘 못하며, 낮을 수록 해당테스트 토큰을 확실하게 측정한다는 뜻이다.

### Time

시간을 측정한 이유는 학습할 파라미터 숫자가 생각보다 많기 때문이다.

$$\theta = (b, d, W, U, H, C)$$

총 학습해야할 파라미터 수는 $\vert V \vert(1+mn+h)+h(1+(n-1)m)$ 로 계산된다.

각 행렬의 크기를 아래에 표시해 두었다.

> $$\begin{aligned} b &= \vert V \vert \\
d &= h \\
U &= \vert V \vert \times h \\
W &= \vert V \vert \times (n-1)m \\
H &= h \times (n-1)m \\
C &= \vert V \vert \times m
\end{aligned}$$

### Result

<img src="https://dl.dropbox.com/s/c975f2j26kzj715/0826_nnlmresult.png">

영어 단어 Brown corpus (약 16000개의 단어)에 대해서, 은닉층 유닛수를 100, 임베딩 차원을 30, direct connection이 없고, 5-Gram 을 사용했을 때 결과는 Perplexity 가 제일 낮았다.

개인적인 실험으로 네이버 영화 corpus [[링크](https://github.com/e9t/nsmc)]를 사용하여 평균 perplexity 측정해보았다. [[노트북링크](https://nbviewer.jupyter.org/github/simonjisu/deepnlp_study/blob/master/notebook/01_NNLM.ipynb)]

한글 데이터를 사용해 최대한 단어갯수를 줄이려고 문장당 부호 및 단일 한글자음모음을 하나로 제약했다. 따라서 총 약 6만개의 단어가 사용됐다. 10 번의 epoch를 훈련 시킨 결과 Perplexity는 계속 떨어지지만 accuracy 는 20% 이후 상승이 멈췄다. 또한 훈련시간이 34 분 가량 걸렸다. 그만큼 학습할 파라미터가 많다는 뜻이다.

아래 문장으로 언어모델링을 해보았다.

> 요즘 나오는 어린이 영화보다 수준 낮은 시나리오 거기다 우리가 아는 윌스미스 보다 어린 윌스미스에 발연기는 보너스

| input | predict | target |
|:-:|:-:|:-:|
| 요즘/Noun 나오는/Verb 어린이/Noun 영화/Noun | 가/Josa |  보다/Josa |
| 나오는/Verb 어린이/Noun 영화/Noun 보다/Josa | 더/Noun |  수준/Noun |
| 어린이/Noun 영화/Noun 보다/Josa 수준/Noun | 이/Josa |  낮은/Adjective |
| 영화/Noun 보다/Josa 수준/Noun 낮은/Adjective | 영화/Noun |  시나리오/Noun |
| 보다/Josa 수준/Noun 낮은/Adjective 시나리오/Noun | 가/Josa |  거기/Noun |
| 수준/Noun 낮은/Adjective 시나리오/Noun 거기/Noun | 에/Josa |  다/Josa |
| 낮은/Adjective 시나리오/Noun 거기/Noun 다/Josa | ./Punctuation |  우리/Noun |
| 시나리오/Noun 거기/Noun 다/Josa 우리/Noun | 는/Josa |  가/Josa |
| 거기/Noun 다/Josa 우리/Noun 가/Josa | 뭐/Noun |  아는/Verb |
| 다/Josa 우리/Noun 가/Josa 아는/Verb | 사람/Noun |  윌스미스/Noun |
| 우리/Noun 가/Josa 아는/Verb 윌스미스/Noun | 인데/Josa |  보다/Verb |
| 가/Josa 아는/Verb 윌스미스/Noun 보다/Verb | 가/Eomi |  어린/Verb |
| 아는/Verb 윌스미스/Noun 보다/Verb 어린/Verb | 애/Noun |  윌스미스/Noun |
| 윌스미스/Noun 보다/Verb 어린/Verb 윌스미스/Noun | 가/Josa |  에/Josa |
| 보다/Verb 어린/Verb 윌스미스/Noun 에/Josa | 대한/Noun |  발연기/Noun |
| 어린/Verb 윌스미스/Noun 에/Josa 발연기/Noun | 에/Josa |  는/Josa |
| 윌스미스/Noun 에/Josa 발연기/Noun 는/Josa | 좋/Adjective |  보너스/Noun |

결과를 살펴보면 문맥은 상당히 못맞추었으나 문장의 구조는 잘 학습한 것을 알 수 있다.