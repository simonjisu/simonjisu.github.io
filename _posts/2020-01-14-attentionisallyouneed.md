---
layout: post
title: "[NLP] Attention Is All You Need - 1"
date: "2020-01-14 14:19:38 +0900"
categories: paper
author: "Soo"
comments: true
toc: true
---

Paper Link: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

# 1. Introduction

그 동안 LSTM([Long Short-term Memory](https://dl.acm.org/citation.cfm?id=1246450), 1997) 과 GRU([Gated Recurrent Unit](https://arxiv.org/abs/1412.3555), 2014) 등의 RNN 계열은 언어 모델링, 기계번역 등의 문제와 같이 시퀀스 모델링(sequence modeling)을 하기에 최고의 알고리즘이었다. 

{% include image.html id="1si3KMBjwZJ3inzTuoeUUsl7mutbDLNbz" desc="[그림 1] RNN의 forward propagation" width="auto" height="auto" %}

`그림 1` 처럼 이전 스텝의 은닉층 유닛인 $h_{t-1}$ 를 현재 스텝의 은닉층 유닛 $h_t$ 로 전달하면서 자연스럽게 시퀀스 데이터의 특징을 유지하지만, 아쉽게도 병렬 처리를 원천적으로 배제한다는 단점이 존재한다. 따라서 만약에 문장이 길어질 수록 훈련 속도가 현저하게 느려진다.

Input 과 Output 문장의 길이와 관계없이 의존성(dependencies)을 해결해주는 **Attention** 매커니즘은 시퀀스 모델링 혹은 변환 모델링<span style="color:gray">(transduction modeling: 각기 다른 특성을 가진 입력-출력 데이터를 변환하는 문제들, 예를 들어 기계번역)</span>에서 필수적인 요소가 됐다. 예시로 다음 논문들을 참고하면 좋다.

- [Neural Machine Translation by Jointly Learning to Align and Translate, Dzmitry Bahdanau 2014](https://arxiv.org/abs/1409.0473)
- [Structured Attention Networks, Yoon Kim, 2017](https://arxiv.org/abs/1702.00887)
- [A Decomposable Attention Model for Natural Language Inference, Ankur P. Parikh, 2016](https://arxiv.org/abs/1606.01933)

위 두 가지를 결합하여 저자들은 Attention 매커니즘만 활용하여 Input 과 Output 의 의존성을 글로벌하게 처리하고, 병렬화까지 가능한 `Transformer`라는 새로운 모델구조를 제안했다.

## 전체 모델구조

대부분의 신경망 시퀀스 변환 모델(transduction models)들은 대체로 Encoder 와 Decoder 로 구성된다. Encoder는 심볼로 표현된 입력 시퀀스(비연속적인 토큰들) $x$ 를 연속 공간(Continuous Space) $z$ 로 맵핑 후, $z$ 를 바탕으로 출력 시퀀스 심볼인 $y$ 를 얻는다. 출력 시퀀스는 이전 타임 스텝($t-1$) 시퀀스를 입력으로 다음 타임 스텝($t$)을 출력하는 자기회귀(auto-regressive) 성격을 가진다. 수식으로 다음과 같다.

$$\begin{aligned} \mathbf{x}&=(x_1, x_2, \cdots, x_n) \rightarrow \mathbf{z}=(z_1, z_2, \cdots, z_n)\\ \mathbf{y}&=(y_1, y_2, \cdots, y_m)\ \text{for}\  y_{t}=f(y_{t-1}, \mathbf{z}) \end{aligned}$$

{% include image.html id="15FPAUru5Rm1x3LUu6pcSjaZiuRrBkj97" desc="[그림 2] 모델구조: Encoder(좌), Decoder(우)" width="75%" height="auto" %}

하지만 **Transformer** 에서는 한 타임 스텝마다 $y$ 를 출력하지 않고 한번에 처리한다. 저자들이 제안한 전체적인 모델구조는 `그림 2` 와 같다(전체적인 느낌만 보고 다음으로 넘어가도록 한다).

## Encoder

Encoder는 각기 다른 N 개의 "Encoder Layer"라는 층으로 구성되며, 각 층에는 두 개의 서브층(SubLayer)이 존재한다. 첫번째는 Self Attention을 수행하는 "Multi-Head Attention", 두번째는 일반적인 "Position-wise Feed Forward"로 구성되며, 각 서브층은 Residual Network([Kaiming He, 2015](https://arxiv.org/abs/1512.03385))처럼 서브층의 입력과 출력을 결합하고, 그 결괏값을 다시 LayerNorm([Jimmy Lei Ba, 2016](https://arxiv.org/abs/1607.06450)) 을 통과시켜 출력을 얻는다. 수식으로 다음과 같다.

$$\text{LayerNorm}(x + \text{SubLayer}(x))$$

## Decoder

Decoder도 Encoder와 마찬가지로 각기 다른 N 개의 "Decoder Layer" 라는 층으로 구성된다. 다만, Encoder의 출력을 받아서 "Multi-Head Attention"을 수행하는 3번째 서브층이 추가된다. Self Attention을 수행하는 첫번째 "Multi-Head Attention"에서는 뒤에 있는 시퀀스정보로 부터 예측을 하지 않게 이를 가리게 됩니다. 따라서 $i$ 번째 토큰은 $i+1$ 번째 이후의 토큰을 참조하지 않게 됩니다. 나머지는 Encoder와 마찬가지로 잔차 연결(residual connection)을 수행하고 LayerNorm을 통과하게 된다.

이제부터 모델의 세부 사항을 살펴보면서 저자가 왜 이렇게 사용했는지, 의도가 무엇인지를 알아보려고 한다.

---

# 2. Scaled Dot-Product Attention

## Attention

Transformer 에서 Attention은 <span style="color:#e25252">**query(Q)**</span> 와 <span style="color:#5470cc">**key(K)**</span>-<span style="color:#cfb648">**value(V)**</span> 세트를 입력으로 집중된 어떤 벡터를 출력하는 함수로 표현할 수 있다. 출력은 <span style="color:#e25252">**Q**</span> 와 <span style="color:#5470cc">**K**</span> 간의 관계(Attention), 즉 <span style="color:#e25252">**Q**</span> 의 정보를 <span style="color:#5470cc">**K**</span> 에 대조 했을 때, 어느 부분을 집중해서 볼 것인지를 계산하고 해당 관계를 <span style="color:#cfb648">**V**</span> 와 결합하여 출력을 만든다. 수식으로 다음과 같다.

$$O = \text{Attention}(Q, K, V)$$

직관적으로 잘 안떠오르는데, 이게 어떤 느낌인지 알아보기위해 예를 들어보면 다음과 같다.

## 기계번역 문제:

영어를 한국어로 번역하는 문제를 예로 들자면, 영어는 소스 문장, 한국어는 타겟 문장이 된다. <span style="color:#e25252">**query(Q)**</span>, <span style="color:#5470cc">**key(K)**</span>, <span style="color:#cfb648">**value(V)**</span> 관계는 `그림 3` 과같이 표현할 수 있다.

{% include image.html id="14tFq4-RDEDFbc9vEABWqiFxG0pI4qq3G" desc="[그림 3] 기계번역 문제로 Q, K-V 의 관계 알아보기" width="auto" height="auto" %}

- <span style="color:#e25252">**query(Q)**</span>: 한국어 문장 정보
- <span style="color:#5470cc">**key(K)**</span>-<span style="color:#cfb648">**value(V)**</span> 세트: 인코딩된 영어 문장 정보, <span style="color:#5470cc">**key(K)**</span> 와 <span style="color:#cfb648">**value(V)**</span> 는 같은 벡터

<span style="color:#e25252">**Q**</span> 는 우리가 알고 싶어하는 문제라고 생각할 수 있다. 명칭도 "query=질의" 그대로 **"한국어로 변역하기 위해 영어 문장에서 집중적으로 봐야하는 단어는 어느 것인가?"** 라는 질문을 인코딩된 영어 문장 정보인 <span style="color:#5470cc">**K**</span>  한테 물어보게 된다. 그 방법은 이 다음에 소개하도록 하고, 그렇게 얻은 결과인 **A** 를 <span style="color:#cfb648">**V**</span> 와 곱하여 그 단어를 집중적으로 보게한다. 그렇게 Attention의 결과물인 <span style="color:#49aa71">**O**</span> 를 얻는다.

## 감성 분석 문제:

꼭 <span style="color:#e25252">**Q**</span>, <span style="color:#5470cc">**K**</span>-<span style="color:#cfb648">**V**</span> 가 다른 성격을 가진 시퀀스가 아니어도 된다. 세 토큰 모두 하나의 시퀀스를 가르킬 수도 있으며, 이를 Self-Attention 이라고 한다. 예를 들어 감성 분석(Sentiment Analysis) 문제를 예로 들면, 모델은 문장을 읽고 이를 사전에 정의해 놓은 감성 카테고리로 판단하게 되는 데, 이때 <span style="color:#e25252">**Q**</span>, <span style="color:#5470cc">**K**</span>, <span style="color:#cfb648">**V**</span> 모두 같은 문장을 지정하여 `그림 4`처럼 Attention 을 사용할 수 있다. 

{% include image.html id="1vFw0wuulHhzu5kwZLQ1QStl24KjnlsgX" desc="[그림 4] 감성 분류 문제를 통해 Self-Attention 에 대해 알아보기" width="auto" height="auto" %}

## Scaled Dot-Product Attention

Attention을 구하는 방법은 사실 다양하지만 Transformer 에서는 제일 기본적인 "Dot Product" 를 사용했으며, 그 수식은 다음과 같으며, 배치크기를 제외한 Q, K, V 의 크기를 표기해서 `그림 5` 와 같다. 

$$\text{Attention}(Q, K, V) = \text{softmax}(\dfrac{QK^T}{\sqrt{d_k}})V$$

{% include image.html id="1CtBsDHkyU8hmFj2MB0IDhEQO7wCKUEkM" desc="[그림 5] Q, K, V크기를 표기한 Scaled-Dot Product Attention" width="auto" height="auto" %}

여기서 주의할 점은 $T_k$ 과 $T_v$가 같다는 점이다. 기계 번역을 예로 들면 소스 문장이 <span style="color:#5470cc">**K**</span>-<span style="color:#cfb648">**V**</span> 세트이기 때문에 같은 길이의 내용을 담고 있지만 각 토큰이 표현하고 있는 차원만 다를 뿐이다. <span style="color:#e25252">**Q**</span> 와 <span style="color:#5470cc">**K**</span> 의 길이는 다를 수 있지만 차원 $d_k$ 로 같다. 두 행렬은 행렬의 곱(matrix multiplication)을 통해서 크기가 $(T_q, T_v)$ 인 점수 행렬 **A** 를 만들어 낸다. 

행렬 **A** 는 스케일링(Scaling), 마스킹(Masking) 후 Softmax 를 통해 확률값을 도출한다. 이 행렬의 뜻은 "문제를 해결하기 위해서 <span style="color:#e25252">**Q**</span> 의 토큰이 <span style="color:#5470cc">**K**</span> 의 어떤 토큰을 가장 많이 참고해야하는가?" 를 뜻한다. 따라서 확률이 높게 부여된 토큰은 <span style="color:#e25252">**Q**</span> 의 해당하는 토큰과 연관성이 높다고 할 수 있다. 물론 이 모든 연산은 학습이 가능하도록 DAG(Directed acyclic graph)로 연결되어 있기 때문에 학습 스텝이 진행됨에 따라 풀고자하는 문제에 최적화된 확률을 계속 도출해낸다<span style="color:gray">((Masking 은 차후에 다룬다)</span>.

스케일링 작업은 행렬 곱을 구한 **A** 를 $\sqrt{d_k}$ 로 나누는데, 그 이유는 다음과 같다. 차원의 크기인 $d_k$ 가 커질 수록 행렬의 곱의 수치는 점점 커지고 Softmax 수식에 의해서 그 확률 값 또한 커진다. 따라서 Softmax 의 경사(gradient) 값도 굉장히 작아지는데, 이를 막기위해서 $\frac{1}{\sqrt{d_k}}$ 값을 곱해줘야한다. 

**왜 $\sqrt{d_k}$ 를 나눌까?** 평균이 0, 표준편차가 1인 랜덤한 값으로 <span style="color:#e25252">**Q**</span> 와 <span style="color:#5470cc">**K**</span> 로 초기화시키고 확률로 표현된 행렬값 **A** 의 경사를 구해보면 $d_k$ 가 커짐에 따라서 평균이 0, 분산이 $d_k$ 를 따르는 분포가 된다. 이러한 시뮬레이션을 다음 코드를 통해 알아 볼 수 있다.

```python
import torch
import torch.nn as nn

def check_dotproduct_dist(d_k, sampling_size=1, seq_len=1, threshold=1e-10):
    """
    to check "https://arxiv.org/abs/1706.03762" Paper page 4, annotation 4
    -------------------------------
    To illustrate why the dot products get large, 
    assume that the components of q and k are independent random variables 
    with mean 0 and variance 1.
    Then their dot product has mean 0 and variance d_k
    
    print("*** notice that the gradient of softmax is y(1-y) ***")
    for d_k in [10, 100, 1000]:
        check_dotproduct_dist(d_k, sampling_size=100000, seq_len=5, threshold=1e-10)
    
    """

    def cal_grad(attn):
        y = torch.softmax(attn, dim=2)
        return y * (1-y)
    
    q = nn.init.normal_(torch.rand((sampling_size, seq_len, d_k)), mean=0, std=1)
    k = nn.init.normal_(torch.rand((sampling_size, seq_len, d_k)), mean=0, std=1)
    attn = torch.bmm(q, k.transpose(1, 2))
    print(f"size of vector d_k is {d_k}, sampling result, dot product distribution has\n")
    print(f" - mean: {attn.mean().item():.4f}, \n - var: {attn.var().item():.4f}")
    grad = cal_grad(attn)
    g_sum = grad.le(threshold).sum()
    g_percent = g_sum.item()/grad.view(-1).size(0)*100
    print(f"count of gradients that smaller than threshod({threshold}) is {g_sum}, {g_percent:.2f}%")
    
    attn2 = attn / torch.sqrt(torch.as_tensor(d_k).float())
    grad2 = cal_grad(attn2)
    g_sum2 = grad2.le(threshold).sum()
    g_percent2 = g_sum2.item()/grad2.view(-1).size(0)*100
    print(f"after divide by sqrt(d_k), count of gradients that smaller than threshod({threshold}) is {g_sum2}, {g_percent2:.2f}% \n")

print("*** notice that the gradient of softmax is y(1-y) ***")
for d_k in [10, 100, 1000]:
    check_dotproduct_dist(d_k, sampling_size=100000, seq_len=5, threshold=1e-10)
```

시뮬레이션 결과: 

```
*** notice that the gradient of softmax is y(1-y) ***
size of vector d_k is 10, sampling result, dot product distribution has

 - mean: -0.0004, 
 - var: 9.9979
count of gradients that smaller than threshod(1e-10) is 193, 0.01%
after divide by sqrt(d_k), count of gradients that smaller than threshod(1e-10) is 0, 0.00% 

size of vector d_k is 100, sampling result, dot product distribution has

 - mean: -0.0028, 
 - var: 99.9868
count of gradients that smaller than threshod(1e-10) is 402283, 16.09%
after divide by sqrt(d_k), count of gradients that smaller than threshod(1e-10) is 0, 0.00% 

size of vector d_k is 1000, sampling result, dot product distribution has

 - mean: 0.0029, 
 - var: 999.6312
count of gradients that smaller than threshod(1e-10) is 1737479, 69.50%
after divide by sqrt(d_k), count of gradients that smaller than threshod(1e-10) is 0, 0.00%
```

해당 모듈(Module) 코드는 [**Link**](https://github.com/simonjisu/annotated-transformer-kr/blob/9c1e4988e5aba3d2b971074590ce49e50c3aa823/transformer/modules.py#L8) 에서 확인할 수 있다.

---

다음편: [Attention Is All You Need - 2](https://simonjisu.github.io/paper/2020/02/02/attentionisallyouneed2.html)