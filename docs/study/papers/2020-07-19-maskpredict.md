---
title: "Mask-Predict: Parallel Decoding of Conditional Masked Language Models"
hide:
  - tags
tags:
  - NLP
  - Decoding
  - Masked Language Models
---

# Mask-Predict: Parallel Decoding of Conditional Masked Language Models

Paper Link: [Mask-Predict: Parallel Decoding of Conditional Masked Language Models](https://arxiv.org/abs/1904.09324)

모두의 연구소에서 진행하는 "beyondBERT" 프로그램에서 참여하다가 본 논문을 정리해보려고 한다. 흥미롭게 생각했던 논문이라 중요 부분만 일단 정리했다.

기존의 기계번역등 작업을 진행할때 Seq2Seq 모델(with Attention)을 사용할 경우 보통 autoregressive하게 토큰을 하나씩 디코딩했다. 예를 들어, "내가 아이언맨이다."라는 문장을 번역하려면, Encoder에 다음과 같이 `source` 토큰들을 넣어주고, Decoder는 문장의 시작을 알리는 `<SOS>` 토큰으로 시작하여 `"I"`를 예측하고, 예측한 `"I"`로 `"am"`을 예측하고, 마지막에 `"<EOS>"` 토큰이 등장하면 끝나는 구조다. sudo-code로 다음과 같이 작성 할 수 있겠다.

```python
source = ["나는" "아이언맨" "이다", "."]
target = ["<SOS>", "I", "am", "IronMan", ".", "<EOS>"]

hiddens = Encoder( source )
token = to_tensor( "<EOS>" )

while token == "<EOS>":
    predict = Decoder( token, hiddens )
    token = to_token( predict )
```

이 논문에서는 non-autoregressive하게 decoding하는 방법을 제시했는데, 구체적인 방법은 `그림1`을 보면 단번에 이해가 되리라고 믿는다. 

> * 일부 스터디에서 나온 의견 및 개인 의견이 섞여서 들어가 있음을 밝힌다.
> * Model Distillation 부분은 비교군이 적절하지 않다고 스터디에서 나온 의견이 있어서 다루지 않았다.

---

# 1. Introduction

생략

# 2. CMLM(Conditional Masked Language Models)

* CMLM은 입력 토큰 $X$와 일부 타겟 토큰 $Y_{obs}$가 주어지면 마스크가 된 타겟 토큰 $Y_{mask}$를 맞추는 문제다.
* 강한 가정: 마스크된 타겟 토큰들 $Y_{mask}$은 입력데이터에 대해서 조건부 독립이다.

    $$\text{Predict: } P(y \vert X, Y_{obs}) \ \forall y \in Y_{mask}$$

    이를 분해해보면 다음과 같다.

    $$\begin{aligned} P(Y_{mask} \vert X, Y_{obs}) &= P(Y_{mask}^{K};Y_{mask}^{1:(K-1)} \vert X, Y_{obs}) P(Y_{mask}^{1:(K-1)} \vert X, Y_{obs}) \\ &= P(Y_{mask}^{K};Y_{mask}^{1:(K-1)} \vert X, Y_{obs}) \cdots P(Y_{mask}^{2};Y_{mask}^{1} \vert X, Y_{obs}) P(Y_{mask}^{1} \vert X, Y_{obs}) \end{aligned} \\ \text{if } Y_{mask} \text{ is conditionally independent by each other } \rightarrow \\ \begin{aligned} P(Y_{mask} \vert X, Y_{obs}) &\approx P(Y_{mask}^{K} \vert X, Y_{obs}) P(Y_{mask}^{K-1} \vert X, Y_{obs}) \cdots P(Y_{mask}^{2} \vert X, Y_{obs}) P(Y_{mask}^{1} \vert X, Y_{obs}) \end{aligned}$$

    (beyondBERT 에서 나온 리뷰: 이러한 최종 예측파트에서는 가정이 맞지만, 훈련시킬때는 아닐 것이다)

* 추가로 마스크 개수는 정해져 있기 때문에 토큰 길이에 대한 제약도 명시적으로 달려있는 셈이다.

## 2.1 Architecture

* 클래식한 Transformer에 Decoder만 Masked-self attention을 제거하기로함
* fair-style Transformer

## 2.2 Training Objective

* $1$~$N$(토큰길이) 만큼의 uniform distribution에서 랜덤하게 숫자를 고른다음에 그 개수만큼 $Y_{mask}$를 선택
* Cross-entropy Loss로 최적화, parallel하게 할 수 있는 이유는 이전의 $Y_{mask}$에 취한 conditionally independent 가정 때문이다.

## 2.3 Predicting Target Sequence Length

* 전통적인 left-to-right 기계번역의 경우, 이전 예측 토큰이 다음 예측 토큰으로 들어가게 된다. 그리고 최종적으로 `EOS`이 나오면 종료가 되는 형태라서 자동적으로 문장의 길이를 알 수 있었지만 , CMLMs에서는 전체 시퀀스를 parallel하게 예측하기 때문에 타겟 문장 전체의 길이를 예측해야한다.
* 논문에서는 BERT 의 `CLS` 토큰처럼, `LENGTH` 토큰을 Encoder에 집어넣기로 한다. 해당 토큰의 loss도 마지막에 추가한다.

---

# 3. Decoding with Mask-Predict

* 요약하면 각 iteration마다 알고리즘은 토큰의 부분집합을 선택하여 masking하고, CMLM으로 예측한다.

## 3.1 Formal Description

* 타겟 시퀀스 $(y_1, \cdots, y_N)$ 와 각 토큰의 확률 $(p_1, \cdots, p_N)$이라는 두 변수가 있고, 미러 정의된 $T$번 동안 알고리즘을 돌린다(이는 상수거나 $N$에 관련된 간단한 함수로 결정된다).
* 각 iteration마다 `mask` 작업을 수행하고, 예측(`predict`)한다.

### **Mask**

* 첫 iteration에는 모든 토큰을 마스킹한다. 그 이후부터는 가장 낮은 확률을 가진 $n$개의 토큰을 masking한다.

    $$\begin{aligned} Y_{mask}^{(t)} &= \arg \underset{i}{\min} (p_i, n) \\ Y_{obs}^{(t)} &= Y \setminus Y_{mask}^{(t)}\end{aligned}$$

* $n$은 $t$의 함수이며 논문에서는 $n=N \cdot \dfrac{T-t}{T}$를 사용했다($T$는 iteration 횟수).

### **Predict**

* Masking후, CMLM은 주어진 입력$X$와 masking 안된 $Y_{obs}^{(t)}$를 기반으로 $Y_{mask}^{(t)}$를 예측하는데, 각 마스킹된 토큰 $y_i \in Y_{mask}^{(t)}$에 대해서 확률이 가장 높은 것을 예측값으로 선택한다.

    $$\begin{aligned} y_i^{(t)} &= \arg \underset{w}{\max} P(y_i = w \vert X, Y_{obs}^{(t)} ) \\ p_i^{(t)} &= \underset{w}{\max} P(y_i = w \vert X, Y_{obs}^{(t)} ) \end{aligned}$$

* 마스크가 안된 token들은 이전 스텝의 값을 그대로 따라간다.

    $$\begin{aligned} y_i^{(t)} &=y_i^{(t-1)} \\ p_i^{(t)} &= p_i^{(t-1)} \end{aligned}$$

* 특정 토큰의 확률이 계속 희박하여 이러한 휴리스틱한 작업에도 불구하고 잘 작동했다.

## 3.2 Example

* 그림으로 보면 조금더 이해가 쉬운데, 차후 3.3에서 이야기하는 Length predict 이후의 예시를 들은 것이다.
* 그림에서 나오는 용어들이 있다.
    * 각 $t$ 스텝 마다 `Mask > Predict` 의 과정을 반복한다.
    * $t$: 현재 스텝
    * $n$: masking 해야할 토큰의 수
    * $probability$(보라색): 각 예측의 확률을 담는 container

{% include image.html id="12HUzuQzCWwkaO4B6H07EOkEIJlKpjMnv" desc="[그림1] Example of parallel decoding" width="100%" height="auto" %}

**장점** 

* 마스킹되지 않았던 것들도 차후에 확률이 다른 토큰에 비해 상대적으로 낮아지면 다시 마스킹될 수도 있다. 즉, 초기에 잘못 예측했더라도, iteration을 통해 점차 바른 예측으로 고쳐질 수도 있다는 것

**문제점: Multi-modality Problem**

{% include image.html id="1VSrcpclqKZ5KxqJF7yKAw-AP8u-hIuFQ" desc="[그림2] Paper Figure 1" width="100%" height="auto" %}

* 논문의 Figure 1 처럼, t=0 인 상황에서 중복된 단어가 생성 될 수가 있음("completed") 이는 non-autoregressive 모델에서 자주 등장하는 문제다. 이는 `5.1`에서 자세히 다룬다.

## 3.3 Deciding Target Sequence Length

* 타켓 문장의 길이인 `LENGTH` 토큰을 예측하기 때문에 배치 연산을 할 수 있다.
* 확률이 가장 높은 길이를 여러개 뽑아서 배치 연산으로 3.2의 과정을 할 수 있다.

{% include image.html id="1WPZ4xsitujEWF9yfeacc6V587DmjGZ2x" desc="[그림3] Length Predict" width="100%" height="auto" %}

* 차후에 제일 높은 평균 로그 확률로 길이를 선택하게 된다(beam search 와 연관)

$$\dfrac{1}{N} \sum_i \log p_i^{(T)}$$

---

# Experiments

## 4.1 Experimental Setup

### Translation Benchmarks

* 총 3개의 데이터 세트를 사용: WMT'14 EN-DE (4.5M sentence pairs), WMT'16 EN-RO (610k pairs), WMT'17 EN-ZH (20M pairs)
* 모든 데이터는 BPE로 인코딩했으며, 퍼포먼스는 BELU score를 계산했다.
* EN-ZH 만 ScareBLEU를 사용했다.

### Hyperparameters

* Attention is All you Need 논문과 똑같이 각 stack마다 6개의 layer, 각 layer마다 8개의 attention heads, 모델 $h_{model}, h_{ffn}$ hidden size는 각 512, 2048로 진행했다.
* 가중치 초기화는 BERT 논문에서 진행한 $\mathcal{N}(0, 0.02)$, bias는 0으로 초기화 했다.
* LayerNorm은 $\beta=0, \gamma=1$
* Regularization은 $\text{dropout}=0.3, \text{weight decay}=0.01$ 로 실험했다.
* Smoothed CV Loss $\varepsilon=0.1$ 
* 훈련은 Adam에 $\beta=(0.9, 0.999), \varepsilon=10^{-6}$으로 진행, warm up 은 $10000$ 스텝에 $5\cdot 10^{-4}$까지 피크로 가다가 역제곱근의 형태로 내려간다.
* 훈련 스텝은 300k 각 epoch 마다 validation 진행하고, 가장 좋은 5개의 checkpoint를 평균내서 최종모델을 만든다.
* Decoding을 비교하기 위해서 autoregressive 모델에서 beam search($b=5$), 논문의 모델은 $l=5$개의 후보를 사용해서 decoding했다.

## 4.2 Translation Quality

{% include image.html id="1sdffQlPXdG9Fgs_O_xgE1xfn-kxIPKmL" desc="[그림4] Paper Table 1 & 2" width="100%" height="auto" %}

* 같은 non-autoregressive 방법들 중에서 논문의 모델이 가장 높은 BLEU score를 달성했다고 주장하고 있다.
* 다른 non-autoregressive 방법들을 확인 해봐야 더 자세히 알것 같다.
    * NAT w/ Fertility ([Gu et al., 2018](https://arxiv.org/abs/1802.06901))
    * CTC Loss ([Libovicky et al., 2018](https://arxiv.org/abs/1811.04719))
    * Iterative Refinement([Lee et al., 2018](https://www.aclweb.org/anthology/D18-1149/))

## 4.3 Decoding Speed

{% include image.html id="1pqAa4SEq-SNleVeUp0IUkcTrzQFqpjLr" desc="[그림5] Paper Figure 2" width="100%" height="auto" %}

* 파란점은 논문저자들의 실험 결과며, `L2R b=1`는 beam search(b=1)를 사용한 Left-to-Right(autoregressive) 모델이다.
* Decoding 스피드와 퍼포먼스간의 trade-off 를 이야기하면, $T=4, l=2$인 경우 2 point의 퍼포먼스를 대가로 `L2R b=5`모델 보다 3배의 스피드를 끌어 올릴 수 있다고 주장한다.
* beyondBERT에서 나온 리뷰중에 하나가 2 point BELU score 면 엄청나게 큰 점수라고 한다(quality 가 상당히 떨어질 수도?!).

---

# 5. Analysis

## 5.1 Why Are Multiple Iterataions Necessary?

* Various non-autoregressive 모델에서는 각 예측 토큰들이 서로 조건부 독립이라는 큰 가정이 들어간다. 때문에 예측할때 서로 다른 토큰에 영향을 받지 않아서 다른 위치라도 높은 확률로 같은 토큰을 반복적으로 예측하는 문제가 생긴다.
* 이러한 문제를 Multi-modality 문제라고 [Gu et al., 2018](https://arxiv.org/abs/1711.02281)의 논문에서 이야기 한적이 있다.
* 저자들은 예측한 토큰을 모델의 입력으로 사용하여, 반복적인 masking-predict 수행을 통해(multi-modal distribution을 uni-modal distribution으로 전환) 이 문제를 완화시키려고 했다.

{% include image.html id="1G5DV3t-J1dp6-nOdx9Nh1Vwo-XXy6xpx" desc="[그림6] Paper Table 3" width="75%" height="auto" %}

* 가설을 검증하기 위해서, Proxy Metric으로 중복 된 예측 토큰의 개수가 몇 퍼센트를 차지하는지 살펴보았다. 확실히 $T$가 높아질 수록 해당 비율은 현저하게 줄어든다.
* $T$가 작을 수록(중복된 토큰 예측이 많아질 수록) BLEU score가 현저하게 낮아지는 것도 이해가 된다.

## 5.2 Do Longer Sequence Need More Iterations?

{% include image.html id="1qf97x-hNgWXtl8NbHJ-z7iUG0KZ-uNg2" desc="[그림7] Paper Table 4" width="75%" height="auto" %}

* 긴 문장일 수록 더 많은 iteration이 도움이 되긴했다. 그러나 $T$가 많아질 수록 연산비용이 많아지는 것을 고려해야 할 것이다.

## 5.3 Do More Length Candidates Help?

{% include image.html id="127K3YP4LlTrPHEiu9XO5v7U_rlrByMlA" desc="[그림8] Paper Table 5" width="75%" height="auto" %}

* 적당한 길이 후보($\mathcal{l}$)는 번역에 도움이 되지만 너무 많은 후보를 두면 도움이 안된다.
* 상식적으로 후보들이 비슷한 길이를 가진다면 예측에 도움이 되겠지만, 많은 후보들 중에 비슷하지 않은 길이들이 있다면, 번역의 품질이 떨어질 수밖에 없을 것 같다.

---

# 개인적 리뷰 및 결론

* non-autoregressive 모델의 decoding은 실무에서 빠르게 decoding 할 수있기 때문에 앞으로 연구할 가치가 있는 분야인것 같다.
* 이런 분야도 있다는 것을 처음 접해서 신선한 decoding 방법이라고 생각했다. 다른 decoding 방법들([챗봇 코리아 게시물](https://www.facebook.com/groups/ChatbotDevKR/permalink/1000241780393951/))도 참고하면 좋을 것 같다.
* 그러나 해결되지 않은 몇 가지 문제(multi-modality)를 해결할 필요가 있어 보인다.