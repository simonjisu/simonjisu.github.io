---
layout: post
title: "[NLP] Attention Is All You Need - 3"
date: "2020-02-23 14:19:38 +0900"
categories: paper
author: "Soo"
comments: true
toc: true
---

Paper Link: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

이전 글: [Attention Is All You Need - 2](https://simonjisu.github.io/paper/2020/02/02/attentionisallyouneed2.html)

---

# 5. Models

## Masking

지금까지 미뤄온 Attention 의 마스킹(Masking)을 이야기 해보려 한다. 마스킹이 필요한 이유는 두 가지다.

1. Decoder 의 Self Attention 

    Decoder 에서는 이전의 타임 스텝(t-1)의 정보를 활용하여 다음 타임 스텝(t)의 정보를 예측하게 되는데 이를 **자기회귀(auto-regressive)**특성이라고 한다. 이러한 특성을 보존하기 위해서 이전 타임 스텝(t-1)을 입력으로 현재 타임 스텝(t)를 예측하려고 할 때, 다음 타임 스텝(t+1)의 정보를 참조하면 안된다. 따라서 이를 Scaled Dot-Product Attention 에서 마스킹을 통해, 음의 무한대(`-np.inf`) 값을 주어서 Softmax 값을 0으로 만들어 준다. 

    예를 들어 `그림 1`처럼 (검은색이 마스킹 위치) Decoder 의 입력 데이터 최대 길이가 4인 경우, <span style="color:#e25252">**Q**</span> 에서 0 번째 토큰은 1 번째 토큰을 예측해야 함으로 Self-Attention 시 <span style="color:#5470cc">**K**</span> 의 1, 2, 3 번째의 토큰의 관계를 무시해야한다. <span style="color:#e25252">**Q**</span> 의 1 번째 토큰을 입력시 2 번째 토큰을 예측하게 되는데, 자기 자신을 포함한 그 이전의 정보를 참조 할 수는 있지만 미래의 2, 3 번째의 정보를 미리 참고하면 안된다.

{% include image.html id="1VnSx8Ct5_NNNoa13zGfA5p-RSgbzBIMn" desc="[그림 1] Decoder Sub-sequence Attention Masking" width="75%" height="auto" %}

2. 실제 토큰의 길이

    앞서 말했듯이 RNN 처럼 recurrance 하지 않기 때문에 최대 입력/출력 길이를 정해야한다. 따라서 실제 문장은 길이가 4인데도 설정한 최대 길이 때문에 그 길이만큼 `Padding`을 하게 되는데, Attention 계산시 `Padding` 은 인위적으로 넣은 토큰이기 때문에 이를 무시해야 한다. 

    예를 들어 Decoder 에 들어가는 타겟 데이터의 최대 길이는 4이지만 실제 토큰의 길이가 3이라면 Attention Matrix 에 해당하는 마스킹은 `그림 2`와 같다. 여기서는 마지막 토큰이 `Padding` 토큰이기 때문에 Self Attention 시 마지막 토큰은 참조하지 않는다. Attention 코드([GitHub](https://github.com/simonjisu/annotated-transformer-kr/blob/master/transformer/modules.py) 참고) 구현하게 되면 3 번째 행은 Softmax 를 통과시 `nan` 값이 된다. 따라서 해당하는 값을 0으로 다시 마스킹하는 과정이 필요하다. 

{% include image.html id="1KOJA8DNlTQjnKb19zn2vRtzEIRbB8Ut2" desc="[그림 2] 실제 토큰 길이에 대한 Masking" width="75%" height="auto" %}

해당 모듈(Module) 코드는 다음과 같다.
* [Encoder Layer](https://github.com/simonjisu/annotated-transformer-kr/blob/9c1e4988e5aba3d2b971074590ce49e50c3aa823/transformer/layers.py#L11)
* [Decoder Layer](https://github.com/simonjisu/annotated-transformer-kr/blob/9c1e4988e5aba3d2b971074590ce49e50c3aa823/transformer/layers.py#L42)
* [Encoder](https://github.com/simonjisu/annotated-transformer-kr/blob/9c1e4988e5aba3d2b971074590ce49e50c3aa823/transformer/models.py#L10)
* [Decoder](https://github.com/simonjisu/annotated-transformer-kr/blob/9c1e4988e5aba3d2b971074590ce49e50c3aa823/transformer/models.py#L54)

## Transformer Model

특이한 점이라면 마지막 예측 토큰을 출력하는 선형 변환 층(`projection`)을 임베딩 층으로 치환하는 방법이 있는데 이를 논문에서 Linear Weight Sharing이라고 했다. 또한, 문제에 따라서 Encoder층의 임베딩과 Decoder층의 임베딩을 공유 할 수도 있는데 Language Modeling 같은 문제가 그 예시라고 할 수 있다. 이를 논문에서 Embed Weight Sharing이라고 했다.

해당 모듈(Module) 코드는 [**Link**](https://github.com/simonjisu/annotated-transformer-kr/blob/9c1e4988e5aba3d2b971074590ce49e50c3aa823/transformer/models.py#L112) 에서 확인할 수 있다.

---

# 6. Loss Function

## Label Smoothing

Discrete한 분포를 예측하는 방법은 주로 Cross Entropy를 많이 사용하지만 논문에서는 [Rethinking the inception architecture for computer vision](https://arxiv.org/abs/1512.00567) 논문에서 언급한 Label Smoothing 기법을 활용했다. 

예측 확률 분포를 **P**, 정답/타겟 확률 분포(ground-truth distribution)를 **Q**라고 하겠다. $x$ 를 입력으로 예측 확률 질량함수 `p(y=k|x)` 에서 구한 확률(Softmax)을 타겟 확률 질량함수 `q(y=k|x)=1` 처럼 만드는 것이 원래의 최종목표다. 이제부터 $x$를 생략해서 쓰겠다. Cross Entropy의 수식은 다음과 같다.

$$Loss = -\sum_{k=1}^{K} \log\big(p(k) \big) q(k)$$

Cross Entropy 를 최소화 하는 것은 $k$ 라벨에 해당하는 log-likelihood 의 기댓값을 $q(k)$로 최대화 하는 것과 같다. 그리고 Cross Entropy 는 Softmax 에 사용되는 예측값의 로짓(logit, $z_k$)에 대해 미분을 구할 수 있는데, 그 미분값은 다음과 같으며 -1 과 1 사이의 치역을 갖는다.

$$\begin{aligned} \dfrac{\partial Loss}{\partial z_k} = q(k)\big(p(k)-1\big) \end{aligned}$$

정답 라벨(ground-truth label)이 $y$인 예시를 들어봅자. 논문에서는 라벨 $y$에 해당하는 로짓($z_y$) 값이 다른 라벨 $k$의 로짓($z_k$) 값에 비해 월등히 클 수록 2가지 문제가 생긴다고 한다. 

1. 오버피팅(over-fitting)이 될 가능성이 있다. 만약에 모델이 전체 확률 분포(모든 라벨)를 학습 시, 일반화(generalize)를 보장할 수 없다.
2. 이러한 구조는 제일 큰 로짓 값과 상대적으로 작은 로짓 값의 차이를 점점 더 크게 만들도록 학습한다. 이는 미분 값을 항상 0에 가깝게 만들어 가중치 업데이트가 안된다. 다음 미분의 수식에서 확인 할 수 있듯이, 정답 라벨 $y$ 의 로짓 값이 높을 수록, 그 확률은 1 에 가까워 경사(gradient)가 0에 가깝다.

    $$\begin{aligned} \dfrac{\partial Loss}{\partial z_y} &= -q(z_y)\cdot\frac{1}{p(z_y)} \times p(z_y)\big(1-p(z_y) \big) \\ &= -q(z_y)+q(z_y)p(z_y) \\&= p(z_y) -1 \end{aligned}$$

결론적으로 말하면, 정답 라벨에 대해서 너무 확실한 예측을 내놓는 다는 것이다. 따라서 이러한 효과를 줄이기 위해서 해당 논문에서는 색다른 정답 확률 분포를 이야기 하는데, 기존의 `q(k)`의 분포는 다음과 같다.

$$q(k)=\delta_{k,y} \begin{cases} 1 \quad \text{if } k=y \\ 0 \quad \text{else} \end{cases}$$

이를 새로운 `q'(k)`로 치환하게 된다. $\epsilon$은 Smoothing을 위한 변수다.

$$q'(k) = (1-\epsilon)\delta_{k,y} + \epsilon \cdot u(k)$$

`u(k)` 의 분포는 훈련데이터와 무관한 분포이며, 보통 `u(k) = 1/K`인 Uniform Distribution 을 사용한다. `q'(k)`와 같은 정규화의 한 방법으로 논문에서는 이를 **Label-Smoothing Regularization (LSR)** 이라고 제시했다.

LSR 의 목적은 원래 목표인 타겟 라벨을 맞추는 목적과 정답 라벨의 로짓(logit) 값이 학습과정에서 과도하게 다른 로짓 값보다 커지는 현상을 방지하는 것이다. 수식을 약간 변형하여 LSR 를 다른 관점에서 볼 수 있다.

$$\begin{aligned} LSR = H(p, q') &= -\sum_{k=1}^K \log \big( p(k)\big) q'(k) \\ &= -\sum_{k=1}^K \log \big( p(k)\big)\big( (1-\epsilon)\delta_{k,y} + \epsilon u(k) \big) \\ &= (1-\epsilon)\big(-\sum_{k=1}^K \log \big( p(k)\big)\delta_{k,y} \big) \epsilon \big( -\sum_{k=1}^K \log \big( p(k)\big)u(k) \big) \\ &= (1-\epsilon) H(q, p) + \epsilon H(u, p) \end{aligned}$$

LSR 은 기존에 Cross Entropy `H(q, p)`를 한 쌍의 `H(q, p)` 와 `H(u, p)`로 대체한 것이다. Smoothing Factor 인 $\epsilon$의 크기의 여부에 따라 정규화의 정도가 달라진다.

**Transformer** 에서도 해당 방법을 정규화 목적으로 $K$ 는 타겟 단어장(vocab)의 크기, $\epsilon$=0.1 로 사용했다.

해당 모듈(Module) 코드는 [**Link**](https://github.com/simonjisu/annotated-transformer-kr/blob/9c1e4988e5aba3d2b971074590ce49e50c3aa823/transformer/labelsmooth.py#L5) 에서 확인할 수 있다.

### References

* [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
* [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html#label-smoothing)

---

# 7. Optimizer

## Learning Rate Variation

논문에서는 Adam Optimizer 를 기반으로 다양한 학습률을 적용하여 사용했다. 학습률 변화의 수식은 다음과 같다.

$$lrate = d_{model}^{-0.5} \cdot \min(\text{step_num}^{-0.5}, \text{step_num} \cdot \text{warmup_steps}^{-1.5} )$$

해당 수식에 따르면 처음 warmup_steps 동안 학습률은 가파르게 상승하다가 차후에 천천히 하강하게 된다.

{% include image.html id="1d0xw7_xjr1rv7-SjuxQKRmML4oio561j" desc="[그림 3] hidden 크기 및 warmup steps 에 따른 학습률의 변화" width="75%" height="auto" %}

해당 모듈(Module) 코드는 [**Link**](https://github.com/simonjisu/annotated-transformer-kr/blob/9c1e4988e5aba3d2b971074590ce49e50c3aa823/transformer/warmupoptim.py#L1) 에서 확인할 수 있다.

---

# 8. Training Multi30k with Transformer

PyTorch의 `torchtext`에 있는 Multi30k 데이터 세트(영어-독일어 번역)로 테스트 해보았다. 큰 데이터는 아니기 때문에, NVIDIA GTX 1080 ti 로 약 36분 훈련시켰다. 기존의 RNN 으로 훈련시키는 것 보다 월등히 빨랐다. 모델에서 Attention에 대한 그림도 [github](https://github.com/simonjisu/annotated-transformer-kr)에 올려두었으니 확인해보길 바란다.

{% include image.html id="1HsVRsp3mMjo8UBSTU81ZE4i_MUZ4Z1Xa" desc="[그림 4] Multi30k 성능 테스트" width="100%" height="auto" %}