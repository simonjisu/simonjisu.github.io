---
layout: post
title: "[NLP] Attention Is All You Need - 2"
date: "2020-02-02 14:19:38 +0900"
categories: paper
author: "Soo"
comments: true
---

# Attention Is All You Need - 2

Paper Link: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

이전 글: [Attention Is All You Need - 1](https://simonjisu.github.io/paper/2020/01/14/attentionisallyouneed.html)

**목차**
* [3. Sub Layers](#3-sub-layers)
    * [Multi-Head Attention](#multi-head-attention)
    * [Position-wise Feed-Forward Networks](#position-wise-feed-forward-networks)
* [4. Embeddings](#4-embeddings)
    * [Input 과 Output](#input-과-output)
    * [Embedding](#embedding)
    * [Positional Encoding](#positional-encoding)

---

## 3. Sub Layers

### Multi-Head Attention

{% include image.html id="1jpQdv3lFrYNRZ5FbCvcXF4RDtpho0og_" desc="[그림 1] Multi-Head Attention" width="75%" height="auto" %}

첫번째 서브층(SubLayer) Multi-Head Attention 의 구조는 `그림 1` 과 같다. 연구자들은 $d_{model}$ 크기의 <span style="color:#e25252">**Q**</span>, <span style="color:#5470cc">**K**</span>, <span style="color:#cfb648">**V**</span> 를 한 번 수행하는 것보다 $h$ 개의 각기 다른 **선형 투영(linear projection)**을 시켜, 크기가 $d_k$(<span style="color:#e25252">**Q**</span>, <span style="color:#5470cc">**K**</span>), $d_v$(<span style="color:#cfb648">**V**</span>) 인 텐서를 사용해서 Attention 을 병렬로 수행하는 것이 더 유리한 것을 찾아냈다. 각기 다른 Attention 을 수행한 $h$ 개의 출력값은 하나로 concatenate 후에 최종 선형결합을 통해 다시 $d_{model}$ 크기로 돌아오는데 이를 수식으로 표현하면 다음과 같다. 

$$\begin{aligned} \text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \cdots \text{head}_h)W^O  \\ \text{where head}_i &= \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)  \end{aligned}$$

W 는 선형결합을 위한 매겨변수이며 각각의 크기는 다음과 같다.

$$\begin{aligned} W^Q_i \in \Bbb{R}^{d_{model}\times d_k}, W^K_i\in \Bbb{R}^{d_{model}\times d_k}, W^V_i\in \Bbb{R}^{d_{model}\times d_v}, W^O \in \Bbb{R}^{h*d_v\times d_{model}}\end{aligned}$$

그렇다면 이렇게 큰 차원의 <span style="color:#e25252">**Q**</span>, <span style="color:#5470cc">**K**</span>, <span style="color:#cfb648">**V**</span> 를 선형 변환 후에 $h$ 개의 Attention 을 나눠서 학습하게 했던 이유는 무엇 일까? 논문에서 이해한 것을 정리하면 다음과 같다.

일단 선형 변환된 <span style="color:#e25252">**Q**</span>, <span style="color:#5470cc">**K**</span>, <span style="color:#cfb648">**V**</span> 를 $h$ 개로 나눠버리는데, 이는 각 토큰을 표현하고 있던 큰 차원의 뉴런들을 $h$ 개의 블록으로 나눴다고 할 수 있다. 이렇게 위치가 상이한 각기 다른 표현 부분공간(representation subspaces) 블록들이 교차하면서(jointly) 정보를 얻게 된다. 이 말은 곧 $h$ 개의 Attention Matrix 가 생기면서 <span style="color:#e25252">**Q**</span> 와 <span style="color:#5470cc">**K**</span> 간의 토큰들이 더 다양한 관점으로 볼 수 있다는 말이다. 만약에 나누지 않았다면 단 하나의 Attention Matrix 를 생성하면서 이러한 효과를 뭉게버림으로, 선형 변환 층(linear projection)은 학습 과정을 반복하면서 최적의 $h$ 개의 Attention Matrix 를 생성하는 역할을 학습하게 된다.

해당 모듈(Module) 코드는 [**Link**](https://github.com/simonjisu/annotated-transformer-kr/blob/9c1e4988e5aba3d2b971074590ce49e50c3aa823/transformer/sublayers.py#L11) 에서 확인할 수 있다.


### Position-wise Feed-Forward Networks

또 다른 서브층으로써 완전 연결층(Fully Connect Layer)인 네트워크를 Encoder, Decoder 뒤에 하나씩 추가했다. 이 완전 연결층은 두 개의 선형변환과 ReLU 활성화 함수를 사용했으며 그 수식은 다음과 같다.

$$\text{FFN}(x) = \max(0, xW_1+b_1)W_2+b_2$$

아마도 입력 텐서가 각 토큰의 위치별로 차원이 커졌다가 다시 원래 모양으로 줄어들어서 이름이 Position-wise 라고 붙여진 것으로 추정되는데, 차원의 크기가 다음과 같이 변하기 때문이다.

$$(B, T, d_{model}) \rightarrow(B, T, d_{ff}) \rightarrow (B, T, d_{model})$$

해당 모듈(Module) 코드는 [**Link**](https://github.com/simonjisu/annotated-transformer-kr/blob/9c1e4988e5aba3d2b971074590ce49e50c3aa823/transformer/sublayers.py#L91) 에서 확인할 수 있다.

---

## 4. Embeddings

### Input 과 Output

Embedding 층과 Position Encoding 을 설명하기 전에 입출력이 어떻게 구성되어 있는지를 살펴봐야한다. 기계 번역 문제를 다시 예시로 들어보면, 다음과 같이 수치화된 문장들이 있다. 0 은 `Padding` 토큰으로써 데이터 처리를 위해 설정한 문장의 최대 길이에 맞춰서 넣은 인위적인 토큰이다.

$$\begin{aligned} \text{src} &= \begin{bmatrix}3&6&4&9 \\ 1&3&5&0 \\ 3&2&0&0 \end{bmatrix} \\ \text{trg} &= \begin{bmatrix}2&5&4&0 \\ 2&5&6&0 \\ 2&7&4&9 \end{bmatrix}  \end{aligned}$$

즉, 위 행렬을 해석하면 현재 Input 데이터는 미니배치가 3, 문장의 최대 길이가 4인 데이터, Target 데이터는 미니배치가 3, 문장의 최대 길이가 4인 데이터다. 각 문장의 토큰들에 순서 인덱스를 부여하여 포지션(Position) 데이터를 얻고자하면 다음과 같다. `Padding` 은 인위적으로 넣은 데이터기 때문에 순서가 없어야 한다.

$$\begin{aligned} \text{src\_pos} &= \begin{bmatrix}1&2&3&4 \\ 1&2&3&0 \\ 1&2&0&0 \end{bmatrix} \\ \text{trg\_pos} &= \begin{bmatrix}1&2&3&0 \\ 1&2&3&0 \\ 1&2&3&4 \end{bmatrix}  \end{aligned}$$

Decoder 의 경우 이전 타임 스텝(t-1)의 토큰들로 다음 타임 스텝(t)의 토큰을 예측하기 때문에 실질적으로 모델에 입력되는 데이터(`trg_input`)와 실제 예측해야하는 타겟 데이터(`gold`)는 다음과 같다. 즉, 예를 들어 1, 2, 3 포지션에 해당하는 타겟 값을 입력으로 주었을때 2, 3, 4 번 포지션에 해당하는 값을 예측하는 것이다.

$$\begin{aligned} \text{trg\_input} &= \begin{bmatrix} 2&5&4 \\ 2&5&6 \\ 2&7&4 \end{bmatrix} \\ \text{gold} &= \begin{bmatrix}5&4&0 \\ 5&6&0 \\ 7&4&9 \end{bmatrix}  \end{aligned}$$

토큰에 순서 정보인 포지션을 구하는 이유는 무엇일까? 그 해답은 RNN 의 구동원리에 있는데, RNN 을 Cell 단위로 만들면 다음 코드와 같다.

```python
    import torch
    import torch.nn as nn
    
    # create inputs tensor
    batch_size = 2
    seq_len = 10
    input_dimension = 7
    inputs = torch.rand(seq_len, batch_size, input_dimension)
    
    # setting rnn cell
    hidden_dimension = 6
    rnn_cell = nn.RNNCell(input_size=input_dimension, hidden_size=hidden_dimension)
    
    # RNN Layer
    outputs = []
    hidden = torch.zeros(batch_size, hidden_dimension)
    for i in range(seq_len):
        hidden = rnn_cell(inputs[i], hidden)
        outputs.append(hidden)
    
    outputs = torch.stack(outputs)
    print(outputs.size())
    # print: torch.Size([10, 2, 6])
```

RNN 의 특징 중 하나는 시퀀스 길이에 상관없이 한 스텝씩 처리하기 때문에 아주 긴 시퀀스도 처리를 할 수 있다. 그러나 이러한 특징은 이전 타입스텝의 정보를 다음 타임스텝에게 전달할 수 있지만 병렬 처리가 불가능 하다. 하지만 Transformer 의 목표중 하나는 시퀀스 데이터의 병렬 처리인데, 즉, 한 번에 지정된 길이의 시퀀스를 모두 모델에게 전달하고 Forward 하게 된다. 그렇다면 시퀀스의 각 토큰간 순서 관계 정보를 모델은 어떻게 알아낼 수 있을까? 바로 **Position Encoding** 을 통해서 각 토큰의 순서 정보를 **Embedding** 된 벡터와 결합하여 모델로 전달하게 된다.

### Embedding

임베딩은 분절된 토큰들을 고정된 $d_{model}$ 차원의 공간으로 표현해주는 방법이다. Decoder 의 출력층에는 선형변환 층과 Softmax 를 섞어서 예측 토큰의 확률을 구하는 기법을 사용했으며, 임베딩된 벡터에 $\sqrt{d_model}$ 를 곱했다.

해당 모듈(Module) 코드는 [**Link**](https://github.com/simonjisu/annotated-transformer-kr/blob/9c1e4988e5aba3d2b971074590ce49e50c3aa823/transformer/layers.py#L107) 에서 확인할 수 있다.

### Positional Encoding

Positional Encoding 은 상대적이거나 절대적인 위치정보를 부여하는 방법이다. 각 Position Encoding 의 차원의 크기는 더할 수 있게 임베딩된 텐서의 차원 크기인 $d_{model}$과 같고 수식은 다음과 같다.

$$\begin{aligned} PE_{pos, 2i} &= \sin(\frac{pos}{10000^{2i/d_{model}}}) \\ PE_{pos, 2i+1} &= \cos(\frac{pos}{10000^{2i/d_{model}}})\end{aligned}$$

결론을 말하자면 각 시퀀스의 순서 인덱서는 PE(Positonal Encoding) 테이블에서 각자의 위치를 조회후에 임베딩된 텐서와 결합하게 된다. pos 는 시퀀스의 위치정보, 예를 들어 텐서의 크기가 $d_{model}$ = 1024 의 경우, 각 1024의 짝수(2i)에 위치한 값들은 sin 함수를 적용하고, 홀수(2i+1) 에 위치한 값들은 cos 함수를 적용한다. PE 테이블을 그리면 `그림 2` 과 같은데, 자세히 보시면 각 포지션에 해당하는 1 줄(1024 크기의 벡터)값은 모두 차별화 되어있다. 

{% include image.html id="1IznpVENdNpwyKqJCD0mWnZRcQ2XaIh22" desc="[그림 2] 최대 길이가 51인 Positional Encoding Table" width="100%" height="auto" %}

해당 모듈(Module) 코드는 [**Link**](https://github.com/simonjisu/annotated-transformer-kr/blob/9c1e4988e5aba3d2b971074590ce49e50c3aa823/transformer/layers.py#L82) 에서 확인할 수 있다.

---

다음편: [NLP] Attention Is All You Need - 3