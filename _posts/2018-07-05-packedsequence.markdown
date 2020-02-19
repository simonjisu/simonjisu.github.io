---
layout: post
title: "Pytorch 의 PackedSequence object 알아보기"
date: "2018-07-05 09:45:37 +0900"
categories: nlp
author: "Soo"
comments: true
toc: true
---

# PackedSequence 란?

> 아래의 일련의 과정을 PackedSequence 라고 할 수 있다.

NLP 에서 매 배치(batch)마다 고정된 문장의 길이로 만들어주기 위해서 `<pad>` 토큰을 넣어야 한다. 아래 그림의 파란색 영역은 `<pad>` 토큰이다.

<img src="https://dl.dropbox.com/s/ctd209m9zlzs0cw/0705img1.png">

> 사진 출처: [Understanding emotions — from Keras to pyTorch](https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983)

그림과 같은 내용을 연산을 하게 되면, 쓸모없는 `<pad>` 토큰까지 연산을 하게 된다.
따라서 `<pad>` 를 계산 안하고 효율적으로 진행하기 위해 병렬처리를 하려고한다. 그렇다면 아래의 조건을 만족해야한다.

* RNN의 히든 스테이트가 이전 타임스텝에 의존해서 최대한 많은 토큰을 병렬적으로 처리해야한다.
* 각 문장의 마지막 토큰이 마지막 타임스텝에서 계산을 멈춰야한다.

아직 어떤 느낌인지 잘 모르겠다면 아래의 그림을 보자.

<img src="https://dl.dropbox.com/s/3ze3svhdz05aakk/0705img3.gif">

즉, 컴퓨터로 하여금 각 **타임스텝**(T=배치내에서 문장의 최대 길이) 마다 일련의 단어를 처리해야한다는 뜻이다.

하지만 $T=2, 3$ 인 부분은 중간에 `<pad>`이 끼어 있어서 어쩔수 없이 연산을 하게 되는데, 이를 방지하기 위해서, 아래의 그림같이 각 배치내에 문장의 길이를 기준으로 <span style="color: #e87d7d">정렬(sorting)</span> 후, 하나의 통합된 배치로 만들어준다.

<img src="https://dl.dropbox.com/s/op87oonnoqegn5c/0705img2.png">

> 사진 출처: [Understanding emotions — from Keras to pyTorch](https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983)

* **data:** `<pad>` 토큰이 제거후 합병된 데이터
* **batch_sizes:** 각 타임스텝 마다 배치를 몇개를 넣는지 기록해 둠

이처럼 PackedSequence 의 **장점**은 `<pad>` 토큰을 계산 안하기 때문에 더 빠른 연산을 처리 할 수 있다.

---

# Pytorch - PackedSequence

Pytorch 에서 사용하는 방법은 의외로 간단하다. 실습 코드는 [nbviewer](https://nbviewer.jupyter.org/github/simonjisu/pytorch_tutorials/blob/master/00_Basic_Utils/02_PackedSequence.ipynb) 혹은 [github](https://github.com/simonjisu/pytorch_tutorials/blob/master/00_Basic_Utils/02_PackedSequence.ipynb)에 있다.


## 과정

전처리를 통해 위 배치의 문장들을 숫자로 바꿔주었다.

```
input_seq2idx
============================================
tensor([[  1,  16,   7,  11,  13,   2],
        [  1,  16,   6,  15,   8,   0],
        [ 12,   9,   0,   0,   0,   0],
        [  5,  14,   3,  17,   0,   0],
        [ 10,   0,   0,   0,   0,   0]])
```

하단의 코드를 통해서 정렬을 해주고, 각 문장의 길이를 담은 list를 만들어준다.

```
input_lengths = torch.LongTensor([torch.max(input_seq2idx[i, :].data.nonzero())+1 for i in range(input_seq2idx.size(0))])
input_lengths, sorted_idx = input_lengths.sort(0, descending=True)
input_seq2idx = input_seq2idx[sorted_idx]
```

모든 `<pad>` 토큰의 인덱스인 0 이 밑으로 내려간 것을 알 수 있다.

```
input_seq2idx, input_lengths
============================================
tensor([[  1,  16,   7,  11,  13,   2],
        [  1,  16,   6,  15,   8,   0],
        [  5,  14,   3,  17,   0,   0],
        [ 12,   9,   0,   0,   0,   0],
        [ 10,   0,   0,   0,   0,   0]])

tensor([ 6,  5,  4,  2,  1])
```

**torch.nn.utils.rnn** 에서 **pack\_padded\_sequence** 를 사용하면 PackedSequence object를 얻을 수 있다. packed\_input 에는 위에서 말한 합병된 데이터와 각 타임스텝의 배치사이즈들이 담겨있다.

```
packed_input = torch.nn.utils.rnn.pack_padded_sequence(input_seq2idx, input_lengths.tolist(), batch_first=True)
```

<br>

## RNN 에서의 사용 방법

실수 벡터공간에 임베딩된 문장들을 pack 한 다음에 RNN 에 input을 넣기만 하면 된다.

```
embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
gru = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)

embeded = embed(input_seq2idx)
packed_input = pack_padded_sequence(embeded, input_lengths.tolist(), batch_first=True)
packed_output, hidden = gru(packed_input)
```
packed\_output 에는 합병된 output 과 batch_sizes 가 포함되어 있다.

```
packed_output[0].size(), packed_output[1]
=========================================================
(torch.Size([18, 2]), tensor([ 5,  4,  3,  3,  2,  1]))
```

이를 다시 원래 형태의 **(배치크기, 문장의 최대 길이, 히든크기)** 로 바꾸려면 **pad\_packed\_sequence** 를 사용하면 된다.

```
output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
output.size(), output_lengths
=========================================================
(torch.Size([5, 6, 2]), tensor([ 6,  5,  4,  2,  1]))
```

실습코드에서 출력 결과를 살펴보면 `<pad>` 토큰과 연관된 행은 모드 0으로 채워져 있다.

---

# RNN Backend 작동 방식

## RNN 안에서 어떤 방법으로 실행되는 것일까?

아래의 그림을 살펴보자

<img src="https://dl.dropbox.com/s/jl1iymxj6fdtvoe/0705img4.gif">

은닉층에서는 매 타임스텝마다 batch\_sizes 를 참고해서 배치수 만큼 은닉층을 골라서 뒤로 전파한다.

기존의 RNN 이라면, **(배치크기 $\times$ 문장의 최대 길이 $\times$ 층의 갯수)** 만큼 연산을 해야하지만, **(실제 토큰의 갯수 $\times$ 층의 갯수)** 만큼 계산하면 된다. 이 예제로 말하면 $(5 \times 6 \times 1)=30 \rightarrow (18 \times 1)=18$ 로 크게 줄었다.

## 그렇다면 Hidden 어떻게 출력 되는가?

기존의 RNN 이라면 마지막 타임스텝 때 hidden vector 만 출력하지만, packed sequence 는 아래의 그림 처럼 골라서 출력하게 된다.

<img src="https://dl.dropbox.com/s/e1kjq4jsehbixiq/0705img5.png">

참고자료: [https://discuss.pytorch.org/t/lstm-hidden-cell-outputs-and-packed-sequence-for-variable-length-sequence-inputs/1183]( https://discuss.pytorch.org/t/lstm-hidden-cell-outputs-and-packed-sequence-for-variable-length-sequence-inputs/1183)
