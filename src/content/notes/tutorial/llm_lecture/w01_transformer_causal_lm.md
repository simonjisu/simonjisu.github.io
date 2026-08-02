---
title: "1주차. Transformer와 Causal LM"
description: "문장을 token으로 나누고, Transformer가 다음 token의 확률과 loss를 계산하는 과정을 배운다."
tags:
  - LLM
  - Transformer
  - causal language modeling
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 16주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

이번 주에는 LLM이 문장을 읽고 다음 token을 맞히는 과정을 배운다. Transformer의 모든 수식을 외울 필요는 없다. 입력 문장이 어떤 모양의 tensor로 바뀌고 모델이 무엇을 정답으로 삼는지 설명하는 데 초점을 둔다.

## 이번 주에 배울 것

- 문장이 token과 embedding으로 바뀌는 과정
- self-attention에서 Query, Key, Value가 맡는 역할
- causal mask가 미래 token을 가리는 이유
- logits와 label로 cross-entropy loss를 계산하는 과정

선수 지식은 Python의 list, 행렬의 행과 열, 확률의 합이 1이라는 사실 정도면 충분하다.

!!! note "다음 token 맞히기"

    “나는 오늘 학교에 ___”라는 문장이 있다고 하자. 빈칸에는 “간다”, “갔다” 같은 말이 올 가능성이 높다. Causal LM은 이런 문제를 아주 많이 풀면서 언어를 배운다. 훈련할 때는 정답 문장이 이미 있으므로, 각 위치의 다음 token을 정답으로 쓸 수 있다.

## 1. 글자는 바로 모델에 들어가지 않는다

모델은 글자를 그대로 읽지 못한다. tokenizer가 문장을 작은 단위인 token으로 나누고, token마다 정수 ID를 붙인다. token은 완전한 단어일 수도 있고 단어의 일부일 수도 있다. 같은 문장도 tokenizer가 다르면 나뉘는 방식이 달라진다.

```text
문장       나는 학교에 간다
token      [나는] [학교] [에] [간다]
token ID   [381, 920, 17, 4421]   # 설명을 위한 가상 값
```

token ID는 출석 번호와 비슷하다. 381이라는 숫자 자체에는 “나는”의 뜻이 없다. embedding table에서 381번 행을 찾으면 여러 실수로 이루어진 vector가 나오고, 모델은 이 vector로 계산한다.

batch 크기를 $B$, token 수를 $T$, embedding 차원을 $d_{\text{model}}$이라고 하면 입력 tensor의 모양은 $X \in \mathbb{R}^{B \times T \times d_{\text{model}}}$이다.

문장 안에서 token의 순서도 알려줘야 한다. 원래 Transformer 논문은 sinusoidal positional encoding을 embedding에 더했다. 요즘 decoder-only LLM은 RoPE 같은 다른 위치 표현도 많이 쓰지만, “token의 내용과 위치를 함께 전달한다”는 생각은 같다.[^1]

## 2. Transformer 블록 안에서는 무슨 일이 일어날까?

![Transformer 원 논문의 encoder-decoder 구조](/notes/tutorial/llm_lecture/images/w01_transformer_architecture.png)

*그림 1. 원래 Transformer의 encoder-decoder 구조. 출처: Vaswani et al. (2017), Figure 1에서 발췌.[^1]*

그림의 왼쪽은 encoder, 오른쪽은 decoder다. GPT나 Qwen 같은 Causal LM은 보통 decoder-only 구조를 사용한다. 원 논문의 decoder를 그대로 떼어 쓴다는 뜻은 아니다. masked self-attention, feed-forward network, residual connection처럼 이어진 핵심 생각을 decoder-only 형태에 맞게 구성한다.

Transformer 블록에는 다음 계산이 반복해서 들어간다.

1. self-attention이 문장 안의 token 관계를 계산한다.
2. feed-forward network가 각 token의 표현을 변환한다.
3. residual connection이 입력을 출력에 더해 정보와 gradient가 흐를 길을 만든다.
4. normalization이 값의 크기를 안정적으로 조절한다.

## 3. Attention은 필요한 정보를 골라 읽는다

![Scaled Dot-Product Attention과 Multi-Head Attention](/notes/tutorial/llm_lecture/images/w01_scaled_dot_product_attention.png)

*그림 2. Scaled Dot-Product Attention과 Multi-Head Attention. 출처: Vaswani et al. (2017), Figure 2에서 발췌.[^1]*

각 token의 vector $X$에 서로 다른 weight를 곱해 Query $Q$, Key $K$, Value $V$를 만든다. 식으로 쓰면 $Q=XW_Q,\; K=XW_K,\; V=XW_V$이다.

도서관을 떠올리면 역할을 구분하기 쉽다.

- Query는 “내가 지금 찾는 정보”다.
- Key는 “각 책이 어떤 정보를 담았는지 알려주는 표지”다.
- Value는 “실제로 꺼내 읽을 내용”이다.

Query와 Key를 곱하면 두 token이 얼마나 관련 있는지 나타내는 점수가 나온다. 점수를 $\sqrt{d_k}$로 나누고 softmax를 적용하면 합이 1인 비율이 된다. 이 비율로 Value를 섞은 값이 attention의 출력이다. 전체 식은 $\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$로 쓴다.

여기서 $d_k$는 Key vector의 차원이고, $M$은 보지 못하게 할 위치를 표시한 mask다. Multi-Head Attention은 이 계산을 여러 관점에서 병렬로 수행한 뒤 결과를 합친다. 한 head는 가까운 단어 관계에, 다른 head는 문장 앞부분의 주어에 더 큰 비중을 둘 수 있다. 다만 head마다 반드시 사람이 이름 붙일 수 있는 문법 규칙을 배운다고 단정해서는 안 된다.

## 4. Causal mask는 미래의 정답을 가린다

“나는 학교에 간다”에서 “학교에” 다음을 예측할 때 모델이 이미 “간다”를 볼 수 있다면 문제를 푸는 의미가 없다. Causal mask는 현재 위치보다 오른쪽에 있는 token의 attention 점수를 $-\infty$에 가까운 값으로 바꾼다. softmax를 거치면 그 위치의 비중은 0이 된다.

```text
예측 위치  읽을 수 있는 token 위치
1          1
2          1 2
3          1 2 3
4          1 2 3 4
```

훈련할 때는 문장 전체를 GPU에 한꺼번에 넣어 계산한다. 그래도 각 위치는 왼쪽 정보만 본다. 이 덕분에 순서를 지키면서 여러 위치의 loss를 병렬로 구한다.

!!! warning "mask와 padding mask는 목적이 다르다"

    causal mask는 미래 token을 가린다. padding mask는 길이가 다른 문장을 한 batch로 묶을 때 채워 넣은 빈칸을 가린다. 둘을 섞어 생각하면 attention 결과가 틀어질 수 있다.

## 5. Logits에서 loss까지

마지막 Transformer 블록의 출력에 linear layer를 적용하면 vocabulary의 모든 token에 대한 점수가 나온다. 이 점수를 logits라고 한다.

batch가 2, 문장 길이가 5, vocabulary 크기가 1,000이라면 logits의 shape는 다음과 같다.

```text
input_ids : [2, 5]
logits    : [2, 5, 1000]
```

각 위치의 정답은 바로 다음 token이다.

```text
입력 위치   [나는]   [학교에]   [간다]
정답         [학교에] [간다]     [<eos>]
```

softmax는 logits를 확률로 바꾼다. cross-entropy loss는 정답 token에 준 확률이 낮을수록 커진다. 식으로 쓰면 $\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(y_i \mid y_{<i})$이다.

$y_i$는 맞혀야 할 token이고, $y_{<i}$는 그보다 왼쪽에 있는 token들이다. 모델은 gradient descent로 loss가 작아지는 방향으로 weight를 조금씩 고친다.

## 6. 직접 shape 확인하기

아래 코드는 GPT-2에 문장 하나를 넣어 logits와 loss의 shape를 확인한다. `labels=input_ids`를 전달하면 `AutoModelForCausalLM`이 내부에서 label을 한 칸 이동해 다음 token loss를 계산한다.[^2]

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

batch = tokenizer("나는 오늘 학교에 간다.", return_tensors="pt")

with torch.inference_mode():
    output = model(
        **batch,
        labels=batch["input_ids"],
    )

print("input_ids:", batch["input_ids"].shape)
print("logits:", output.logits.shape)
print("loss:", output.loss.item())
print("vocab size:", model.config.vocab_size)

assert output.logits.shape[:2] == batch["input_ids"].shape
assert output.logits.shape[-1] == model.config.vocab_size
```

!!! note "loss 하나만 보고 판단하지 않기"

    짧은 문장 하나의 loss는 모델 품질을 평가하는 점수가 아니다. 문장과 tokenizer가 달라지면 값도 달라진다. 이번 실습에서는 숫자의 크기보다 input과 logits의 shape, label이 이동하는 위치를 확인한다.

## 7. 실제 실행 결과

GPT-2에 영어 문장과 한국어 문장을 하나씩 넣어 forward pass를 실행했다. 두 문장의 token 수와 loss는 다음과 같았다.[^4]

| 입력 문장 | 입력 token 수 | vocabulary 크기 | loss | forward 시간 |
| --- | ---: | ---: | ---: | ---: |
| `The quick brown fox jumps over the lazy dog.` | 10 | 50,257 | 5.091 | 1.248초 |
| `나는 오늘 학교에 간다.` | 28 | 50,257 | 1.984 | 0.153초 |

같은 GPT-2 tokenizer를 썼는데 한국어 문장이 더 많은 token으로 나뉘었다. 영어 중심으로 학습한 tokenizer가 한국어 글자를 잘게 나눈 결과다. 두 문장의 길이와 token 구성이 다르므로 loss만 보고 어느 언어를 더 잘 안다고 결론 내릴 수는 없다.

영어 문장을 먼저 실행했기 때문에 1.248초에는 초기 준비 시간이 섞였다. 두 시간 값도 언어별 속도 차이로 비교하지 않는다.

![GPT-2가 영어 문장 다음에 예측한 token의 확률](/notes/tutorial/llm_lecture/images/w01_forward_next_token_result.png)

*그림 3. 영어 문장 뒤에 올 token의 확률 상위 10개. 출처: GPT-2 직접 실행 결과(2026-08-01, Apple MPS). 모델 정보는 GPT-2 model card를 참고했다.[^4]*

그래프의 막대는 모델이 다음 token 후보마다 나눠 준 확률이다. 가장 높은 후보도 약 0.26이므로 모델이 하나의 답만 확신한 상황은 아니다. 다음 token 생성은 이 확률 분포에서 하나를 고르는 과정이다.

## 확인 문제

1. token ID 381이라는 숫자 자체에 단어의 뜻이 없는 이유는 무엇인가?
2. Causal LM이 훈련 중에 문장 전체를 입력받아도 미래 token을 훔쳐보지 못하는 이유는 무엇인가?
3. logits의 shape가 `[2, 5, 1000]`일 때 1,000은 무엇을 뜻하는가?
4. Query와 Key의 곱에 softmax를 적용하는 까닭은 무엇인가?
5. padding mask와 causal mask는 각각 무엇을 가리는가?

## 완료 체크

- [ ] 문장을 token ID와 embedding으로 바꾸는 과정을 설명했다.
- [ ] Transformer 원 논문의 Figure 1과 Figure 2를 읽었다.
- [ ] attention 식의 $Q$, $K$, $V$, $M$을 설명했다.
- [ ] 코드에서 input, logits, loss의 shape를 확인했다.
- [ ] 확인 문제에 답하고 `Forward/loss 분석 노트`를 남겼다.

---

[^1]: Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 특히 Figure 1, Figure 2, §3을 참고했다.
[^2]: Hugging Face. [Transformers: Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling). 확인일: 2026-07-31.
[^3]: PyTorch. [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). 확인일: 2026-07-31.
[^4]: Hugging Face. [openai-community/gpt2 model card](https://huggingface.co/openai-community/gpt2). 실행 모델과 tokenizer 정보를 참고했다. 확인일: 2026-08-01.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 절별로 5,000자 이하로 나누어 점검
원본/윤문본: 6336자 / 6264자, 변경률 1.14%
탐지/수정: A-10 1→0, E-2 1→0, A-18 1→0, 그 밖의 S1 0→0
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 신규 작성문을 보수적으로 다듬음
주요 변경: 긴 목표 문장 단순화, 모든 수식을 inline math 형식으로 통일, 직접 실행 결과의 수치와 해석 추가
-->
