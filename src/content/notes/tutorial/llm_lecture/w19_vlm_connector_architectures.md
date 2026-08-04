---
title: "19주차. Vision encoder와 LLM을 잇는 구조"
description: "LLaVA의 projector, BLIP-2의 Q-Former, Flamingo의 Perceiver Resampler가 이미지 특징을 LLM의 visual token으로 바꾸는 방식을 비교한다."
tags:
  - VLM
  - LLaVA
  - BLIP-2
  - Flamingo
  - multimodal connector
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 26주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

18주차의 CLIP은 이미지와 문장이 얼마나 가까운지 점수를 매겼지만 긴 답변을 만들지는 않았다. 사진을 보고 대화하려면 vision encoder가 만든 특징을 LLM이 읽을 수 있는 token으로 바꿔야 한다. 이번 주에는 그 사이를 잇는 connector를 살펴본다.

## 이번 주에 배울 것

- Vision encoder, connector, LLM이 맡는 역할
- Visual token의 개수와 hidden dimension을 따로 읽는 방법
- LLaVA의 linear projector가 차원을 맞추는 방식
- BLIP-2의 Q-Former가 32개 query로 이미지 정보를 고르는 방식
- Flamingo의 Perceiver Resampler와 gated cross-attention
- Vision encoder와 LLM을 고정하거나 함께 학습할 때의 차이
- 세 connector의 tensor shape와 LLM 쪽 계산량을 비교하는 방법

선수 지식은 17주차의 patch embedding, 18주차의 image embedding, 1주차의 self-attention이다. Query, Key, Value가 낯설다면 Query가 질문하고 Key와 Value가 참고할 정보를 들고 있다고 생각하면 된다.

!!! note "서로 다른 두 교실 사이의 통역사"

    Vision encoder는 사진을 숫자로 설명하고 LLM은 text token을 읽는다. 두 model은 숫자의 크기와 개수가 서로 다르다. Connector는 이미지 특징을 LLM이 받을 수 있는 visual token으로 바꾸는 통역사다.

## 1. Connector는 차원과 token 수를 다룬다

VLM의 큰 흐름은 다음과 같다.

```text
image
  → vision encoder
  → image features [batch, vision tokens, vision dimension]
  → connector
  → visual tokens [batch, connector tokens, LLM dimension]
  → LLM + text tokens
  → answer tokens
```

Shape를 읽을 때는 token 수와 hidden dimension을 구분한다. Vision encoder 출력이 `[B, N_v, D_v]`, connector 출력이 `[B, N_c, D_lm]`이라고 하자.

| 기호 | 뜻 |
| --- | --- |
| $B$ | 한 번에 처리하는 이미지 수 |
| $N_v$ | Vision encoder가 만든 token 수 |
| $D_v$ | Vision feature 한 개의 차원 |
| $N_c$ | Connector가 LLM에 전달하는 visual token 수 |
| $D_{lm}$ | LLM의 word embedding 차원 |

Connector는 적어도 $D_v$를 $D_{lm}$에 맞춰야 한다. 구조에 따라 $N_v$를 그대로 두거나, 더 적은 $N_c$로 압축하기도 한다. Visual token이 LLM에 들어가는 위치도 다르다. Text token 앞에 붙일 수도 있고, LLM layer 사이의 cross-attention으로 따로 읽힐 수도 있다.

## 2. LLaVA는 projection으로 차원을 맞춘다

![LLaVA의 vision encoder, projection, language model 구조](/notes/tutorial/llm_lecture/images/w19_llava_projector.png)

*그림 1. CLIP vision feature를 projection한 뒤 language instruction token과 함께 LLM에 넣는 LLaVA 구조. 출처: Liu et al. (2023), Figure 1에서 발췌.[^1]*

원래 LLaVA 논문은 CLIP ViT-L/14의 grid feature $Z_v$를 linear projection $W$에 통과시킨다. 결과 $H_v=WZ_v$는 LLM의 word embedding과 같은 차원을 갖는다. Projection은 각 visual token에 같은 linear layer를 적용하므로 token 수는 유지된다.[^1]

이번 shape 실습에서는 visual feature `[1, 256, 1024]`를 `[1, 256, 4096]`로 바꿨다.

```python
import torch
from torch import nn

visual_features = torch.randn(1, 256, 1024)
projector = nn.Linear(1024, 4096)
visual_tokens = projector(visual_features)

print(visual_tokens.shape)
# torch.Size([1, 256, 4096])
```

이 `nn.Linear`는 bias를 포함해 parameter가 4,198,400개다. LLM이 수십억 parameter인 것과 비교하면 작은 연결부지만 출력 token 256개는 모두 LLM sequence에 들어간다.

LLaVA의 첫 학습 단계에서는 vision encoder와 LLM을 고정하고 projection만 학습한다. 두 번째 단계에서는 vision encoder를 계속 고정한 채 projection과 LLM을 함께 fine-tuning한다.[^1]

!!! note "단순하다는 말은 정보가 적다는 뜻이 아니다"

    Linear projector는 구조가 단순하지만 vision feature 256개를 그대로 전달한다. 압축해서 일부만 고르는 connector보다 많은 위치 정보를 보존하는 대신 LLM이 읽을 token도 많아진다.

## 3. BLIP-2는 Q-Former로 정보를 고른다

![BLIP-2의 frozen image encoder, Q-Former, frozen LLM 구조](/notes/tutorial/llm_lecture/images/w19_blip2_qformer.png)

*그림 2. Frozen image encoder와 frozen LLM 사이에서 Q-Former를 두 단계로 pre-training하는 BLIP-2 구조. 출처: Li et al. (2023), Figure 1에서 발췌.[^2]*

BLIP-2는 Querying Transformer, 줄여서 Q-Former를 connector로 쓴다. 학습 가능한 query embedding이 image feature를 cross-attention으로 읽고, text에 필요한 시각 정보를 고른다. Vision encoder와 LLM은 고정한다.[^2]

논문 실험에서는 768차원 query 32개를 사용한다. 예로 든 ViT-L/14 image feature는 `[257, 1024]`이고 Q-Former 출력은 `[32, 768]`이다. 입력 image token 수와 관계없이 LLM으로 넘어갈 visual representation을 32개로 줄이는 bottleneck이다.[^2]

```python
queries = learned_queries.expand(batch_size, 32, hidden_size)
query_output, _ = cross_attention(
    query=queries,
    key=image_features,
    value=image_features,
)

print(query_output.shape)
# torch.Size([1, 32, hidden_size])
```

Q-Former는 단순한 평균이 아니다. Query끼리 self-attention을 하고, 일정한 layer마다 image feature를 cross-attention으로 읽는다. 논문의 Q-Former는 BERTbase에서 초기화한 Transformer와 새 cross-attention layer로 구성되며 총 188M parameter다.[^2]

학습도 두 단계로 나뉜다. 첫 단계는 image-text contrastive learning, image-grounded text generation, image-text matching으로 시각과 언어 표현을 맞춘다. 두 번째 단계에서는 Q-Former 출력을 linear projection해 frozen LLM의 soft visual prompt로 넣고 text generation을 학습한다.

## 4. Flamingo는 visual token을 따로 읽게 한다

![Flamingo의 Perceiver Resampler와 gated cross-attention 구조](/notes/tutorial/llm_lecture/images/w19_flamingo_architecture.png)

*그림 3. 여러 이미지와 text를 섞어 입력하고, Perceiver Resampler의 visual token을 frozen LM 사이의 gated cross-attention layer로 읽는 Flamingo 구조. 출처: Alayrac et al. (2022), Figure 3에서 발췌.[^3]*

Flamingo의 Perceiver Resampler는 이미지나 영상에서 나온 길이가 다른 feature sequence를 고정된 visual token 64개로 바꾼다. 학습 가능한 latent query 64개가 image feature를 cross-attention으로 읽는다.[^3]

이 visual token을 text token 앞에 그대로 붙이지는 않는다. Flamingo는 frozen LM block 사이에 새 `GATED XATTN-DENSE` block을 넣는다. Language feature가 Query가 되고 visual token이 Key와 Value가 되어, text가 필요한 이미지 정보를 읽는다.

새 cross-attention의 residual update는 $y=x+\tanh(\alpha)f(x,v)$로 나타낼 수 있다. $x$는 기존 language feature, $v$는 visual token, $f$는 새로 넣은 cross-attention block의 출력이다. 학습 가능한 scalar $\alpha$는 0에서 시작하므로 초기에는 $\tanh(0)=0$이다. 새 block을 넣은 직후에도 frozen LM의 출력이 바뀌지 않아 학습을 안정적으로 시작한다.[^3]

```python
visual_update, _ = cross_attention(
    query=language_features,
    key=visual_tokens,
    value=visual_tokens,
)
output = language_features + torch.tanh(alpha) * visual_update
```

Flamingo는 image와 text가 번갈아 나오는 입력을 처리하도록 설계됐다. 사진 한 장과 질문 하나를 넘어서, 예시 이미지와 답을 몇 쌍 보여준 뒤 새 이미지에 답하는 multimodal in-context learning에 어울리는 구조다.

## 5. 세 connector를 한 표에서 비교한다

| 구조 | Image feature를 바꾸는 법 | LLM이 이미지를 읽는 법 | 원 논문의 주요 학습 부분 | 특징 |
| --- | --- | --- | --- | --- |
| LLaVA | 각 token을 linear projection | Visual token과 text token을 한 sequence로 입력 | 1단계 projection, 2단계 projection과 LLM | 단순하고 token 수를 보존 |
| BLIP-2 | 학습 query 32개로 cross-attention | Projected query를 soft visual prompt로 입력 | Q-Former와 projection | 고정된 작은 bottleneck |
| Flamingo | Perceiver latent 64개로 resampling | LM layer 사이의 gated cross-attention | Resampler와 새 gated block | Interleaved image-text 입력 |

세 논문은 vision encoder와 LLM을 이미 학습된 model에서 가져와 활용한다. 하지만 어느 부분을 학습하는지는 다르다. LLaVA는 두 번째 단계에서 LLM도 학습하고, BLIP-2는 Q-Former를 통해 frozen LLM에 맞춘다. Flamingo는 LM block을 고정한 채 사이에 새 cross-attention block을 넣는다.

어느 구조가 항상 낫다고 정할 수는 없다. Visual token을 줄이면 LLM 쪽 비용이 작아지지만 connector가 무엇을 남길지 골라야 한다. Token을 많이 보존하면 세밀한 정보를 전달할 여지가 커지는 대신 긴 sequence를 처리해야 한다.

## 6. 같은 입력으로 tensor shape를 비교한다

![세 VLM connector의 visual token 수와 Flamingo gate 비교](/notes/tutorial/llm_lecture/images/w19_connector_token_comparison.png)

*그림 4. 같은 256개 vision token을 linear projector, 32-query Q-Former 형태, 64-latent Perceiver Resampler 형태로 처리한 shape 실습. 오른쪽은 Flamingo의 $\tanh(\alpha)$ gate다. macOS CPU, PyTorch 2.13.0, Matplotlib 3.11.1에서 직접 실행했다.[^5]*

| Connector | 입력 shape | 출력 shape | LLM에 닿는 visual token | BF16 visual activation 추정 | LLM 쪽 attention score proxy |
| --- | --- | --- | ---: | ---: | ---: |
| LLaVA projector | `[1, 256, 1024]` | `[1, 256, 4096]` | 256 | 2.00MiB | 76,176 |
| BLIP-2 형태 | `[1, 256, 64]` | `[1, 32, 64]` | 32 | 0.25MiB | 2,704 |
| Flamingo 형태 | `[1, 256, 64]` | `[1, 64, 64]` | 64 | 0.50MiB | 1,680 |

BF16 activation 추정은 visual token을 LLM hidden dimension 4,096으로 바꿔 한 벌만 저장한다고 계산한 값이다. LLaVA는 $256 \times 4096 \times 2$ byte로 2.00MiB, BLIP-2 형태는 0.25MiB, Flamingo 형태는 0.50MiB다.

Attention proxy에는 text token 20개를 사용했다. Visual token을 앞에 붙이는 LLaVA와 BLIP-2 형태는 각각 $(256+20)^2=76{,}176$, $(32+20)^2=2{,}704$로 계산했다. Flamingo 형태는 text self-attention $20^2$과 visual cross-attention $20 \times 64$를 더해 1,680이다.

이 숫자는 connector 선택에 따른 LLM 쪽 shape 차이를 보기 위한 계산이다. Vision encoder, Q-Former, Perceiver Resampler 자체의 계산과 layer 수, attention head 수, KV cache는 포함하지 않았다. 서로 다른 전체 model의 실제 속도나 memory를 나타내지 않는다.

실습의 Q-Former와 Perceiver는 `nn.MultiheadAttention` 하나로 만든 작은 모형이다.[^4] 논문의 전체 block을 재현하지 않는다. 그래도 query 길이가 출력 token 수를 정한다는 점은 확인된다. Flamingo 형태에서 $\alpha=0$일 때 residual 출력과 원래 language feature의 최대 차이는 정확히 0이었다.

## 7. Frozen과 trainable을 먼저 표시한다

Architecture 그림을 볼 때는 tensor shape만큼 weight update 범위가 중요하다.

```text
[F] frozen: forward에는 참여하지만 gradient로 weight를 바꾸지 않음
[T] trainable: loss의 gradient를 받아 optimizer가 weight를 바꿈
```

| 학습 단계 | Vision encoder | Connector | LLM |
| --- | --- | --- | --- |
| LLaVA stage 1 | `[F]` | `[T]` projection | `[F]` |
| LLaVA stage 2 | `[F]` | `[T]` projection | `[T]` |
| BLIP-2 pre-training | `[F]` | `[T]` Q-Former와 projection | `[F]` |
| Flamingo training | `[F]` | `[T]` Resampler와 gated block | `[F]` |

Frozen weight도 forward와 backward 경로에서 activation이나 입력 gradient가 필요할 수 있다. `requires_grad=False`만 보고 전체 GPU memory가 거의 들지 않는다고 판단하면 안 된다. 실제 peak memory는 dtype, batch, token 수, gradient checkpointing, optimizer state를 함께 측정한다.

## 8. 자주 생기는 실수

| 실수 | 생기는 문제 | 확인 방법 |
| --- | --- | --- |
| Token 수와 hidden dimension을 바꿔 읽음 | Projection과 resampling의 역할을 혼동함 | `[B, N, D]`에서 `N`, `D`를 따로 기록 |
| Vision output을 LLM 차원에 맞추지 않음 | Embedding을 합칠 때 shape 오류 발생 | Connector 전후 마지막 차원 출력 |
| Visual token의 위치와 mask를 빠뜨림 | LLM이 image token을 읽지 못하거나 엉뚱한 위치를 봄 | 최종 `input_ids`, embedding, attention mask 확인 |
| Q-Former를 평균 pooling으로 설명함 | 학습 query와 cross-attention의 역할을 놓침 | Query, Key, Value shape를 따로 출력 |
| Flamingo gate를 처음부터 1로 둠 | 새 random layer가 frozen LM 출력을 크게 바꿈 | `alpha` 초기값과 `tanh(alpha)` 확인 |
| Frozen module을 optimizer에 넣음 | 불필요한 gradient와 optimizer state가 생김 | Trainable parameter 이름과 수를 출력 |
| Token proxy를 실제 latency로 보고함 | Connector 계산과 hardware 차이를 무시함 | 같은 장치에서 end-to-end benchmark 실행 |

Model 이름만 보고 connector를 단정하지 않는다. 같은 계열도 version에 따라 projector layer 수, image resolution, token merging, 학습 범위가 바뀐다. 사용한 checkpoint의 config와 구현을 함께 기록한다.

## 확인 문제

1. Vision encoder 출력의 hidden dimension을 LLM word embedding 차원에 맞춰야 하는 이유는 무엇인가?
2. `[1, 256, 1024]`에서 256과 1024는 각각 무엇을 뜻하는가?
3. Linear projector가 256개 visual token에 적용되면 출력 token 수도 256개인 이유는 무엇인가?
4. Q-Former의 32개 query는 257개 image feature에서 어떤 역할을 하는가?
5. Q-Former가 단순 average pooling과 다른 점을 Query, Key, Value로 설명해보자.
6. Flamingo가 visual token을 text 앞에 붙이지 않고 cross-attention으로 읽는 방식은 interleaved 입력에 왜 알맞은가?
7. Flamingo에서 $\alpha=0$으로 시작하면 새 cross-attention block이 초기 출력에 어떤 영향을 주는가?
8. Visual token 수를 줄였는데도 전체 VLM이 반드시 빨라진다고 말할 수 없는 이유는 무엇인가?
9. Frozen weight가 많아도 GPU memory가 0에 가까워지지 않는 이유는 무엇인가?

## 완료 체크

- [x] Vision encoder, connector, LLM의 역할을 하나의 흐름으로 연결했다.
- [x] LLaVA 원 논문의 linear projection과 두 학습 단계를 확인했다.
- [x] BLIP-2의 32-query Q-Former와 두 pre-training 단계를 정리했다.
- [x] Flamingo의 64-token Resampler와 gated cross-attention을 설명했다.
- [x] 세 구조의 tensor shape와 visual token 수를 코드로 비교했다.
- [x] Frozen module과 trainable module을 표에 표시했다.
- [x] 교육용 attention proxy를 실제 latency와 구분했다.

---

[^1]: Liu, H. et al. (2023). [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485). Figure 1, §4.1의 linear projection과 §4.2의 두 학습 단계를 참고했다.
[^2]: Li, J. et al. (2023). [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597). Figure 1, §3.1의 32개 query와 Q-Former, §3.2-§3.3의 두 pre-training 단계를 참고했다.
[^3]: Alayrac, J.-B. et al. (2022). [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198). Figure 3, §2.1의 64개 visual output, §2.2의 gated cross-attention을 참고했다.
[^4]: PyTorch. [`torch.nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html). `batch_first=True`의 Query, Key, Value shape를 참고했다. 확인일: 2026-08-04.
[^5]: 직접 실행한 `llm_lecture/week19_vlm_connector_demo.py`의 결과다. LLaVA linear layer는 실제 1,024→4,096 projection을 실행했다. Q-Former와 Perceiver Resampler는 hidden size 64인 작은 cross-attention 모형이며 원 논문의 전체 model을 재현하지 않는다. 원본 PDF, crop 중간 파일, CSV와 JSON은 Git에서 제외했다. 실행일: 2026-08-04.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
원본/윤문본: 12,703자 / 12,417자, 변경률 2.25%
카테고리별 탐지/수정: A-10 3→1, A-18 2→0, C-11 1→0, D-1 0→0, H-1 1→0
정량 점검: humanize-korean metrics v2.0 risk band low
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 논문 수치, tensor shape, 직접 실행 결과를 그대로 보존함
주요 변경: 세 connector의 차원을 같은 기호로 맞추고 실제 구조와 작은 shape 실습을 명확히 구분함
-->
