---
title: "13주차. 추론 메모리와 KV cache"
description: "Model weight와 KV cache 크기를 직접 계산하고 context, batch, GQA가 추론 메모리를 어떻게 바꾸는지 살펴본다."
tags:
  - LLM
  - inference
  - KV cache
  - GQA
  - PagedAttention
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 16주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

12주차까지는 model이 답을 잘 만드는지 평가했다. 이제 같은 model을 실제로 실행할 때 필요한 메모리를 살펴본다. 추론 메모리는 model weight만 더해서 구하지 않는다. 대화가 길어지고 동시에 처리하는 요청이 늘면 KV cache가 weight보다 커지기도 한다.[^1][^2]

## 이번 주에 배울 것

- 학습 메모리와 추론 메모리의 차이
- FP32, BF16·FP16, INT8, INT4 weight 저장량 계산
- KV cache가 이전 token의 K와 V를 저장하는 이유
- Context, batch, layer, KV head로 cache 크기를 구하는 식
- MHA, GQA, MQA의 KV cache 차이
- PagedAttention이 메모리 낭비를 줄이는 원리
- 계산값과 실제 GPU 사용량이 다른 까닭

선수 지식은 1주차의 causal language model과 attention, 4주차의 dtype과 양자화다.

!!! note "책과 메모지"

    Model weight는 이미 배운 지식이 적힌 책이다. KV cache는 지금 대화의 앞부분을 매번 다시 계산하지 않으려고 적어두는 메모지다. 책을 INT4로 줄여도 대화가 길고 사용자가 많으면 메모지가 책보다 커진다.

## 1. 학습할 때와 답할 때 들고 있는 것이 다르다

| 메모리 항목 | 학습 | 추론 | 크기를 키우는 값 |
| --- | :---: | :---: | --- |
| Model weight | O | O | Parameter 수, dtype |
| Gradient | O | X | Parameter 수, gradient dtype |
| Optimizer state | O | X | Optimizer와 state dtype |
| Backward용 activation | O | X | Batch, sequence, layer, hidden size |
| KV cache | 보통 사용 안 함 | O | Batch, 현재 token 수, layer, KV head |
| Temporary buffer | O | O | Kernel, tensor shape, backend |

학습은 정답 방향으로 weight를 고쳐야 하므로 gradient와 optimizer state가 필요하다. 추론은 weight를 고치지 않아 이 둘이 없다. 대신 autoregressive generation에서 이전 token의 K와 V를 보관한다. Hugging Face 문서도 KV cache는 추론에서만 쓰고 학습에서는 예상하지 못한 오류를 만들 수 있다고 설명한다.[^1][^3]

## 2. Weight 저장량은 parameter 수와 bit 수로 계산한다

Parameter가 $P$개이고 하나를 $b$bit로 저장한다면 이상적인 weight 크기는 $P \times b / 8$byte다. GiB로 바꾸려면 다시 $1024^3$으로 나눈다.

```python
def weight_gib(parameters: int, bits: int) -> float:
    return parameters * bits / 8 / 1024**3
```

가상의 7B model을 계산했다.[^4] 여기서 7B는 정확히 70억 parameter라는 가정이다.

| Weight dtype | Bit/parameter | 이상적인 저장량 |
| --- | ---: | ---: |
| FP32 | 32 | 26.08GiB |
| BF16 | 16 | 13.04GiB |
| FP16 | 16 | 13.04GiB |
| INT8 | 8 | 6.52GiB |
| INT4 | 4 | 3.26GiB |

!!! warning "INT4 model이 반드시 3.26GiB 파일은 아니다"

    실제 양자화 model에는 scale, zero point, group metadata와 padding이 붙는다. 일부 layer를 BF16으로 남기기도 한다. 표는 weight 값만 빈틈없이 저장했을 때의 하한에 가깝다.

## 3. KV cache는 이미 계산한 K와 V를 기억한다

Token을 하나 만들 때 attention은 지금까지 나온 token을 읽는다. Cache가 없으면 다음 token을 만들 때마다 과거 token의 K와 V를 다시 계산한다. Cache를 쓰면 과거 K와 V는 불러오고 새 token의 값만 추가한다.[^1]

한 layer의 K와 V tensor shape은 보통 `[batch, kv_heads, sequence, head_dim]`으로 생각하면 된다. Layer마다 K 하나와 V 하나가 있으므로 2를 곱한다.

KV cache byte 수는 $2 \times L \times B \times S \times H_{kv} \times D_{head} \times d$다.

| 기호 | 뜻 |
| --- | --- |
| $L$ | Transformer layer 수 |
| $B$ | 동시에 cache에 놓인 sequence 수 |
| $S$ | Sequence의 현재 token 수 |
| $H_{kv}$ | K·V head 수 |
| $D_{head}$ | Head 하나의 차원 |
| $d$ | 원소 하나의 byte 수, BF16·FP16은 2 |

```python
def kv_cache_gib(layers, batch, sequence, kv_heads, head_dim, bytes_per_element):
    size_bytes = (
        2 * layers * batch * sequence
        * kv_heads * head_dim * bytes_per_element
    )
    return size_bytes / 1024**3
```

## 4. Context와 batch를 늘리면 곱셈으로 커진다

32개 layer, hidden size 4096, query head 32개, KV head 8개의 GQA model을 가정했다. Head 하나의 차원은 $4096 / 32 = 128$이다. KV dtype은 BF16으로 두었다.[^4]

![7B weight 저장량과 BF16 GQA KV cache 계산 결과](/notes/tutorial/llm_lecture/images/w13_weight_kv_memory.png)

*그림 1. 왼쪽은 정확히 70억 parameter라는 가정에서 구한 weight 저장량이다. 가운데는 BF16 GQA KV cache, 오른쪽은 batch 8에서 weight와 cache를 더한 값이다. 공식이 만든 추정값이며 GPU에 model을 올려 잰 값이 아니다. macOS CPU, Python 3.12, Matplotlib 3.10.8에서 직접 계산했다.[^4]*

| Batch | Context 2K | Context 8K | Context 32K |
| ---: | ---: | ---: | ---: |
| 1 | 0.25GiB | 1.00GiB | 4.00GiB |
| 8 | 2.00GiB | 8.00GiB | 32.00GiB |
| 32 | 8.00GiB | 32.00GiB | 128.00GiB |

Context를 2K에서 8K로 네 배 늘리면 cache도 네 배가 된다. Batch 1을 8로 늘려도 여덟 배다. BF16 weight 13.04GiB는 요청 수와 상관없이 한 번 올라가지만, 요청별 KV cache는 각 대화 길이에 맞춰 늘어난다.

오른쪽 그래프의 24GiB 선은 model 실행 가능 여부를 판정하는 선이 아니다. Weight와 KV cache 외의 메모리를 빼놓았기 때문이다. 계산값이 23GiB여도 실제 실행은 OOM일 수 있다.

## 5. MHA, GQA, MQA는 KV head 수가 다르다

Query head가 32개일 때 세 attention 방식을 비교해보자.[^5]

| 방식 | KV head 예시 | Batch 8·8K의 BF16 KV cache | 생각하는 법 |
| --- | ---: | ---: | --- |
| MHA | 32 | 32GiB | Query마다 K·V head가 있음 |
| GQA | 8 | 8GiB | Query 여러 개가 K·V head를 공유 |
| MQA | 1 | 1GiB | 모든 query가 K·V head 하나를 공유 |

GQA는 MHA와 MQA의 중간이다. Ainslie et al.은 query head 수보다 많지 않고 1보다는 많은 KV head를 쓰는 방식을 GQA로 설명했다.[^5] 이미 학습된 model의 `num_key_value_heads`를 임의로 줄이면 안 된다. Architecture와 weight가 그 구조에 맞게 학습돼 있어야 한다.

## 6. 미리 크게 잡으면 빈 공간이 생긴다

![기존 연속 KV cache 할당에서 생기는 reserved, internal, external fragmentation](/notes/tutorial/llm_lecture/images/w13_pagedattention_fragmentation.png)

*그림 2. 요청마다 최대 길이의 연속 공간을 잡는 방식에서 예약 공간, 내부 단편화, 외부 단편화가 생기는 예시. 출처: Kwon et al. (2023), Figure 3에서 발췌.[^2]*

응답 길이는 시작 전에 정확히 알기 어렵다. 최대 2048 token을 만든다고 공간을 먼저 잡았는데 10 token에서 끝나면 대부분이 빈자리로 남는다. 작은 빈 공간이 여기저기 흩어져 큰 요청을 넣지 못하는 상황도 생긴다.

PagedAttention은 KV cache를 고정 크기 block으로 나누고 logical block과 실제 GPU의 physical block을 table로 연결한다. 운영체제가 virtual memory page를 다루는 생각과 비슷하다. 요청의 token이 늘어날 때 block을 붙이고, 요청이 끝나면 block을 다른 요청에 돌려준다.[^2]

| 연속 할당 | PagedAttention |
| --- | --- |
| 최대 길이에 맞춰 큰 연속 공간을 예약 | 필요한 block을 조금씩 추가 |
| 흩어진 빈 공간을 쓰기 어려움 | Physical block이 붙어 있을 필요 없음 |
| Shared prefix 복제가 생기기 쉬움 | Block 단위 공유와 copy-on-write 가능 |

PagedAttention이 KV cache 자체의 수학적 크기를 없애는 것은 아니다. 같은 token의 K와 V는 여전히 저장한다. 주된 효과는 예약과 단편화, 중복 복사에서 생기는 낭비를 줄이는 데 있다.

## 7. Transformers cache 방식에도 교환 조건이 있다

현재 Transformers는 DynamicCache, StaticCache, QuantizedCache 등을 제공한다.[^6]

| Cache | 특징 | 대가 |
| --- | --- | --- |
| DynamicCache | Token이 생길 때마다 크기가 자람 | Shape이 바뀌어 compile 최적화가 어려움 |
| StaticCache | 최대 길이를 미리 할당 | 사용하지 않는 공간과 계산이 생김 |
| Offloaded cache | 일부 layer cache를 CPU로 이동 | 전송 때문에 throughput이 낮아질 수 있음 |
| QuantizedCache | KV를 더 낮은 bit로 저장 | 짧은 context에서는 변환 비용이 더 클 수 있음 |

메모리를 줄였다는 말만으로는 최적화가 끝나지 않는다. 줄인 대신 TTFT나 TPOT가 얼마나 달라졌는지 14주차에서 함께 측정한다.

## 8. 계산값과 실제 GPU 사용량은 왜 다를까

| 차이를 만드는 항목 | 예시 |
| --- | --- |
| Temporary buffer | Attention, sampling, matrix multiplication workspace |
| Activation | 현재 forward의 hidden state와 logits |
| Memory allocator | 예약 pool과 block rounding |
| Runtime | CUDA context, library handle, kernel code |
| Graph·compile | CUDA graph pool, compiled artifact |
| Quantization | Scale, metadata, 일부 고정밀 layer |
| Serving policy | KV block 크기, preemption, prefix cache |

실제 측정에서는 세 값을 구분한다.

```text
theoretical_weight_gib: 공식으로 계산한 weight 저장량
theoretical_kv_gib: model config와 token 수로 계산한 KV cache
peak_gpu_allocated_gib: 실행 중 framework가 보고한 peak allocation
```

`nvidia-smi`는 process가 예약한 메모리를 보여주고, `torch.cuda.max_memory_allocated()`는 PyTorch tensor allocation을 중심으로 보여준다. 서로 다른 질문에 답하는 값이므로 표에 측정 도구까지 적는다.

## 확인 문제

1. INT4 weight가 작아져도 긴 대화 여러 개에서 OOM이 날 수 있는 이유는 무엇인가?
2. KV cache 식에서 K와 V 때문에 곱하는 숫자는 무엇인가?
3. Batch 8·context 8K의 cache가 8GiB라면 batch 16에서는 몇 GiB인가?
4. MHA, GQA, MQA 중 KV cache가 가장 작은 방식은 무엇이며 왜 그런가?
5. PagedAttention은 KV cache의 어떤 낭비를 줄이고, 무엇은 그대로 저장하는가?
6. 이론값과 `nvidia-smi` 값이 다를 때 확인할 항목을 세 가지 말해보자.

## 완료 체크

- [x] 학습 메모리와 추론 메모리의 차이를 설명했다.
- [x] FP32, BF16·FP16, INT8, INT4 weight 크기 계산기를 만들었다.
- [x] Context 2K·8K·32K와 batch 1·8·32의 KV cache를 계산했다.
- [x] 실제 GPU 값과 공식의 추정값을 구분하고 차이가 생기는 까닭을 기록했다.
- [x] 결과물로 `Weight/KV memory calculator`를 완성했다.

---

[^1]: Hugging Face. [Transformers: Caching](https://huggingface.co/docs/transformers/cache_explanation). KV cache의 K·V 재사용, tensor shape와 inference-only 주의를 참고했다. 확인일: 2026-08-03.
[^2]: Kwon, W. et al. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). Figure 3과 block 기반 KV cache 관리, 단편화 문제를 참고했다.
[^3]: Hugging Face. [Transformers: GPU memory usage](https://huggingface.co/docs/transformers/model_memory_anatomy). Weight, optimizer state, gradient, activation과 temporary tensor의 구분을 참고했다. 확인일: 2026-08-03.
[^4]: 직접 실행한 `llm_lecture/week13_inference_memory_demo.py`의 결과다. 정확히 70억 parameter, 32 layer, hidden size 4096, query head 32, KV head 8인 가상 GQA model을 계산했다. GPU allocation을 실행하지 않은 공식 기반 추정값이며 원본 CSV는 Git에서 제외했다. 실행일: 2026-08-03.
[^5]: Ainslie, J. et al. (2023). [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). MHA와 MQA 사이에서 여러 KV head를 공유하는 GQA 정의를 참고했다.
[^6]: Hugging Face. [Transformers: Cache strategies](https://huggingface.co/docs/transformers/kv_cache). Dynamic, Static, offloaded, quantized cache의 지원 범위와 trade-off를 참고했다. 확인일: 2026-08-03.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
원본/윤문본: 작성 단계에서 규칙 적용 / 후처리 변경률 0.0%
카테고리별 탐지/수정: A-10 0→0, C-11 0→0, D-1 0→0, H-1 0→0, I-1 0→0
정량 점검: humanize-korean metrics v2.0 risk band low
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 공식, dtype, 실험 수치를 그대로 보존함
주요 확인: 가상 계산과 GPU 실측을 구분하고 KV cache 식을 짧은 문장과 표로 나눔
-->
