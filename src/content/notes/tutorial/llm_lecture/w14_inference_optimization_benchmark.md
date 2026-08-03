---
title: "14주차. 추론 최적화와 성능 측정"
description: "Prefill과 decode를 나누고 TTFT, TPOT, throughput, goodput으로 추론 서버의 속도를 공정하게 측정한다."
tags:
  - LLM
  - inference
  - vLLM
  - FlashAttention
  - benchmark
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 16주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

13주차에는 weight와 KV cache가 GPU 메모리를 어떻게 나눠 쓰는지 계산했다. 메모리에 들어간다고 사용자가 빠르게 답을 받는 것은 아니다. 첫 token이 늦게 나오는 문제와 token 사이가 끊기는 문제, 많은 요청을 처리하지 못하는 문제는 서로 다른 숫자로 측정해야 한다.[^1][^2]

## 이번 주에 배울 것

- Prefill과 decode가 하는 일
- TTFT, TPOT, ITL, end-to-end latency의 차이
- Throughput과 goodput을 구분하는 이유
- FlashAttention이 GPU의 memory IO를 줄이는 원리
- Continuous batching이 동시 요청을 묶는 방식
- Prompt 길이, output 길이, concurrency를 고정하는 benchmark 설계
- Chunked prefill의 TTFT·TPOT trade-off

선수 지식은 1주차의 attention, 13주차의 GPU memory와 KV cache다.

!!! note "첫 숟가락과 식사 속도"

    TTFT는 주문하고 첫 숟가락을 받기까지 걸린 시간이다. TPOT는 그다음 숟가락이 오는 간격에 가깝다. 첫 접시는 빨리 왔지만 이후 음식이 자꾸 끊길 수도 있다. 둘을 한 숫자로 합치면 이런 차이가 사라진다.

## 1. 한 응답에는 prefill과 decode가 있다

| 단계 | 입력 | 한 번에 처리하는 token | 주된 병목 |
| --- | --- | ---: | --- |
| Prefill | User prompt 전체 | 여러 prompt token | 큰 matrix multiplication, compute |
| Decode | 지금까지의 context와 KV cache | Sequence마다 새 token 1개 | Weight·KV 읽기, memory bandwidth |

Prefill은 prompt의 모든 token을 처리해 첫 KV cache를 채운다. 긴 prompt일수록 할 일이 늘고 첫 token이 늦어진다. Decode는 한 step마다 다음 token 하나를 만든다. 같은 weight를 반복해서 읽으므로 작은 batch에서는 GPU 계산기보다 메모리 통로가 먼저 한계에 닿기 쉽다.[^3]

```text
request arrives
    -> queue
    -> prefill(prompt tokens)
    -> first output token
    -> decode one token at a time
    -> response ends
```

## 2. 빠르다는 말을 네 숫자로 나눈다

| Metric | 계산 | 사용자가 느끼는 것 |
| --- | --- | --- |
| TTFT | 첫 token 시각 - 요청 시각 | 답이 시작될 때까지의 기다림 |
| ITL | 이웃한 output token 사이 시간 | Streaming이 매끄러운가 |
| TPOT | `(E2E - TTFT) / (output_tokens - 1)` | 첫 token 뒤 평균 생성 간격 |
| E2E latency | 마지막 token 시각 - 요청 시각 | 전체 답이 끝날 때까지의 시간 |

Output token이 하나뿐이면 분모가 0이므로 TPOT를 따로 다뤄야 한다. 현재 vLLM benchmark 구현도 output length가 1보다 클 때 TPOT를 계산한다.[^2]

Throughput은 일정 시간에 끝낸 request 수나 만든 token 수다. 예를 들어 10초 동안 output token 5000개를 만들었다면 output throughput은 500 token/s다.

Goodput은 정해둔 SLO를 통과한 request만 센다. 서버가 100 request/s를 끝냈어도 TTFT 500ms와 TPOT 20ms를 함께 만족한 요청이 초당 30개라면 goodput은 30 request/s다.[^2]

## 3. 평균 하나보다 분포를 본다

요청 100개의 TTFT가 대부분 100ms지만 5개가 10초라면 평균만으로 긴 기다림을 숨길 수 있다.

| 값 | 의미 |
| --- | --- |
| P50 | 절반의 요청이 이 값 이하 |
| P95 | 95%의 요청이 이 값 이하 |
| P99 | 100개 중 느린 쪽 1개 수준 |

대화형 서비스라면 TTFT P95와 TPOT P95를 함께 본다. Batch 작업이라면 전체 throughput과 완료 시간을 더 중요하게 둘 수 있다. 어떤 metric을 우선할지는 benchmark 전에 정한다.

## 4. FlashAttention은 계산식을 근사하지 않는다

![GPU memory hierarchy와 FlashAttention의 tiling](/notes/tutorial/llm_lecture/images/w14_flashattention_io.png)

*그림 1. FlashAttention은 Q, K, V를 block으로 나눠 빠른 SRAM에 올리고, 큰 attention matrix 전체를 HBM에 쓰지 않는다. 출처: Dao et al. (2022), Figure 1에서 발췌.[^4]*

GPU 안에서도 메모리마다 크기와 속도가 다르다. HBM은 크지만 on-chip SRAM보다 느리다. Standard attention이 큰 $N \times N$ attention matrix를 HBM에 쓰고 다시 읽으면 계산기보다 데이터 이동에 많은 시간을 쓴다.[^4]

FlashAttention은 작은 block을 SRAM에 올려 attention을 나눠 계산한다. 큰 중간 matrix 전체를 HBM에 만들지 않는다. 이름에 Flash가 붙었지만 답을 대충 계산하는 근사 attention은 아니다. 같은 exact attention 결과를 data movement가 적은 순서로 계산한다.[^4]

!!! warning "논문의 7.6배를 내 GPU의 예상 속도로 쓰지 않는다"

    그림 오른쪽 수치는 논문의 GPT-2 attention 연산과 당시 hardware·software 조건에서 나온 값이다. Model, dtype, sequence, GPU, backend가 달라지면 속도도 달라진다. 자신의 workload를 다시 재야 한다.

## 5. Continuous batching은 끝난 자리에 새 요청을 넣는다

Static batch는 가장 긴 sequence가 끝날 때까지 batch 모양을 유지하기 쉽다. 먼저 끝난 자리는 놀게 된다. Continuous batching은 decode step 사이에서 끝난 요청을 빼고 기다리던 요청을 넣는다. 길이가 다른 요청이 섞인 온라인 서비스에 유리하다.[^5]

```text
step 1: [A decode] [B decode] [C prefill]
step 2: [A decode] [B finished] [C decode]
step 3: [A decode] [D prefill] [C decode]
```

Concurrency를 높이면 GPU가 한 번에 처리할 일이 늘어 throughput이 오를 수 있다. 한계를 넘으면 queue와 KV cache 압력이 커져 TTFT P99가 나빠지고 preemption이 생긴다. 최대 throughput 지점과 좋은 사용자 경험을 주는 지점이 같지 않을 수 있다.[^3]

## 6. 먼저 같은 시험 조건을 만든다

다음 grid를 Transformers와 vLLM에 똑같이 적용한다.

| 축 | 값 예시 |
| --- | --- |
| Model | 같은 checkpoint와 revision |
| Dtype | BF16 |
| Prompt length | 128, 2048, 8192 token |
| Output length | 32, 256 token |
| Concurrency | 1, 4, 16, 32 |
| Decoding | Greedy, 같은 stop 조건 |
| 반복 | Warm-up 뒤 각 조건 3회 이상 |
| 기록 | TTFT·TPOT·E2E P50/P95/P99, token/s, peak memory |

Random text를 tokenizer에 넣었다고 정확히 2048 token이 되는 것은 아니다. Token ID를 직접 만들거나 tokenize한 뒤 길이를 확인한다. 두 engine의 chat template과 special token 처리도 맞춘다.

Transformers의 단순한 `generate()` loop는 request를 하나씩 처리하는 baseline으로 둔다. vLLM은 server를 띄우고 같은 prompt를 streaming으로 보낸다. Client tokenization 시간과 network 시간을 포함할지 보고서에 적는다.

## 7. GPU가 없는 환경에서는 실행 여부를 숨기지 않는다

이번 환경은 CUDA GPU와 vLLM server가 없어 실제 engine benchmark를 실행하지 않았다. 대신 metric 관계를 확인하는 deterministic scheduler model을 실행했다.[^6] 아래 숫자는 software 성능이 아니라 benchmark 표를 읽는 연습용 synthetic 값이다.

![Sequential generation과 continuous batching의 synthetic 비교, chunked prefill trade-off](/notes/tutorial/llm_lecture/images/w14_latency_throughput_simulation.png)

*그림 2. 왼쪽은 2048 input·64 output token workload의 교육용 latency-throughput model이다. 가운데와 오른쪽은 8K prefill과 decode 요청을 섞었다고 가정한 chunk budget sweep이다. 실제 Transformers·vLLM 측정값이 아니다. macOS CPU, Python 3.12, Matplotlib 3.10.8에서 직접 계산했다.[^6]*

| Concurrency | Sequential output tok/s | Continuous batching tok/s | Continuous TTFT | Continuous TPOT |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 42.4 | 43.4 | 89.7ms | 22.0ms |
| 4 | 42.4 | 234.0 | 115.5ms | 12.0ms |
| 16 | 42.4 | 418.0 | 221.3ms | 8.2ms |
| 32 | 42.4 | 510.0 | 416.7ms | 8.5ms |

Sequential baseline은 한 요청씩 처리한다는 식을 써서 concurrency가 올라가도 throughput이 42.4 token/s에 머문다. Continuous batching 식은 batch 효율과 queue 비용을 함께 넣어 concurrency 32에서 510.0 token/s가 됐다. 이 숫자로 vLLM이 Transformers보다 몇 배 빠르다고 결론 내리면 안 된다. 실제 측정 전에는 구현이 아니라 가정의 차이일 뿐이다.

## 8. 실제 GPU에서는 vLLM benchmark CLI로 바꾼다

현재 vLLM은 `vllm bench serve`로 online serving throughput을 측정한다.[^1] 설치한 version에 따라 option이 바뀔 수 있으므로 `vllm bench serve --help`를 먼저 확인한다.

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192
```

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 64 \
  --num-prompts 200 \
  --request-rate 8 \
  --max-concurrency 16 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result
```

실행할 때는 두 terminal의 로그를 모두 남긴다. Server log에는 preemption과 KV cache 상태가 있고 client 결과에는 latency 분포가 있다. Model revision, vLLM commit, CUDA, driver, GPU 이름도 결과 JSON 옆에 적는다.

## 9. Chunked prefill은 큰 prompt를 잘라 decode와 섞는다

긴 prefill 하나가 GPU를 오래 잡으면 이미 답을 생성 중인 요청의 token 사이가 벌어진다. Chunked prefill은 긴 prompt를 작은 조각으로 나눠 decode와 같은 batch에 배치한다. 현재 vLLM V1은 가능한 경우 이 기능을 기본으로 켜고 decode를 먼저 scheduling한다.[^3]

공식 문서는 `max_num_batched_tokens`가 작으면 decode 사이에 들어오는 prefill 양이 줄어 ITL이 좋아지고, 값이 크면 prefill을 한 번에 더 많이 처리해 TTFT가 좋아진다고 설명한다.[^3]

| Token budget | Prefill 조각 수 | Synthetic TTFT | Synthetic TPOT | Output tok/s | SLO 통과 |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 2048 | 4 | 755.0ms | 9.9ms | 350 | X |
| 4096 | 2 | 465.0ms | 12.7ms | 410 | O |
| 8192 | 1 | 320.0ms | 18.5ms | 470 | O |
| 16384 | 1 | 310.0ms | 29.9ms | 530 | X |

여기서는 TTFT 500ms 이하와 TPOT 20ms 이하를 동시에 만족해야 통과하도록 했다. 16K는 throughput이 가장 높지만 TPOT SLO를 어겼다. 이처럼 가장 높은 token/s가 운영에 가장 알맞은 설정은 아니다.

실제 sweep에서는 server를 설정마다 다시 시작하고 warm-up한다. Compile cache가 첫 실행과 다음 실행을 다르게 만들 수 있어 시작 순서도 기록한다.

## 10. Benchmark가 쉽게 틀리는 지점

| 실수 | 왜 문제인가 | 고치는 법 |
| --- | --- | --- |
| Warm-up 없음 | Compile과 cache 준비 시간을 섞음 | 별도 warm-up 뒤 측정 |
| Output 길이가 다름 | 짧게 끝난 model이 빨라 보임 | 생성 token 수를 고정하거나 함께 공개 |
| Non-streaming으로 TTFT 측정 | 첫 token 도착 시각을 알 수 없음 | Streaming event timestamp 기록 |
| 평균만 공개 | 긴 tail latency가 숨음 | P50·P95·P99 공개 |
| Concurrency만 쓰고 arrival rate 누락 | 부하 형태를 재현하지 못함 | Request rate와 burstiness 기록 |
| Engine마다 prompt가 다름 | Token 수와 template 비용이 달라짐 | Rendered prompt와 token 수 저장 |
| 실패 요청 제외 | 빠른 대신 오류가 많은 설정이 유리함 | Success rate와 goodput 공개 |

## 확인 문제

1. Prefill과 decode 중 긴 prompt가 TTFT에 직접 영향을 주는 단계는 무엇인가?
2. TTFT가 낮고 TPOT가 높다면 사용자는 응답을 어떻게 느낄까?
3. Throughput 100 request/s와 goodput 30 request/s가 함께 나올 수 있는 이유는 무엇인가?
4. FlashAttention이 근사 attention이 아닌데도 빨라지는 핵심 이유는 무엇인가?
5. `max_num_batched_tokens`를 크게 했을 때 TTFT와 TPOT는 어떤 방향으로 움직일 수 있는가?
6. Synthetic 결과를 실제 vLLM 성능 순위로 쓰면 안 되는 이유는 무엇인가?

## 완료 체크

- [x] Prefill과 decode의 GPU 사용 차이를 설명했다.
- [x] Prompt 길이, output 길이, concurrency를 바꾸는 benchmark를 설계했다.
- [x] Transformers와 vLLM의 공통 측정표와 실행 명령을 만들고 CUDA 실측은 `미실행`으로 표시했다.
- [x] 교육용 chunked prefill sweep에서 TTFT와 TPOT trade-off를 확인했다.
- [x] 결과물로 `Latency-throughput benchmark` 설계와 synthetic baseline을 완성했다.

---

[^1]: vLLM. [vLLM CLI Guide](https://docs.vllm.ai/en/latest/cli/). `vllm bench latency`, `serve`, `throughput`의 현재 subcommand와 online serving benchmark 예시를 참고했다. 확인일: 2026-08-03.
[^2]: vLLM. [Online serving benchmark implementation](https://docs.vllm.ai/en/stable/api/vllm/benchmarks/serve/). TTFT, TPOT, ITL, E2E latency, throughput과 SLO 기반 goodput 계산을 참고했다. 확인일: 2026-08-03.
[^3]: vLLM. [Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/). Prefill의 compute-bound 성격, decode의 memory-bound 성격, V1 chunked prefill scheduling과 `max_num_batched_tokens` trade-off를 참고했다. 확인일: 2026-08-03.
[^4]: Dao, T. et al. (2022). [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). Figure 1, GPU memory hierarchy와 tiling으로 HBM 접근을 줄이는 exact attention을 참고했다.
[^5]: Kwon, W. et al. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). Iteration-level scheduling, continuous batching과 vLLM serving 구조를 참고했다.
[^6]: 직접 실행한 `llm_lecture/week14_inference_optimization_demo.py`의 결과다. 2048 input·64 output token, concurrency 1·4·16·32의 deterministic scheduler model과 8K prompt의 chunk budget 2048·4096·8192·16384를 계산했다. 실제 model, CUDA kernel, Transformers, vLLM server를 실행하지 않은 synthetic 값이며 원본 CSV는 Git에서 제외했다. 실행일: 2026-08-03.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
원본/윤문본: 작성 단계에서 규칙 적용 / 후처리 변경률 0.0%
카테고리별 탐지/수정: A-10 0→0, C-11 0→0, D-1 0→0, H-1 0→0, I-1 0→0
정량 점검: humanize-korean metrics v2.0 risk band low
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 실제 측정과 synthetic 값을 섞지 않음
주요 확인: TTFT·TPOT·throughput을 나눠 설명하고 CUDA 미실행 상태를 본문·표·캡션에 표시함
-->
