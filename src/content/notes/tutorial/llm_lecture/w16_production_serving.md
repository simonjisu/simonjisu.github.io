---
title: "16주차. 양자화와 Production 서빙"
description: "양자화와 multi-GPU parallelism을 구분하고 latency, goodput, KV cache, 오류를 함께 보는 Production serving report를 만든다."
tags:
  - LLM
  - inference
  - quantization
  - distributed serving
  - observability
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 16주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

15주차에는 요청이 API server와 scheduler를 지나 model runner에 도착하는 흐름을 배웠다. 이제 제한된 GPU에 모델을 올리고, 여러 GPU에 일을 나누고, 느려지거나 고장 나는 순간을 찾는다. 이번 주의 목표는 한 번 실행되는 demo가 아니라 상태를 계속 관찰할 수 있는 서비스를 만드는 것이다.

## 이번 주에 배울 것

- BF16, FP8, INT8, 4-bit weight-only quantization의 차이
- GPTQ와 AWQ가 양자화 오차를 줄이는 방식
- Tensor, pipeline, data, expert parallelism의 쓰임
- Prefix caching, speculative decoding, prefill-decode 분리
- TTFT, TPOT, queue time, KV cache usage, OOM을 함께 읽는 법
- Throughput보다 SLO goodput을 먼저 보는 이유
- 재현 가능한 Production serving report 작성법

선수 지식은 13주차의 GPU memory·KV cache, 14주차의 benchmark, 15주차의 추론 서버다.

!!! note "작은 상자와 작은 자"

    양자화는 숫자를 더 작은 상자에 담는 일과 비슷하다. BF16 숫자를 4-bit로 줄이면 같은 GPU에 더 많은 weight를 담을 수 있다. 대신 눈금이 성긴 자로 재는 것처럼 오차가 생긴다. 어떤 눈금을 쓰고 오차를 어떻게 보정하는지가 GPTQ와 AWQ의 차이다.

## 1. 양자화는 무엇을 몇 bit로 줄였는지부터 말한다

`INT4 model`이라는 말만으로는 부족하다. Weight만 4-bit인지, activation도 줄였는지, KV cache dtype은 무엇인지 적어야 한다.

| 표기 예 | Weight | Activation | 쉬운 설명 |
| --- | --- | --- | --- |
| BF16 | 16-bit | 16-bit | 품질과 호환성을 확인할 기준점 |
| W8A8 INT8 | 8-bit | 8-bit | Weight와 activation을 모두 8-bit로 계산 |
| FP8 | 8-bit floating point | 주로 FP8 또는 혼합 | 지원 GPU와 kernel에서 속도 이점을 노림 |
| W4A16 | 4-bit | 16-bit | Weight 저장량을 크게 줄이고 계산 때 풀어 사용 |
| KV cache quantization | 별도 | 별도 | 긴 context와 많은 동시 요청의 cache를 줄임 |

정확히 70억 parameter의 weight만 저장한다고 가정하면 필요한 공간은 `parameter 수 × bit 수 ÷ 8`이다. GiB로 바꾸려면 다시 $1024^3$으로 나눈다.

| Format | Weight bit | 이상적인 weight 저장량 | 실제 GPU 측정 |
| --- | ---: | ---: | --- |
| BF16 | 16 | 13.04GiB | `미실행` |
| FP8 | 8 | 6.52GiB | `미실행` |
| INT8 | 8 | 6.52GiB | `미실행` |
| AWQ W4 | 4 | 3.26GiB | `미실행` |
| GPTQ W4 | 4 | 3.26GiB | `미실행` |

이는 weight만 센 이상적인 계산값이다.[^1] 실제 checkpoint에는 scale, zero point, group metadata가 들어간다. 실행 중에는 activation, KV cache, CUDA graph, temporary buffer도 필요하다. 이 계산만 보고 4-bit model의 실제 peak memory가 BF16의 정확히 4분의 1이라고 단정하면 안 된다.

## 2. GPTQ와 AWQ는 같은 INT4라는 이름 뒤에서 다르게 일한다

GPTQ는 이미 학습한 model을 한 번에 양자화하는 post-training quantization 방법이다. 작은 calibration data에서 얻은 정보를 사용하고, 한 weight를 줄이며 생긴 오차가 다음 weight에 미칠 영향을 approximate second-order 정보로 보정한다.[^2]

AWQ도 post-training weight-only quantization이다. 모든 weight가 같은 중요도를 갖는다고 보지 않는다. Activation을 관찰해 출력에 큰 영향을 주는 channel을 찾고, 그 weight가 양자화에서 덜 손상되도록 scaling한다.[^3]

![AWQ의 activation-aware scaling](/notes/tutorial/llm_lecture/images/w16_awq_activation_scaling.png)

*그림 1. 단순 반올림, 중요한 weight만 FP16으로 남기는 방법, 중요한 channel을 scaling한 AWQ를 비교한다. 출처: Lin et al. (2023), Figure 2에서 발췌.[^3]*

그림 가운데 방법은 중요한 1%를 FP16으로 남겨 perplexity를 낮추지만 mixed precision 때문에 hardware에서 효율적으로 계산하기 어렵다. AWQ는 channel 전체를 scaling한 뒤 같은 낮은 bit 형식을 유지한다. 논문의 OPT-6.7B, INT3-g128 수치는 원리를 보여주는 사례다. 다른 model과 bit 설정에 그대로 적용되는 보장값은 아니다.

| 질문 | GPTQ | AWQ |
| --- | --- | --- |
| 중요하게 보는 정보 | Weight 오차와 Hessian 근사 | Activation이 보여주는 salient channel |
| 학습 방식 | One-shot post-training quantization | Backpropagation 없는 weight-only PTQ |
| 확인할 설정 | Bit, group size, calibration, checkpoint format | Bit, group size, calibration, scaling, checkpoint format |
| 공통 주의 | Engine과 GPU가 해당 kernel을 지원하는지 확인 | 같은 평가 문제로 품질을 다시 측정 |

!!! warning "파일이 작아졌다고 서버가 빨라졌다고 말하지 않는다"

    Weight를 풀어 쓰는 dequantization 비용이나 지원되지 않는 kernel 때문에 작은 checkpoint가 더 느릴 수 있다. Batch 크기와 GPU 세대에 따라 결과도 달라진다. Peak memory, TTFT, TPOT, throughput, task quality를 같은 workload에서 함께 잰다.

## 3. 품질 평가는 12주차 시험지를 다시 쓴다

양자화 전 BF16 model과 양자화 model에 같은 prompt, chat template, decoding 설정을 쓴다. 12주차의 일반 대화, 지시 따르기, reasoning, tool calling 평가를 그대로 사용하면 압축 전후를 비교할 수 있다.

| 항목 | 왜 필요한가 |
| --- | --- |
| Task success | 답이 실제 과제를 해결했는지 확인 |
| Format accuracy | JSON-only, 길이, 필수 문구 같은 지시 확인 |
| Tool execution success | Tool 이름과 인자가 실행 가능한지 확인 |
| Perplexity | 다음 token 확률 분포의 전체 변화를 확인 |
| 사람이 읽은 표본 | 자동 채점기가 놓친 의미 오류 확인 |

실제 실험표에는 다음 열을 둔다.

```text
format, checkpoint, quantization config, engine version, GPU,
peak memory, TTFT P50/P95, TPOT P50/P95, output tokens/s,
task success, format accuracy, tool success, error rate
```

이번 환경에서는 quantized checkpoint와 CUDA kernel을 실행하지 않았다. 위 표의 이상적인 저장량은 quantization 품질이나 속도 실측값이 아니다.

## 4. 여러 GPU는 일을 나누는 기준부터 정한다

| 방법 | 무엇을 나누나 | 잘 맞는 상황 | 주요 비용 |
| --- | --- | --- | --- |
| Tensor parallelism, TP | 한 layer의 큰 weight matrix | Model 한 장이 GPU 하나에 안 들어감 | Layer마다 GPU 통신 |
| Pipeline parallelism, PP | Layer 묶음 | 여러 node에 깊은 model을 나눔 | Stage 사이 전달과 pipeline bubble |
| Data parallelism, DP | Model 전체 복제본과 서로 다른 request | Model은 한 GPU에 들어가고 요청이 많음 | 복제본별 memory와 load balancing |
| Expert parallelism, EP | MoE의 expert | Expert가 많은 MoE model | Token routing과 expert load imbalance |

TP를 4로 설정하면 같은 layer의 matrix를 GPU 4개가 나눠 계산하고 결과를 자주 합친다. PP는 앞쪽 layer와 뒤쪽 layer를 다른 GPU에 둔다. DP는 model 복제본을 여러 개 두고 각 복제본이 다른 요청을 처리한다. EP는 MoE model에서 expert를 나눈다.[^4]

큰 TP가 언제나 빠른 것은 아니다. GPU 사이 연결이 느리거나 batch가 작으면 통신 시간이 계산 이득보다 커질 수 있다. `GPU 개수`만 기록하지 말고 TP·PP·DP·EP 크기와 node 간 network도 적는다.

```bash
# vLLM의 tensor parallel 예시
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 8192
```

## 5. Weight를 줄이는 것 말고도 방법이 있다

### Prefix caching

같은 system prompt나 문서 앞부분의 KV cache를 재사용한다. Shared prefix가 길고 반복될 때 TTFT를 줄이는 데 유리하다. Cache hit가 낮은 서비스라면 관리 비용만 늘 수 있으므로 cached token과 hit를 측정한다.

### Speculative decoding

작은 draft model이 다음 token 후보를 여러 개 먼저 쓰고, 큰 target model이 한 번에 확인한다. Target이 후보를 많이 받아들이면 target 호출 횟수가 줄어든다. 검증 규칙을 올바르게 쓰면 target model과 같은 분포를 유지할 수 있다.[^5] Draft가 자주 틀리거나 두 model을 함께 올릴 memory가 부족하면 이득이 작다.

### Prefill-decode 분리

긴 prompt를 처리하는 prefill worker와 한 token씩 만드는 decode worker를 나눈다. 두 단계의 자원 성격이 다를 때 각각 따로 확장할 수 있다. 대신 KV cache를 worker 사이에 옮기는 network 비용과 장애 지점이 늘어난다. TensorRT-LLM과 현재 vLLM에는 disaggregated serving 관련 기능이 있지만, workload에 맞춘 검증이 필요하다.[^6]

## 6. Production에서는 `/metrics`를 먼저 본다

vLLM server는 `/metrics`에 Prometheus 형식의 지표를 내보낸다.[^7]

```bash
curl http://localhost:8000/metrics
```

| 운영 질문 | vLLM metric 예 | 해석 |
| --- | --- | --- |
| 지금 몇 요청을 처리하나? | `vllm:num_requests_running` | GPU에서 실행 중인 request 수 |
| 줄이 길어지는가? | `vllm:num_requests_waiting`, `vllm:request_queue_time_seconds` | Scheduler 앞 대기 상태 |
| KV cache가 가득 찼나? | `vllm:kv_cache_usage_perc` | 1에 가까울수록 여유가 적음 |
| 첫 token이 늦나? | `vllm:time_to_first_token_seconds` | TTFT histogram |
| Token 사이가 느린가? | `vllm:request_time_per_output_token_seconds` | TPOT histogram |
| 요청 전체가 늦나? | `vllm:e2e_request_latency_seconds` | End-to-end latency |
| 다시 계산하는 일이 생기나? | `vllm:num_preemptions` | KV 부족과 scheduling 압력의 신호 |
| Cache를 실제로 재사용하나? | `vllm:prefix_cache_hits` | Prefix cache hit 수 |
| 처리한 양은 얼마인가? | `vllm:prompt_tokens`, `vllm:generation_tokens` | Input·output token 누적량 |

Metric 이름은 version에 따라 바뀔 수 있다. Dashboard를 만들기 전에 실행 중인 server의 `/metrics`와 공식 문서를 함께 확인한다. Label에 request ID나 user prompt를 넣으면 시계열 종류가 끝없이 늘고 개인정보가 노출될 수 있으므로 피한다.

## 7. Load가 늘면 평균보다 꼬리가 먼저 무너진다

실제 GPU server를 실행할 수 없어 deterministic load model을 만들었다.[^1] 동시 요청을 1에서 64까지 늘리고, TTFT 500ms 이하, TPOT 25ms 이하, KV cache usage 90% 이하, OOM 0건을 SLO로 정했다. 이 기준은 결과를 계산하기 전에 고정했다.

![7B weight 저장량과 synthetic production serving report](/notes/tutorial/llm_lecture/images/w16_production_serving_report.png)

*그림 2. 왼쪽은 7B weight의 이상적인 저장량 공식, 가운데와 오른쪽은 교육용 load model의 throughput·goodput과 tail latency·KV 압력이다. Quantized checkpoint, LLM engine, CUDA GPU를 실행한 결과가 아니다. macOS CPU, Python 3.12, Matplotlib 3.11.1에서 직접 계산했다.[^1]*

| Concurrency | Output tok/s | Queue P95 | TTFT P95 | TPOT P95 | KV usage | OOM | SLO |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 56.0 | 0.0ms | 88.2ms | 11.78ms | 11.5% | 0 | O |
| 8 | 427.4 | 0.0ms | 131.6ms | 10.24ms | 21.6% | 0 | O |
| 16 | 658.4 | 0.0ms | 181.2ms | 8.48ms | 33.2% | 0 | O |
| 32 | 850.7 | 205.8ms | 486.2ms | 17.44ms | 56.4% | 0 | O |
| 64 | 923.2 | 1130.0ms | 1608.8ms | 89.12ms | 100.0% | 3 | X |

Concurrency 64에서는 raw throughput이 923.2 token/s로 가장 높다. 하지만 요청 3개가 OOM을 냈고 TTFT·TPOT·KV 기준도 모두 어겼다. SLO를 통과한 요청만 세는 goodput은 0이 된다. 이 가정에서는 concurrency 32가 마지막 통과 지점이다.

이 숫자로 특정 engine의 한계를 예측하면 안 된다. 곡선 모양과 경보 판단을 연습하기 위한 식일 뿐이다. 실제 server에서는 concurrency 24, 28, 32, 36처럼 경계 주변을 더 촘촘하게 재고 반복 측정한다.

## 8. 지표를 묶어 원인을 좁힌다

| 관찰 | 가능한 원인 | 다음 확인 |
| --- | --- | --- |
| Queue와 TTFT 상승, TPOT 안정 | 요청 유입이 prefill 처리량보다 많음 | Arrival rate, prompt 길이, running·waiting request |
| KV usage 상승, preemption 증가 | 긴 context나 sequence가 너무 많음 | Context 분포, `max-num-seqs`, cache dtype |
| TTFT 안정, TPOT 상승 | Decode batch·memory bandwidth 압력 | Output 길이, active sequence, GPU utilization |
| Throughput 상승, goodput 하락 | 과도한 batching으로 tail SLO 위반 | P95/P99와 실패 요청 포함 여부 |
| Error·OOM 증가 | Memory 여유 부족 또는 설정 한계 | Peak memory, engine log, 재시작 횟수 |

경보 하나만 보고 바로 server를 늘리지 않는다. 예를 들어 TTFT가 높아도 GPU가 비어 있다면 model 계산보다 queue 앞의 gateway나 network가 문제일 수 있다. Metrics는 `무슨 일이 생겼나`, logs는 `어떤 오류였나`, traces는 `어디서 시간이 걸렸나`를 찾는 데 쓴다.

## 9. Production serving report를 완성한다

```markdown
## Environment
- model ID와 revision:
- engine와 version:
- CUDA, driver, GPU와 node 수:
- dtype, quantization, TP·PP·DP·EP:

## Workload
- rendered prompt와 input/output token 분포:
- request rate, concurrency, warm-up, 반복 횟수:
- streaming, decoding, stop 조건:

## SLO
- success rate:
- TTFT P95 / TPOT P95 / E2E P99:
- error와 OOM 허용치:

## Results
- latency, throughput, goodput, peak memory, KV usage:
- quality regression:

## Decision
- 선택한 설정과 근거:
- 포기한 설정과 이유:
- 알려진 한계와 rollback 조건:
```

최종 선택은 `가장 빠른 설정`이 아니라 `품질과 SLO를 지키면서 필요한 traffic을 처리하는 가장 단순한 설정`으로 적는다. Model revision, prompt, 실행 명령, raw JSON을 함께 보관해야 다음 사람이 결과를 재현할 수 있다.

## 확인 문제

1. W4A16에서 4-bit와 16-bit는 각각 무엇을 가리키는가?
2. 7B W4 weight가 이상적으로 3.26GiB여도 실제 peak memory가 더 큰 이유는 무엇인가?
3. GPTQ와 AWQ가 양자화 오차를 다루는 관점은 어떻게 다른가?
4. Model이 GPU 하나에 들어가고 요청이 아주 많을 때 TP보다 DP를 먼저 검토할 수 있는 이유는 무엇인가?
5. Raw throughput이 올랐는데 goodput이 떨어질 수 있는 이유는 무엇인가?
6. Queue time과 TTFT가 함께 오르고 TPOT는 안정적이라면 어느 단계를 먼저 살펴볼까?
7. Speculative decoding이 항상 빨라지지 않는 이유 두 가지는 무엇인가?

## 완료 체크

- [x] 양자화와 TP·PP·DP·EP의 선택 기준을 정리했다.
- [x] BF16, FP8, INT8, AWQ, GPTQ의 이상적인 weight 저장량을 계산했다. 실제 GPU memory·속도·품질은 `미실행`이다.
- [x] Concurrency 1~64의 교육용 latency-throughput 곡선을 그렸다. 실제 engine 부하 시험은 `미실행`이다.
- [x] Queue time, TTFT, TPOT, token 수, KV cache, preemption, OOM을 기록할 metric 표를 만들었다.
- [x] 결과물로 작성 가능한 `Production serving report` 양식을 완성했다.
- [ ] 실제 quantized model과 GPU server로 최종 프로젝트 보고서를 채운다.

---

[^1]: 직접 실행한 `llm_lecture/week16_production_serving_demo.py`의 결과다. 정확히 7B parameter의 이상적인 weight 저장량과 concurrency 1·2·4·8·16·32·64의 deterministic load model을 계산했다. Quantized checkpoint, serving engine, CUDA GPU는 실행하지 않았다. 실행일: 2026-08-03.
[^2]: Frantar, E. et al. (2022). [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323). Approximate second-order 정보를 이용한 one-shot weight quantization을 참고했다.
[^3]: Lin, J. et al. (2023). [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978). Activation으로 salient weight channel을 찾고 scaling하는 원리와 Figure 2를 참고했다.
[^4]: NVIDIA. [TensorRT-LLM Parallelism](https://nvidia.github.io/TensorRT-LLM/1.2.0rc6.post2/features/parallel-strategy.html). Tensor, pipeline, data parallelism의 구조를 참고했다. MoE의 expert 배치는 [Expert Parallelism](https://nvidia.github.io/TensorRT-LLM/advanced/expert-parallelism.html)을 참고했다. 확인일: 2026-08-03.
[^5]: Leviathan, Y. et al. (2022). [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). Draft model의 후보를 target model이 병렬로 검증하면서 target 분포를 유지하는 방법을 참고했다.
[^6]: NVIDIA. [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html). Quantization, speculative decoding, KV cache reuse, chunked prefill, disaggregated serving 기능을 참고했다. vLLM의 현재 기능 범위는 [vLLM Documentation](https://docs.vllm.ai/en/latest/)을 확인했다. 확인일: 2026-08-03.
[^7]: vLLM. [Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/). Running·waiting requests, queue time, TTFT, TPOT, KV cache usage, preemption, prefix cache와 token metric 이름을 참고했다. 확인일: 2026-08-03.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
원본/윤문본: 초안 / 후처리 변경률 0.2%
카테고리별 탐지/수정: A-10 0→0, C-11 0→0, D-1 0→0, H-1 0→0, I-1 0→0
정량 점검: humanize-korean metrics v2.0 risk band low
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 계산값·synthetic 값·CUDA 미실행 값을 구분함
주요 확인: 양자화와 분산 서빙 용어를 짧은 정의와 표로 풀고 실제 성능으로 오해할 표현을 제거함
-->
