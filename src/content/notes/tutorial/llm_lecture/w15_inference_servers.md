---
title: "15주차. vLLM으로 추론 서버 만들기"
description: "OpenAI-compatible API 요청이 scheduler와 model runner를 거쳐 streaming 응답이 되는 흐름을 익히고 vLLM, SGLang, TensorRT-LLM을 비교한다."
tags:
  - LLM
  - inference
  - vLLM
  - SGLang
  - TensorRT-LLM
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 16주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

14주차에는 TTFT, TPOT, throughput을 이용해 추론 속도를 재는 법을 배웠다. 이번에는 측정 대상인 서버를 직접 살펴본다. 사용자가 보낸 대화가 HTTP 요청이 되고, GPU에서 token을 만든 뒤, 화면에 한 조각씩 도착하는 전 과정을 따라간다.

## 이번 주에 배울 것

- OpenAI-compatible API가 편리한 이유
- Chat template이 `role`과 `content`를 실제 prompt로 바꾸는 과정
- Scheduler, KV cache manager, model runner가 맡는 일
- Streaming에서 첫 글자를 빨리 보여주는 방법
- vLLM의 `max-model-len`, `gpu-memory-utilization`, `max-num-seqs`
- Prefix caching과 structured output의 쓰임
- vLLM, SGLang, TensorRT-LLM, TGI의 선택 기준

선수 지식은 13주차의 KV cache와 14주차의 latency·throughput이다.

!!! note "식당의 주문서와 주방"

    API는 주문서 양식이고 추론 엔진은 주방이다. 같은 주문서 양식을 쓰면 손님 쪽 프로그램을 거의 바꾸지 않고 주방을 교체할 수 있다. 주방 안에서는 scheduler가 여러 주문의 순서를 정하고, model runner가 GPU에서 실제 계산을 한다.

## 1. OpenAI-compatible은 모델 이름이 아니다

OpenAI-compatible server는 요청과 응답의 모양을 OpenAI API와 비슷하게 맞춘 서버다. vLLM에서는 `/v1/chat/completions`, `/v1/completions`, `/v1/responses` 같은 endpoint를 쓸 수 있다.[^1] 여기서 compatible은 통신 규격을 뜻한다. 안쪽 모델이 OpenAI 모델이라는 뜻은 아니다.

```text
client
  -> HTTP API와 입력 검사
  -> chat template로 prompt 만들기
  -> request queue와 scheduler
  -> KV cache manager
  -> model runner와 GPU worker
  -> sampler
  -> JSON 응답 또는 SSE streaming
```

Client는 `messages` 배열을 보낸다. API server는 잘못된 필드가 없는지 확인하고 chat template을 적용한다. Scheduler는 새 prompt의 prefill과 이미 생성 중인 요청의 decode를 batch로 묶는다. Model runner는 GPU 계산을 맡고 sampler는 다음 token을 고른다. Streaming 요청이면 완성된 답 전체를 기다리지 않고 작은 조각을 차례로 보낸다.

## 2. PagedAttention 서버 안에서는 무슨 일이 일어날까

![vLLM system overview](/notes/tutorial/llm_lecture/images/w15_vllm_system_overview.png)

*그림 1. API server가 요청을 scheduler로 보내고, scheduler가 여러 GPU worker에 명령한다. KV cache block은 block manager가 관리한다. 출처: Kwon et al. (2023), Figure 4에서 발췌.[^2]*

그림의 scheduler는 매 step에 어떤 요청을 실행할지 정한다. 각 worker에는 model과 KV cache가 있다. Block manager는 논리적인 KV block과 GPU의 물리 block을 연결한다. 운영체제가 파일을 작은 page로 나눠 관리하듯, PagedAttention도 KV cache를 block으로 나눠 빈 공간을 덜 낭비한다.[^2]

!!! warning "API server와 engine을 같은 것으로 보지 않는다"

    API server는 HTTP, 인증, 입력 검사를 담당한다. Engine은 scheduling과 model 실행을 맡는다. HTTP 응답이 느린 원인을 모두 GPU 탓으로 돌리면 queue, JSON 변환, network 문제를 놓칠 수 있다.

## 3. vLLM 서버를 띄운다

CUDA GPU와 vLLM이 준비된 환경에서 다음처럼 시작할 수 있다.[^1] 설치한 version마다 option이 바뀔 수 있으므로 먼저 `vllm serve --help`를 확인한다.

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64
```

서버가 뜨면 health와 model 목록을 먼저 확인한다.

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Python client는 `base_url`만 local server로 바꾼다. `stream=True`이면 chunk가 도착할 때마다 글자를 바로 출력할 수 있다.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local",
)

stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "system", "content": "Answer in one short sentence."},
        {"role": "user", "content": "Why is the sky blue?"},
    ],
    temperature=0,
    max_tokens=64,
    stream=True,
)

for chunk in stream:
    text = chunk.choices[0].delta.content
    if text:
        print(text, end="", flush=True)
```

## 4. Chat template은 대화에 교복을 입힌다

Model은 Python의 dictionary를 직접 읽지 않는다. `system`, `user`, `assistant` message를 학습 때 보던 특수 token 문자열로 바꿔야 한다. 이 변환 규칙이 chat template이다.

```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "messages": [
    {"role": "system", "content": "Answer briefly."},
    {"role": "user", "content": "What is a KV cache?"}
  ],
  "temperature": 0,
  "stream": true
}
```

같은 JSON도 model마다 다음과 같이 다른 문자열이 될 수 있다.

```text
<|im_start|>system
Answer briefly.<|im_end|>
<|im_start|>user
What is a KV cache?<|im_end|>
<|im_start|>assistant
```

vLLM의 chat completion은 tokenizer에 chat template이 있어야 한다. Template이 없는 model에는 `--chat-template`로 알맞은 규칙을 지정해야 한다.[^1] 엉뚱한 template을 쓰면 서버는 실행돼도 model이 role 경계를 잘 이해하지 못할 수 있다.

## 5. GPU가 없어도 HTTP와 streaming은 연습할 수 있다

이번 환경에는 CUDA GPU와 vLLM server가 없다. 그래서 Python 표준 라이브러리로 `/v1/chat/completions`의 아주 작은 일부를 흉내 낸 mock server를 실행했다.[^3] 이 실습은 JSON 입력 검사, HTTP 400 오류, SSE 형식의 streaming을 확인한다. Model 추론이나 vLLM 성능을 재는 실험은 아니다.

![OpenAI-compatible mock server 실행 결과](/notes/tutorial/llm_lecture/images/w15_openai_mock_server.png)

*그림 2. 왼쪽은 인위적인 지연을 둔 streaming과 non-streaming 응답 시간이다. 오른쪽은 thread 기반 mock server의 동시 요청 처리량이다. LLM engine이나 GPU benchmark가 아니다. macOS CPU, Python 3.12, Matplotlib 3.11.1에서 직접 실행했다.[^3]*

| Mode | HTTP status | 첫 content | 전체 응답 | 받은 문자열 |
| --- | ---: | ---: | ---: | --- |
| Streaming | 200 | 23.1ms | 81.1ms | `Mock server response.` |
| Non-streaming | 200 | 70.4ms | 70.4ms | `Mock server response.` |

Streaming은 전체 응답 시간이 10.7ms 더 길었지만 첫 content는 47.3ms 빨랐다. 조각을 여러 번 보내는 비용이 조금 들어도 사용자는 답이 시작됐다는 사실을 일찍 알 수 있다. 실제 LLM에서는 첫 chunk에 role만 있고 content가 비어 있을 수 있으므로, TTFT는 첫 HTTP byte가 아니라 첫 실제 token 시각으로 정한다.

| 동시 요청 | 완료 요청 | Mock 처리량 | TTFT 중앙값 | E2E 중앙값 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 13.0 request/s | 19.4ms | 76.8ms |
| 4 | 4 | 48.0 request/s | 22.6ms | 82.4ms |
| 16 | 16 | 132.3 request/s | 25.3ms | 82.5ms |

이 값은 `ThreadingHTTPServer`가 잠깐 기다리는 작업을 겹쳐 처리한 결과다. GPU batch 효율, KV cache, model 크기와는 관계가 없다.

## 6. 세 설정은 서로 연결되어 있다

| 설정 | 쉬운 뜻 | 너무 크게 잡았을 때 | 너무 작게 잡았을 때 |
| --- | --- | --- | --- |
| `max-model-len` | 한 request가 쓸 수 있는 최대 context | 긴 KV cache 여유가 필요함 | 긴 prompt가 거절됨 |
| `gpu-memory-utilization` | Model executor가 쓸 GPU memory 비율 | 다른 process와 충돌하거나 여유가 부족함 | KV cache block이 줄어듦 |
| `max-num-seqs` | 한 번에 처리할 sequence 상한 | Queue는 줄어도 KV 압력과 tail latency가 커질 수 있음 | GPU가 덜 차서 throughput이 낮을 수 있음 |

이 세 값을 하나씩만 바꾸고 prompt·output 길이와 arrival rate를 고정한다. `max-model-len`을 늘렸다고 실제 요청이 모두 길어지는 것은 아니다. 반대로 긴 context를 허용해도 KV cache 공간이 부족하면 동시에 받을 수 있는 sequence 수가 줄어든다.[^4]

```text
고정: model, prompt 2048 tokens, output 128 tokens, request rate 8/s
변경: max-num-seqs = 8, 16, 32, 64
기록: success rate, TTFT P50/P95/P99, TPOT P95, output token/s,
      KV cache usage, preemption, peak GPU memory
```

## 7. Prefix caching은 같은 앞부분을 다시 쓰는 기술이다

여러 요청이 같은 system prompt나 긴 문서 앞부분을 공유하면 prefill에서 계산한 KV block을 다시 쓸 수 있다. 이를 prefix caching이라고 한다. SGLang은 공유 prefix를 radix tree로 관리하는 RadixAttention을 제안했다.[^5]

```text
공통 prefix: 학교 규칙 8,000 tokens
요청 A: 공통 prefix + "급식 시간을 알려줘"
요청 B: 공통 prefix + "도서관 규칙을 알려줘"
```

요청 B는 A와 겹치는 앞부분의 KV를 재사용할 수 있다. 하지만 새 질문의 prefill과 답을 만드는 decode까지 사라지는 것은 아니다. 효과를 보려면 cached token 수, cache hit, TTFT를 함께 기록한다.

Structured output은 JSON schema나 정규식처럼 허용할 출력 형태를 정해두는 기능이다. Tool calling에서는 model의 chat template, tool-call parser, schema가 모두 맞아야 한다. 단순히 JSON처럼 보이는 문장을 만들었다고 실행 가능한 tool call이 되는 것은 아니다.

## 8. 어떤 서버를 먼저 고를까

| Engine | 먼저 살펴볼 상황 | 확인할 점 |
| --- | --- | --- |
| vLLM | 범용 OpenAI-compatible server와 넓은 model 지원이 필요함 | 현재 model·dtype·quantization 지원, scheduler 설정 |
| SGLang | 긴 shared prefix, agent, structured generation workload가 많음 | RadixAttention hit, frontend/runtime 사용법 |
| TensorRT-LLM | NVIDIA GPU에서 kernel·multi-GPU 최적화를 깊게 다룸 | Engine build, GPU 세대, TensorRT 버전, parallelism |
| TGI | 기존 Hugging Face TGI 시스템을 유지함 | 공식 문서가 maintenance mode라고 밝힌 현재 상태 |

TensorRT-LLM은 in-flight batching, paged attention, quantization, speculative decoding, KV cache reuse, multi-GPU parallelism을 제공한다.[^6] TGI는 현재 maintenance mode이므로 새 프로젝트에서는 vLLM이나 SGLang 같은 대안을 먼저 검토하라는 안내가 공식 문서에 있다.[^7]

표는 기능의 방향을 정리한 것이지 속도 순위가 아니다. 같은 model, 같은 rendered prompt, 같은 input·output token 수, 같은 concurrency와 SLO로 다시 재야 한다.

## 9. 실제 비교 실험표를 만든다

```bash
# SGLang 예시
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --context-length 8192 \
  --mem-fraction-static 0.90

# 설치한 버전의 인자 확인
python -m sglang.launch_server --help
```

| 항목 | vLLM | SGLang | TensorRT-LLM |
| --- | --- | --- | --- |
| Model revision | 기록 | 기록 | 기록 |
| Rendered prompt와 token 수 | 동일 | 동일 | 동일 |
| Warm-up | 동일 횟수 | 동일 횟수 | 동일 횟수 |
| TTFT·TPOT P50/P95/P99 | 측정 | 측정 | 측정 |
| Output token/s·goodput | 측정 | 측정 | 측정 |
| Peak memory·KV usage | 측정 | 측정 | 측정 |
| 오류·OOM·timeout | 포함 | 포함 | 포함 |

이번 환경에서는 이 GPU 비교를 실행하지 않았다. 설치와 model download만 끝냈다고 체크하지 말고, 세 engine에 같은 요청을 보내 결과 JSON을 얻은 뒤 완료로 판단한다.

## 확인 문제

1. OpenAI-compatible server를 쓴다고 안쪽 model도 OpenAI model인 것은 아닌 이유는 무엇인가?
2. Chat template이 없는 tokenizer로 chat completion을 보내면 어떤 문제가 생길 수 있는가?
3. Streaming의 E2E latency가 조금 길어도 사용자 경험이 나아질 수 있는 이유는 무엇인가?
4. `max-model-len`과 `max-num-seqs`를 모두 크게 잡으면 KV cache에 어떤 일이 생길 수 있는가?
5. Prefix caching이 decode 계산까지 없애 주지 않는 이유는 무엇인가?
6. Engine의 속도를 비교할 때 rendered prompt를 저장해야 하는 이유는 무엇인가?
7. Mock server의 132.3 request/s를 vLLM 성능이라고 부르면 안 되는 이유는 무엇인가?

## 완료 체크

- [x] Client에서 GPU worker까지 요청 흐름을 설명했다.
- [x] Chat completion과 streaming 요청 코드를 작성하고 mock server로 전송 형식을 확인했다. 실제 vLLM CUDA 실행은 `미실행`이다.
- [x] 세 가지 server 설정을 바꾸는 실험표를 만들었다. GPU 실측값은 `미실행`이다.
- [x] vLLM, SGLang, TensorRT-LLM의 공통 비교 기준을 만들었다. Cross-engine 실측은 `미실행`이다.
- [x] 결과물로 `OpenAI-compatible server 실행 및 비교 노트`를 완성했다.

---

[^1]: vLLM. [Online Serving](https://docs.vllm.ai/en/latest/serving/online_serving/). OpenAI-compatible endpoint, chat template, streaming 사용법을 참고했다. 확인일: 2026-08-03.
[^2]: Kwon, W. et al. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). Figure 4의 vLLM 구조와 scheduler, worker, block manager 설명을 참고했다.
[^3]: 직접 실행한 `llm_lecture/week15_openai_compatible_server_demo.py`의 결과다. Python 표준 라이브러리로 chat completion 일부와 SSE 형식을 흉내 냈고, prefill 18ms와 chunk 사이 16ms의 인위적 지연을 넣었다. Model, vLLM, SGLang, GPU는 실행하지 않았다. 실행일: 2026-08-03.
[^4]: vLLM. [Engine Configuration](https://docs.vllm.ai/en/latest/api/vllm/config/). Maximum model length, GPU memory utilization, KV cache와 scheduling 설정을 참고했다. 확인일: 2026-08-03.
[^5]: Zheng, L. et al. (2023). [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104). RadixAttention, compressed finite state machine, frontend와 runtime 구성을 참고했다. 실행 인자는 [Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)와 [Serving Benchmark Guide](https://docs.sglang.io/docs/developer_guide/bench_serving)를 확인했다. 확인일: 2026-08-03.
[^6]: NVIDIA. [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html). In-flight batching, paged attention, KV cache reuse, quantization과 multi-GPU 기능을 참고했다. 확인일: 2026-08-03.
[^7]: Hugging Face. [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/main/index). TGI가 maintenance mode이며 기존 시스템 유지 중심이라는 안내를 확인했다. 확인일: 2026-08-03.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
원본/윤문본: 초안 / 후처리 변경률 0.1%
카테고리별 탐지/수정: A-10 0→0, C-11 0→0, D-1 0→0, H-1 0→0, I-1 0→0
정량 점검: humanize-korean metrics v2.0 risk band low
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 mock server와 실제 추론 엔진을 구분함
주요 확인: API와 engine의 역할을 짧은 문장으로 나누고 CUDA 미실행 상태를 본문·표·체크리스트에 표시함
-->
