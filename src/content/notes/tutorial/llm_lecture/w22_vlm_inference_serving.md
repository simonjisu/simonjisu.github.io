---
title: "22주차. VLM 추론과 멀티모달 서빙"
description: "이미지가 VLM의 시각 토큰으로 바뀌는 과정부터 TTFT, TPOT, 메모리 측정, 여러 이미지 요청, vLLM 서빙과 외부 미디어 보안까지 다룬다."
tags:
  - VLM
  - multimodal serving
  - inference
  - vLLM
  - TTFT
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 26주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

21주차에는 VLM의 답을 과제별 지표와 대조 실험으로 분석했다. 이번에는 model을 서비스에 올렸을 때 생기는 문제를 살펴본다. 사진 한 장을 질문에 붙였을 뿐인데 첫 답이 늦게 나오거나, 여러 장을 넣자 memory가 갑자기 부족해질 수 있다. 원인은 사진이 그대로 LLM에 들어가는 것이 아니라 수십 개에서 수천 개의 visual token으로 바뀌기 때문이다.

## 이번 주에 배울 것

- 이미지 URL을 읽은 뒤 첫 token을 만들기까지 VLM이 거치는 단계
- Image 수와 tile 수가 visual token 수를 늘리는 방식
- TTFT, TPOT, end-to-end latency, throughput의 차이
- 한 장, 여러 장, 고해상도 분할 입력의 latency와 memory 측정법
- 여러 이미지가 섞인 chat template과 image 순서 관리
- vLLM에서 multimodal 요청 수와 허용 media domain을 제한하는 법
- 같은 이미지를 다시 처리하지 않는 cache의 역할
- 외부 URL, 큰 파일, 잘못된 MIME type을 받는 서버의 방어 항목

선수 지식은 11주차의 추론 최적화와 KV cache, 17주차의 patch와 visual token, 20주차의 multimodal chat template이다.

!!! note "사진의 파일 크기와 visual token 수는 같은 값이 아니다"

    5MB짜리 JPEG가 항상 1MB짜리 PNG보다 많은 token을 만드는 것은 아니다. 압축을 푼 뒤 Processor가 어떤 크기로 줄이고 몇 개 tile로 나누는지가 중요하다. 서버에서는 전송할 때의 byte, 압축을 푼 pixel, model에 들어가는 visual token을 따로 제한한다.

## 1. VLM 요청은 다섯 구간을 지나간다

사용자가 이미지 URL과 질문을 보내면 바로 다음 단어를 만드는 것이 아니다. 대개 다음 순서로 처리한다.

1. **Media load**: URL이나 파일에서 byte를 읽고 image로 decode한다.
2. **Preprocess**: 회전 정보를 적용하고 RGB 변환, resize, crop, tile 분할을 한다.
3. **Vision encode**: Vision encoder가 각 patch를 feature로 바꾼다.
4. **Project and prefill**: Connector가 feature를 LLM 차원에 맞추고, text token과 함께 prompt 전체를 읽는다.
5. **Decode**: 첫 output token부터 한 token씩 이어서 만든다.

Text-only LLM에는 1번부터 3번까지가 없다. VLM의 첫 token이 늦다면 LLM만 보지 말고 다운로드, image decode, vision encoder, text prefill을 구간별로 재야 한다.

| 구간 | 대표 원인 | 따로 남길 값 |
| --- | --- | --- |
| Media load | 느린 원격 서버, 큰 파일, redirect | Download와 decode 시간, byte 수 |
| Preprocess | Resize, crop, tile 수 증가 | 원본 크기, 입력 크기, tile 수 |
| Vision encode | Image와 patch 증가 | Visual token 수, device 시간 |
| Text prefill | 긴 질문, 여러 이미지 설명 | 전체 input token, prefill 시간 |
| Decode | 긴 답변, 작은 batch | TPOT, output token 수 |

!!! warning "어디서부터 시간을 쟀는지 쓰지 않으면 TTFT를 비교할 수 없다"

    어떤 팀은 HTTP 요청을 보낸 때부터 재고, 어떤 팀은 image를 모두 읽은 뒤 `generate()` 직전부터 잰다. 두 값의 이름이 모두 TTFT여도 범위가 다르다. 시작점과 끝점을 결과표에 함께 적는다.

## 2. Chat template은 image 자리와 실제 image를 맞춘다

Multimodal chat의 `content`는 text 한 덩어리가 아니라 image와 text가 섞인 목록이다. Transformers의 multimodal template은 `type: image` 자리에 model별 특수 token을 넣고 Processor가 실제 image tensor와 연결한다.[^1]

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Image 1."},
            {"type": "image"},
            {"type": "text", "text": "Image 2. What changed?"},
        ],
    }
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = processor(text=prompt, images=[before, after], return_tensors="pt")
```

위 입력에서 첫 번째 placeholder는 `before`, 두 번째는 `after`와 연결된다. Image 목록의 순서를 바꾸면서 본문에 적은 `Image 1`, `Image 2` 표시는 그대로 두면 model이 비교 대상을 거꾸로 이해한다.

![여러 이미지, video frame, 3D view, 한 이미지의 여러 patch를 interleaved 형식으로 나타낸 예시](/notes/tutorial/llm_lecture/images/w22_interleaved_multimodal_inputs.png)

*그림 1. 여러 이미지, video frame, 3D view, 한 이미지의 여러 patch를 image-text interleaved 형식으로 통일한 예시. 출처: Li et al. (2024), Figure 2에서 발췌.[^2]*

LLaVA-NeXT-Interleave는 multi-image, video, multi-view 3D, 한 장을 여러 patch로 나눈 입력을 같은 interleaved 구조로 다뤘다.[^2] 형식은 같아도 뜻은 다르다. Video에서는 frame 순서, 3D에서는 카메라 시점, 문서에서는 page 번호를 함께 알려줘야 한다.

## 3. 한 장의 image가 여러 visual token이 된다

17주차에서 본 것처럼 vision encoder는 image를 patch로 나눈다. 고해상도 image를 여러 tile로 자르는 model이라면 각 tile이 다시 visual token 묶음이 된다. 간단히 $N_{\mathrm{visual}} = N_{\mathrm{image}} \times N_{\mathrm{tile}} \times N_{\mathrm{token/tile}}$로 생각할 수 있다. 실제 값에는 전역 thumbnail이나 구분 token이 붙기도 한다. 그래서 Processor output을 직접 세는 편이 정확하다.

```python
visual_tokens = (inputs.input_ids == processor.image_token_id).sum().item()
image_tiles = inputs.pixel_values.shape[1]

print(visual_tokens, image_tiles)
```

이번 실습의 SmolVLM checkpoint에서 compact 설정은 image 한 장을 1개 tile, 64 visual token으로 만들었다. `do_image_splitting=True`를 켜자 한 장이 16개 지역 tile과 전역 image 1개, 모두 17개로 바뀌었다. Visual token도 $17 \times 64 = 1{,}088$개가 됐다. 이 숫자는 이 checkpoint와 Processor의 결과이며 모든 VLM에 그대로 적용되지 않는다.[^3]

| 입력 조건 | Image 수 | Tile 수 | Visual token | 전체 input token |
| --- | ---: | ---: | ---: | ---: |
| 1 image compact | 1 | 1 | 64 | 101 |
| 2 images compact | 2 | 2 | 128 | 173 |
| 4 images compact | 4 | 4 | 256 | 317 |
| 1 image tiled | 1 | 17 | 1,088 | 1,161 |
| 2 images tiled | 2 | 34 | 2,176 | 2,293 |

Image 두 장의 visual token만 두 배가 되는 것이 아니다. `Image 1`, `Image 2` 같은 label과 image separator도 늘어 전체 input token은 2,176보다 큰 2,293개가 됐다.

## 4. 빠르다는 말을 네 가지 숫자로 나눈다

Latency를 한 숫자로만 적으면 사용자가 느끼는 기다림과 서버의 처리 능력을 구분하기 어렵다.

| 지표 | 뜻 | 사용자가 느끼는 부분 |
| --- | --- | --- |
| TTFT | 요청 시작부터 첫 output token까지 | 답변이 시작되기 전의 침묵 |
| TPOT | 첫 token 뒤에 token 하나가 추가되는 평균 시간 | 글자가 이어지는 속도 |
| End-to-end latency | 요청 시작부터 마지막 token까지 | 답변 전체를 받는 시간 |
| Throughput | 단위 시간에 처리한 request 또는 token 수 | 여러 사용자를 함께 처리하는 능력 |

Output token이 $M$개라면 model 실행 시간은 대략 $TTFT + (M - 1) \times TPOT$로 볼 수 있다. HTTP 전송과 queue가 있다면 그 시간도 더해진다. 이번 실습에서는 16개 token을 고정했기 때문에 조건별 decode 길이가 같다.

Throughput이 높다고 모든 사용자가 빠른 것은 아니다. Batch를 오래 모으면 GPU 사용률과 전체 token/s는 좋아져도 먼저 도착한 요청의 TTFT가 길어진다. 운영에서는 P50뿐 아니라 P95, P99와 시간 초과 비율을 함께 본다.

## 5. 작은 VLM로 입력 모양을 바꿔 측정한다

`HuggingFaceTB/SmolVLM-256M-Instruct`의 revision을 고정하고 1024×1024 합성 image를 사용했다.[^3][^4] Greedy decoding으로 정확히 16개 token을 만들었고, warm-up 1회 뒤 각 조건을 3회 실행해 중앙값을 냈다. 장치는 macOS MPS였다.[^3]

측정 시작점은 image가 이미 memory에 있고 Processor 처리와 device 이동까지 끝난 뒤, `model.generate()`를 호출하기 직전이다. 따라서 표의 TTFT에는 vision encode, text prefill, 첫 output token 생성이 들어가지만 file read, image decode, Processor, device transfer는 빠져 있다.

```python
synchronize(device)
started = time.perf_counter()

with torch.inference_mode():
    output = model.generate(
        **inputs,
        min_new_tokens=16,
        max_new_tokens=16,
        do_sample=False,
        use_cache=True,
        stopping_criteria=StoppingCriteriaList([token_clock]),
    )

synchronize(device)
ttft_ms = (token_clock.times[0] - started) * 1000
tpot_ms = (token_clock.times[-1] - token_clock.times[0]) * 1000 / 15
```

![이미지 수와 tile 분할에 따른 visual token, TTFT, TPOT, 요청 중 memory 증가량](/notes/tutorial/llm_lecture/images/w22_vlm_inference_benchmark.png)

*그림 2. SmolVLM-256M-Instruct를 macOS MPS에서 실행한 작은 진단 실험. Warm-up 뒤 3회 중앙값이며 engine 사이의 우열을 비교한 benchmark가 아니다.[^3]*

| 조건 | Visual token | Processor | TTFT | TPOT | 전체 생성 | 출력 속도 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 image compact | 64 | 12.1ms | 65.4ms | 10.3ms | 227.8ms | 70.2 token/s |
| 2 images compact | 128 | 20.7ms | 120.2ms | 11.1ms | 296.4ms | 54.0 token/s |
| 4 images compact | 256 | 36.2ms | 221.6ms | 11.4ms | 397.7ms | 40.2 token/s |
| 1 image tiled | 1,088 | 18.9ms | 850.5ms | 17.6ms | 1,119.2ms | 14.3 token/s |
| 2 images tiled | 2,176 | 33.2ms | 1,605.7ms | 26.2ms | 2,007.1ms | 8.0 token/s |

Compact 입력에서 image를 1장에서 4장으로 늘리자 visual token은 4배, TTFT는 약 3.4배가 됐다. 한 장을 17개 tile로 나눈 조건은 compact 한 장보다 visual token이 17배, TTFT는 약 13배였다. 숫자가 정확히 같은 비율로 늘지 않는 까닭은 vision encoder와 LLM 연산, 고정 비용, hardware 병렬 처리가 함께 작용하기 때문이다.

!!! note "이 결과는 model 비교표가 아니라 병목을 찾는 실험이다"

    작은 합성 image와 한 대의 MPS 장치에서 얻은 값이다. CUDA의 kernel, vLLM batching, production network를 반영하지 않는다. 여기서 말할 수 있는 것은 같은 환경에서 입력 모양을 바꾸면 visual token과 TTFT가 크게 달라졌다는 사실까지다.

## 6. Memory 숫자는 범위를 적고 조심해서 읽는다

Model weight가 차지한 memory와 요청 한 건이 추가로 사용한 memory를 섞지 않는다. 이번 실습은 model을 먼저 올린 뒤 `generate()` 직전 값을 baseline으로 잡고, 5ms마다 process RSS와 MPS allocated memory를 읽었다.[^3]

| 조건 | Process RSS peak delta | Device allocation peak delta |
| --- | ---: | ---: |
| 1 image compact | 0.05MiB | 8.2MiB |
| 2 images compact | 0.03MiB | 30.2MiB |
| 4 images compact | 0.03MiB | 36.4MiB |
| 1 image tiled | 0.03MiB | 178.6MiB |
| 2 images tiled | 0.05MiB | 157.2MiB |

Process RSS가 거의 늘지 않은 것은 큰 tensor가 MPS device memory에 잡혔고 model weight를 baseline에서 제외했기 때문이다. 두 장 tiled의 peak delta가 한 장보다 작게 나온 결과도 요청이 가볍다는 뜻이 아니다. MPS allocator 재사용, 이전 실행의 cache, 5ms sampling 간격, 3회 중앙값이 섞인 작은 실험이라 세부 순위를 해석하기 어렵다. Tiled 조건에서 compact보다 훨씬 큰 device allocation 증가가 관찰됐다는 수준으로 읽는다.

CUDA 서버에서는 `torch.cuda.max_memory_allocated()`를 요청 전후로 초기화해 재고, process 전체 memory와 engine이 미리 예약한 block도 따로 기록한다. 동시 요청에서는 개별 request보다 server 전체의 peak와 out-of-memory 횟수가 중요하다.

## 7. 같은 image를 다시 쓰면 encode 결과를 재사용할 수 있다

대화가 길어질 때 사용자가 같은 사진을 매 turn 다시 보내기도 한다. 매번 download, decode, vision encode를 반복하면 시간과 비용이 낭비된다. Transformers 문서는 반복 생성에서 이미 계산한 multimodal 표현과 cache를 재사용해 같은 media를 다시 처리하지 않는 방식을 안내한다.[^4]

Cache key에는 URL 문자열만 넣지 않는다. URL의 내용이 바뀔 수 있기 때문이다. 가능하면 image byte의 hash, model revision, Processor 설정, resize와 tile 설정을 함께 사용한다.

```text
cache key = sha256(image bytes)
          + model revision
          + processor revision
          + resize and tile configuration
```

Cache에는 만료 시간, 사용자별 접근 권한, 최대 용량이 필요하다. 다른 사용자의 비공개 image feature가 잘못 반환되면 원본 image가 노출되지 않아도 정보 유출이다. 삭제 요청이 들어왔을 때 원본과 feature cache를 함께 지울 수 있어야 한다.

## 8. vLLM에서 image 개수와 출처를 제한한다

vLLM은 OpenAI 호환 server에서 multimodal 입력을 받을 수 있다. Model과 revision을 고정하고 한 request에 허용할 image 수를 제한한다.[^5]

```bash
VLLM_MEDIA_URL_ALLOW_REDIRECTS=0 \
vllm serve HuggingFaceTB/SmolVLM-256M-Instruct \
  --revision 7e3e67edbbed1bf9888184d9df282b700a323964 \
  --limit-mm-per-prompt '{"image": 4}' \
  --allowed-media-domains images.example.com
```

Client 요청은 text와 `image_url`을 같은 content 목록에 넣는다.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local-token")
response = client.chat.completions.create(
    model="HuggingFaceTB/SmolVLM-256M-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "두 image의 차이를 한 문장으로 답해줘."},
                {"type": "image_url", "image_url": {"url": "https://images.example.com/a.png"}},
                {"type": "image_url", "image_url": {"url": "https://images.example.com/b.png"}},
            ],
        }
    ],
    max_tokens=64,
)
```

이 작업 공간에는 CUDA GPU와 vLLM이 없어 위 server benchmark를 실행하지 않았다. Transformers MPS 결과를 vLLM 성능이라고 부르거나, 실행하지 않은 숫자를 채우지 않는다. vLLM 문서는 빠르게 바뀌므로 실제 배포에서는 설치한 version의 multimodal 문서와 engine argument를 확인한다.[^5]

### Production에서는 동시 요청을 따로 잰다

Single request가 잘 돌아가도 16명이 함께 image를 보내면 queue와 memory 부족이 생긴다. 다음 표처럼 같은 입력 묶음을 concurrency별로 반복한다.

| 동시 요청 | 성공률 | P50 TTFT | P95 TTFT | P50 TPOT | Output token/s | Peak GPU memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 실행 후 기록 | 실행 후 기록 | 실행 후 기록 | 실행 후 기록 | 실행 후 기록 | 실행 후 기록 |
| 4 | 실행 후 기록 | 실행 후 기록 | 실행 후 기록 | 실행 후 기록 | 실행 후 기록 | 실행 후 기록 |
| 16 | 실행 후 기록 | 실행 후 기록 | 실행 후 기록 | 실행 후 기록 | 실행 후 기록 | 실행 후 기록 |

각 조건을 바로 이어 실행하면 앞선 image cache가 뒤 조건을 도울 수 있다. Cold cache와 warm cache를 분리하고 요청 순서를 섞는다. Error가 난 요청을 latency 통계에서 몰래 빼지 말고 timeout, 4xx, 5xx, out-of-memory 비율로 남긴다.

## 9. 외부 image URL은 작은 browser처럼 다룬다

Server가 사용자가 준 URL을 대신 읽으면 내부 주소나 cloud metadata endpoint에 접근하는 통로가 된다. vLLM은 `--allowed-media-domains`로 원격 media domain을 제한하고 redirect 허용 여부도 통제한다.[^5][^6] Local file 경로 허용 옵션은 보안 위험이 있으므로 신뢰할 수 있는 환경에서만 사용한다.[^7]

운영 server에서는 다음 항목을 함께 점검한다.

- `https`와 허용 domain 목록만 받고 redirect는 기본적으로 막는다.
- DNS가 가리키는 IP를 확인해 loopback, 사설망, link-local 주소를 차단한다.
- Download timeout과 압축된 file byte 상한을 둔다.
- Header만 믿지 않고 실제 MIME type과 decode 성공 여부를 확인한다.
- 압축을 푼 뒤 width, height, 전체 pixel 수, frame 수를 다시 제한한다.
- 한 request의 image 수와 tile 수, 전체 visual token 상한을 둔다.
- 인증, 사용자별 rate limit, queue 길이 제한을 적용한다.
- 원본 image와 log의 보존 기간을 정하고 민감 정보가 log에 남지 않게 한다.

!!! warning "작은 압축 파일도 큰 memory를 쓸 수 있다"

    단색에 가까운 거대한 image는 압축 파일이 작아도 decode 뒤에는 수억 pixel이 될 수 있다. File byte 제한만 두면 decompression bomb를 막지 못한다. Decode 전에 header 크기를 검사하고, decoder의 pixel 제한도 켠다.

## 10. 배포 전에 한 장짜리 보고서를 만든다

VLM production serving report에는 다음 내용을 넣는다.

1. Model ID, revision, Processor, chat template, engine version
2. 허용 image 형식, byte, pixel, 장수, tile, visual token 상한
3. 대표 요청의 media load, preprocess, vision encode, prefill, decode 시간
4. Image 수와 해상도별 P50, P95 TTFT와 TPOT
5. Concurrency별 throughput, queue time, peak memory, error 비율
6. Cold cache와 warm cache 조건, cache key와 만료 정책
7. 허용 domain, redirect, timeout, local file 접근 정책
8. 대표 성공, timeout, 잘못된 MIME type, 너무 큰 image 요청 log

한 숫자로 “빠르다”고 쓰기보다 입력 크기와 동시 사용자 수에 따라 어디서 느려지는지 보여주는 보고서가 운영에 도움이 된다.

## 확인 문제

1. 5MB JPEG가 1MB PNG보다 항상 visual token이 많다고 말할 수 없는 이유는 무엇인가?
2. VLM의 TTFT 안에는 어떤 계산 구간이 들어갈 수 있으며, 비교 전에 무엇을 확인해야 하는가?
3. 한 image가 17개 tile, tile당 64 token이라면 visual token은 몇 개인가?
4. 이번 실험에서 compact image를 1장에서 4장으로 늘렸을 때 TTFT와 visual token은 어떻게 달라졌는가?
5. TTFT는 크게 늘었지만 TPOT가 비교적 적게 변할 수 있는 이유를 prefill과 decode로 나누어 설명해보자.
6. 두 장 tiled 조건의 memory peak delta가 한 장보다 작게 측정됐다고 해서 더 가볍다고 결론 내리면 안 되는 이유는 무엇인가?
7. Multi-image chat에서 placeholder와 실제 image 목록의 순서가 어긋나면 어떤 문제가 생기는가?
8. 같은 image의 vision feature를 cache할 때 URL만 key로 쓰면 위험한 이유는 무엇인가?
9. `--limit-mm-per-prompt`와 `--allowed-media-domains`는 각각 무엇을 제한하는가?
10. 외부 image server에서 file byte와 decoded pixel을 모두 제한해야 하는 까닭은 무엇인가?
11. Single request benchmark만으로 production throughput을 판단할 수 없는 이유는 무엇인가?
12. 이 글의 MPS 측정값으로 vLLM과 Transformers의 속도를 비교할 수 없는 이유는 무엇인가?

## 완료 체크

- [x] VLM 요청을 media load, preprocess, vision encode, prefill, decode로 나눴다.
- [x] Chat template의 image placeholder와 실제 image 순서를 확인했다.
- [x] Image 수와 tile 분할에 따른 visual token 수를 직접 셌다.
- [x] TTFT, TPOT, end-to-end latency, throughput의 차이를 설명했다.
- [x] SmolVLM을 실행해 다섯 입력 조건의 latency와 memory를 기록했다.
- [x] Memory 측정 범위와 작은 진단 실험의 한계를 밝혔다.
- [x] vLLM 요청 예제와 concurrency benchmark 표를 준비했다.
- [x] 외부 media domain, redirect, file 크기, pixel, timeout 제한을 점검했다.
- [x] VLM production serving report에 들어갈 항목을 정리했다.

---

[^1]: Hugging Face. [Multimodal chat templates](https://huggingface.co/docs/transformers/en/chat_templating_multimodal). Image와 text content를 섞은 message 구조와 `apply_chat_template()` 사용법을 참고했다. 확인일: 2026-08-04.
[^2]: Li, F. et al. (2024). [LLaVA-NeXT-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models](https://arxiv.org/abs/2407.07895). Figure 2와 interleaved multi-image 형식, multi-image·multi-frame·multi-view·multi-patch 구성을 참고했다.
[^3]: 직접 실행한 `llm_lecture/week22_vlm_inference_serving.py`의 결과다. SmolVLM-256M-Instruct, macOS MPS, 16 output token, warm-up 1회, 조건별 3회 측정의 중앙값을 사용했다. Script, raw CSV, JSON과 합성 image는 Git에서 제외하고 최종 plot만 공개했다. 실행일: 2026-08-04.
[^4]: Hugging Face. [Image-text-to-text](https://huggingface.co/docs/transformers/main/tasks/image_text_to_text). `AutoModelForImageTextToText`, Processor, multimodal message, 반복 생성에서 encoded multimodal 표현을 재사용하는 방법을 참고했다. 확인일: 2026-08-04.
[^5]: vLLM. [Multimodal Inputs](https://docs.vllm.ai/en/v0.21.0/features/multimodal_inputs/). OpenAI 호환 multimodal request, request별 media 개수 제한, 허용 domain 설정을 참고했다. 확인일: 2026-08-04.
[^6]: vLLM. [Security](https://docs.vllm.ai/en/latest/usage/security/). 원격 media redirect와 server 보안 지침을 참고했다. 확인일: 2026-08-04.
[^7]: vLLM. [Engine Arguments](https://docs.vllm.ai/en/v0.14.1/configuration/engine_args/). Local media path 허용의 보안 경고를 참고했다. 확인일: 2026-08-04.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
원본/윤문본: 15,171자 / 15,237자, 글자 수 기준 변경률 0.43%
카테고리별 탐지/수정: A-10 모호한 가능 표현 2→0, C-11 긴 복합문 1→0, D-1 관용구 0→0, H-1 번역투 피동 0→0
정량 점검: humanize-korean metrics v2.0 risk band low
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 model revision, 실험 수치, 측정 범위, 보안 설정과 참고 문헌을 보존함
주요 변경: 가능 표현을 직접 서술로 바꾸고 visual token 계산 문장을 둘로 나누어 읽기 쉽게 다듬음
-->
