---
title: "20주차. 멀티모달 데이터와 Instruction Tuning"
description: "이미지와 글이 섞인 message를 VLM 입력으로 바꾸고, 정답 부분만 학습하는 LoRA 실습으로 데이터 구성과 대조 평가를 익힌다."
tags:
  - VLM
  - multimodal data
  - instruction tuning
  - chat template
  - LoRA
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 26주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

19주차에는 vision encoder와 LLM 사이를 잇는 connector를 살펴봤다. 이제 VLM에 어떤 문제와 답을 보여줄지 정해야 한다. 이번 주에는 이미지와 질문을 한 message에 담고, Processor가 이를 model 입력으로 바꾸는 과정을 따라간다. 마지막에는 2억 5천만 parameter 규모의 작은 VLM에 LoRA를 붙여 정해진 형식으로 답하는 연습을 시킨다.

## 이번 주에 배울 것

- Caption, VQA, OCR, 대화형 instruction data의 차이
- `content` 안에서 image와 text의 순서를 나타내는 방법
- Multimodal chat template과 image placeholder의 역할
- Processor가 만드는 `input_ids`, `pixel_values`, `attention_mask`
- User 질문을 가리고 assistant 답에만 loss를 계산하는 방법
- Vision encoder, connector, LLM LoRA 가운데 학습 범위를 고르는 기준
- Train, validation, test 분리와 이미지 중복 점검
- 이미지를 바꾸는 대조 실험으로 model이 사진을 실제로 쓰는지 확인하는 방법

선수 지식은 2주차의 chat template, 3주차의 SFT, 4주차의 LoRA, 17주차의 image tensor, 19주차의 connector다. 이 글에서는 사진 한 장과 짧은 영어 답을 사용한다. 실습 model이 영어 지시에 익숙하고 출력 형식을 자동으로 판정하기 쉽기 때문이다.

!!! note "사진 문제집 만들기"

    글만 배우는 학생에게는 질문과 모범 답안을 적은 문제집을 준다. VLM 문제집에는 사진도 들어간다. 사진, 질문, 답 중 하나라도 서로 맞지 않으면 model은 엉뚱한 관계를 배운다. 멀티모달 학습에서는 사진과 글의 짝을 정확히 맞추는 일이 먼저다.

## 1. 같은 이미지도 학습 목표에 따라 답이 달라진다

이미지 한 장에서도 여러 종류의 데이터를 만든다. 무엇을 답하게 하느냐에 따라 model이 연습하는 능력이 달라진다.

| 데이터 종류 | 입력 예시 | 답 예시 | 주로 연습하는 능력 |
| --- | --- | --- | --- |
| Caption | 사진을 설명해 줘 | 빨간 공 두 개가 탁자 위에 있다 | 장면을 문장으로 요약하기 |
| VQA | 공은 몇 개인가? | 2개 | 질문과 관련된 시각 정보 고르기 |
| OCR | 간판에 무엇이라고 적혀 있나? | OPEN | 이미지 속 글자 읽기 |
| 자세한 설명 | 위치와 색도 포함해 자세히 설명해 줘 | 왼쪽에는…, 오른쪽에는… | 여러 물체와 관계를 길게 말하기 |
| Instruction data | 정해진 JSON으로 답해 줘 | `{"count": 2}` | 이미지 이해와 지시 형식을 함께 지키기 |

Caption만 많이 학습했다고 해서 VQA나 OCR을 자동으로 잘하는 것은 아니다. Caption은 눈에 띄는 장면을 자연스럽게 말하면 되지만 VQA는 질문이 가리키는 정보만 골라야 한다. OCR은 작은 글자를 보존할 해상도와 읽기 자료가 따로 필요하다.

![LLaVA 논문의 conversation, detailed description, complex reasoning 데이터 예시](/notes/tutorial/llm_lecture/images/w20_llava_instruction_data.png)

*그림 1. Caption과 bounding box를 바탕으로 conversation, detailed description, complex reasoning 답을 만든 예시. 출처: Liu et al. (2023), Table 1에서 발췌.[^1]*

LLaVA 논문은 COCO 이미지의 caption과 bounding box를 글로 표현한 뒤, 이를 바탕으로 세 종류의 시각 instruction data를 만들었다. 논문이 보고한 고유 sample은 conversation 58K, detailed description 23K, complex reasoning 77K로 모두 158K다.[^1] 이 수치는 모든 VLM에 필요한 정답 수가 아니다. 한 연구가 데이터 종류를 어떻게 나눴는지 보여주는 사례로 읽어야 한다.

## 2. 한 message 안에서 image와 text의 순서를 표시한다

Text chat에서는 `content`가 문자열인 경우가 많다. Multimodal chat에서는 `content`를 여러 조각의 목록으로 만든다. 각 조각에는 `type`을 붙여 Processor가 사진과 글을 구분하게 한다.[^2]

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "red_circle.png"},
            {
                "type": "text",
                "text": "What color and shape are shown?",
            },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "COLOR=red; SHAPE=circle; COUNT=1",
            }
        ],
    },
]
```

목록의 순서도 입력의 일부다. 사진을 먼저 보여주고 질문을 쓰면 `image → text`가 된다. 여러 사진을 번갈아 넣는 model이라면 어느 질문이 어느 사진을 가리키는지 이 순서가 더 중요해진다.

Dataset에는 경로, URL, PIL image 중 무엇을 저장할지 정한다. 공개 데이터를 만들 때는 경로만 남기지 말고 image ID, 원본 출처, license, 수집 시각도 함께 보관한다. 나중에 중복을 찾거나 삭제 요청을 처리할 때 필요하다.

## 3. Chat template은 image placeholder와 역할 표시를 넣는다

`apply_chat_template`은 `user`, `assistant`, image, text 조각을 해당 checkpoint가 학습할 때 본 문자열 형식으로 바꾼다. Model마다 제어 token이 다르므로 다른 model의 template을 임의로 복사하면 안 된다.[^3]

이번 실습의 user message를 SmolVLM template에 넣은 결과는 다음과 같다.

```text
<|im_start|>User:<image>Reply with exactly one line in this format:
COLOR=<color>; SHAPE=<shape>; COUNT=<number>. What is shown?<end_of_utterance>
Assistant:
```

추론할 때는 아직 assistant 답이 없으므로 `add_generation_prompt=True`를 쓴다. 마지막의 `Assistant:`가 이제 답을 생성할 차례임을 알린다. 반대로 학습 자료에는 정답까지 넣고 `add_generation_prompt=False`로 처리했다.

```python
prompt_text = processor.apply_chat_template(
    user_messages,
    tokenize=False,
    add_generation_prompt=True,
)

training_text = processor.apply_chat_template(
    user_and_assistant_messages,
    tokenize=False,
    add_generation_prompt=False,
)
```

`<image>`는 한 개의 일반 text token과 같지 않다. Processor는 image placeholder를 visual token이 들어갈 자리만큼 펼친다. 실습에서는 이미지 분할을 껐고, 한 장이 image token 64개로 바뀌었다. 그래서 질문 prompt는 text가 짧아도 `input_ids` 길이가 109였다. 정답까지 붙인 학습 입력은 125 token이었다.[^4]

!!! warning "문자열에 `<image>`만 적으면 끝나는가?"

    아니다. Placeholder는 visual feature가 들어갈 위치를 알려줄 뿐이다. 실제 사진도 Processor에 넘겨 `pixel_values`를 만들어야 한다. Placeholder 개수와 이미지 수가 다르면 오류가 나거나 엉뚱한 사진을 읽는다.

## 4. Processor는 글과 사진을 서로 다른 tensor로 만든다

Processor는 tokenizer와 image processor를 한데 묶은 입구다. Text는 `input_ids`로, 이미지는 `pixel_values`로 바꾸고 두 입력을 함께 돌려준다.[^2]

```python
inputs = processor(
    text=prompt_text,
    images=[image],
    return_tensors="pt",
    do_image_splitting=False,
)

for name, tensor in inputs.items():
    print(name, tensor.shape)
```

이번 입력에서 확인한 shape는 다음과 같다.

| 항목 | Shape | 뜻 |
| --- | --- | --- |
| `input_ids` | `[1, 109]` | Image 자리와 질문을 포함한 token ID |
| `attention_mask` | `[1, 109]` | 실제 token과 padding을 구분하는 표시 |
| `pixel_values` | `[1, 1, 3, 512, 512]` | Batch 1, 이미지 조각 1, RGB, 높이와 너비 512 |
| `pixel_attention_mask` | `[1, 1, 512, 512]` | 실제 pixel과 padding pixel을 구분하는 표시 |

`do_image_splitting=True`인 기본 처리에서는 같은 작은 그림이 17개 image tile로 늘어났고 `input_ids`도 1,169개 안팎으로 길어졌다. 작은 도형 한 장의 pipeline을 확인하는 실습이라 분할을 껐다. 문서의 작은 글자나 큰 사진의 세부를 읽는 과제에서는 tile을 줄이면 필요한 정보도 사라진다. 비용만 보고 항상 끄면 안 된다.

## 5. Assistant 답에만 loss를 계산한다

학습 입력에는 user 질문과 assistant 정답이 모두 들어간다. 모든 token을 정답으로 두면 model은 답뿐 아니라 user 질문까지 외우는 loss를 받는다. 이번 실습에서는 assistant가 답하기 시작하기 전의 label을 `-100`으로 바꿨다. PyTorch의 cross-entropy는 이 값을 무시한다.

```python
prompt_length = int(prompt.attention_mask.sum())
labels = full.input_ids.clone()
labels[:, :prompt_length] = -100

outputs = model(
    input_ids=full.input_ids,
    attention_mask=full.attention_mask,
    pixel_values=full.pixel_values,
    labels=labels,
)
loss = outputs.loss
```

Token마다 loss를 계산할지 여부를 간단히 적으면 다음과 같다.

```text
User: <image> What is shown?  Assistant: COLOR=red ...
      xxxxxxxxxxxxxxxxxxxxxxxxx          ooooooooooo
      -100이라 loss에서 제외             정답 token이라 loss에 포함
```

TRL의 `SFTTrainer`도 VLM dataset의 `image`나 `images` 열을 처리한다. 대화형 dataset에서는 model의 chat template을 적용하며 일부 template은 `{% generation %}` 영역을 이용해 assistant-only loss를 지원한다.[^5] 어느 방식을 쓰든 첫 batch에서 `input_ids`, label, image token 위치를 사람이 직접 확인하는 편이 안전하다.

## 6. 무엇을 학습할지 먼저 정한다

VLM에는 vision encoder, connector, LLM이 있다. 모두 바꿀 수도 있고 일부만 바꿀 수도 있다.

| 학습 범위 | 장점 | 주의할 점 | 어울리는 경우 |
| --- | --- | --- | --- |
| Connector만 | Parameter와 저장량이 작음 | LLM의 말투나 복잡한 지시 습관은 덜 바뀔 수 있음 | 새 vision encoder와 LLM을 처음 맞출 때 |
| LLM에 LoRA | 출력 형식과 지시 수행을 적은 비용으로 조정 | 작은 시각 특징을 새로 배우는 데 한계가 있음 | 이미 이미지를 읽는 Instruct VLM을 과제에 맞출 때 |
| Connector와 LLM LoRA | 연결부와 답변 방식을 함께 조정 | 학습 설정과 저장할 adapter가 늘어남 | 새 domain의 표현과 지시가 모두 달라질 때 |
| 전체 fine-tuning | 모든 weight를 바꿀 수 있음 | GPU memory와 데이터가 많이 필요하고 기존 능력이 흐려질 수 있음 | 충분한 데이터와 계산 자원이 있을 때 |

이번 실습 model은 이미 색과 도형을 말할 수 있었다. 문제는 지정한 형식을 지키지 않는다는 점이었다. 그래서 vision encoder와 connector는 고정하고 LLM self-attention의 `q_proj`, `v_proj`에만 rank 8 LoRA를 붙였다. 학습 parameter는 460,800개로 adapter를 포함한 전체 256,945,728개의 약 0.179%였다.[^4]

학습 범위는 model 이름만 보고 정하지 않는다. 먼저 base output을 읽는다. 사진 자체를 잘못 본다면 connector나 vision 쪽 조정이 필요하다. 사진은 맞게 읽고 형식만 어긴다면 LLM LoRA부터 시도한다.

## 7. Image ID를 기준으로 split하고 누수를 찾는다

같은 사진에서 질문만 여러 개 만들었다면 질문별 무작위 분할은 위험하다. 거의 같은 사진이 train과 test에 함께 들어가면 model이 답을 외워도 높은 점수가 나온다. 촬영 원본, 문서 페이지, 영상 장면처럼 서로 묶인 단위의 ID를 먼저 만들고 그 ID로 split한다.

이번 합성 dataset은 빨강, 초록, 파랑과 원, 사각형, 1개, 2개의 조합 12가지를 사용했다. 각 조합에서 도형 위치만 다르게 그렸다.

| Split | 위치 variant | Sample 수 | 쓰임 |
| --- | ---: | ---: | --- |
| Train | 0, 1, 2 | 36 | LoRA weight update |
| Validation | 4 | 12 | Epoch를 고르는 중간 비교 |
| Test | 5 | 12 | 설정을 정한 뒤 한 번만 확인 |

모든 색, 모양, 개수 조합은 train에도 있다. Test는 새로운 개념을 알아보는 평가가 아니라, 학습 때와 다른 위치에서도 같은 지시를 따르는지 보는 작은 확인이다. 이 제한을 밝히지 않고 일반 VLM 성능으로 부르면 안 된다.

데이터를 나누기 전에 다음 항목을 점검한다.

- 같은 image hash나 가까운 perceptual hash가 여러 split에 있는가?
- 같은 영상의 이웃 frame이나 같은 문서의 연속 page가 갈라졌는가?
- 질문 안에 파일명, object label, OCR 정답이 들어가 있는가?
- 특정 답만 특정 문장 길이, 배경색, template과 묶였는가?
- Image를 제거해도 text만으로 정답을 쉽게 맞힐 수 있는가?

실습 질문은 모든 사진에서 똑같다. 질문에는 색, 모양, 개수의 정답이 없다. 따라서 답을 고르려면 이미지를 읽어야 한다.

## 8. 작은 VLM LoRA를 직접 학습한다

실습은 `HuggingFaceTB/SmolVLM-256M-Instruct` revision `7e3e67e`를 사용했다.[^6] Seed 20, batch 3, learning rate 0.0002, 10 epoch로 36개 training sample을 돌렸다. 총 optimizer step은 120번이다. MPS에서 LoRA update에 걸린 시간은 약 13.81초였으며 model load와 image 전처리와 평가는 이 시간에서 뺐다.[^4]

정답은 영어 한 줄로 고정했다.

```text
COLOR=red; SHAPE=circle; COUNT=1
```

Base model은 사진 내용을 짧게 맞혔지만 형식은 지키지 않았다.

```text
Target: COLOR=red; SHAPE=circle; COUNT=1
Before: Red circle
After:  COLOR=red; SHAPE=circle; COUNT=1
```

![SmolVLM LoRA의 학습 전후 정확도와 loss](/notes/tutorial/llm_lecture/images/w20_vlm_lora_results.png)

*그림 2. 합성 도형 36개로 LLM LoRA를 학습하고 위치가 다른 test 12개를 평가한 결과. macOS MPS, PyTorch 2.13.0, Transformers 5.14.1, PEFT 0.20.0에서 직접 실행했다.[^4]*

| 조건 | 엄격한 형식 | 전체 정답 일치 | 색 | 모양 | 개수 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 학습 전 | 0% | 0% | 0% | 0% | 0% |
| LoRA 학습 후 | 100% | 75% | 100% | 83.3% | 91.7% |
| 학습 후, 이미지 교체 | 100% | 0% | 75% | 33.3% | 8.3% |

색, 모양, 개수는 엄격한 형식에 맞은 답에서만 parser로 꺼냈다. 그래서 학습 전 `Red circle`처럼 의미상 맞는 부분이 있어도 항목 정확도는 0%다. 이 표는 내용 인식만 평가하는 표가 아니라 내용과 지시 형식을 함께 보는 결과다.

이미지 교체 대조군에서는 질문과 원래 정답은 그대로 두고 사진만 다음 sample의 사진으로 바꿨다. Model은 바뀐 사진에 맞춰 답을 바꾸었고 전체 정답 일치는 0%로 떨어졌다. 형식은 계속 100%였다. 이 차이는 LoRA가 답 형식을 배웠고, 답의 내용은 image input에 따라 바뀌었다는 증거다.

!!! warning "100% 형식 정확도를 크게 해석하지 않는다"

    Test가 12개뿐이고, train에서 이미 본 색·모양·개수 조합의 위치만 바꿨다. 실제 사진, 복잡한 배경, 처음 보는 물체, OCR, 긴 대화 능력을 증명하지 않는다. 작은 pipeline에서 template, label masking, LoRA update, 대조 평가가 연결됐는지 확인한 smoke experiment다.

## 9. 멀티모달 데이터 카드를 남긴다

Model adapter만 저장하면 어떤 사진과 규칙을 배웠는지 알 수 없다. 최소한 다음 내용을 데이터 카드에 함께 적는다.

| 항목 | 이번 실습의 기록 |
| --- | --- |
| 목적 | 이미지의 색, 모양, 개수를 지정한 영어 형식으로 답하기 |
| 출처 | PIL로 직접 만든 256×256 합성 도형 |
| 정답 범위 | 색 3종, 모양 2종, 개수 2종 |
| 전체 수 | 60개: train 36, validation 12, test 12 |
| Split 기준 | 같은 조합에서 위치 variant를 분리 |
| Template | SmolVLM checkpoint의 multimodal chat template |
| Image 처리 | 512×512, image splitting 끔, image token 64개 |
| Loss | Assistant answer token만 계산 |
| 학습 범위 | LLM `q_proj`, `v_proj`의 LoRA, rank 8 |
| 알려진 한계 | 단순한 흰 배경, 12개 조합, 실제 사진과 OCR 없음 |
| 대조 평가 | Test 질문과 정답을 고정하고 image만 순환 교체 |

실제 사용자 사진을 다룬다면 개인정보와 얼굴, 위치 정보, 저작권, 삭제 절차도 적어야 한다. 자동 생성 답은 틀릴 수 있으므로 생성 model과 prompt, 사람이 검수한 비율, 거절하거나 수정한 기준도 남긴다.

## 10. 자주 생기는 실수

| 실수 | 생기는 문제 | 확인 방법 |
| --- | --- | --- |
| Text-only chat template을 그대로 사용 | Image 위치가 빠짐 | Rendered template에서 placeholder 확인 |
| `add_generation_prompt`를 학습과 추론에서 뒤섞음 | Assistant marker가 겹치거나 사라짐 | 두 문자열을 나란히 출력 |
| User token에도 loss를 계산 | 질문을 베껴 쓰는 학습이 섞임 | `labels == -100` 구간 출력 |
| Image 수와 placeholder 수가 다름 | 사진과 질문의 대응이 깨짐 | Batch마다 두 개수 검사 |
| 같은 이미지의 질문을 split별로 흩음 | Test 누수로 점수가 부풀어 오름 | Image ID나 hash로 묶어 분할 |
| 질문에 정답 label을 넣음 | Image 없이도 답을 맞힘 | Text-only baseline 실행 |
| Train loss만 보고 끝냄 | 형식은 배웠지만 사진을 무시할 수 있음 | 원본, image 제거, image 교체를 비교 |
| 작은 smoke test의 100% 형식 정확도를 일반화 | 실제 VLM 품질을 과장함 | Dataset 범위와 실패 조건 함께 공개 |

## 확인 문제

1. Caption과 VQA는 같은 사진을 쓰더라도 model에 어떤 다른 연습을 시키는가?
2. Multimodal message의 `content`를 문자열 하나가 아니라 typed list로 만드는 이유는 무엇인가?
3. `<image>` placeholder가 한 개여도 `input_ids` 안의 image token이 여러 개인 이유는 무엇인가?
4. 추론에서 `add_generation_prompt=True`가 필요한 까닭을 chat template의 마지막 문자열로 설명해보자.
5. User 질문 token을 `-100`으로 가리면 loss 계산이 어떻게 달라지는가?
6. 사진은 맞게 읽지만 JSON 형식만 자주 어기는 model에는 어떤 학습 범위부터 시도할 수 있는가?
7. 같은 이미지의 caption은 train, VQA는 test에 넣으면 왜 데이터 누수가 될 수 있는가?
8. Test에서 정확도가 높아도 image를 바꿨을 때 답이 그대로라면 무엇을 의심해야 하는가?
9. 이번 image 교체 대조군에서 형식 정확도는 100%지만 전체 정답은 0%였다. 두 수치가 함께 나타난 까닭은 무엇인가?
10. 이번 실험의 100% 형식 정확도를 실제 사진의 VQA 성능이라고 부를 수 없는 이유를 세 가지 적어보자.

## 완료 체크

- [x] Caption, VQA, OCR, instruction data의 학습 목적을 구분했다.
- [x] Image와 text가 섞인 message와 실제 rendered chat template을 확인했다.
- [x] `input_ids`, `pixel_values`, image token 수와 tensor shape를 기록했다.
- [x] Assistant answer token에만 loss를 계산했다.
- [x] Image ID 관점에서 split과 데이터 누수 점검 항목을 정리했다.
- [x] SmolVLM의 LLM `q_proj`, `v_proj`에 LoRA를 학습했다.
- [x] 같은 test에서 학습 전후를 비교하고 image 교체 대조군을 실행했다.
- [x] Model revision, 설정, 한계가 담긴 멀티모달 데이터 카드를 작성했다.

---

[^1]: Liu, H. et al. (2023). [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485). §3과 Table 1의 multimodal instruction-following data 구성과 158K sample 내역을 참고했다.
[^2]: Hugging Face. [Multimodal chat templates](https://huggingface.co/docs/transformers/en/chat_templating_multimodal). Typed content, image input, Processor 사용법을 참고했다. 확인일: 2026-08-04.
[^3]: Hugging Face. [Chat templates](https://huggingface.co/docs/transformers/chat_templating). `apply_chat_template`, `add_generation_prompt`, model별 control token 차이를 참고했다. 확인일: 2026-08-04.
[^4]: 직접 실행한 `llm_lecture/week20_multimodal_instruction_tuning.py`의 결과다. 원본 합성 image, CSV, adapter, 실행 JSON은 Git에서 제외하고 최종 plot만 공개했다. Model은 `HuggingFaceTB/SmolVLM-256M-Instruct`, revision `7e3e67edbbed1bf9888184d9df282b700a323964`다. 실행일: 2026-08-04.
[^5]: Hugging Face TRL. [SFT Trainer](https://huggingface.co/docs/trl/en/sft_trainer). Vision-language dataset, conversational data, assistant-only loss 설명을 참고했다. 확인일: 2026-08-04.
[^6]: Hugging FaceTB. [SmolVLM-256M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct). Model 규모, image-text-to-text 사용 예, 학습 목적과 한계를 참고했다. 확인일: 2026-08-04.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
원본/윤문본: 13,586자 / 13,553자, 변경률 1.18%
카테고리별 탐지/수정: A-10 8→0, C-11 3→0, D-1 0→0, H-1 0→0
정량 점검: humanize-korean metrics v2.0 risk band low
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 model revision, tensor shape, 데이터 수, 학습 설정, 전후 지표를 그대로 보존함
주요 변경: 반복되던 가능 표현을 직접 서술로 바꾸고 연결어미 뒤의 불필요한 쉼표를 덜어냄
-->
