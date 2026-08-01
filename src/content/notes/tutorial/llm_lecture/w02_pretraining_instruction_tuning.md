---
title: "2주차. Pre-training과 Instruction Tuning"
description: "Base model이 다음 token을 배우고, instruction model이 질문에 맞춰 답하는 방식을 익히는 과정을 비교한다."
tags:
  - LLM
  - pre-training
  - instruction tuning
---

[← 16주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

1주차에서는 모델이 다음 token을 맞히며 학습한다는 사실을 배웠다. 이번 주에는 같은 학습 원리가 두 종류의 데이터에서 어떻게 쓰이는지 살펴본다. 일반 문서로 언어의 넓은 규칙을 익히는 단계가 pre-training이고, 질문과 모범 답안으로 지시를 따르는 연습을 하는 단계가 instruction tuning이다.

## 이번 주에 배울 것

- Base model과 instruction model의 차이
- pre-training 문서가 다음 token 정답으로 바뀌는 과정
- `system`, `user`, `assistant` 메시지가 token 열로 바뀌는 과정
- 같은 계열의 Base/Instruct model을 공정하게 비교하는 방법

선수 지식은 1주차의 token, logits, next-token loss다.

!!! note "많이 읽은 학생과 답하는 법을 연습한 학생"

    Base model은 책과 웹 문서를 아주 많이 읽고 “다음에는 어떤 말이 올까?”를 연습한 학생과 비슷하다. Instruction model은 그 학생에게 질문과 모범 답안을 더 보여주며 “요청을 읽고 이런 형식으로 답해”라고 가르친 모델이다.

## 1. Pre-training은 다음 token 문제를 대량으로 만든다

pre-training 데이터는 책, 문서, 코드처럼 이어진 text다. 학습 파이프라인은 보통 text를 모으고 정리한 뒤 token으로 바꾸고, 일정한 길이의 묶음으로 잘라 모델에 넣는다.

```text
원문 수집
  ↓
품질 검사·중복 제거·민감 정보 처리
  ↓
tokenization
  ↓
고정 길이 sequence 구성
  ↓
다음 token loss 계산
```

“고양이가 창가에 앉았다”라는 문장 하나에서도 여러 문제가 생긴다.

```text
입력                         정답
[고양이가]                   [창가에]
[고양이가, 창가에]           [앉았다]
[고양이가, 창가에, 앉았다]   [<eos>]
```

사람이 문장마다 정답표를 따로 만들 필요는 없다. 원문에서 오른쪽으로 한 칸 옮긴 token이 곧 정답이 된다. 이런 방식을 self-supervised learning이라고 부른다. “아무 도움 없이 스스로 배운다”는 뜻은 아니다. 데이터 수집, 정제, tokenizer 설계에는 여전히 사람의 판단이 들어간다.

pre-training을 마친 Base model은 문장을 자연스럽게 잇고 다양한 지식을 흉내 낸다. 다만 사용자의 질문에 짧고 정확하게 답하는 규칙은 충분히 배우지 못했을 수 있다. 그래서 질문을 이어 쓰거나 답 대신 비슷한 문서를 계속 생성하기도 한다.

## 2. Instruction tuning은 모범 답안의 형식을 가르친다

instruction tuning 데이터에는 보통 사용자의 요청과 assistant의 답이 함께 들어간다.

```json
{
  "messages": [
    {"role": "system", "content": "쉽고 짧게 설명한다."},
    {"role": "user", "content": "달이 왜 모양을 바꾸는지 알려줘."},
    {"role": "assistant", "content": "달이 지구를 돌면서 햇빛을 받는 부분이 다르게 보이기 때문이다."}
  ]
}
```

모델이 JSON을 직접 이해하는 것은 아니다. chat template이 role과 message 사이에 특별한 control token을 넣어 하나의 token 열로 만든다. 모델마다 학습에 사용한 control token이 다르므로, 다른 모델의 template을 섞어 쓰면 성능이 떨어질 수 있다.[^2]

```text
<|system|>쉽고 짧게 설명한다.<|end|>
<|user|>달이 왜 모양을 바꾸는지 알려줘.<|end|>
<|assistant|>달이 지구를 돌면서 ... 때문이다.<|end|>
```

위 control token은 원리를 설명하기 위한 예시다. 실제 문자열은 model tokenizer의 `chat_template`로 확인해야 한다.

```python
from transformers import AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "쉽고 짧게 설명한다."},
    {"role": "user", "content": "달이 왜 모양을 바꾸는지 알려줘."},
]

formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print(repr(formatted))
```

`add_generation_prompt=True`는 이제 assistant가 답할 차례라는 표시를 끝에 붙인다. 학습 데이터를 만들 때와 답을 생성할 때 이 값의 쓰임이 다르므로 공식 문서의 예제를 함께 확인한다.[^2]

## 3. InstructGPT 논문에서 보는 전체 흐름

![InstructGPT의 SFT, Reward Model, PPO 학습 흐름](/notes/tutorial/llm_lecture/images/w02_instructgpt_training_pipeline.png)

*그림 1. InstructGPT의 세 학습 단계. 출처: Ouyang et al. (2022), Figure 2에서 발췌.[^1]*

그림의 첫 번째 열이 이번 주와 다음 주에 집중할 SFT 단계다. 사람이 prompt에 맞는 답을 작성하고, 모델은 그 demonstration을 따라 하도록 학습한다. 두 번째 열의 Reward Model과 세 번째 열의 PPO는 6~7주차에 자세히 다룬다.

이 논문은 모델 크기만 늘린다고 사람의 의도를 더 잘 따르는 것은 아니라고 설명한다. 연구진은 demonstration으로 supervised fine-tuning을 하고, 답변 순위로 Reward Model을 학습한 뒤, PPO로 policy를 조정했다.[^1] Base model이 instruction model로 바뀌어도 새로운 언어 학습 장치가 생기지는 않는다. 모델은 여전히 다음 token의 확률을 계산하지만, 어떤 text를 정답으로 보여주는지가 달라진다.

## 4. Base model과 Instruct model 비교하기

Qwen2.5에는 같은 크기의 Base model과 Instruct model이 공개되어 있다. `Qwen/Qwen2.5-0.5B`는 pre-training 단계의 Base model이고, `Qwen/Qwen2.5-0.5B-Instruct`는 post-training을 거친 instruction model이다.[^3][^4]

두 모델에 같은 질문을 주되, 각 모델이 학습한 입력 형식을 지켜야 한다. Base model에는 평범한 text prompt를 넣고, Instruct model에는 chat template을 적용한다.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

question = "물의 끓는점을 중학생에게 두 문장으로 설명해줘."

def load(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return tokenizer, model

def generate(model, tokenizer, inputs):
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

base_id = "Qwen/Qwen2.5-0.5B"
base_tokenizer, base_model = load(base_id)
base_prompt = f"질문: {question}\n답변:"
base_inputs = base_tokenizer(base_prompt, return_tensors="pt")
print("[Base]", generate(base_model, base_tokenizer, base_inputs))

instruct_id = "Qwen/Qwen2.5-0.5B-Instruct"
inst_tokenizer, inst_model = load(instruct_id)
messages = [{"role": "user", "content": question}]
inst_inputs = inst_tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
print("[Instruct]", generate(inst_model, inst_tokenizer, inst_inputs))
```

GPU 메모리가 부족하면 두 모델을 동시에 올리지 말고 Base model의 결과를 저장한 뒤 메모리에서 내리고 Instruct model을 실행한다.

!!! warning "비교할 때 생성 설정을 맞춘다"

    한 모델은 무작위 sampling을 쓰고 다른 모델은 가장 확률이 높은 token만 고르면 학습 단계의 차이와 생성 방식의 차이가 섞인다. 두 모델의 `max_new_tokens`, `do_sample`, temperature 같은 설정을 같게 둔다. chat template은 각 모델이 학습한 형식을 따른다.

## 5. 무엇을 관찰해야 할까?

한 질문만 보고 어느 모델이 더 좋다고 결론 내리지 않는다. 아래처럼 성격이 다른 질문을 5개 이상 준비한다.

IFEval은 사람이 느낌으로 답을 평가하는 대신, 프로그램으로 확인할 수 있는 지시를 사용한다. 원래 benchmark에는 약 500개의 영어 prompt와 25가지 지시 유형이 들어 있다.[^5][^6] 이번 로컬 실험은 그중 10가지 유형의 원리를 짧게 옮긴 smoke test다. 공식 prompt와 평가 코드를 전부 사용한 IFEval 점수는 아니다.

| 지시 유형 | 통과 조건 |
| --- | --- |
| punctuation | 쉼표를 하나도 쓰지 않는가? |
| placeholder | 대괄호 placeholder를 정확히 2개 쓰는가? |
| bullet | 다른 문장 없이 markdown bullet을 정확히 3개 쓰는가? |
| heading | `Benefits`, `Risks` heading만 순서대로 쓰는가? |
| 대소문자 | 알파벳을 모두 대문자로 쓰는가? |
| keyword 횟수 | `orbit`을 정확히 2번 쓰는가? |
| 문장 수 | 정확히 2문장으로 답하는가? |
| 끝 문구 | `Keep practicing.`으로 끝나는가? |
| 첫 단어 | `Because`로 시작하는가? |
| 금지 단어 | `plant`, `plants`를 쓰지 않는가? |

결과표에는 답변 전문, token 수, 생성 시간, 지시 준수 여부를 적는다. 사람이 읽은 판단에는 이유를 한 문장으로 붙인다.

## 6. 실제 비교 결과

Qwen2.5-0.5B의 Base model과 Instruct model에 같은 영어 task 10개를 입력했다. Qwen2.5는 한국어를 포함한 여러 언어를 지원하지만, 이번 실험은 한국어 문장 생성 능력보다 학습 단계의 차이에 집중하려고 영어로 맞췄다. 두 모델 모두 greedy decoding을 사용하고 `max_new_tokens`를 96으로 설정했다.[^3][^4]

| 모델 | 자동 지시 준수 비율 | 평균 생성 token 수 | 평균 생성 시간 |
| --- | ---: | ---: | ---: |
| Base | 0.00 | 53.5 | 1.068초 |
| Instruct | 0.30 | 54.2 | 1.114초 |

| 질문 | Base | Instruct | 관찰한 차이 |
| --- | --- | --- | --- |
| 쉼표 금지 | 실패 | 실패 | 두 답 모두 쉼표를 썼다. |
| placeholder 2개 | 실패 | 통과 | Instruct만 대괄호 묶음을 정확히 2개 만들었다. |
| bullet 3개 | 실패 | 실패 | 두 모델 모두 markdown bullet 형식을 지키지 않았다. |
| heading 2개 | 실패 | 실패 | heading의 제목이나 개수가 조건과 달랐다. |
| 대문자만 쓰기 | 실패 | 통과 | Instruct는 `OCEAN OF VIBES`라고 답했다. |
| `orbit` 2번 | 실패 | 실패 | 두 모델 모두 정확한 단어 `orbit`을 2번 쓰지 않았다. |
| 2문장 | 실패 | 실패 | Base는 1문장, Instruct는 3문장을 썼다. |
| 정해진 문구로 끝내기 | 실패 | 실패 | Instruct는 문구를 처음에 썼지만 끝에는 쓰지 않았다. |
| `Because`로 시작하기 | 실패 | 통과 | Instruct만 첫 단어를 정확히 지켰다. |
| 금지 단어 피하기 | 실패 | 실패 | 두 모델 모두 `plant` 또는 `plants`를 썼다. |

![Qwen2.5 Base model과 Instruct model의 질문별 지시 준수, 평균 출력 길이와 생성 시간 비교](/notes/tutorial/llm_lecture/images/w02_base_instruct_result.png)

*그림 2. IFEval에서 착안한 영어 task 10개의 자동 지시 준수 여부, 평균 출력 길이, 평균 생성 시간. 출처: Qwen2.5-0.5B와 Qwen2.5-0.5B-Instruct 직접 실행 결과(2026-08-02, Apple MPS). Task 유형은 IFEval을 참고했다.[^3][^4][^5][^6]*

Base model은 10개 조건을 하나도 통과하지 못했고 Instruct model은 3개를 통과했다. placeholder 개수, 대문자, 첫 단어처럼 답의 내용보다 출력 규칙이 중요한 task에서 차이가 났다. 이 결과는 instruction tuning이 자연어로 적은 제약을 따르는 행동을 가르친다는 설명과 맞는다.

Instruct model도 7개 조건은 지키지 못했다. 모델 크기가 0.5B로 작고 평가 task도 10개뿐이므로 0.30을 일반적인 instruction-following 성능으로 해석하면 안 된다. 출력 상한에 닿았는지는 별도로 기록했으며, 자동 규칙은 지정된 조건만 검사한다. 내용의 정확성과 답변 완성도는 답변 전문을 읽어 따로 평가해야 한다.

## 7. 데이터가 모델의 행동을 만든다

pre-training과 instruction tuning은 데이터의 목적이 다르지만, 데이터 품질이 중요하다는 점은 같다.

- 중복 문서가 많으면 일부 표현을 지나치게 외울 수 있다.
- 개인정보가 섞이면 모델이 민감한 내용을 재생할 위험이 생긴다.
- 틀린 모범 답안을 넣으면 모델도 틀린 행동을 배운다.
- 특정 언어나 관점이 부족하면 그 영역의 성능이 약해질 수 있다.

학습 데이터의 양만 기록해서는 실험을 이해하기 어렵다. 출처, 정제 규칙, 중복 제거 방법, train/validation/test 분리 기준을 함께 남긴다.

## 확인 문제

1. pre-training을 self-supervised learning이라고 부르는 이유는 무엇인가?
2. Base model이 질문 뒤에 답 대신 새로운 질문을 이어 쓸 수 있는 까닭은 무엇인가?
3. chat template이 role 정보를 특별한 token으로 바꾸는 이유는 무엇인가?
4. Base model과 Instruct model을 비교할 때 생성 설정을 맞춰야 하는 이유는 무엇인가?
5. instruction tuning 뒤에도 모델이 하는 기본 계산이 next-token prediction인 이유를 설명해보자.

## 완료 체크

- [ ] pre-training과 instruction tuning의 데이터 차이를 설명했다.
- [ ] Qwen2.5 Instruct tokenizer의 chat template 결과를 출력했다.
- [ ] Base/Instruct model에 같은 질문 묶음을 실행했다.
- [ ] 답변, token 수, 생성 설정, 지시 준수 여부를 표로 정리했다.
- [ ] 데이터 품질 문제와 비교 실험의 한계를 기록했다.

---

[^1]: Ouyang, L. et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). 특히 Figure 2와 §3을 참고했다.
[^2]: Hugging Face. [Transformers: Chat templates](https://huggingface.co/docs/transformers/chat_templating). 확인일: 2026-07-31.
[^3]: Qwen Team. [Qwen/Qwen2.5-0.5B model card](https://huggingface.co/Qwen/Qwen2.5-0.5B). 확인일: 2026-07-31.
[^4]: Qwen Team. [Qwen/Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). 확인일: 2026-07-31.
[^5]: Zhou, J. et al. (2023). [Instruction-Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911). 검증 가능한 지시 유형과 평가 원칙을 참고했다.
[^6]: Google. [google/IFEval dataset](https://huggingface.co/datasets/google/IFEval). 541개 영어 prompt의 필드와 지시 유형을 확인했다. 확인일: 2026-08-02.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 절별로 5,000자 이하로 나누어 점검
원본/윤문본: 7322자 / 7615자, 변경률 2.55%
탐지/수정: E-2 2→0, I-2 1→0, A-10 2→1, 그 밖의 S1 0→0
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 신규 작성문을 보수적으로 다듬음
주요 변경: 반복되는 “~수 있다” 종결을 분산하고 “기억할 부분은 ~점이다”를 직접 서술로 수정, IFEval-style 영어 task 10개의 Base/Instruct 비교와 실패 원인 추가
-->
