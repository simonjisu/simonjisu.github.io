---
title: "3주차. Supervised Fine-Tuning"
description: "질문과 모범 답안을 token으로 바꾸고, assistant 답변의 loss로 작은 언어 모델을 학습하는 과정을 익힌다."
tags:
  - LLM
  - SFT
  - exposure bias
  - supervised fine-tuning
  - TRL
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 16주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

2주차에서는 Base model과 instruction model의 차이를 살펴봤다. 이번 주에는 그 사이에 있는 SFT 학습을 직접 따라간다. 질문과 모범 답안 한 쌍이 어떻게 token, label, loss로 바뀌는지 확인하고 작은 데이터로 학습을 실행한다.

## 이번 주에 배울 것

- SFT 데이터 한 행에 들어가는 `system`, `user`, `assistant` 메시지
- assistant가 쓴 token에만 loss를 계산하는 방법
- teacher forcing과 exposure bias가 생기는 이유
- padding, truncation, packing이 필요한 이유
- train, validation, test 데이터를 나누는 기준
- `SFTTrainer`로 작은 SFT 실험을 실행하고 기록하는 방법

선수 지식은 1주차의 next-token loss와 2주차의 chat template이다.

!!! note "모범 답안을 보고 푸는 연습"

    수학 문제와 풀이 예시를 함께 보면 학생은 정답뿐 아니라 답을 쓰는 방식도 익힌다. SFT도 prompt와 모범 response를 한 쌍으로 보여준다. 모델은 정답 token의 확률을 높이는 쪽으로 weight를 고친다.

## 1. SFT는 좋은 답의 예시를 따라 배운다

![InstructGPT의 demonstration 수집과 SFT 단계](/notes/tutorial/llm_lecture/images/w03_sft_demonstration_stage.png)

*그림 1. 사람이 prompt에 맞는 demonstration을 작성하고 SFT model을 학습하는 단계. 출처: Ouyang et al. (2022), Figure 2의 첫 번째 열에서 발췌.[^1]*

그림에서는 사람이 prompt를 읽고 원하는 답을 직접 쓴다. 이 답이 demonstration, 곧 모델이 따라 배울 모범 답안이다. 여러 demonstration을 모아 학습하면 Base model은 질문에 답하고 형식 지시를 따르는 습관을 익힌다.[^1]

SFT 데이터 한 행은 보통 다음처럼 대화를 담는다. 모델이 JSON 문법을 배우는 것은 아니다. `SFTTrainer`가 대화형 데이터를 받으면 tokenizer의 chat template을 적용해 하나의 token 열로 바꾼다.[^2]

```json
{
  "messages": [
    {"role": "system", "content": "중학생에게 두 문장으로 설명한다."},
    {"role": "user", "content": "무지개는 왜 생겨?"},
    {"role": "assistant", "content": "햇빛이 빗방울 안에서 꺾이고 여러 색으로 나뉘기 때문이다. 나뉜 빛이 눈에 들어오면 둥근 색 띠로 보인다."}
  ]
}
```

좋은 SFT 데이터는 답이 맞는지만 보지 않는다. 사용자의 조건을 지켰는지, 읽기 쉬운지, 위험한 요청을 알맞게 다루는지도 살핀다. 틀린 모범 답안은 모델이 그대로 따라 배울 수 있으므로 학습 전에 사람이 표본을 읽어봐야 한다.

## 2. 모든 token을 똑같이 채점하지 않는다

chat template을 적용하면 system, user, assistant 메시지와 control token이 한 줄로 이어진다.

```text
token 역할   system system user user assistant assistant assistant
loss mask       0      0     0    0       1         1         1
label          -100   -100  -100 -100     842       19       731
```

PyTorch의 cross-entropy loss에서는 label이 `-100`인 위치를 무시한다. user token을 `-100`으로 가리고 assistant token만 원래 ID로 남기면, 질문 내용을 외우는 대신 답변을 생성하는 부분에 학습 신호를 집중한다.

assistant token인지 표시하는 mask를 $m_t$라고 하면 SFT loss는 $\mathcal{L}_{\text{SFT}}=-\frac{1}{\sum_t m_t}\sum_{t=1}^{T}m_t \log p_\theta(y_t \mid y_{<t})$로 쓴다.

$m_t=1$인 token만 loss에 들어간다. 다만 어떤 부분을 가릴지는 데이터 형식과 실험 목적에 따라 달라진다. TRL의 `assistant_only_loss=True`는 대화형 데이터에서 assistant 메시지만 학습한다. 이 기능을 쓰려면 chat template이 assistant 구간을 표시해야 한다.[^2][^3]

Qwen3가 배포한 기본 template에는 학습용 assistant 표시가 없을 수 있다. 현재 TRL은 알려진 model family의 template을 학습용으로 자동 보완한다. 아래 코드는 같은 보완 함수를 먼저 적용한 뒤 assistant mask와 label을 확인한다. `assistant_masks`의 0은 `-100`, 1은 token ID로 바꾼다.[^2][^3][^4]

```python
from transformers import AutoTokenizer
from trl.chat_template_utils import get_training_chat_template

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
training_template = get_training_chat_template(tokenizer) or tokenizer.chat_template

messages = [
    {"role": "user", "content": "무지개는 왜 생겨?"},
    {"role": "assistant", "content": "햇빛이 빗방울에서 꺾이고 여러 색으로 나뉘기 때문이다."},
]

encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_dict=True,
    return_assistant_tokens_mask=True,
    chat_template=training_template,
)

input_ids = encoded["input_ids"]
assistant_mask = encoded["assistant_masks"]
labels = [token_id if mask else -100 for token_id, mask in zip(input_ids, assistant_mask)]

for token_id, mask, label in zip(input_ids, assistant_mask, labels):
    token = tokenizer.decode([token_id])
    print(repr(token), "mask=", mask, "label=", label)
```

!!! warning "mask를 눈으로 확인한다"

    `assistant_only_loss=True`를 켰다고 끝난 것이 아니다. template이 assistant 구간을 지원하지 않으면 원하는 mask가 나오지 않기도 한다. 첫 batch의 token, mask, label을 출력해 질문은 가려지고 답변은 남았는지 확인한다.

## 3. Teacher forcing과 exposure bias

SFT loss 식의 $y_{<t}$에는 중요한 조건이 숨어 있다. 학습할 때 $y_{<t}$는 사람이 작성한 모범 답안의 왼쪽 token이다. 모델이 앞 위치에서 엉뚱한 token을 예측했더라도 다음 위치를 학습할 때는 다시 올바른 token을 보여준다. 이 방식을 teacher forcing이라고 부른다.[^7]

Decoder-only Transformer는 학습 중에 이전 token을 한 개씩 되먹이지 않는다. 문장 전체를 넣고 causal mask로 미래를 가린 채 여러 위치를 병렬로 계산한다. 그래도 각 위치가 읽는 왼쪽 문맥은 모범 답안에서 왔으므로 teacher forcing과 같은 조건이다.[^7]

```text
모범 답안:  물은 1기압에서 섭씨 100도에 끓는다

학습할 때
입력 prefix: 물은 1기압에서 섭씨
다음 정답:   100도에

생성할 때
모델 출력:   물은 보통 섭씨 90도에
다음 입력:   물은 보통 섭씨 90도에
```

학습에서는 실제 데이터의 올바른 prefix만 읽는다. 생성에서는 모델이 조금 전에 고른 token이 다음 입력에 포함된다. 이처럼 학습과 생성에서 접하는 prefix의 분포가 달라지는 현상이 exposure bias다. 앞에서 잘못 고른 token 때문에 다음 예측의 문맥까지 달라진다. 긴 답에서는 오류가 이어지기도 한다.[^7]

| 구분 | 이전 token의 출처 | 특징 |
| --- | --- | --- |
| SFT 학습 | 모범 답안 | 모든 위치의 loss를 병렬로 계산하기 쉬움 |
| 실제 생성 | 모델의 출력 | 한 token씩 생성하며 이전 선택의 영향을 받음 |

Exposure bias는 causal mask가 고장 나서 생기는 문제가 아니다. Causal mask는 미래 token을 가린다. Teacher forcing은 현재 위치보다 왼쪽에 있는 정답 token을 학습 입력으로 쓰는 방법이다. 미래를 훔쳐보지는 않는다. 다만 생성 중에 만날 틀린 prefix를 학습에서는 거의 보지 않는다는 차이가 남는다.

### 오류가 항상 끝까지 커지는 것은 아니다

Exposure bias를 설명할 때 “한 번 틀리면 뒤가 모두 무너진다”라고 단정하면 안 된다. He et al.은 open-ended text generation을 평가했다. 모델이 만든 prefix 때문에 생긴 품질 저하는 제한적이었고 생성 길이에 따라 계속 커지지 않았다. 모델이 어긋난 문맥에서 다시 자연스러운 문장으로 돌아오는 self-recovery도 관찰했다.[^9]

반면 Arora et al.은 imitation learning 관점의 실험에서 exposure bias가 오류를 누적시키고 생성 품질을 낮추는 증거를 제시했다.[^10] 두 결과는 평가 task와 오류 측정법이 중요하다는 뜻이다. Exposure bias는 학습과 생성 사이에 실제로 존재하는 차이다. 그렇다고 모든 모델과 문장에서 같은 크기의 연쇄 오류를 만든다고 보지는 않는다.

### Scheduled Sampling은 간단한 해법처럼 보인다

Scheduled Sampling은 학습 초기에 모범 답안 token을 주로 사용하다가, 시간이 지날수록 모델이 생성한 token을 입력에 섞는 방법이다. 모델이 자신의 출력 뒤에서도 다음 token을 예측하도록 연습시키려는 생각이다.[^7]

```python
if random.random() < teacher_forcing_ratio:
    next_input = gold_token
else:
    next_input = sampled_model_token
```

그렇다고 modern LLM의 SFT에 이 코드를 바로 넣지는 않는다. Huszár는 Scheduled Sampling의 학습 objective가 일관되지 않으며 데이터의 올바른 joint distribution을 배우지 못한다고 지적했다.[^8] Model token을 하나씩 뽑아 다시 넣으면 병렬 학습의 장점도 줄어든다. Scheduled Sampling은 exposure bias를 이해하는 대표 연구이지만 언제나 적용하는 표준 처방은 아니다.

실무에서는 validation loss만 보지 않고 free-running generation을 함께 평가한다. Validation loss도 모범 답안 prefix에서 계산되므로 모델이 자신의 출력 뒤에서 얼마나 잘 회복하는지는 충분히 드러나지 않는다. 긴 답변을 직접 생성해 반복, 모순, 지시 이탈, 앞의 오류 뒤 회복 여부를 기록한다. 필요하면 정상 prefix와 일부러 흐트러뜨린 prefix를 각각 주고 이어지는 답을 비교한다.

!!! note "Exposure bias와 hallucination은 같은 말이 아니다"

    Exposure bias는 학습과 생성에서 읽는 prefix 분포가 다르다는 설명이다. Hallucination은 근거가 없거나 사실과 다른 내용을 생성한 결과를 가리킨다. Exposure bias가 일부 생성 오류와 관계가 있더라도 모든 hallucination의 원인으로 단정하지 않는다.

## 4. 길이가 다른 대화를 batch로 묶는 방법

대화마다 token 수가 다르다. GPU는 같은 batch 안의 tensor 크기가 맞아야 하므로 짧은 문장 뒤에 padding token을 붙인다.

```text
예시 A  [질문] [답] [답] [PAD] [PAD]
예시 B  [질문] [질문] [답] [답] [답]
```

padding token은 빈칸이므로 attention과 loss에서 제외한다. 너무 긴 문장은 `max_length`에 맞춰 자르는데, 이를 truncation이라고 한다. 답변 끝이 잘리면 학습 목표 자체가 사라진다. 길이 분포를 먼저 확인한 뒤 `max_length`를 정해야 하는 이유다.

packing은 여러 짧은 예시를 하나의 긴 sequence에 빈틈없이 넣는다.[^2]

```text
packing 전  [예시 A][PAD][PAD][PAD]  [예시 B][PAD][PAD]
packing 후  [예시 A][예시 B][예시 C 일부 ...]
```

padding이 줄어 GPU가 실제 token 계산에 더 많은 시간을 쓴다. 처음부터 packing을 켜면 오류를 찾기 어렵다. 첫 실험은 `packing=False`로 mask와 길이를 검증하고, 같은 설정에서 `packing=True`만 바꿔 처리 속도를 비교한다.

## 5. train, validation, test를 먼저 나눈다

한 문제를 학습할 때도 보고 평가할 때도 쓰면 모델이 외운 답을 잘한 것으로 착각한다. 데이터는 학습 전에 세 묶음으로 나눈다.

| split | 쓰임 | 학습 중 weight 변경 |
| --- | --- | --- |
| train | gradient를 계산해 모델을 학습 | 예 |
| validation | 설정을 고르고 과적합을 확인 | 아니요 |
| test | 마지막 결과를 한 번 평가 | 아니요 |

거의 같은 질문과 답이 서로 다른 split에 들어가는 중복 누수도 막아야 한다. 문장 하나가 완전히 같지 않더라도 이름이나 숫자만 바꾼 예시는 사실상 중복인 경우가 있다. 먼저 중복을 묶고, 그 묶음 단위로 split을 나누는 편이 안전하다.

## 6. 작은 SFT 실험 실행하기

아래 예시는 공식 TRL 문서에서 사용하는 `trl-lib/Capybara`의 일부와 `Qwen/Qwen3-0.6B`를 사용한다.[^2][^4][^5] 여기서는 SFT loss로 모든 weight를 바꾸는 full-parameter SFT를 실행한다. SFT는 학습 목적이고 LoRA는 바꿀 parameter의 범위를 줄이는 방법이므로 둘 중 하나만 골라야 하는 관계가 아니다. GPU 메모리가 부족하거나 작은 adapter만 보관하려면 [4주차의 LoRA SFT](/notes/tutorial/llm_lecture/w04_lora_qlora/)로 넘어간다.

```bash
pip install -U transformers datasets accelerate trl
```

```python
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

if not torch.cuda.is_available():
    raise RuntimeError("이 실습은 CUDA GPU를 기준으로 작성했다.")

dataset = load_dataset("trl-lib/Capybara", split="train[:1000]")
split = dataset.train_test_split(test_size=0.1, seed=42)

args = SFTConfig(
    output_dir="outputs/w03_qwen3_sft",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    max_length=512,
    packing=False,
    assistant_only_loss=True,
    eval_strategy="steps",
    eval_steps=50,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    model_init_kwargs={
        "dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    },
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    args=args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
)

train_result = trainer.train()
eval_result = trainer.evaluate()

print(train_result.metrics)
print(eval_result)
trainer.save_model()
```

이 코드는 학습 절차를 확인하는 작은 실험이다. 1,000개 예시와 1 epoch의 결과를 일반적인 모델 성능으로 해석하면 안 된다. `transformers`, `datasets`, `trl`, model revision, GPU 이름과 peak memory를 함께 기록한다.

## 7. loss가 내려가도 답변을 읽어봐야 한다

train loss는 모델이 학습 예시를 얼마나 잘 맞히는지 알려준다. validation loss는 처음 보는 예시에서도 비슷하게 맞히는지 보여준다. train loss만 계속 내려가고 validation loss가 오르면 학습 데이터를 외우기 시작했을 가능성이 있다.

수치만으로 지시 준수를 모두 평가할 수는 없다. 학습 전후 모델에 같은 질문 묶음을 넣고 아래 항목을 비교한다.

| 기록할 값 | 확인할 질문 |
| --- | --- |
| train/eval loss | 학습과 검증의 차이가 벌어지는가? |
| 답변 전문 | 사실이 맞고 문장이 자연스러운가? |
| 형식 준수 | “두 문장”, “JSON” 같은 조건을 지켰는가? |
| 생성 token 수 | 무조건 길게 답하는 습관이 생겼는가? |
| peak GPU memory | 다음 실험을 같은 장비에서 반복할 수 있는가? |

!!! warning "test를 설정 고르기에 쓰지 않는다"

    test 결과를 보고 learning rate나 epoch를 계속 바꾸면 test도 사실상 학습 과정에 들어간다. 설정은 validation으로 고르고, test는 마지막 비교에 남겨둔다.

## 8. 실제 학습 결과

로컬에서 빠르게 확인하려고 `HuggingFaceTB/SmolLM2-135M-Instruct`와 직접 만든 한국어 instruction 예시 24개를 사용했다. 20개는 학습에, 4개는 평가에 넣고 모든 weight를 16 step 동안 학습했다. 이 모델은 주로 영어를 이해하고 생성하므로 한국어 결과에는 분명한 한계가 있다.[^6]

![SmolLM2의 SFT train loss와 eval loss 변화](/notes/tutorial/llm_lecture/images/w03_sft_loss_result.png)

*그림 2. 16 step SFT의 train loss와 eval loss. 출처: SmolLM2-135M-Instruct 직접 실행 결과(2026-08-01, Apple MPS).[^6]*

| step | train loss | eval loss |
| ---: | ---: | ---: |
| 1 | 1.683 | 1.451 |
| 12 | 0.817 | 1.382 |
| 16 | 0.808 | 1.438 |

train loss는 1.683에서 0.808로 내려갔다. eval loss는 12 step까지 낮아지다가 마지막에 다시 1.438로 올랐다. 학습 예시는 더 잘 맞히지만 처음 보는 예시에서는 좋아지지 않는 과적합 신호로 볼 수 있다.

| 시점 | `무지개는 왜 생겨?`에 대한 생성 결과 |
| --- | --- |
| 학습 전 | `무지개는 왜 생겨?` |
| 학습 후 | `무지개는 아니다.` |

학습 후에도 답은 틀렸다. loss가 낮아졌다는 사실은 학습 데이터의 다음 token을 더 잘 맞혔다는 뜻일 뿐, 좋은 한국어 답변을 만들었다는 보장은 아니다. 데이터 24개와 16 step은 학습 코드가 움직이는지 확인하는 smoke test에 가깝다.

## 확인 문제

1. SFT가 pre-training과 같은 next-token loss를 쓰면서도 다른 행동을 가르치는 이유는 무엇인가?
2. user token의 label을 `-100`으로 바꾸면 loss 계산에서 어떤 일이 생기는가?
3. truncation이 assistant 답변 끝을 자르면 왜 문제가 되는가?
4. packing은 왜 속도를 높일 수 있으며, 첫 실험에서 끄는 편이 좋은 이유는 무엇인가?
5. train loss는 내려가는데 validation loss가 오르면 무엇을 의심해야 하는가?
6. Teacher forcing으로 학습할 때와 실제 생성할 때 이전 token의 출처는 어떻게 다른가?
7. Exposure bias가 있다고 해서 모든 오류가 생성 길이에 따라 계속 커진다고 단정하면 안 되는 이유는 무엇인가?
8. Scheduled Sampling을 modern LLM의 기본 학습법으로 바로 사용하면 안 되는 이유는 무엇인가?
9. SFT와 LoRA를 둘 중 하나만 고르는 관계로 보면 안 되는 이유는 무엇인가?

## 완료 체크

- [ ] 대화형 SFT 데이터 한 행을 직접 작성했다.
- [ ] token, assistant mask, label을 나란히 출력했다.
- [ ] teacher forcing과 실제 생성에서 prefix가 어떻게 달라지는지 설명했다.
- [ ] validation loss와 free-running generation을 함께 평가해야 하는 이유를 정리했다.
- [ ] train, validation, test를 나누고 중복 누수를 확인했다.
- [ ] 작은 `SFTTrainer` 학습을 실행하고 loss와 peak memory를 기록했다.
- [ ] SFT의 학습 목적과 full fine-tuning·LoRA의 parameter update 범위를 구분했다.
- [ ] 학습 전후 답변을 같은 생성 설정으로 비교했다.
- [ ] 결과물로 `작은 instruction model과 학습 기록`을 남겼다.

---

[^1]: Ouyang, L. et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). Figure 2와 §3을 참고했다.
[^2]: Hugging Face. [TRL: SFT Trainer](https://huggingface.co/docs/trl/sft_trainer). 데이터 형식, loss masking, packing, `assistant_only_loss`를 참고했다. 확인일: 2026-07-31.
[^3]: Hugging Face. [Transformers: Tokenizer `apply_chat_template`](https://huggingface.co/docs/transformers/main_classes/tokenizer). `return_assistant_tokens_mask`의 조건을 참고했다. 확인일: 2026-07-31.
[^4]: Qwen Team. [Qwen/Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B). 확인일: 2026-07-31.
[^5]: Hugging Face. [trl-lib/Capybara dataset](https://huggingface.co/datasets/trl-lib/Capybara). 확인일: 2026-07-31.
[^6]: Hugging Face. [HuggingFaceTB/SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct). 모델 크기, 사용법, 언어 한계를 참고했다. 확인일: 2026-08-01.
[^7]: Bengio, S. et al. (2015). [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://papers.nips.cc/paper_files/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html). 정답인 이전 token으로 학습하고 모델이 만든 이전 token으로 추론하면서 생기는 차이, 오류 누적, Scheduled Sampling을 참고했다.
[^8]: Huszár, F. (2015). [How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?](https://arxiv.org/abs/1511.05101). Scheduled Sampling objective가 improper하며 inconsistent learning으로 이어질 수 있다는 이론적 비판을 참고했다.
[^9]: He, T. et al. (2021). [Exposure Bias versus Self-Recovery: Are Distortions Really Incremental for Autoregressive Text Generation?](https://aclanthology.org/2021.emnlp-main.415/). Open-ended generation에서 prefix 차이로 인한 왜곡이 제한적이고 계속 누적되지 않았다는 결과와 self-recovery 분석을 참고했다.
[^10]: Arora, K. et al. (2022). [Why Exposure Bias Matters: An Imitation Learning Perspective of Error Accumulation in Language Generation](https://aclanthology.org/2022.findings-acl.58/). Exposure bias에 따른 오류 누적과 생성 품질 저하를 imitation learning 관점에서 분석한 결과를 참고했다.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: SFT와 LoRA의 관계를 설명하는 실행 안내
원본/윤문본: 13,677자 / 13,925자, metrics v2.0 변경률 1.22%
탐지/수정: C-11 연결어미 뒤 쉼표 0→0, A-10 가능 표현 0→0, D-1 결산 표현 0→0, H-1 문두 접속사 0→0, A-8 이중 피동 0→0
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 기존 SFT 실험의 설정과 결과를 보존함
주요 변경: “full fine-tuning은 모든 weight를 학습” → SFT 목적과 parameter update 범위를 구분하고 4주차를 연결함
-->
