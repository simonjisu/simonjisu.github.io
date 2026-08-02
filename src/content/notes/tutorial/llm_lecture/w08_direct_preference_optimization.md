---
title: "8주차. Direct Preference Optimization"
description: "chosen·rejected 답변의 상대 log probability를 비교해 Reward Model과 rollout 없이 선호를 학습하는 DPO를 익힌다."
tags:
  - LLM
  - DPO
  - preference optimization
  - alignment
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 16주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

7주차의 PPO 기반 RLHF는 Reward Model로 답변을 채점하고 policy가 새 답변을 만들며 학습했다. 이 과정에는 여러 model과 rollout이 필요했다. 이번 주에는 같은 종류의 preference pair를 policy에 바로 가르치는 DPO를 살펴본다.[^1]

## 이번 주에 배울 것

- DPO와 PPO 기반 RLHF의 차이
- `prompt`, `chosen`, `rejected`로 이루어진 preference dataset
- Chat template이 두 답변을 token sequence로 바꾸는 과정
- Policy와 reference model의 sequence log probability
- DPO loss와 `beta`의 역할
- `rewards/margins`, `rewards/accuracies`를 읽는 방법

선수 지식은 2주차의 chat template, 3주차의 SFT, 6주차의 preference pair, 7주차의 KL과 reference model이다.

!!! note "별도 채점기를 만들지 않고 답안 비교를 바로 배운다"

    DPO에는 먼저 학습한 Reward Model이 없다. Policy가 chosen을 rejected보다 얼마나 더 그럴듯하게 보는지 계산한 다음 고정된 reference model의 판단과 비교해 loss를 만든다.

## 1. DPO는 RLHF pipeline을 짧게 만든다

![PPO 기반 RLHF와 DPO의 학습 흐름 비교](/notes/tutorial/llm_lecture/images/w08_dpo_vs_rlhf_pipeline.png)

*그림 1. PPO 기반 RLHF는 preference data로 Reward Model을 만든 뒤 response를 sampling하며 policy를 학습한다. DPO는 preference data에서 policy를 직접 최적화한다. 출처: Rafailov et al. (2023), Figure 1에서 발췌.[^1]*

| 항목 | PPO 기반 RLHF | DPO |
| --- | --- | --- |
| 학습 데이터 | prompt와 Reward Model score | prompt, chosen, rejected |
| 별도 Reward Model | 필요 | 필요 없음 |
| 학습 중 response 생성 | 필요 | 필요 없음 |
| Value model | 필요 | 필요 없음 |
| Reference model | KL 계산에 사용 | 상대 log probability 계산에 사용 |
| 학습 성격 | online policy optimization | offline preference optimization |
| 주요 위험 | rollout 비용, 불안정한 PPO update, reward hacking | preference data 품질, label noise, chosen·rejected 확률의 이상한 변화 |

DPO가 사람의 선호 없이 학습한다는 뜻은 아니다. 사람이나 다른 평가자가 이미 chosen과 rejected를 골라 둔 dataset이 필요하다. 줄어드는 것은 별도의 Reward Model 학습과 PPO rollout이다.[^1]

## 2. 한 행에는 같은 prompt의 두 답이 들어간다

현재 TRL의 DPOTrainer는 explicit prompt가 있는 preference format을 권장한다. 일반 문자열과 `role`·`content`를 쓰는 conversational format을 모두 받는다. Conversational data에는 chat template을 자동으로 적용한다.[^2][^3]

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "Explain why sleep matters in exactly two sentences."
    }
  ],
  "chosen": [
    {
      "role": "assistant",
      "content": "Sleep helps the brain store memories and regulate attention. It also gives the body time to repair tissues and balance hormones."
    }
  ],
  "rejected": [
    {
      "role": "assistant",
      "content": "Sleep is important because it is good for you."
    }
  ]
}
```

이 pair에서는 chosen만 두 문장 조건을 지켰다. 두 답의 사실성, 길이, 말투가 한꺼번에 다르면 model은 사람이 무엇을 선호했는지 알기 어렵다. 지시 준수 실험이라면 가능하면 다른 조건을 비슷하게 두고 지시를 지켰는지만 다르게 만든 pair도 섞는다.

!!! warning "chosen이 언제나 완벽한 정답은 아니다"

    DPO는 dataset에 적힌 순서를 배운다. Label이 뒤집혔거나 chosen과 rejected가 사실상 같은 답이면 잘못된 방향으로 update한다. 중복, 동점, 답변 위치 편향, 개인정보를 학습 전에 확인한다.

## 3. Chat template은 두 답을 같은 문법으로 감싼다

대화형 model도 실제로는 token sequence를 이어 쓴다. Chat template은 `user`와 `assistant` 같은 role을 control token으로 바꾼다.[^4] Qwen 계열 template의 형태를 단순화해 쓰면 다음과 같다.

```text
<|im_start|>user
Explain why sleep matters in exactly two sentences.<|im_end|>
<|im_start|>assistant
Sleep helps the brain store memories and regulate attention. It also gives the body time to repair tissues and balance hormones.<|im_end|>
```

Rejected sequence도 prompt 부분은 같고 assistant 답변만 바뀐다.

```text
<|im_start|>user
Explain why sleep matters in exactly two sentences.<|im_end|>
<|im_start|>assistant
Sleep is important because it is good for you.<|im_end|>
```

Model마다 control token이 다르므로 문자열을 직접 조립하지 않는다. Tokenizer에 저장된 template을 사용해 실제 결과를 확인한다.

```python
from transformers import AutoTokenizer

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = [
    {
        "role": "user",
        "content": (
            "Explain why sleep matters in exactly two sentences."
        ),
    }
]
chosen = [
    {
        "role": "assistant",
        "content": (
            "Sleep helps the brain store memories and regulate attention. "
            "It also gives the body time to repair tissues and balance hormones."
        ),
    }
]

rendered = tokenizer.apply_chat_template(
    prompt + chosen,
    tokenize=False,
    add_generation_prompt=False,
)
token_ids = tokenizer.apply_chat_template(
    prompt + chosen,
    tokenize=True,
    add_generation_prompt=False,
)

print(rendered)
print("token count:", len(token_ids))
```

완성된 답변을 학습 자료로 만들 때는 `add_generation_prompt=False`를 쓴다. 이미 assistant 답변이 들어 있는데 새 assistant 차례를 또 붙일 필요가 없기 때문이다. 문자열로 template을 적용한 뒤 따로 tokenize한다면 special token이 중복되지 않도록 `add_special_tokens=False`를 사용한다.[^4]

## 4. 답변의 확률은 token log probability의 합이다

Chosen 답변이 token 세 개 $c_1,c_2,c_3$로 나뉜다고 하자. Policy가 이 답변 전체에 주는 log probability는 $\log\pi_\theta(y_{chosen}\mid x)=\log p(c_1\mid x)+\log p(c_2\mid x,c_1)+\log p(c_3\mid x,c_1,c_2)$다.

확률을 그대로 곱하면 아주 작은 수가 되기 쉽다. Log를 쓰면 곱셈이 덧셈으로 바뀌어 계산하기 편하다. DPO는 policy의 chosen·rejected log probability와 reference model의 같은 두 값을 모두 사용한다.

```text
policy chosen logp    = -1.7
policy rejected logp  = -3.0
policy gap            = -1.7 - (-3.0) = 1.3

reference chosen logp   = -2.2
reference rejected logp = -2.4
reference gap           = -2.2 - (-2.4) = 0.2

relative margin = policy gap - reference gap
                = 1.3 - 0.2 = 1.1
```

Log probability는 0 이하인 경우가 보통이다. 덜 음수인 값이 더 높은 확률을 뜻한다. 위 예시에서 policy는 chosen을 rejected보다 훨씬 더 그럴듯하게 본다. Reference가 원래 갖고 있던 0.2의 차이를 빼도 relative margin은 양수다.

## 5. DPO loss는 reference보다 선호를 더 잘 구분하게 만든다

Chosen을 $y_w$, rejected를 $y_l$이라고 쓰자. Policy의 차이에서 reference의 차이를 뺀 relative margin은 $m=(\log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x))-(\log\pi_{ref}(y_w\mid x)-\log\pi_{ref}(y_l\mid x))$다.

DPO loss는 $\mathcal{L}_{DPO}=-\log\sigma(\beta m)$이다.[^1][^2]

- $m>0$: policy가 reference보다 chosen 쪽으로 더 기울었다.
- $m=0$: policy와 reference의 chosen·rejected 간격이 같다.
- $m<0$: policy가 reference보다 선호 순서를 덜 잘 구분한다.

$m=0$이면 sigmoid 입력도 0이므로 loss는 $-\log(0.5)\approx0.693$이다. Margin이 양수로 커지면 loss가 줄어든다. 음수로 내려가면 loss가 커진다.

!!! note "Reward Model이 사라졌지만 reward라는 생각은 남는다"

    DPO는 $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}$를 implicit reward로 해석한다. 별도 Reward Model이 숫자를 출력하는 것이 아니라 policy와 reference의 log probability 차이가 reward 역할을 한다.[^1]

## 6. `beta`는 reference에서 벗어나는 정도를 조절한다

현재 TRL 문서는 `beta`의 기본값을 0.1로 두며 값이 클수록 reference model에서 덜 벗어나도록 제어한다고 설명한다.[^2] 같은 relative margin을 loss에 넣으면 `beta`가 클수록 sigmoid 입력의 크기도 커진다.

![Relative preference margin과 beta에 따른 DPO loss](/notes/tutorial/llm_lecture/images/w08_dpo_beta_loss_curve.png)

*그림 2. 고정된 relative margin에 `beta` 0.1, 0.5, 1.0을 적용한 DPO loss. 출처: PyTorch 2.13.0으로 직접 계산한 결과(2026-08-02, macOS CPU).[^6]*

| relative margin | `beta=0.1` | `beta=0.5` | `beta=1.0` |
| ---: | ---: | ---: | ---: |
| -1.2 | 0.755 | 1.037 | 1.463 |
| 0.0 | 0.693 | 0.693 | 0.693 |
| +1.1 | 0.640 | 0.455 | 0.287 |

이 표는 학습이 끝난 뒤 policy가 얼마나 달라질지를 예측한 결과가 아니다. 고정된 margin을 loss에 넣은 수식 실험이다. 실제 학습에서는 `beta`가 gradient의 크기와 reference 제약에 함께 관여한다. 여러 값을 비교할 때는 validation margin, 실제 생성 답변, reference와의 KL을 함께 기록한다.

## 7. DPOTrainer로 작은 학습을 구성한다

TRL 공식 quick start는 `Qwen/Qwen3-0.6B`와 `trl-lib/ultrafeedback_binarized`를 사용한다.[^2][^5] 아래 코드는 5,000개 pair만 뽑고 LoRA를 붙인 실습용 설정이다. 자신의 3주차 SFT checkpoint가 있다면 `model_id`를 그 경로로 바꾼다.

```bash
pip install -U "trl[peft]" transformers datasets accelerate
```

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

model_id = "Qwen/Qwen3-0.6B"
dataset = load_dataset(
    "trl-lib/ultrafeedback_binarized",
    split="train[:5000]",
)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

compute_dtype = (
    torch.bfloat16
    if torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
    else torch.float32
)

args = DPOConfig(
    output_dir="outputs/w08_qwen3_dpo",
    beta=0.1,
    loss_type="sigmoid",
    learning_rate=5e-7,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    max_length=512,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=10,
    model_init_kwargs={"dtype": compute_dtype},
    report_to="none",
    seed=42,
)

trainer = DPOTrainer(
    model=model_id,
    ref_model=None,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
    ),
)

trainer.model.print_trainable_parameters()
trainer.train()
print(trainer.evaluate())
trainer.save_model()
```

`ref_model=None`이면 현재 DPOTrainer는 DPO 학습이 시작되기 전의 initial policy를 reference로 사용한다.[^2] PEFT를 쓰면 adapter만 학습하고 base weight는 고정한다. 긴 sequence가 많은 dataset에서 `max_length`를 너무 작게 잡으면 chosen이나 rejected의 중요한 뒷부분이 잘린다. Token 길이 분포를 먼저 확인한다.

!!! warning "Validation pair를 학습에 섞지 않는다"

    같은 prompt나 거의 같은 답변이 train과 validation에 동시에 들어가면 margin과 accuracy가 부풀 수 있다. 단순한 무작위 분할 뒤에도 중복 prompt를 찾아본다. 가능하면 prompt 단위로 묶어 분할한다.

## 8. DPO 로그의 reward는 Reward Model score가 아니다

현재 DPOTrainer는 다음 지표를 기록한다.[^2]

| 로그 | 뜻 |
| --- | --- |
| `loss` | batch의 평균 DPO loss |
| `logps/chosen` | policy가 chosen completion에 준 평균 log probability |
| `logps/rejected` | policy가 rejected completion에 준 평균 log probability |
| `rewards/chosen` | chosen의 implicit reward |
| `rewards/rejected` | rejected의 implicit reward |
| `rewards/margins` | chosen과 rejected implicit reward의 평균 차이 |
| `rewards/accuracies` | chosen implicit reward가 더 높은 pair의 비율 |
| `entropy` | token 분포가 얼마나 퍼져 있는지 나타내는 값 |

`rewards/chosen`과 `rewards/rejected`는 6주차 Reward Model이 출력한 score가 아니다. Policy와 reference의 log probability 비율로 계산한 implicit reward다. 이름만 보고 두 실험의 숫자를 직접 비교하면 안 된다.

Chosen log probability만 보는 것도 부족하다. DPO는 상대 차이를 학습하므로 chosen과 rejected가 함께 내려가면서 rejected가 더 빠르게 내려갈 수 있다. 실제로 현재 TRL 문서는 DPO가 보통 chosen 확률을 올리기보다 rejected 확률을 누르는 방식으로 목표를 달성하기도 한다고 설명한다.[^2] 학습 전후 generation을 반드시 비교한다.

## 9. 지시 준수 pair로 model을 시험한다

영어 model의 지시 준수를 확인하려면 자동으로 판정하기 쉬운 영어 task를 만든다. 각 task에는 chosen과 rejected를 함께 두고 무엇이 다른지 한 가지 기준으로 설명한다.

| task | chosen 조건 | rejected의 문제 |
| --- | --- | --- |
| 정확히 두 문장 | 문장 수가 2개 | 1개 또는 3개 |
| JSON만 출력 | parse 가능한 JSON | 설명문이나 Markdown fence 포함 |
| 세 단어로 답변 | whitespace 기준 3개 token | 단어 수 위반 |
| 금지 단어 피하기 | 지정 단어가 없음 | 금지 단어 포함 |
| 항목 순서 지키기 | `name`, `reason`, `risk` 순서 | key 누락 또는 순서 위반 |

학습 전후에 같은 prompt와 decoding 설정을 사용한다. 자동 검사 결과와 답변 전문을 함께 저장한다. JSON 형식은 맞아도 내용은 틀릴 수 있다. 두 문장을 지켰지만 사실이 틀린 답도 있다. 형식 점수와 내용 점수를 나누어 기록한다.

## 10. 실제 DPO smoke 학습을 실행했다

`HuggingFaceTB/SmolLM2-135M-Instruct`에 LoRA를 붙이고, 영어 지시 준수 pair 12개로 DPO smoke 학습을 실행했다.[^7][^8] 10개 pair는 학습에, 2개는 평가에 사용했다. `beta=0.1`, learning rate `5e-5`, 최대 길이 192 token, batch size 1로 12번 update했다. 실행 장치는 Apple Silicon의 MPS였고 dtype은 `float32`였다.

![SmolLM2 DPO smoke test의 loss와 implicit reward 지표](/notes/tutorial/llm_lecture/images/w08_dpo_training_smoke.png)

*그림 3. 12번의 DPO update에서 기록한 loss, implicit reward margin과 accuracy. Accuracy가 오르내리므로 마지막 숫자 하나만으로 학습 성공을 판단할 수 없다. TRL 1.9.2, PyTorch 2.13.0, macOS MPS에서 직접 실행했다.[^8]*

| 항목 | 측정값 |
| --- | ---: |
| 학습 / 평가 pair | 10개 / 2개 |
| optimizer step | 12회 |
| 실행 시간 | 4.50초 |
| 최대 메모리 | 1,239MB |
| train loss | 0.6860 |
| validation loss | 0.6967 |
| 영어 지시 준수율 | 학습 전 0% → 학습 후 0% |

학습에 쓰지 않은 영어 지시 다섯 개로 답변을 다시 만들었다. 정확히 두 문장, JSON만 출력, 세 단어만 출력, `ANSWER:`로 시작, `SAFE`만 대문자로 출력하는 task다. 다섯 답변은 학습 전후가 같았고 모두 자동 검사를 통과하지 못했다. 예를 들어 JSON task에는 `The answer is 1000 seconds.`라고 답했고, 대문자 task에는 `Safely`라고 답했다.[^8]

!!! note "실패한 결과도 학습 결과다"

    이 smoke test는 DPOTrainer가 preference pair를 읽고 LoRA adapter를 저장할 수 있다는 점을 확인했다. 하지만 pair 10개와 update 12번만으로 처음 보는 지시를 더 잘 따르게 만들지는 못했다. 다음 실험에서는 학습 pair를 늘리고 validation prompt와 겹치지 않는지 확인한 뒤, `beta`와 update 수를 하나씩 바꾸어 비교해야 한다.

## 11. 400개 pair로 늘리자 지시 준수율이 올랐다

Smoke test보다 학습 효과가 잘 보이도록 실험을 한 번 더 구성했다. 시작 model은 `Qwen/Qwen2.5-0.5B-Instruct`이고, 영어 지시 준수 task 10종을 만들었다.[^9][^10] 대문자만 출력하기, JSON만 반환하기, 정확히 두 개의 bullet 쓰기처럼 자동 판정이 가능한 task다.

Task마다 60개씩 만들고 앞의 40개는 train, 다음 10개는 validation, 마지막 10개는 test에 넣었다. 문장 틀은 같지만 안에 들어가는 숫자와 label은 split마다 겹치지 않는다. 전체 자료는 train 400쌍, validation 100쌍, test 100개 prompt다. Chat template을 적용한 chosen·rejected 문자열과 token 수도 CSV에 함께 저장했다.[^10]

```text
prompt:   Write code158x in uppercase and output nothing else.
chosen:   CODE158X
rejected: CODE158X!
```

LoRA의 rank는 16, learning rate는 `1e-5`, `beta`는 0.1, effective batch size는 8로 두었다. 1에포크는 50회, 3에포크는 150회 optimizer update에 해당한다. 평가는 학습에 쓰지 않은 test prompt 100개에 greedy decoding을 적용하고, 정답과 형식이 모두 맞는지 Python 함수로 판정했다.

![Qwen2.5-0.5B-Instruct의 DPO loss와 task별 지시 준수율](/notes/tutorial/llm_lecture/images/w08_dpo_demo_result.png)

*그림 4. 400개 preference pair로 3에포크 DPO 학습을 수행한 결과. 왼쪽은 train·validation loss, 오른쪽은 보지 않은 값으로 만든 10개 task의 통과율이다. TRL 1.9.2, PyTorch 2.13.0, macOS MPS에서 직접 실행했다.[^10]*

| 설정 | optimizer update | 실행 시간 | test 통과율 |
| --- | ---: | ---: | ---: |
| 학습 전 | 0회 | - | 24% |
| DPO 1에포크 | 50회 | 132.25초 | 47% |
| DPO 3에포크 | 150회 | 396.23초 | 49% |

3에포크 뒤 전체 통과율은 24%에서 49%로 25%p 올랐다. `lowercase_exact`, `number_only`, `two_bullets`는 test 10개를 모두 통과했다. 반면 `three_words`와 `uppercase_exact`는 하나도 통과하지 못했다. 처음에 모두 맞힌 `comma_no_spaces`는 10%로 떨어졌다. 평균 점수만 보면 가려지는 변화다.

| task | 학습 전 | 3에포크 뒤 |
| --- | ---: | ---: |
| `lowercase_exact` | 100% | 100% |
| `number_only` | 0% | 80% |
| `two_bullets` | 0% | 100% |
| `result_prefix` | 40% | 60% |
| `comma_no_spaces` | 100% | 10% |
| `three_words` | 0% | 0% |
| `uppercase_exact` | 0% | 0% |

Validation preference accuracy는 100%였고 loss도 0.0022까지 내려갔다. 그런데 실제 생성 통과율은 49%에 머물렀다. Chosen과 rejected를 구분하는 능력과 처음 보는 지시에 맞는 문장을 직접 만드는 능력은 같은 지표가 아니다. 이 실험에서 3에포크가 1에포크보다 얻은 이득도 2%p뿐이었다. Validation loss만 보고 더 오래 학습하기보다 task별 생성 평가를 멈춤 기준으로 삼아야 한다.[^2][^10]

!!! note "이 정도면 학습은 보이지만 완성된 model은 아니다"

    400쌍을 1에포크만 학습해도 23%p가 올라 smoke test보다 효과가 뚜렷했다. 하지만 목표로 잡은 70%에는 닿지 못했고 한 task는 오히려 나빠졌다. 좋은 demo에는 평균 통과율과 함께 좋아진 task, 나빠진 task, 그대로인 task를 모두 남겨야 한다.

## 12. Preference-tuned model 카드를 작성한다

| 항목 | 기록할 내용 |
| --- | --- |
| 시작 model | SFT checkpoint ID와 revision |
| reference | initial policy 사용 여부와 별도 checkpoint |
| dataset | 출처, pair 수, 언어, 선호 기준 |
| template | tokenizer ID, 실제 rendered chosen·rejected |
| 학습 | `beta`, loss type, learning rate, batch, epoch, seed |
| 효율 | trainable parameter, peak memory, 학습 시간 |
| 지표 | validation loss, margin, accuracy, chosen·rejected logp |
| 생성 평가 | 지시 준수율, 사람 선호, 대표 성공·실패 답변 |
| 제한 | label noise, 길이 편향, 다루지 못한 언어와 주제 |

## 확인 문제

1. DPO가 별도 Reward Model을 쓰지 않아도 사람의 선호 자료가 필요한 이유는 무엇인가?
2. `prompt`를 chosen과 rejected에 따로 복사하기보다 explicit field로 두면 무엇을 확인하기 쉬운가?
3. Policy gap이 0.8이고 reference gap이 0.3이면 relative margin은 얼마인가?
4. Relative margin이 0일 때 DPO loss가 약 0.693인 이유를 설명해보자.
5. `rewards/accuracies`가 높아도 실제 생성 답변을 읽어야 하는 이유는 무엇인가?

## 완료 체크

- [x] PPO 기반 RLHF와 DPO의 구성 요소를 표로 비교했다.
- [x] 영어 지시 준수 task로 `prompt`, `chosen`, `rejected` pair를 다섯 개 이상 만들었다.
- [x] Chat template을 적용한 chosen·rejected 문자열과 token 수를 저장했다.
- [x] 작은 숫자로 policy gap, reference gap, relative margin, DPO loss를 계산했다.
- [ ] 같은 dataset에서 `beta`를 바꾸어 margin과 생성 답변을 비교했다.
- [x] Validation의 chosen·rejected log probability를 함께 확인했다.
- [x] 결과물로 `Preference-tuned model과 model 카드`를 저장했다.

---

[^1]: Rafailov, R. et al. (2023). [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290). Figure 1, Eq. 5~7과 §4의 DPO 유도를 참고했다.
[^2]: Hugging Face. [TRL: DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer). DPO loss, `beta`, dataset 처리, reference model, PEFT와 logged metrics를 참고했다. 확인일: 2026-08-02.
[^3]: Hugging Face. [TRL: Dataset formats and types](https://huggingface.co/docs/trl/dataset_formats). Explicit prompt preference format과 conversational data 구조를 참고했다. 확인일: 2026-08-02.
[^4]: Hugging Face. [Transformers: Chat templates](https://huggingface.co/docs/transformers/chat_templating). Control token, `apply_chat_template`, `add_generation_prompt`와 special token 중복 주의를 참고했다. 확인일: 2026-08-02.
[^5]: Qwen Team. [Qwen/Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B). 실습 model의 구조와 Transformers 사용법을 참고했다. 확인일: 2026-08-02.
[^6]: 직접 실행한 `llm_lecture/week08.py`의 계산 결과다. PyTorch 2.13.0, macOS CPU에서 `beta` 0.1, 0.5, 1.0을 비교했다. 실행일: 2026-08-02.
[^7]: Hugging FaceTB. [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct). Model 구조, chat template과 사용 조건을 참고했다. 확인일: 2026-08-02.
[^8]: 직접 실행한 `llm_lecture/week08_dpo_smoke.py`의 결과다. SmolLM2-135M-Instruct, TRL 1.9.2, Transformers 5.14.1, PyTorch 2.13.0, macOS MPS를 사용했다. 원본 CSV와 adapter는 Git에서 제외했다. 실행일: 2026-08-02.
[^9]: Qwen Team. [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). Model 크기, chat template과 사용법을 참고했다. 확인일: 2026-08-02.
[^10]: 직접 실행한 `llm_lecture/week08_dpo_demo.py`의 결과다. Qwen2.5-0.5B-Instruct, train 400쌍, validation 100쌍, test 100개, `beta=0.1`, TRL 1.9.2, Transformers 5.14.1, PyTorch 2.13.0, macOS MPS를 사용했다. 원본 CSV와 adapter는 Git에서 제외했다. 실행일: 2026-08-02.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: DPO 1·3에포크 본 실험을 포함한 전체 문서를 절별로 나누어 점검
카테고리별 탐지/수정: A-8 0→0, C-11 0→0, D-1 0→0, H-1 0→0, I-1 0→0
정량 점검: humanize-korean metrics v2.0 risk score 1, low
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 DPO 수식, 모델명, 실측값과 task별 회귀를 그대로 보존함
주요 변경: preference accuracy와 실제 생성 통과율을 분리하고, 평균 뒤에 숨은 task별 변화를 짧게 풀어 씀
-->
