---
title: "6주차. Reward Model과 RLHF"
description: "사람이 고른 chosen·rejected 답변으로 Reward Model을 학습하고 pairwise loss와 reward accuracy를 읽는 방법을 익힌다."
tags:
  - LLM
  - RLHF
  - reward model
  - preference learning
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 16주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

5주차에는 agent가 reward를 높이도록 policy를 바꾸는 원리를 배웠다. LLM의 답변에는 게임 점수처럼 미리 정해진 reward가 없는 경우가 많다. 이번 주에는 사람이 두 답을 비교한 자료로 Reward Model을 만들고 이 모델이 새 답변에 점수를 주는 과정을 살펴본다.

## 이번 주에 배울 것

- SFT, Reward Model, 강화학습으로 이어지는 RLHF의 세 단계
- `chosen`과 `rejected`로 이루어진 preference dataset
- Bradley–Terry 확률과 pairwise loss
- reward accuracy와 reward margin을 읽는 방법
- labeler disagreement, length bias, reward hacking을 점검하는 방법

선수 지식은 3주차의 SFT, 4주차의 LoRA, 5주차의 policy와 reward다.

!!! note "좋은 답을 직접 채점하기보다 둘 중 하나를 고른다"

    “이 답은 83점”이라고 정확히 매기기는 어렵다. 두 답을 나란히 놓고 어느 쪽이 더 나은지 고르는 일은 비교적 쉽다. Reward Model은 이런 비교를 많이 보고 사람이 더 자주 고른 답에 높은 숫자를 주도록 학습한다.

## 1. RLHF는 사람의 비교를 reward로 바꾼다

![InstructGPT의 SFT, Reward Model, PPO 학습 흐름](/notes/tutorial/llm_lecture/images/w06_instructgpt_rm_rlhf_pipeline.png)

*그림 1. demonstration으로 SFT model을 만들고 사람이 정렬한 답변으로 Reward Model을 학습한 뒤 PPO로 policy를 update하는 흐름. 출처: Ouyang et al. (2022), Figure 2에서 발췌.[^1]*

RLHF는 Reinforcement Learning from Human Feedback의 약자다. InstructGPT가 사용한 과정은 세 단계로 나뉜다.[^1]

1. 사람이 쓴 모범 답안으로 SFT model을 학습한다.
2. 같은 prompt에서 여러 답변을 만들고 사람이 선호 순서를 정한다. 이 자료로 Reward Model을 학습한다.
3. Reward Model의 점수를 reward로 사용해 policy를 강화학습한다.

이번 주의 초점은 두 번째 단계다. 세 번째 단계의 PPO는 7주차에서 다룬다.

Reward Model은 “모든 사람이 동의하는 절대적인 좋은 답”을 판정하지 않는다. 학습 데이터에 참여한 labeler가 어떤 기준으로 답을 골랐는지 근사한다. InstructGPT 실험에서도 labeler 사이의 일치율은 100%가 아니었다.[^1]

## 2. Preference dataset은 같은 질문의 두 답을 담는다

한 행에는 prompt, 더 선호한 답 `chosen`, 덜 선호한 답 `rejected`가 들어간다. 대화형 데이터는 다음처럼 쓴다.

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "식물이 햇빛을 필요로 하는 이유를 한 문장으로 설명해줘."
    }
  ],
  "chosen": [
    {
      "role": "assistant",
      "content": "식물은 햇빛 에너지를 이용해 물과 이산화탄소로 양분을 만들기 때문이다."
    }
  ],
  "rejected": [
    {
      "role": "assistant",
      "content": "식물은 햇빛을 좋아하기 때문이다."
    }
  ]
}
```

두 답은 같은 prompt에 대한 것이어야 한다. 질문이 다르면 Reward Model이 답의 품질이 아니라 질문의 차이를 단서로 삼는다. 답변 위치도 매번 무작위로 섞어야 “항상 위쪽 답을 고른다” 같은 편향을 막는다.

!!! warning "동점과 애매한 비교를 억지로 고르지 않는다"

    두 답의 품질이 비슷하거나 둘 다 틀렸다면 labeler가 확신 없이 고르기도 한다. 동점을 허용하거나 해당 예시를 따로 표시하고 여러 labeler가 얼마나 자주 동의했는지 기록한다. 개인정보와 유해한 내용도 학습 전에 걸러야 한다.

현재 TRL의 `RewardTrainer`는 standard·conversational preference format과 explicit·implicit prompt 형식을 지원한다. conversational format을 받으면 chat template을 자동으로 적용한다.[^3]

## 3. Reward Model은 답변 하나를 scalar로 바꾼다

Reward Model은 보통 Transformer가 prompt와 response를 함께 읽은 뒤 마지막에 score head를 붙여 숫자 하나를 출력한다. 분류 label의 이름을 맞히는 모델이라기보다 답변의 순서를 정하는 채점기에 가깝다.

같은 prompt $x$에 chosen 답변 $y^+$와 rejected 답변 $y^-$가 있다고 하자. Reward Model의 점수를 각각 $r_\theta(x,y^+)$와 $r_\theta(x,y^-)$로 쓴다. Bradley–Terry model은 chosen이 선택될 확률을 $P(y^+\succ y^-\mid x)=\sigma(r_\theta(x,y^+)-r_\theta(x,y^-))$로 나타낸다.[^2][^3]

두 점수의 차이가 클수록 sigmoid의 출력은 1에 가까워진다. chosen 점수가 더 낮으면 확률은 0.5보다 작아진다. 학습 loss는 $\mathcal{L}(\theta)=-\log\sigma(r_\theta(x,y^+)-r_\theta(x,y^-))$다.

작은 숫자로 계산해보자.

```text
chosen reward       = 1.2
rejected reward     = 0.4
reward margin       = 1.2 - 0.4 = 0.8
preference 확률     = sigmoid(0.8) ≈ 0.69
pairwise loss       = -log(0.69) ≈ 0.37
```

chosen reward를 올리거나 rejected reward를 내리면 margin이 커지고 loss가 줄어든다. 두 reward에 같은 수를 더해도 margin은 변하지 않는다. reward의 0점 자체에는 절대적인 의미가 없다. 현재 TRL은 reward 평균을 0 근처로 유도하는 선택적 보조 항 `center_rewards_coefficient`도 제공한다.[^3]

## 4. Accuracy와 margin을 함께 본다

pairwise accuracy는 전체 비교 중 $r_{\text{chosen}}>r_{\text{rejected}}$인 비율이다. 100쌍 가운데 73쌍의 순서를 맞혔다면 accuracy는 0.73이다. reward margin은 $r_{\text{chosen}}-r_{\text{rejected}}$의 평균이다.

| 지표 | 뜻 | 주의할 점 |
| --- | --- | --- |
| loss | 관찰한 선호의 negative log-likelihood | 데이터와 batch가 다르면 단순 비교하기 어려움 |
| accuracy | chosen에 더 높은 점수를 준 비율 | class balance와 중복 예시를 확인해야 함 |
| margin | chosen과 rejected reward의 평균 차이 | 지나치게 커져도 실제 일반화가 좋아졌다고 단정할 수 없음 |
| mean reward | model이 출력한 reward의 평균 | reward의 절대 위치는 임의적일 수 있음 |

train accuracy만 계속 오르고 validation accuracy가 멈추면 비교 자료를 외우고 있을 수 있다. 같은 prompt나 거의 같은 답변이 train과 validation에 동시에 들어가지 않았는지도 확인한다.

## 5. 작은 Reward Model을 학습한다

TRL 공식 문서는 `Qwen/Qwen3-0.6B`와 `trl-lib/ultrafeedback_binarized`를 RewardTrainer의 quick start 예제로 사용한다.[^3][^4][^5] 여기서는 4주차의 LoRA를 적용해 학습할 parameter를 줄인다. Base model에 원래 없던 `score` head도 저장해야 하므로 `modules_to_save=["score"]`를 지정한다.[^3]

```bash
pip install -U "trl[peft]" transformers datasets accelerate
```

```python
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from trl import RewardConfig, RewardTrainer

train_dataset = load_dataset(
    "trl-lib/ultrafeedback_binarized",
    split="train[:2000]",
)
eval_dataset = load_dataset(
    "trl-lib/ultrafeedback_binarized",
    split="test[:500]",
)

compute_dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["score"],
)

args = RewardConfig(
    output_dir="outputs/w06_qwen3_reward_model",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-3,
    max_length=512,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=10,
    bf16=compute_dtype == torch.bfloat16,
    fp16=compute_dtype == torch.float16,
    model_init_kwargs={"dtype": compute_dtype},
    report_to="none",
    seed=42,
)

trainer = RewardTrainer(
    model="Qwen/Qwen3-0.6B",
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)

trainer.model.print_trainable_parameters()
trainer.train()

metrics = trainer.evaluate()
print("eval accuracy:", metrics["eval_accuracy"])
print("eval margin:", metrics["eval_margin"])

trainer.save_model()
```

`max_length`보다 긴 chosen 또는 rejected sequence는 현재 RewardTrainer의 전처리 단계에서 제외될 수 있다.[^3] 긴 답변이 많은 dataset에서 이 값을 너무 작게 잡으면 데이터 분포가 달라진다. 학습 전후의 남은 행 수와 token 길이 분포를 기록한다.

!!! warning "이 예제의 숫자는 성능 기준이 아니다"

    2,000개 학습 예시와 1 epoch는 pipeline을 확인하기 위한 작은 실습이다. 좋은 Reward Model을 보장하는 설정이 아니다. GPU 종류, package version, 실제 학습 행 수, peak memory, validation 지표를 함께 남긴다.

## 6. Length bias를 직접 찔러본다

Reward Model이 “좋은 답은 길다”라는 지름길을 배웠을 수 있다. 정확한 짧은 답과 같은 내용을 불필요하게 반복한 긴 답을 만들어 점수를 비교한다.

```python
import torch

tokenizer = trainer.processing_class
model = trainer.model
model.eval()


def reward_score(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    batch = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.inference_mode():
        return model(**batch).logits.squeeze().item()


prompt = {
    "role": "user",
    "content": "Why does rain fall from clouds?",
}
short_answer = {
    "role": "assistant",
    "content": "Water droplets fall when they become too heavy for rising air to hold.",
}
padded_answer = {
    "role": "assistant",
    "content": (
        "Water droplets fall when they become too heavy for rising air to hold. "
        "In other words, they fall because they are heavy enough to fall. "
        "This repeats the same point without adding useful information."
    ),
}

print("short:", reward_score([prompt, short_answer]))
print("padded:", reward_score([prompt, padded_answer]))
```

긴 답의 점수가 한 번 높았다고 곧바로 length bias라고 결론 내리지는 않는다. 사실성, 문체, 안전성은 그대로 두고 길이만 바꾼 예시를 여러 개 만든다. 답변 token 수와 reward의 상관관계도 validation set에서 확인한다.

## 7. Reward hacking은 채점기의 빈틈을 이용한다

Reward Model도 제한된 데이터로 학습한 model이다. 사람이 중요하게 생각한 모든 기준을 완벽히 담지 못한다. policy를 reward에 지나치게 맞추면 실제 답변 품질은 나빠지는데 예측 reward만 오르는 reward overoptimization이 생긴다.[^6]

예를 들어 학습 데이터에서 공손한 답이 자주 chosen으로 뽑혔다면 Reward Model이 정확성보다 “물론입니다” 같은 표현에 점수를 주기 쉽다. policy는 이 표현을 반복해 reward를 높이려 한다. 다음 항목을 따로 점검한다.

- 답변 길이와 reward의 상관관계
- 특정 인사말, 사과, 글머리표를 붙였을 때의 점수 변화
- 사실을 한 군데 틀리게 바꾸었을 때 reward가 내려가는지
- train에 없던 주제와 언어에서도 chosen 순서를 맞히는지
- 여러 labeler가 동의하지 않은 예시에서 점수가 과도하게 확신하는지

Reward Model의 accuracy가 높아도 마지막 판단은 사람이 실제 답변을 읽어 확인한다. 자동 지표와 사람 평가가 어긋난 사례를 실패 기록에 남겨 두면 다음 학습에서 reward 기준을 고칠 근거가 된다.

## 8. 실제 학습 결과

`HuggingFaceTB/SmolLM2-135M-Instruct`에 LoRA를 붙여 작은 Reward Model을 만들었다. 직접 작성한 preference pair 12개 중 9개로 24 step 학습하고, 나머지 3개로 평가했다.[^3][^7]

![Reward Model의 pairwise loss와 평가 정확도 및 reward margin](/notes/tutorial/llm_lecture/images/w06_reward_model_result.png)

*그림 2. 학습 중 pairwise loss와 작은 평가 묶음의 정확도·평균 margin. 출처: SmolLM2-135M-Instruct Reward Model 직접 실행 결과(2026-08-01, Apple MPS).[^7]*

| 항목 | 결과 |
| --- | ---: |
| 평가 pair 수 | 3 |
| 마지막 eval loss | 0.000023 |
| preference accuracy | 1.000 |
| 평균 reward margin | 20.406 |

세 평가 pair에서 chosen의 reward가 모두 rejected보다 높았다. 하지만 평가 문제가 3개뿐이고 학습 데이터와 주제도 비슷하다. accuracy 1.000은 코드와 loss가 의도대로 움직였다는 확인값이지, 새로운 질문에서도 사람의 선호를 정확하게 맞힌다는 증거는 아니다.

길이만 늘린 답을 세 쌍 만들어 reward 변화를 살펴봤다.

| 질문 | 짧은 답 token | 늘인 답 token | 늘인 답의 reward 변화 |
| --- | ---: | ---: | ---: |
| 비가 구름에서 내리는 이유 | 14 | 37 | -0.720 |
| 달이 스스로 빛을 내는가 | 10 | 24 | +1.198 |
| tokenizer가 하는 일 | 10 | 29 | +3.577 |

불필요한 문장을 붙인 세 답 중 두 답의 reward가 오히려 높아졌다. 이 결과만으로 모델이 항상 긴 답을 좋아한다고 단정할 수는 없지만, 정확한 내용을 반복하기만 해도 점수가 오르는 length bias 후보를 찾았다. 더 많은 주제와 문장 길이에서 같은 현상이 반복되는지 확인해야 한다.

## 9. Reward Model 카드를 작성한다

| 항목 | 기록할 내용 |
| --- | --- |
| Base model | model ID와 revision |
| Preference 기준 | labeler가 chosen을 고른 규칙 |
| 데이터 | dataset 이름, train/eval 행 수, 언어 |
| 전처리 | chat template, `max_length`, 제외된 행 수 |
| 학습 | LoRA 설정, learning rate, batch, seed, GPU |
| 성능 | validation loss, accuracy, margin |
| 편향 점검 | 길이, 말투, 언어, 주제별 결과 |
| 제한 | 점수를 믿으면 안 되는 입력과 알려진 실패 |

숫자만 적지 말고 어떤 답에서 순서를 틀렸는지 예시를 함께 남긴다. Reward Model 카드는 다음 주 PPO 실습에서 reward가 이상해졌을 때 원인을 찾는 기준이 된다.

## 확인 문제

1. 점수를 직접 매기는 방식보다 두 답을 비교하는 방식이 쉬운 이유는 무엇인가?
2. chosen과 rejected는 왜 같은 prompt에 대한 답이어야 하는가?
3. chosen reward가 0.3, rejected reward가 0.8이면 preference 확률은 0.5보다 큰가, 작은가?
4. reward에 같은 상수를 더해도 pairwise loss가 바뀌지 않는 이유를 설명해보자.
5. validation accuracy가 높아도 Reward Model을 그대로 믿으면 안 되는 이유는 무엇인가?

## 완료 체크

- [ ] SFT→Reward Model→강화학습의 흐름을 설명했다.
- [ ] preference dataset 한 행을 `prompt`, `chosen`, `rejected`로 만들었다.
- [ ] reward margin과 pairwise loss를 작은 숫자로 계산했다.
- [ ] 작은 Reward Model을 학습하고 validation accuracy와 margin을 기록했다.
- [ ] 짧은 답과 불필요하게 늘인 답으로 length bias를 점검했다.
- [ ] Reward Model이 틀린 비교를 유형별로 다섯 개 이상 모았다.
- [ ] 결과물로 `Preference Reward Model 카드`를 완성했다.

---

[^1]: Ouyang, L. et al. (2022). [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155). Figure 2와 §3.1, §3.4, §3.5를 참고했다.
[^2]: Bradley, R. A. and Terry, M. E. (1952). [Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons](https://doi.org/10.2307/2334029). pairwise preference probability를 참고했다.
[^3]: Hugging Face. [TRL: Reward Modeling](https://huggingface.co/docs/trl/reward_trainer). dataset format, Bradley–Terry loss, metrics, `RewardConfig`와 PEFT 설정을 참고했다. 확인일: 2026-08-01.
[^4]: Cui, G. et al. (2023). [UltraFeedback: Boosting Language Models with Scaled AI Feedback](https://arxiv.org/abs/2310.01377). preference dataset의 배경을 참고했다.
[^5]: Qwen Team. [Qwen/Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B). 확인일: 2026-08-01.
[^6]: Gao, L., Schulman, J. and Hilton, J. (2022). [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760). reward를 지나치게 최적화할 때 생기는 성능 저하를 참고했다.
[^7]: Hugging Face. [HuggingFaceTB/SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct). 실행 모델의 구조, 사용법, 언어 한계를 참고했다. 확인일: 2026-08-01.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 절별로 5,000자 이하로 나누어 점검
원본/윤문본: 10542자 / 10515자, 변경률 0.34%
탐지/수정: A-10 7→1, C-11 5→0, D-1 1→0, H-1 0→0, 그 밖의 S1 0→0
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 논문 수치·모델명·실습 설정을 보존하며 번역투만 줄임
주요 변경: “쓸 수 있다”→“쓴다”, “생길 수 있다”→“생긴다”, 문두 “따라서” 삭제, 연결어미 뒤 불필요한 쉼표 삭제, Reward Model 실측표와 length bias 결과 추가
-->
