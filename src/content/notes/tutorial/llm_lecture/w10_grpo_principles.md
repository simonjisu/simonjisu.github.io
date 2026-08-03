---
title: "10주차. GRPO의 원리"
description: "같은 prompt에서 여러 답을 뽑아 그룹 점수로 advantage를 만드는 GRPO를 손계산과 작은 정책 학습으로 익힌다."
tags:
  - LLM
  - GRPO
  - reinforcement learning
  - reward
  - reasoning
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 16주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

7주차의 PPO는 Value Model이 각 token 이후에 받을 점수를 예상했다. 큰 언어 모델만 한 번 더 올려도 메모리 부담이 크다. 이번 주에는 Value Model 대신 같은 문제에서 뽑은 여러 답의 평균을 기준선으로 삼는 GRPO를 배운다. DeepSeekMath는 수학 추론 학습에 이 방법을 사용했다.[^1]

## 이번 주에 배울 것

- PPO와 GRPO의 구성 요소 차이
- 한 prompt에서 여러 completion을 만드는 group sampling
- Group-relative advantage 손계산
- 정확도 reward와 형식 reward를 나누는 법
- Group size가 학습 신호와 메모리에 미치는 영향
- GRPO 학습 로그에서 reward와 분산을 읽는 법

선수 지식은 5주차의 policy gradient와 advantage, 7주차의 PPO·KL·clipping이다.

!!! note "반 전체의 평균을 임시 기준점으로 쓴다"

    같은 문제를 푼 답 네 개가 90점, 70점, 50점, 30점이라면 평균 60점보다 잘한 답과 못한 답을 나눌 수 있다. GRPO는 별도의 Value Model 대신 이 그룹 안의 상대 점수를 이용한다.

## 1. GRPO는 Value Model을 빼고 여러 답을 뽑는다

![PPO와 GRPO의 구성 요소 비교](/notes/tutorial/llm_lecture/images/w10_ppo_grpo_comparison.png)

*그림 1. PPO는 Value Model로 baseline을 추정한다. GRPO는 같은 질문에서 만든 여러 output의 reward를 묶어 상대 advantage를 계산하므로 Value Model이 필요 없다. 출처: Shao et al. (2024), Figure 4에서 발췌.[^1]*

| 구성 요소 | PPO | GRPO |
| --- | --- | --- |
| 한 prompt의 response | 보통 하나 이상 rollout | 같은 prompt에서 $G$개 completion |
| Baseline | 학습하는 Value Model | 그룹 reward의 평균 |
| Advantage | GAE 사용 | 그룹 안에서 reward 표준화 |
| Policy update | Probability ratio를 clip | Probability ratio를 clip |
| Reference와 KL | 보통 사용 | 원 논문은 사용, 구현 설정에 따라 생략 가능 |
| 큰 부담 | Value Model 학습 | Completion을 $G$개 생성·보관 |

GRPO가 항상 메모리를 적게 쓴다는 뜻은 아니다. Value Model은 빠지지만 prompt마다 여러 completion의 token, log probability와 reward를 저장해야 한다. 긴 reasoning을 큰 group으로 만들면 생성 비용이 커진다.[^1][^2]

## 2. 한 prompt에서 G개의 completion을 만든다

질문이 “3+5는?”이고 group size $G=4$라고 하자. Policy가 다음 네 답을 만들 수 있다.

```text
prompt: What is 3 + 5?

completion 1: <answer>8</answer>
completion 2: 8
completion 3: <answer>9</answer>
completion 4: I think nine
```

이 네 답은 서로 다른 질문의 답이 아니다. 같은 prompt에서 sampling한 한 묶음으로 보아야 한다. 이제 “이 문제에서 다른 답보다 얼마나 잘했는가?”를 계산할 수 있다.

Greedy decoding은 매번 확률이 가장 높은 token을 골라 여러 답이 같아지기 쉽다. GRPO rollout에는 보통 sampling과 적절한 temperature를 사용한다. 그래야 서로 다른 풀이와 오답도 나와 비교 신호가 생긴다.

## 3. 정확도와 형식을 따로 채점한다

수학 답이 맞더라도 약속한 `<answer>...</answer>` 형식을 지키지 않을 수 있다. 반대로 형식만 맞고 숫자는 틀릴 수도 있다. 하나의 reward 함수에서 둘을 섞어 버리면 어느 능력이 변했는지 알기 어렵다.

| Completion | 정확도 reward | 형식 reward | 합계 |
| --- | ---: | ---: | ---: |
| `<answer>8</answer>` | 1.0 | 0.2 | 1.2 |
| `8` | 1.0 | 0.0 | 1.0 |
| `<answer>9</answer>` | 0.0 | 0.2 | 0.2 |
| `I think nine` | 0.0 | 0.0 | 0.0 |

형식 reward를 1.0처럼 정확도와 같은 크기로 주면 model이 쉬운 태그만 맞추고 계산은 소홀히 할 수 있다. 이 예에서는 정확도 1.0, 형식 0.2로 두었다. 숫자 자체는 task와 validation 결과를 보고 정해야 한다.

```python
import re

def content_of(completion):
    # Conversational format의 completion은 assistant message 목록이다.
    return completion[0]["content"]

def accuracy_reward(completions, answer, **kwargs):
    rewards = []
    for completion, target in zip(completions, answer):
        match = re.search(r"<answer>(-?\d+)</answer>", content_of(completion))
        rewards.append(float(match is not None and int(match.group(1)) == target))
    return rewards

def format_reward(completions, **kwargs):
    return [
        0.2 if re.fullmatch(r"<answer>-?\d+</answer>", content_of(item).strip()) else 0.0
        for item in completions
    ]
```

TRL의 GRPOTrainer는 여러 reward function을 받을 수 있고 함수별 평균과 표준편차를 따로 기록한다. `rewards/accuracy/mean`, `rewards/format/mean`처럼 이름을 나누면 형식 점수만 오르는 reward hacking을 찾기 쉽다.[^2]

## 4. Group-relative advantage를 손으로 계산한다

앞의 total reward는 `[1.2, 1.0, 0.2, 0.0]`이다. 평균은 $\mu=0.6$, 모집단 표준편차는 $\sigma\approx0.510$이다. 각 답의 advantage는 $A_i=(r_i-\mu)/\sigma$로 계산한다.[^1][^2]

| Reward $r_i$ | $r_i-0.6$ | Advantage $A_i$ | 뜻 |
| ---: | ---: | ---: | --- |
| 1.2 | +0.6 | +1.177 | 확률을 올릴 강한 근거 |
| 1.0 | +0.4 | +0.784 | 확률을 올릴 근거 |
| 0.2 | -0.4 | -0.784 | 확률을 내릴 근거 |
| 0.0 | -0.6 | -1.177 | 확률을 내릴 강한 근거 |

평균보다 좋은 답은 양수, 나쁜 답은 음수 advantage를 얻는다. Reward의 단위를 그대로 쓰지 않고 그룹 안에서 표준화하므로 문제마다 점수 범위가 조금 달라도 상대 비교를 할 수 있다.

```python
import torch

rewards = torch.tensor([1.2, 1.0, 0.2, 0.0])
advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False)
print(advantages)
```

## 5. 모든 답의 점수가 같으면 배울 신호가 없다

네 답이 모두 틀려 reward가 `[0, 0, 0, 0]`이면 평균도 0이고 표준편차도 0이다. 어느 답이 상대적으로 나은지 정할 수 없다. 구현에서는 0으로 나누지 않도록 advantage를 0으로 처리한다.

```python
std = rewards.std(unbiased=False)
advantages = torch.where(
    std > 1e-8,
    (rewards - rewards.mean()) / (std + 1e-8),
    torch.zeros_like(rewards),
)
```

이런 zero-variance group이 많다면 다음 항목을 살핀다.

- Reward가 너무 엄격해 모든 completion에 0점을 주는가?
- Policy가 거의 같은 답만 반복하는가?
- Group size가 너무 작아 우연히 같은 점수만 뽑히는가?
- 부분 점수를 줄 수 있는 검증 가능한 과정이 있는가?

현재 TRL은 reward 표준편차가 0인 prompt 비율을 로그로 남긴다. 이 값이 높으면 optimizer가 움직였더라도 많은 그룹에서 유용한 상대 신호를 받지 못했다는 뜻이다.[^2]

## 6. Policy update에는 clipping과 KL을 쓸 수 있다

Advantage가 양수라고 새 policy가 그 답의 확률을 한 번에 크게 올리게 두면 불안정해진다. GRPO도 old policy와 현재 policy의 확률 비율을 clip해 한 update의 폭을 제한한다.[^1]

DeepSeekMath의 원래 GRPO는 reference model과의 KL도 loss에 넣었다. 현재 TRL의 기본 `beta`는 0이라 reference model을 불러오지 않는다. 원 논문 수식과 현재 library 기본값을 같은 것으로 생각하면 안 된다.[^1][^2]

현재 TRL에는 원래 GRPO 외에도 길이 편향과 정규화 문제를 고친 여러 loss type이 있다. 공식 문서는 원래 `grpo` loss보다 `dapo`, `dr_grpo` 같은 대안을 설명한다. 설치한 version의 기본값과 논문에서 재현하려는 식을 실험 기록에 함께 적는다.[^2]

## 7. Group size를 키우면 비교 기회와 비용이 함께 늘어난다

작은 categorical policy가 `0+0`부터 `9+9`까지 100개 prompt의 답을 고르는 실험을 만들었다.[^3] Action 25개 중 0~18은 올바른 형식의 숫자, 나머지 6개는 형식 오류다. 정확도 reward는 1.0, 형식 reward는 0.2다.

각 설정은 800번 update했고, 한 update마다 prompt 16개를 골랐다. 바꾼 값은 group size뿐이다. 이것은 언어 모델 품질 실험이 아니라 group-relative update의 움직임을 확인하는 작은 계산 실험이다.

![GRPO toy policy의 group size별 학습과 비용](/notes/tutorial/llm_lecture/images/w10_grpo_group_size_result.png)

*그림 2. Group size 2·4·8·16으로 같은 categorical policy를 학습한 결과. 왼쪽은 덧셈 정답률, 가운데는 reward 표준편차가 0인 그룹의 비율, 오른쪽은 completion 저장량의 상대값이다. PyTorch 2.13.0, macOS CPU에서 직접 실행했다.[^3]*

| Group size | 마지막 정답률 | 평균 group reward 분산 | Zero-variance group | 뽑은 completion | 저장량 상대값 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 100% | 0.0345 | 80.59% | 25,600개 | 1배 |
| 4 | 100% | 0.0371 | 75.87% | 51,200개 | 2배 |
| 8 | 100% | 0.0350 | 73.21% | 102,400개 | 4배 |
| 16 | 100% | 0.0321 | 71.48% | 204,800개 | 8배 |

네 설정 모두 이미 본 100개 prompt를 정확히 고르는 데 성공했다. Group이 커질수록 서로 다른 reward를 만날 기회가 늘어 zero-variance 비율은 낮아졌다. 동시에 생성하고 저장한 completion 수는 group size에 비례해 늘었다. 이 실험에서는 $G=16$이 $G=2$보다 학습 신호가 없는 그룹을 약 9.1%p 줄이는 대신 completion을 8배 썼다.

!!! warning "100%는 LLM 수학 성능이 아니다"

    이 결과는 25개 action 중 하나를 고르는 작은 정책이 이미 본 덧셈 prompt를 외운 값이다. 자유로운 문장을 생성하거나 처음 보는 문제를 푸는 능력을 측정하지 않았다.

## 8. GRPOTrainer에는 chat 형태의 prompt를 넣을 수 있다

Conversational dataset에서는 `prompt`를 `system`·`user` message의 목록으로 둔다. Trainer가 tokenizer의 chat template을 적용하고 한 prompt에서 `num_generations`개의 completion을 만든다.[^2][^4]

```python
dataset_row = {
    "prompt": [
        {"role": "system", "content": "Return only <answer>number</answer>."},
        {"role": "user", "content": "What is 17 + 26?"},
    ],
    "answer": 43,
}
```

```python
from trl import GRPOConfig, GRPOTrainer

args = GRPOConfig(
    output_dir="outputs/w10_grpo",
    num_generations=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_completion_length=64,
    learning_rate=5e-6,
    beta=0.01,
    loss_type="dr_grpo",
    report_to="none",
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[accuracy_reward, format_reward],
    args=args,
    train_dataset=dataset,
)
trainer.train()
```

Effective batch size는 `num_processes × per_device_train_batch_size × gradient_accumulation_steps`다. 이 값은 `num_generations`로 나누어떨어져야 한다. 작은 model이라도 completion 생성이 학습 loop 안에 들어가므로 SFT보다 오래 걸릴 수 있다.[^2]

## 9. Group reward 분석 노트를 작성한다

| 구분 | 남길 값 |
| --- | --- |
| Model | ID, revision, chat template, dtype |
| Prompt | train·validation 분리, 길이, 언어, task 종류 |
| Sampling | `num_generations`, temperature, maximum completion length |
| Reward | 함수별 범위, 가중치, 실패 처리, 검증 코드 |
| Optimization | loss type, `beta`, clip, learning rate, batch, update 수 |
| Reward log | 함수별 mean·std, 전체 reward, zero-variance 비율 |
| Policy log | KL, entropy, clip ratio, completion length |
| 비용 | 생성 token 수, peak memory, wall-clock time |
| 실제 답변 | 성공, 계산 오류, 형식 오류, reward hacking 사례 |

Reward 평균만 오르면 성공이라고 판단하지 않는다. 정확도와 형식 reward가 함께 올랐는지, zero-variance group과 KL이 지나치게 커지지 않았는지, 실제 답변이 더 나아졌는지 함께 본다.

## 확인 문제

1. GRPO에서 Value Model을 쓰지 않고 무엇을 baseline으로 삼는가?
2. Reward가 `[1.2, 1.0, 0.2, 0.0]`일 때 평균보다 높은 두 답의 advantage 부호는 무엇인가?
3. 정확도와 형식 reward를 따로 기록하면 어떤 편법을 찾기 쉬운가?
4. 모든 completion의 reward가 같으면 왜 policy가 배울 비교 신호가 사라지는가?
5. Group size를 키울 때 얻는 이점과 치르는 비용을 하나씩 설명해보자.

## 완료 체크

- [x] PPO와 GRPO의 구성 요소를 그림과 표로 비교했다.
- [x] 한 prompt의 completion 네 개로 group-relative advantage를 계산했다.
- [x] 정확도 reward와 형식 reward를 따로 구현하고 기록했다.
- [x] Group size 2·4·8·16에서 reward 분산과 completion 저장량을 비교했다.
- [x] 결과물로 `Group reward 분석 노트`를 완성했다.

---

[^1]: Shao, Z. et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). §3.2의 GRPO, Figure 4와 수식을 참고했다.
[^2]: Hugging Face. [TRL: GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer). Group sampling, reward function, advantage, loss type, 설정값과 logged metrics를 참고했다. 확인일: 2026-08-02.
[^3]: 직접 실행한 `llm_lecture/week10_grpo_demo.py`의 결과다. 100개 덧셈 prompt와 25개 action을 가진 categorical policy를 group size 2·4·8·16에서 각각 800 update했다. PyTorch 2.13.0, macOS CPU, seed 42를 사용했다. 원본 CSV와 코드는 Git에서 제외했다. 실행일: 2026-08-02.
[^4]: Hugging Face. [Transformers: Chat templates](https://huggingface.co/docs/transformers/chat_templating). Conversational message와 chat template 적용 방식을 참고했다. 확인일: 2026-08-02.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
카테고리별 탐지/수정: A-8 0→0, C-11 0→0, D-1 0→0, H-1 0→0, I-1 0→0
정량 점검: humanize-korean metrics v2.0 risk band low
자체검증: 고유명사·수식·실측값 보존 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음 / 변경률 30% 이하
등급: B — 자체검증 6/6을 통과했고 손계산과 group size별 실측값을 그대로 보존함
주요 변경: 같은 prompt의 completion이 한 그룹이라는 설명과 advantage의 부호를 자연스럽게 이어 씀
-->
