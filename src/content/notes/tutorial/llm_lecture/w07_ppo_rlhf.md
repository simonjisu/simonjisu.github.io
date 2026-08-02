---
title: "7주차. PPO로 배우는 RLHF"
description: "Policy, reference, Reward Model, value model이 함께 답변을 만들고 PPO clipping과 KL penalty로 policy를 조금씩 고치는 과정을 익힌다."
tags:
  - LLM
  - RLHF
  - PPO
  - policy optimization
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 16주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

6주차에는 사람이 고른 답변 쌍으로 Reward Model을 만들었다. 이제 policy가 직접 답변을 만들고 Reward Model이 준 점수에 따라 그 답변을 더 자주 또는 덜 자주 만들도록 바꿀 차례다. 이번 주에는 InstructGPT에서 사용한 PPO 기반 RLHF를 한 단계씩 따라간다.[^1]

## 이번 주에 배울 것

- Policy, reference, Reward Model, value model의 역할
- prompt에서 response를 만드는 rollout
- reward와 value의 차이, advantage의 뜻
- PPO probability ratio와 clipped objective
- KL penalty가 기준 model과의 거리를 조절하는 방법
- reward, KL, ratio, clip fraction, value loss를 함께 읽는 방법

선수 지식은 5주차의 policy gradient와 advantage, 6주차의 Reward Model이다.

!!! note "좋은 방향으로 가되 한 번에 뛰지 않는다"

    Reward Model이 높은 점수를 준 답을 발견했다고 policy를 크게 바꾸면, 우연히 찾은 편법까지 강하게 배울 수 있다. PPO는 이전 policy와 새 policy의 확률 비율을 살피고 한 번의 update가 지나치게 커질 때 이득을 잘라낸다.

## 1. PPO 단계에는 네 model이 참여한다

![InstructGPT의 SFT, Reward Model, PPO 학습 흐름](/notes/tutorial/llm_lecture/images/w07_instructgpt_ppo_pipeline.png)

*그림 1. SFT model에서 출발한 policy가 답변을 만들고 Reward Model의 점수로 PPO update를 받는 흐름. 출처: Ouyang et al. (2022), Figure 2에서 발췌.[^1]*

현재 TRL의 PPOTrainer도 policy, reference, Reward Model, value model을 따로 받는다.[^4] 네 model은 같은 문장을 보더라도 맡은 일이 다르다.

| 구성 요소 | 학습 중 하는 일 | weight가 바뀌는가 |
| --- | --- | --- |
| policy model | prompt를 읽고 response token을 뽑는다 | 바뀐다 |
| reference model | 학습 시작점의 token 확률을 계산한다 | 고정한다 |
| Reward Model | 완성된 prompt와 response에 score를 준다 | 고정한다 |
| value model | 아직 끝나지 않은 문장에서 앞으로 받을 reward를 예상한다 | 바뀐다 |

Reference model은 보통 SFT model의 복사본이다. Policy와 같은 출발점에서 시작하지만 update하지 않는다. Value model은 답변을 직접 만들지 않고 각 시점의 예상 점수 $V(s_t)$를 낸다.

!!! note "old policy와 reference model은 다르다"

    PPO 수식의 old policy는 이번 update에 쓸 rollout을 만든 직전 policy다. Reference model은 RLHF가 시작될 때 정해 둔 기준점이다. old policy는 update마다 새로 바뀌지만 reference model은 계속 고정된다.

네 model을 모두 GPU에 올리면 메모리를 많이 쓴다. Policy와 value에 LoRA를 쓰거나 reference log probability를 미리 계산하면 메모리가 줄어든다. 그래도 PPO는 3~4주차의 SFT나 8주차의 DPO보다 실습 환경을 꾸리기 어렵다.

## 2. Rollout은 policy가 직접 만든 학습 자료다

PPO의 한 cycle은 다음 순서로 움직인다.

1. dataset에서 prompt를 뽑는다.
2. 현재 policy가 response를 생성한다.
3. Policy와 reference model이 각 response token의 log probability를 계산한다.
4. Reward Model이 완성된 response에 score를 준다.
5. Value model의 예상값과 실제 reward를 비교해 advantage를 구한다.
6. 같은 rollout을 작은 minibatch로 나누어 PPO update를 몇 번 한다.

SFT는 사람이 적어 둔 정답 token을 학습한다. PPO의 response는 현재 policy가 방금 만든 것이다. 그래서 policy가 변하면 다음 cycle에 모이는 학습 자료도 달라진다. 이를 online learning이라고 부른다.

LLM에서는 prompt와 지금까지 만든 token이 state $s_t$, 다음 token이 action $a_t$다. Response가 끝날 때 Reward Model score를 받더라도, 그 점수를 어떤 token 선택에 돌려줄지 계산해야 한다.

## 3. Reward와 value의 차이가 advantage다

Reward Model이 한 답변에 2.0점을 주었다고 하자. Value model이 그 상태에서 평균 1.4점을 받을 것으로 예상했다면 간단한 advantage는 $A=2.0-1.4=0.6$이다. 예상보다 좋은 결과이므로 선택한 token의 확률을 올릴 근거가 된다.[^3]

반대로 reward가 0.8이고 value가 1.4라면 $A=-0.6$이다. 선택한 token의 확률을 내리는 쪽으로 update한다. 실제 PPO 구현은 여러 token에 reward를 나누고 GAE로 advantage를 계산한다. TRL의 PPOConfig에는 이 계산에 쓰는 `gamma`와 `lam`이 있다.[^4]

| advantage | 뜻 | policy가 배우는 방향 |
| ---: | --- | --- |
| 양수 | 예상보다 결과가 좋았다 | 그 action의 확률을 높임 |
| 0 근처 | 예상과 비슷했다 | update가 작음 |
| 음수 | 예상보다 결과가 나빴다 | 그 action의 확률을 낮춤 |

Value model이 엉뚱한 값을 내면 advantage도 흔들린다. Policy loss만 보지 않고 value loss를 함께 확인해야 하는 이유다.

## 4. Probability ratio는 확률이 얼마나 바뀌었는지 잰다

Rollout을 만들 때 old policy가 어떤 token에 0.20의 확률을 주었다고 하자. 한 번 update한 새 policy가 같은 token에 0.26을 주었다면 ratio는 $r_t(\theta)=0.26/0.20=1.3$이다.

- ratio가 1이면 확률이 그대로다.
- ratio가 1보다 크면 해당 token의 확률이 올랐다.
- ratio가 1보다 작으면 해당 token의 확률이 내려갔다.

기본 policy gradient는 $r_t(\theta)A_t$를 크게 만드는 방향으로 움직인다. 좋은 action의 확률을 너무 크게 올리거나 나쁜 action의 확률을 너무 빠르게 내릴 수 있다는 문제가 있다. PPO는 ratio를 $1-\epsilon$과 $1+\epsilon$ 사이로 자른 값도 계산한다. Clipped objective는 $L^{clip}=\mathbb{E}[\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)]$다.[^2]

`cliprange=0.2`라면 중심 구간은 0.8~1.2다. 양의 advantage에서 ratio가 1.2보다 커져도 objective의 이득은 더 늘지 않는다. 음의 advantage에서는 ratio가 0.8보다 작아질 때 이득을 막는다.

```python
import torch

ratio = torch.tensor([0.5, 0.8, 1.0, 1.2, 1.5])
advantage = torch.tensor(1.0)
clip_range = 0.2

unclipped = ratio * advantage
clipped = torch.clamp(
    ratio,
    1 - clip_range,
    1 + clip_range,
) * advantage
objective = torch.minimum(unclipped, clipped)

print(objective)
# tensor([0.5000, 0.8000, 1.0000, 1.2000, 1.2000])
```

## 5. Clipping이 모든 큰 변화를 막는 것은 아니다

![Advantage 부호에 따른 PPO clipped objective](/notes/tutorial/llm_lecture/images/w07_ppo_clipping_curve.png)

*그림 2. `cliprange=0.2`에서 probability ratio와 surrogate objective의 관계. 출처: PyTorch 2.13.0으로 직접 계산한 결과(2026-08-02, macOS CPU).[^5]*

그림의 회색 영역은 ratio 0.8~1.2다. Advantage가 +1인 왼쪽 그래프에서는 ratio가 1.2를 넘어갈 때 주황색 objective가 평평해진다. Advantage가 -1인 오른쪽 그래프에서는 ratio가 0.8 아래로 내려갈 때 더 이상 이득을 주지 않는다.

| advantage | ratio | unclipped | PPO clipped | 해석 |
| ---: | ---: | ---: | ---: | --- |
| +1 | 1.0 | 1.0 | 1.0 | 확률 변화 없음 |
| +1 | 1.5 | 1.5 | 1.2 | 좋은 action을 지나치게 올린 이득을 제한 |
| -1 | 1.0 | -1.0 | -1.0 | 확률 변화 없음 |
| -1 | 0.5 | -0.5 | -0.8 | 나쁜 action을 지나치게 내린 이득을 제한 |

Clipping은 update를 완전히 금지하는 벽이 아니다. Objective가 더 좋아질 유인을 없애는 장치에 가깝다. 여러 token과 minibatch가 얽힌 실제 학습에서는 ratio, approximate KL, clip fraction을 함께 봐야 한다.[^2][^4]

## 6. KL penalty는 reference model을 기준으로 삼는다

PPO clipping은 직전 policy와의 변화를 살핀다. 조금씩 여러 번 움직이면 처음 SFT model에서는 멀어질 수 있다. Reference model은 이 장기 기준점을 잡아준다.

Policy가 만든 답변의 score를 $s(x,y)$, reference model과의 차이를 $KL(\pi_\theta\|\pi_{ref})$라고 쓰자. 학습에 쓰는 reward는 $r_{RLHF}=s(x,y)-\beta KL(\pi_\theta\|\pi_{ref})$다. InstructGPT도 SFT model에서 너무 멀어지는 일을 줄이려고 token마다 KL penalty를 더했다.[^1]

```text
Reward Model score = 2.40
KL penalty         = 0.35
RLHF reward        = 2.40 - 0.35 = 2.05
```

`beta` 또는 `kl_coef`가 너무 작으면 policy가 Reward Model의 빈틈을 좇아 reference에서 빠르게 멀어질 수 있다. 너무 크면 거의 움직이지 못해 reward가 오르지 않는다. 하나의 숫자를 외우기보다 reward와 KL의 곡선을 보며 조절한다.

!!! warning "PPO clipping과 KL penalty를 같은 장치로 보지 않는다"

    PPO clipping은 한 update 전후의 확률 변화를 제한한다. KL penalty는 고정된 reference model과 얼마나 달라졌는지 reward에서 뺀다. 둘은 서로 다른 거리를 살핀다.

## 7. 로그는 한 줄이 아니라 묶음으로 읽는다

현재 TRL의 PPO 문서는 다음 지표를 기록한다.[^4]

| 로그 | 확인할 질문 |
| --- | --- |
| `objective/scores` | Reward Model score가 오르는가 |
| `objective/kl` | Policy가 reference model에서 얼마나 멀어지는가 |
| `objective/rlhf_reward` | KL penalty를 뺀 뒤에도 reward가 오르는가 |
| `val/ratio` | 새 policy와 old policy의 확률 비율이 1 근처인가 |
| `policy/clipfrac_avg` | update 중 얼마나 많은 항이 clip되었는가 |
| `loss/value_avg` | Value model의 예상이 실제 reward에 가까워지는가 |
| `policy/entropy_avg` | Token 선택이 너무 빨리 한쪽으로 굳지 않는가 |
| `val/num_eos_tokens` | Response가 EOS로 제대로 끝나는가 |

Reward Model score만 오르고 KL도 급격히 커진다면 좋은 신호라고 단정하기 어렵다. Policy가 채점기의 지름길을 찾았을 수 있다. 정해 둔 평가 prompt의 실제 답변을 주기적으로 저장하고 사람이 읽는다. 6주차에서 만든 length bias probe도 다시 사용한다.

## 8. TRL의 PPO smoke test를 실행한다

PPO API는 현재 TRL 문서에서 experimental로 분류된다. 아래 명령은 2026-08-02 공식 문서의 minimal 예제를 바탕으로 했다. 설치한 TRL version에 맞는 문서를 다시 확인해야 한다.[^4]

```bash
git clone https://github.com/huggingface/trl.git
cd trl
pip install -e .

python examples/scripts/ppo/ppo.py \
  --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
  --dataset_train_split descriptiveness \
  --learning_rate 3e-6 \
  --num_ppo_epochs 1 \
  --num_mini_batches 1 \
  --output_dir models/w07_ppo_smoke \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --total_episodes 10000 \
  --model_name_or_path EleutherAI/pythia-1b-deduped \
  --sft_model_path EleutherAI/pythia-1b-deduped \
  --reward_model_path EleutherAI/pythia-1b-deduped \
  --missing_eos_penalty 1.0
```

이 명령은 trainer가 움직이는지 확인하는 dummy Reward Model 예제다. 6주차에서 만든 Reward Model을 사용한 실험 결과로 해석하면 안 된다. 실제 RLHF 실습에서는 다음을 바꿔야 한다.

- `sft_model_path`: 3주차에 만든 SFT checkpoint
- `reward_model_path`: 6주차에 만든 Reward Model checkpoint
- dataset: 여러 도메인의 prompt-only dataset
- batch와 episode 수: GPU 메모리에 맞춘 값
- 평가: 고정 prompt의 답변, Reward Model score, KL, 사람 평가

!!! warning "작은 GPU에서 바로 시작하지 않는다"

    PPO는 여러 model을 동시에 사용하고 rollout도 생성한다. 먼저 이 글의 clipping 계산으로 수식을 확인한다. 공식 dummy 예제로 pipeline을 점검한 뒤 작은 SFT·Reward Model checkpoint를 연결한다. Out of Memory가 나면 batch를 줄이고 gradient accumulation을 늘린다.[^4]

## 9. 실제 135M model로 pipeline을 확인했다

`HuggingFaceTB/SmolLM2-135M-Instruct`에 LoRA를 붙여 PPO policy update를 두 번 실행했다.[^6][^7] 영어 prompt 8개를 사용했고, 한 batch에는 4개씩 넣었다. Response 길이는 16 token, learning rate는 `1e-5`, `cliprange`는 0.2, KL coefficient는 0.05로 정했다. 실행 장치는 Apple Silicon의 MPS였고 dtype은 `float32`였다.

![SmolLM2 PPO smoke test의 reward와 update 지표](/notes/tutorial/llm_lecture/images/w07_ppo_training_smoke.png)

*그림 3. 무작위 Reward Model로 실행한 PPO smoke test. 두 번의 update 동안 기록한 reward, KL, ratio, clip fraction이다. TRL 1.9.2, PyTorch 2.13.0, macOS MPS에서 직접 실행했다.[^7]*

| 항목 | 측정값 |
| --- | ---: |
| policy update / episode | 2회 / 8개 |
| 실행 시간 | 4.60초 |
| 최대 메모리 | 2,104MB |
| 마지막 dummy RM score | 1.1475 |
| 마지막 reference KL | -0.000262 |
| 마지막 policy ratio | 1.000001 |
| 마지막 clip fraction | 0.0 |

두 차례 update 동안 dummy RM score는 0.6624에서 1.1475로 올랐다. 하지만 고정한 평가 prompt 네 개의 답변은 학습 전후가 모두 같았다. Reward Model의 분류 head가 무작위로 초기화됐고 update도 두 번뿐이었다. 따라서 이 점수 상승을 답변 품질 향상으로 해석할 수 없다. 이 실험이 확인한 것은 작은 model에서 rollout, reward 계산, value 계산, PPO update, adapter 저장이 끝까지 이어진다는 사실뿐이다.[^7]

!!! warning "smoke test 점수는 성능표가 아니다"

    무작위 Reward Model은 사람이 좋아할 답을 판별하도록 학습되지 않았다. 제대로 된 RLHF 실험을 하려면 6주차 Reward Model을 연결하고, 별도의 평가 prompt와 사람 평가로 답변 품질을 확인해야 한다.

첫 실행은 원문 `prompt` 문자열 열이 token과 함께 padding 단계로 넘어가 멈췄다. PPOTrainer에 `input_ids`만 전달하도록 고친 뒤 같은 설정으로 다시 실행했다. 실패 원인과 수정 내용을 남겨야 smoke test를 다시 돌릴 때 같은 문제를 피할 수 있다.

## 10. 학습한 Reward Model로 PPO를 다시 시험했다

무작위 head가 아닌 Reward Model을 연결해 본 실험을 구성했다. Policy는 `Qwen/Qwen2.5-0.5B-Instruct`이고, 영어 지시 준수 task 10종을 task마다 60개씩 만들었다.[^8][^9] Train에는 400개 prompt, validation과 test에는 각각 100개씩 넣었다. 문장 틀은 같지만 숫자와 label은 split마다 겹치지 않는다.

첫 Reward Model은 train 400쌍을 1에포크 학습했다. 보지 않은 test 100쌍의 pairwise accuracy는 100%였지만 평균 reward margin이 14.15로 크게 벌어졌다. 이 모델로 PPO 400 episode를 실행하자 test 답변의 평균 reward는 0.81에서 1.06으로 올랐지만 자동 지시 준수율은 24%에서 22%로 떨어졌다.[^9]

높은 reward를 받은 실패 답변을 읽어 보니 원인이 드러났다. 예를 들어 `72+9`를 묻고 `RESULT:` 뒤에 합을 쓰라고 한 prompt에 policy는 `RESULT:81`이라고 답했다. 합은 맞지만 요구한 공백이 없어 자동 검사를 통과하지 못했고 reward는 6.38이었다. 더 심한 경우에는 틀린 합을 써도 높은 점수를 받았다. Reward Model이 학습 때 본 오답 종류가 너무 적었다.

그래서 정답 하나에 오답을 여러 종류로 붙였다. 틀린 계산, Markdown code fence로 감싼 JSON, 대소문자 위반, 번호 목록, 항목 순서 변경 등을 추가해 train 1,320쌍, validation 330쌍, test 330쌍으로 늘렸다. 강화한 Reward Model은 165회 update, 152.26초 동안 학습했고 test pairwise accuracy 94.85%, 평균 margin 6.18을 기록했다.

![여러 오답 유형으로 학습한 Qwen2.5 Reward Model](/notes/tutorial/llm_lecture/images/w07_reward_model_demo_result.png)

*그림 4. 여러 종류의 오답을 넣은 Reward Model의 loss와 task별 test accuracy. JSON, 숫자, yes/no task가 상대적으로 어려웠다. PyTorch 2.13.0, macOS MPS에서 직접 실행했다.[^9]*

학습 전 policy가 만든 test 답변 100개에도 새 Reward Model을 적용했다. 자동 검사를 통과한 답변의 평균 reward는 9.42, 실패한 답변은 0.10이었다. 통과 여부와 reward의 상관계수도 0.885였다. Pair만 잘 구분하는 데 그치지 않고 실제 생성 답변에도 전보다 알맞은 점수를 주었다.

이 Reward Model을 고정하고 PPO를 400 episode와 1,600 episode로 나누어 실행했다. Policy에는 rank 8 LoRA를 붙였다. Batch size는 8, learning rate는 `5e-6`, response 길이는 32 token, `cliprange`는 0.2, KL coefficient는 0.05로 두었다. Reward whitening을 켰고 한 rollout은 PPO update에 한 번 사용했다.[^4][^9]

![Qwen2.5-0.5B-Instruct의 1,600 episode PPO 결과](/notes/tutorial/llm_lecture/images/w07_ppo_demo_result.png)

*그림 5. 강화한 Reward Model로 1,600 episode PPO를 실행한 결과. Reward와 KL은 크게 흔들렸고, 보지 않은 100개 지시의 통과율은 좋아지지 않았다. TRL 1.9.2, PyTorch 2.13.0, macOS MPS에서 직접 실행했다.[^9]*

| Reward Model | PPO episode / update | 실행 시간 | 평균 reward | test 통과율 |
| --- | ---: | ---: | ---: | ---: |
| 단순 오답 | 400 / 50 | 180.94초 | 0.81 → 1.06 | 24% → 22% |
| 여러 오답 | 400 / 50 | 179.76초 | 2.34 → 2.38 | 24% → 23% |
| 여러 오답 | 1,600 / 200 | 724.89초 | 2.34 → 2.00 | 24% → 22% |

Episode를 네 배로 늘려도 지시 준수율은 오르지 않았다. `comma_no_spaces`와 `lowercase_exact`는 계속 모두 맞혔지만 `result_prefix`는 40%에서 20%로 떨어졌고, 나머지 일곱 task는 0%에 머물렀다. 이 조건에서는 PPO가 지시 준수를 배웠다고 말할 수 없다.

!!! warning "Reward Model 정확도가 높아도 PPO 성공을 보장하지 않는다"

    Pairwise accuracy 94.85%는 정해 둔 두 답 중 좋은 쪽을 고르는 시험이다. PPO는 policy가 새로 만든 답을 채점하므로 Reward Model이 보지 못한 문장도 다뤄야 한다. 작은 batch의 noisy reward, sparse한 문장 단위 점수, value 오차도 policy 학습을 어렵게 한다. 더 오래 돌리기 전에 실제 생성 답변을 오답 자료에 넣고, 별도 검증자가 reward와 task 성공을 함께 확인해야 한다.[^1][^4]

이 실험의 좋은 결과는 통과율 상승이 아니라 중단 기준을 찾은 것이다. 400 episode에서 개선이 없었고 1,600 episode에서도 같은 결과가 반복됐다. 이 상태에서 episode만 계속 늘리면 시간은 더 쓰지만 성공 근거는 생기지 않는다. 8주차 DPO처럼 chosen token을 직접 비교해 배우는 방법이 이 작은 실습에서는 훨씬 효율적이었다.

## 11. PPO-RLHF 보고서를 남긴다

| 항목 | 기록할 내용 |
| --- | --- |
| model | policy, reference, Reward Model, value model ID와 revision |
| rollout | prompt 수, response 길이, temperature, EOS 처리 |
| PPO | learning rate, `cliprange`, PPO epoch, minibatch 수 |
| reward | Reward Model score, KL coefficient, RLHF reward |
| 안정성 | ratio, approximate KL, clip fraction, value loss, entropy |
| 품질 | 고정 prompt 답변과 사람의 선호 평가 |
| 실패 | reward는 올랐지만 답변이 나빠진 사례 |

비교 실험에서는 한 번에 하나만 바꾼다. `cliprange`를 비교할 때 KL coefficient와 prompt, seed, generation 설정까지 함께 바꾸면 원인을 알 수 없다.

## 확인 문제

1. Policy model과 reference model은 같은 weight에서 시작해도 역할이 왜 다른가?
2. PPO 수식의 old policy와 고정된 reference model의 차이는 무엇인가?
3. Reward가 1.8이고 value가 2.1이면 advantage의 부호는 무엇인가?
4. Advantage가 양수일 때 ratio가 1.2를 크게 넘어가도 clipped objective가 더 커지지 않는 이유는 무엇인가?
5. Reward Model score만 오르고 KL이 급격히 커질 때 어떤 답변을 먼저 확인해야 하는가?

## 완료 체크

- [x] 네 model이 주고받는 값을 rollout 순서대로 설명했다.
- [x] Reward와 value로 advantage를 계산했다.
- [x] `cliprange=0.2`에서 양수·음수 advantage의 objective를 손으로 계산했다.
- [x] TRL PPO smoke test를 실행하고 package version을 기록했다.
- [ ] `cliprange`와 KL coefficient를 하나씩 바꾸어 로그를 비교했다.
- [x] Reward가 올라도 실제 답변이 나빠진 사례를 찾았다.
- [x] 결과물로 `작은 PPO-RLHF pipeline 보고서`를 완성했다.

---

[^1]: Ouyang, L. et al. (2022). [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155). Figure 2와 §3.5의 PPO, token-level KL penalty, value initialization을 참고했다.
[^2]: Schulman, J. et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). probability ratio와 clipped surrogate objective를 참고했다.
[^3]: Sutton, R. S. and Barto, A. G. (2018). [Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html). value와 advantage의 배경을 참고했다.
[^4]: Hugging Face. [TRL: PPO Trainer](https://huggingface.co/docs/trl/ppo_trainer). 현재 PPOTrainer의 네 model, PPOConfig, minimal 실행 명령과 로그 정의를 참고했다. 확인일: 2026-08-02.
[^5]: 직접 실행한 `llm_lecture/week07.py`의 계산 결과다. PyTorch 2.13.0, macOS CPU, `cliprange=0.2`를 사용했다. 실행일: 2026-08-02.
[^6]: Hugging FaceTB. [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct). Model 구조, chat template과 사용 조건을 참고했다. 확인일: 2026-08-02.
[^7]: 직접 실행한 `llm_lecture/week07_ppo_smoke.py`의 결과다. SmolLM2-135M-Instruct, TRL 1.9.2, Transformers 5.14.1, PyTorch 2.13.0, macOS MPS를 사용했다. 원본 CSV와 adapter는 Git에서 제외했다. 실행일: 2026-08-02.
[^8]: Qwen Team. [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). Model 크기, chat template과 사용법을 참고했다. 확인일: 2026-08-02.
[^9]: 직접 실행한 `llm_lecture/week07_reward_model_demo.py`와 `llm_lecture/week07_ppo_demo.py`의 결과다. Qwen2.5-0.5B-Instruct, TRL 1.9.2, Transformers 5.14.1, PyTorch 2.13.0, macOS MPS를 사용했다. PPO는 최대 1,600 episode와 200 update까지 실행했다. 원본 CSV, model, adapter는 Git에서 제외했다. 실행일: 2026-08-02.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: PPO 본 실험을 포함한 전체 문서를 절별로 나누어 점검
카테고리별 탐지/수정: A-8 0→0, C-11 0→0, D-1 0→0, H-1 0→0, I-1 0→0
정량 점검: humanize-korean metrics v2.0 risk score 1, low
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 PPO 실패 수치와 기술 용어를 그대로 보존함
주요 변경: reward와 실제 지시 준수율을 분리해 쓰고, 긴 원인 설명을 문단별로 나눔
-->
