---
title: "5주차. 강화학습의 기초"
description: "agent가 환경과 상호작용하며 reward를 모으는 과정을 배우고 작은 길 찾기 문제에서 REINFORCE와 baseline을 구현한다."
tags:
  - LLM
  - reinforcement learning
  - policy gradient
  - REINFORCE
---

[← 16주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

4주차까지는 모델에게 모범 답안을 직접 보여줬다. 이번 주부터는 모델이 먼저 행동하고 그 결과에 받은 점수로 학습한다. 강화학습의 state, action, reward를 짧은 길 찾기 게임에서 익힌 뒤 LLM의 token 생성 과정과 연결해본다.

## 이번 주에 배울 것

- agent, environment, state, action, reward, trajectory의 관계
- reward를 미래까지 더한 return의 뜻
- policy, value, advantage가 맡는 역할
- REINFORCE가 선택한 action의 확률을 바꾸는 방법
- sparse reward와 dense reward, baseline이 학습에 미치는 영향

선수 지식은 확률, 평균, PyTorch tensor와 1주차의 다음 token 예측이다.

!!! note "미로를 푸는 로봇"

    로봇이 갈림길에서 왼쪽이나 오른쪽을 고른다고 하자. 출구에 도착하면 점수를 받는다. 처음에는 아무 방향이나 고르지만 여러 번 시도하면서 점수를 받은 경로의 선택 확률을 높인다. 강화학습은 정답 action을 매 순간 알려주지 않아도 결과를 보고 행동을 고친다.

## 1. Agent와 environment가 차례로 움직인다

![Agent와 environment 사이의 state, reward, action 흐름](/notes/tutorial/llm_lecture/images/w05_agent_environment_interaction.png)

*그림 1. Markov decision process에서 agent와 environment가 state, reward, action을 주고받는 구조. 출처: Sutton and Barto (2018), Figure 3.1에서 발췌.[^1]*

agent는 현재 state $S_t$를 보고 action $A_t$를 고른다. environment는 그 action을 받은 뒤 다음 state $S_{t+1}$와 reward $R_{t+1}$를 돌려준다. 이 순서를 episode가 끝날 때까지 반복한다.[^1]

| 용어 | 쉬운 뜻 | 길 찾기 예시 |
| --- | --- | --- |
| agent | 행동을 고르는 주체 | 로봇 |
| environment | action의 결과를 정하는 세계 | 복도와 벽 |
| state | 지금 상황을 나타내는 정보 | 현재 칸 |
| action | agent가 고른 행동 | 왼쪽 또는 오른쪽 |
| reward | 방금 행동에 받은 점수 | 출구 도착 시 $+1$ |
| policy | state마다 action을 고르는 규칙 | 각 칸에서 오른쪽으로 갈 확률 |
| episode | 시작부터 종료까지 한 번의 시도 | 출발점에서 출구까지 한 판 |
| trajectory | 한 episode에 지나간 기록 | $S_0,A_0,R_1,S_1,\ldots$ |

reward는 방금 받은 점수다. return은 현재부터 episode가 끝날 때까지 받을 reward를 모은 값이다. 할인율 $\gamma$를 쓰면 $G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots$로 계산한다. $\gamma$가 1에 가까우면 먼 미래의 reward도 크게 반영하고 0에 가까우면 바로 다음 reward를 더 중시한다.

!!! warning "reward는 목표 그 자체가 아니다"

    agent는 사람이 마음속에 둔 목표를 직접 알지 못한다. 코드로 주어진 reward만 높인다. 출구에 빨리 도착하는 것이 목표인데 이동할 때마다 점수를 준다면 agent가 출구로 가지 않고 제자리 근처에서 점수만 모으는 길을 찾을 수도 있다.

## 2. Policy, value, advantage를 구분한다

policy $\pi_\theta(a\mid s)$는 state $s$에서 action $a$를 고를 확률이다. $\theta$는 이 확률을 만드는 학습 가능한 parameter다. action이 두 개라면 policy의 출력은 다음처럼 보인다.

```text
현재 state: 2번 칸
P(왼쪽) = 0.25
P(오른쪽) = 0.75
확률의 합 = 1.00
```

value $V(s)$는 state $s$에서 출발했을 때 앞으로 받을 return의 평균을 예상한다. advantage $A(s,a)$는 특정 action이 평소 예상보다 얼마나 좋았는지 나타낸다. 가장 단순하게는 $A(s,a)\approx G_t-V(s)$로 생각하면 된다.

- return $G_t$가 예상보다 크면 advantage는 양수다. 선택한 action의 확률을 높인다.
- return $G_t$가 예상보다 작으면 advantage는 음수다. 선택한 action의 확률을 낮춘다.
- 예상과 비슷하면 update가 작다.

value는 미래를 완벽히 맞히는 정답표가 아니다. policy update가 어느 방향으로 얼마나 움직일지 판단할 때 쓰는 기준점이다.

## 3. LLM의 token 생성을 강화학습과 연결한다

길 찾기와 LLM은 겉모습이 다르지만 같은 용어로 대응한다.

| 강화학습 | LLM 생성 |
| --- | --- |
| state $S_t$ | prompt와 지금까지 생성한 token |
| action $A_t$ | 다음에 생성할 token |
| policy $\pi_\theta$ | vocabulary의 다음 token 확률 |
| trajectory | 완성된 답변과 token 순서 |
| environment | token을 문장 뒤에 붙이고 다음 state를 만드는 과정 |
| reward | 답변의 정확성, 유용성, 안전성 등에 매긴 점수 |

LLM의 action 수는 왼쪽과 오른쪽 두 개가 아니라 vocabulary에 있는 수만 개의 token이다. 한 token의 선택은 뒤에 이어질 모든 token에 영향을 준다. 답변을 다 만든 뒤 reward 하나를 받으면 어떤 token이 점수에 기여했는지 바로 알기 어렵다. 이를 credit assignment 문제라고 부른다.

!!! note "답변 전체에 받은 점수를 token에 나누어 준다"

    축구팀이 경기 뒤에 점수 하나만 받는 상황과 비슷하다. 어느 선수가 어느 순간에 잘했는지 알려면 경기 과정을 함께 봐야 한다. LLM 강화학습도 답변 전체 reward를 각 token의 선택과 연결해야 한다.

## 4. REINFORCE는 선택한 action의 log probability를 이용한다

REINFORCE는 episode를 끝까지 실행한 뒤 실제로 고른 action의 확률을 return에 따라 바꾼다. 핵심 gradient는 $\nabla_\theta J(\theta)\approx\sum_t G_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)$로 쓴다.[^1][^2]

코드에서는 gradient ascent 대신 최소화 방식의 optimizer를 쓰므로 loss 앞에 음수를 붙인다. baseline이 없다면 $\mathcal{L}_{\text{policy}}=-\sum_tG_t\log\pi_\theta(A_t\mid S_t)$다.

REINFORCE의 return은 episode마다 크게 흔들린다. 같은 state에서도 우연히 좋은 action이 연달아 나오면 높은 점수를 받고 나쁜 action이 섞이면 낮은 점수를 받는다. 이전 episode의 평균 return 같은 baseline $b$를 빼면 loss는 $\mathcal{L}_{\text{policy}}=-\sum_t(G_t-b)\log\pi_\theta(A_t\mid S_t)$가 된다.

baseline이 action에 따라 달라지지 않으면 기대 gradient의 방향은 그대로 두면서 분산을 줄인다.[^1] 다만 baseline이 부정확하거나 학습 설정이 나쁘면 실제 실험 곡선이 항상 매끈해지지는 않는다.

## 5. 두 action으로 REINFORCE를 구현한다

아래 환경에는 0번부터 4번까지 다섯 칸이 있다. agent는 0번에서 출발해 왼쪽 또는 오른쪽으로 움직인다. 4번 칸에 도착하면 episode가 끝난다.

```bash
pip install torch
```

```python
from statistics import pstdev

import torch
from torch.distributions import Categorical

N_STATES = 5
GOAL = N_STATES - 1
MAX_STEPS = 8


def train(reward_mode, use_baseline, episodes=800, seed=42):
    torch.manual_seed(seed)

    # state마다 [왼쪽, 오른쪽] action의 logit을 둔다.
    logits = torch.zeros(N_STATES, 2, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.05)

    baseline = 0.0
    successes = []
    gradient_norms = []

    for _ in range(episodes):
        state = 0
        done = False
        log_probs = []
        rewards = []

        for _ in range(MAX_STEPS):
            policy = Categorical(logits=logits[state])
            action = policy.sample()
            log_probs.append(policy.log_prob(action))

            old_state = state
            move = 1 if action.item() == 1 else -1
            state = max(0, min(GOAL, state + move))
            done = state == GOAL

            if reward_mode == "sparse":
                reward = 1.0 if done else 0.0
            else:
                moved_right = state > old_state
                reward = 1.0 if done else (0.05 if moved_right else -0.05)

            rewards.append(reward)
            if done:
                break

        # 각 step에서 미래 reward를 거꾸로 더한다.
        returns = []
        G = 0.0
        for reward in reversed(rewards):
            G = reward + 0.99 * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        advantages = returns - baseline if use_baseline else returns
        policy_loss = -(
            torch.stack(log_probs) * advantages.detach()
        ).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        gradient_norms.append(logits.grad.norm().item())
        optimizer.step()

        # 현재 episode는 다음 episode의 baseline에만 반영한다.
        if use_baseline:
            baseline = 0.9 * baseline + 0.1 * returns.mean().item()

        successes.append(float(done))

    probabilities = logits.softmax(dim=-1)
    return {
        "reward": reward_mode,
        "baseline": use_baseline,
        "last_100_success": sum(successes[-100:]) / 100,
        "p_right_at_start": probabilities[0, 1].item(),
        "gradient_std": pstdev(gradient_norms[-200:]),
    }


for reward_mode in ["sparse", "dense"]:
    for use_baseline in [False, True]:
        print(train(reward_mode, use_baseline))
```

`Categorical`은 logit을 확률로 바꾸고 그 확률에 따라 action을 뽑는다. `log_prob(action)`은 선택한 action의 log probability를 돌려주므로 policy gradient loss에 바로 쓴다.[^3]

실행 결과에서 다음 세 값을 비교한다.

| 값 | 확인할 내용 |
| --- | --- |
| `last_100_success` | 마지막 100회 중 출구에 도착한 비율 |
| `p_right_at_start` | 시작점에서 오른쪽을 고를 확률 |
| `gradient_std` | 최근 gradient 크기가 얼마나 흔들렸는지 나타내는 보조값 |

한 seed에서 나온 숫자만으로 baseline의 효과를 단정하지 않는다. seed를 5개 이상 바꿔 평균과 표준편차를 기록한다. dense reward가 더 빨리 학습되더라도, 그 reward가 원래 목표와 같은 행동을 유도하는지도 확인해야 한다.

## 6. Sparse reward와 dense reward는 장단점이 다르다

sparse reward는 출구 도착처럼 중요한 사건에서만 점수를 준다. 목표는 분명하지만 성공 경험이 드물면 학습 신호를 거의 얻지 못한다. dense reward는 목표에 가까워질 때마다 작은 점수를 주므로 초반 학습이 쉬워진다. 대신 사람이 잘못 만든 중간 점수를 agent가 편법으로 이용할 위험이 있다.

| 비교 | Sparse reward | Dense reward |
| --- | --- | --- |
| 점수를 받는 때 | 성공하거나 실패했을 때 | 중간 과정에서도 자주 |
| 초반 학습 | 느릴 수 있음 | 비교적 빠를 수 있음 |
| 설계 난이도 | 목표 정의가 단순함 | 중간 행동의 점수까지 정해야 함 |
| 주요 위험 | 성공 경험 부족 | reward hacking |

LLM에서도 비슷하다. 최종 답의 정답 여부만 채점하면 reward가 드물다. 문장 형식, 도구 선택, 중간 계산에 점수를 나누어 주면 신호는 많아진다. 대신 모델이 형식 점수만 챙기고 내용은 틀리는 편법을 배우기도 한다.

## 7. 실험 기록표를 남긴다

seed 5개에서 각각 800 episode를 실행한 뒤 평균을 냈다. 성공률의 표준편차는 seed에 따라 마지막 100회 성공률이 얼마나 달라졌는지를 나타낸다.[^2][^3]

| reward | baseline | 마지막 100회 성공률 | 성공률 표준편차 | 시작점 오른쪽 확률 | gradient 표준편차 |
| --- | --- | ---: | ---: | ---: | ---: |
| sparse | 없음 | 0.996 | 0.0055 | 0.955 | 0.408 |
| sparse | 사용 | 0.998 | 0.0045 | 0.927 | 0.051 |
| dense | 없음 | 0.998 | 0.0045 | 0.985 | 0.339 |
| dense | 사용 | 0.998 | 0.0045 | 0.985 | 0.058 |

성공률 그래프의 가로축은 episode, 세로축은 최근 100회 이동 평균으로 맞춘다. 네 실험의 축과 episode 수가 같아야 곡선 비교가 공정하다.

![Sparse reward와 dense reward에서 baseline 사용 전후의 REINFORCE 학습 곡선](/notes/tutorial/llm_lecture/images/w05_reinforce_result.png)

*그림 2. reward와 baseline 조건별 성공률의 100 episode 이동 평균. 선은 seed 5개의 평균이고 옅은 영역은 seed 사이의 표준편차다. 출처: 복도 환경 직접 실행 결과(2026-08-01, Apple MPS). REINFORCE 구현은 Williams(1992)와 PyTorch `Categorical` 문서를 참고했다.[^2][^3]*

네 조건 모두 마지막 성공률은 0.996 이상이었다. 마지막 점수만 보면 baseline의 차이가 거의 없어 보이지만 gradient 표준편차에서는 차이가 컸다. sparse reward에서는 0.408에서 0.051로, dense reward에서는 0.339에서 0.058로 줄었다. baseline이 gradient의 흔들림을 낮춘다는 설명과 맞는 결과다.

dense reward 곡선은 초반에 더 빨리 오른다. 중간 위치에서도 점수를 받아 학습 신호를 자주 얻었기 때문이다. 다만 이 복도는 선택지가 두 개뿐인 작은 환경이다. 실제 LLM의 긴 문장 생성에서도 같은 속도 차이가 난다고 일반화할 수는 없다.

## 확인 문제

1. reward와 return은 어떻게 다른가?
2. 길 찾기에서 state, action, policy를 하나씩 예로 들어보자.
3. LLM이 다음 token을 고르는 일을 강화학습의 state와 action으로 설명해보자.
4. baseline을 빼도 기대 gradient의 방향이 바뀌지 않는 조건은 무엇인가?
5. dense reward가 sparse reward보다 위험할 수 있는 사례를 하나 만들어보자.

## 완료 체크

- [ ] agent, environment, state, action, reward, trajectory를 token 생성과 연결했다.
- [ ] return $G_t$와 advantage $G_t-b$의 차이를 설명했다.
- [ ] 작은 복도 환경에서 REINFORCE를 실행했다.
- [ ] sparse reward와 dense reward의 성공률 곡선을 비교했다.
- [ ] baseline 사용 전후의 gradient 흔들림을 여러 seed에서 측정했다.
- [ ] reward를 잘못 설계했을 때 생길 편법을 하나 찾았다.
- [ ] 결과물로 `Policy gradient 실습 노트`를 완성했다.

---

[^1]: Sutton, R. S. and Barto, A. G. (2018). [Reinforcement Learning: An Introduction, 2nd edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/). Figure 3.1과 Chapter 3, Chapter 13을 참고했다.
[^2]: Williams, R. J. (1992). [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://doi.org/10.1007/BF00992696). REINFORCE의 policy gradient update를 참고했다.
[^3]: PyTorch. [`torch.distributions.Categorical`](https://docs.pytorch.org/docs/stable/distributions.html#categorical). action sampling과 `log_prob` 사용법을 참고했다. 확인일: 2026-08-01.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 절별로 5,000자 이하로 나누어 점검
원본/윤문본: 9311자 / 9270자, 변경률 0.65%
탐지/수정: A-10 8→1, C-11 7→0, D-1 0→0, H-1 0→0, 그 밖의 S1 0→0
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 수식·코드·수치를 보존하며 반복 표현만 보수적으로 다듬음
주요 변경: “생각할 수 있다”→“생각하면 된다”, “쓸 수 있다”→“쓴다”, 연결어미 뒤 불필요한 쉼표 삭제, seed 5개의 실험표와 학습 곡선 추가
-->
