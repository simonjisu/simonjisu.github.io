---
title: "5. Monte Carlo Method"
hide:
  - tags
tags:
  - "reinforcement learning"
  - "monte carlo"
---

> 데이터사이언스 대학원 강화학습 수업을 듣고 정리한 내용입니다.

## Monte Carlo Method

**Monte Carlo** 방법은 여러 곳에서 쓰이는 용어로 전체를 알기 위해 random sampling을 통해 근사해서 추정하는 방법이다. 강화학습에서 어떻게 사용이 되는지 알아본다.

지금까지는 complete model, 즉, MDP의 dynamics $p(s', r \vert s, a)$를 완전히 아는 상태를 가정했지만, 현실은 그렇지 않다. 현실에서 우리는 **경험(experience)**으로부터 배운다. 경험이란 상태, 행동, 그리고 보상으로 이루어진 $\lbrace s, a, r, s' \rbrace$ 일련의 시퀀스다. 

우선 이번 챕터에서 episodic, undiscounted($\gamma = 1$), 그리고 episode-by-episode incremental update를 진행한다고 가정해보자. 또한, bandit 알고리즘 처럼 각 state-action pair 별로 평균 수익(average return)을 구한다고 가정한다.

## Monte Carlo Prediction

Monte Carlo Prediction를 수행하려면 [Policy Evaluation](../chapter4/#policy-evaluation), 즉, 주어진 policy $\pi$ 에서 state-value function을 학습하면 된다.

예를 들어 $A, B, T$ 세 개의 state($T$는 종료)가 존재하고, Agent가 경험한 두 개의 episode가 관찰되었다. 

| Episode | $s_0$ | $r_0$ | $s_1$ | $r_1$ | $s_2$ | $r_2$ | $s_3$ | $r_3$ | $s_4$ | $r_4$ | $s_5$ |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | A | +3 | A | +2 | B | -4 | A | +4 | B | -3 | T |
| 1 | B | -2 | A | +3 | B | -3 | T |   |   |   |   |

Return을 계산하는 방법에 따라 두 개의 알고리즘으로 나뉜다.   

!!! info "First-visit MC prediction"

    === "pseduo code"
        
        ![HeadImg](https://drive.google.com/uc?id=18PPPbW18TSVP6lLKpWar-2HMGDt90mCA){ class="skipglightbox" width="100%" }

    === "every-visit 차이" 
    
        [StackExchange](https://ai.stackexchange.com/questions/10812/what-is-the-difference-between-first-visit-monte-carlo-and-every-visit-monte-car)

### First-visit MC

$\gamma = 1$, episodic return을 사용해서 계산시([3. Finite Markov Decision Processes](../chapter3/#reward-hypothesis) 참고) A의 가치를 first-visit MC로 계산하는 방법: 각 episode 별로 처음 A라는 state가 등장 했을 때 부터 계산하면 된다. 따라서 다음과 같이 계산된다.

$$\begin{aligned} 
G^{e_1} &= 3 + 2 - 4 + 4 - 3 = 2 \\
G^{e_2} &= 3 - 3 = 0 \\
V(A) &= \dfrac{G^{e_1} + G^{e_2}}{2} = \dfrac{2 + 0}{2} = 1
\end{aligned}$$

마찬가지로 B의 가치를 계산하면 다음과 같다.

$$\begin{aligned} 
G^{e_1} &= -4 + 4 - 3 = -3 \\
G^{e_2} &= 3 - 3 = 0 \\
V(B) &= \dfrac{G^{e_1} + G^{e_2}}{2} = \dfrac{-3 - 2}{2} = -2.5
\end{aligned}$$

### Every-visit MC

Every-visit MC로 계산하는 방법은 매번 해당 state가 나올 때 마다 계산 해준다는 뜻이다. episode의 return을 계산시 매번 state가 나올 때 마다 해당 시퀀스를 return 계산에 포함하게 되는데, A의 가치를 예를 들면 다음과 같다.

$$\begin{aligned} 
G^{e_1}_1 &= 3 + 2 - 4 + 4 - 3 = 2 ,\quad G^{e_1}_2 = 2 - 4 + 4 - 3 = -1 \\
G^{e_1}_3 &= 4 - 3 = 1 ,\quad G^{e_2} = 3 - 3 = 0 \\
V(A) &= \dfrac{G^{e_1}_1 + G^{e_1}_2 + G^{e_1}_3 + G^{e_2}}{4} = \dfrac{2 -1 + 1 + 0}{4} = 0.5
\end{aligned}$$

마찬가지로 B의 가치를 계산하면 다음과 같다.

$$\begin{aligned} 
G^{e_1}_1 &= -4 + 4 - 3 = -3 ,\quad G^{e_1}_2 = -3 \\
G^{e_2}_1 &= 3 - 3 = 0 ,\quad G^{e_2}_1 = 3 \\
V(B) &= \dfrac{G^{e_1}_1 + G^{e_1}_2 + G^{e_2}_1 + G^{e_2}}{4} = \dfrac{-3 -3 -2 - 3}{4} = -2.75
\end{aligned}$$

## Monte Carlo Prediction of Action Values

만약에 model이 불분명하다면(=dynamics를 이용할 수 없다면), state-value function $v_\pi(s)$ 대신 action-value function $q_\pi(s, a)$를 사용해야한다. State $s$에서 action $a$를 선택하고 기대 수익을 계산하면 된다.

$$q_{\pi}(s, a) := \Bbb{E}_{\pi} \lbrack G_t \vert S_t = s, A_t = a\rbrack = \Bbb{E}_{\pi} \Bigg\lbrack \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \vert S_t = s, A_t = a \Bigg\rbrack$$

그러나 어떤 state-action 쌍은 한번도 선택되지 않을 수 있는 경우가 생길 수 있다. 따라서 state-action 쌍에 0이 아닌 확률을 부여하여 **탐색(exploration)**할 수 있게 해야한다. 이러한 결과로 policy도 확률적(stochastic)으로 선택하게 된다. 

## Monte Carlo Control

기본적으로 [Generalized Policy Iteration(GPI)](../chapter4/#generalized-policy-iteration)와 같은 패턴을 가진다. 여기서 $E$는 policy evaluation, $I$는 policy improvement다.

$$\pi_0 \xrightarrow[]{E} q_{\pi_0} \xrightarrow[]{I} \pi_1 \xrightarrow[]{E} q_{\pi_1} \cdots \xrightarrow[]{I} \pi_* \xrightarrow[]{E} q_{\pi_*} $$

그러나 GPI 과 다른점은 (1) 평가시 epidosic 하게 업데이트, (2) $v_\pi$ 대신에 $q_\pi$를 사용, 그리고 (3) $I$ 과정에서 $q(s,a)$를 이용해 최적의 행동을 선택하여 policy를 선택한다. 

$$\pi_{k+1}(s)=\underset{a}{\arg \max}\ q_{\pi_k}(s, a)$$

또한 policy improvement은 모든 $s\in \mathcal{S}$에서 현재 주어진 policy $\pi_k$의 value function을 최대화 하는 action을 선택하면 된다.

$$\begin{aligned}q_{\pi_k} \big(s, \pi_{k+1}(s) \big) &= q_{\pi_k} \big(s, \underset{a}{\arg \max}\ q_{\pi_k}(s, a) \big) \\
&= \underset{a}{\max}\ q_{\pi_k}(s, a) \geq q_{\pi_k}\big(s, \pi_k(s)\big) \geq v_{\pi_k}(s)
\end{aligned}$$

### Monte carlo with Exploring Starts

!!! info "Monte Carlo ES"

    === "pseduo code"
        
        ![HeadImg](https://drive.google.com/uc?id=18hhd4TQyk1o5BAY6ChUvl9Vszub3eWlI){ class="skipglightbox" width="100%" }

Exploring starts는 확률이 0이 상인 state-action 쌍을 고른다는 뜻이다. 그러나 해당 알고리즘은 실용적이지 않다. 모든 state-action 쌍마다 모든 return을 유지하고 중복적으로 평균을 계산하기 때문이다. 

### On-policy and Off-policy

**On-policy** 방법은 의사 결정에 사용되는 policy를 평가하거나 개선하려고 시도한다. 즉, 업데이트 하려는 policy와 최종 의사결정에 사용되는 policy(action 선택)가 같다고 보면 된다. **Off-policy** 방법은 데이터 생성에 사용되는 것과 다른 정책을 평가하거나 개선한다. 이 개념은 차후에 behavior policy와 target policy 연결지어서 설명된다.

보통 on-policy 방법에서 policy 는 soft한 성격을 듸고 있다. 즉, 모든 state $s \in \mathcal{S}$ 와 action $a \in \mathcal{A}$에 대해서 $\pi(a \vert s) > 0$ 이며, 점진적으로 deterministic optimal policy에 다가간다는 뜻이다. 예로 $\epsilon$-greedy policy를 들 수가 있다.

$$\begin{cases}
\dfrac{\epsilon}{\vert \mathcal{A}(s) \vert} & \text{non-greedy action} \\
1 - \epsilon + \dfrac{\epsilon}{\vert \mathcal{A}(s) \vert} & \text{greedy action}
\end{cases}$$

![HeadImg](https://drive.google.com/uc?id=18mpw3St6NI0TP9vBLgG6E1lIE62vQsBE){ class="skipglightbox" width="65%" }

이러니저러니해도 on-policy MC control 는 GPI의 한 종류다. Exploring starts 이라는 가정 없이는 policy를 개선 시킬 수 없는데 왜냐면 non-greedy actions의 선택을 방지 해주기 때문이다. 다행이 GPI는 policy가 greedy policy로 향하게 끔만 하면되지 강제로 greedy policy가 되어야 한다는 필요 조건이 없다. 어떠한 $\epsilon$-soft policy $\pi$에 대해서 $\epsilon$-greedy policy에 관한 $q_\pi$는 $\pi$ 보다 항상 좋다는 것을 보장한다.

!!! info "On-policy first-visit MC control $\epsilon$-soft policies"

    === "pseduo code"
        
        ![HeadImg](https://drive.google.com/uc?id=18mZ2An-k0x9p-t29lxpgyvUWwUukEsIP){ class="skipglightbox" width="100%" }

### Policy Improvement Guarantees with ε-soft Policy

왜 보장되는지 살펴보자. $\epsilon$-greedy policy인 $\pi'(a \vert s)$가 있다.

* Greedy: $1 - \epsilon + \dfrac{\epsilon}{\vert \mathcal{A}(s)\vert}$ 확률로 $A=\underset{a}{\arg \max}\ q_\pi(s, a)$ 선택
* Exploration: $\dfrac{\epsilon}{\vert \mathcal{A}(s)\vert}$ 확률로 $A \neq \underset{a}{\arg \max}\ q_\pi(s, a)$ 선택하지 않음

$$\begin{aligned}q_\pi \big(s, \pi'(s) \big) &= \sum_a \pi'(a \vert s) q_\pi(s, a) \\
&= \dfrac{\epsilon}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a) + (1 - \epsilon) \cdot \underset{a}{\max}\ q_\pi(s, a) \\
&\geq \dfrac{\epsilon}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a) + (1 - \epsilon) \sum_a \dfrac{\pi(a \vert s) - \dfrac{\epsilon}{\vert \mathcal{A}(s)\vert}}{1 - \epsilon} q_\pi(s, a) \\
&= \dfrac{\epsilon}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a) - \dfrac{\epsilon}{\vert \mathcal{A}(s)\vert} \sum_a q_\pi(s, a) + \sum_a \pi(a \vert s) q_\pi(s, a) \\
&= \sum_a \pi(a \vert s) q_\pi(s, a)  = v_\pi(s), \quad \forall s \in \mathcal{S}
\end{aligned}$$

즉, [policy improvement theorem](../chapter4/#policy-improvement)에 의해서 action-value가 state-value보다 같거나 높기 때문에 $\pi'$는 $\pi$보다 좋은 policy이며, action-value function을 최대화 하는 policy가 결국에 좋은 policy를 선택하는 것과 같게 된다. 그림으로 개념을 표현하면 다음과 같다.

![HeadImg](https://drive.google.com/uc?id=18n6jlc2x6iGZwMiPuFVX_wcD2Ta3alzp){ class="skipglightbox" width="65%" }

## Off-policy Prediction

**Off-policy**는 {++target++} policy $\pi(a\vert s)$ 를 평가 혹은 개선하는데 다른 {++behavior++} policy $b(a\vert s)$가 생성하는 데이터를 바탕으로 진행한다는 점에서 On-policy와 다르다. 즉, target policy는 학습되고 최적화의 대상이라면, behavior policy는 환경과 상호작용한다. 사실 On-policy는 Off-policy의 특별한 케이스다 $\pi(a \vert s) = b(a \vert s)$.

![HeadImg](https://drive.google.com/uc?id=18rD0i1KAgszxddFRTHOi8AQEXFy1z-OQ){ class="skipglightbox" width="100%" }

Grid-World 예를 들자면, 그림과 같다. 우리의 목적은 behavior policy로 target policy $\pi$의 value function $v_\pi$을 학습하고자 한다. 조건으로 모든 action $a \in \mathcal{A}(s), \pi(a \vert s) > 0$ 에 대해서 $b(a \vert s) > 0$이어야 한다.

### Importance Sampling

**Importance Sampling**은 behavior policy로부터 $v_\pi$를 학습시킬 수 있는 방법 중 하나다. Behavior policy로부터 random sampling을 한 후에 {++중요도 가중치(importance weight)++} $\rho$를 사용하여 분포를 조정한다(=policy control). Random variable $X \backsim b$이 있고, $\Bbb{E}_\pi \lbrack X \rbrack$가 목적인 상황에서 $\rho$를 도출 할 수 있다.

$$\begin{aligned} 
\Bbb{E}_\pi \lbrack X \rbrack &= \sum_{x\in \mathcal{X}} \pi(x) \cdot x = \sum_{x\in \mathcal{X}} \dfrac{b(x)}{b(x)}\pi(x) \cdot x = \sum_{x\in \mathcal{X}} \dfrac{\pi(x)}{b(x)} b(x) \cdot x \\
& := \sum_{x\in \mathcal{X}} \rho(x) b(x) \cdot x = \Bbb{E}_b\lbrack \rho(X)X \rbrack \\
&\approx \dfrac{1}{n} \sum_{i=1}^n \rho(x_i) x_i \quad \text{with } x_i \backsim b
\end{aligned}$$

![HeadImg](https://drive.google.com/uc?id=18v3YCtLF3cJ9jidOjiQ1L-y6EdME9TrC){ class="skipglightbox" width="100%" }

예를 들어, 확률이 균일한 파란 주사위와 균일하지 않은 빨간 주사위가 있다. 우리의 목적은 파란색 주사위를 굴려서 빨간색 주사위의 기댓값을 알려고한다. 3번 굴렸을 때 R.V. 의 realization은 다음과 같다.

1. $X_1^{blue} = 1, \pi(1)=0.7, b(1)=0.167$
1. $X_2^{blue} = 2, \pi(1)=0.1, b(2)=0.167$
1. $X_3^{blue} = 1, \pi(1)=0.7, b(1)=0.167$

따라서 기댓값은 다음과 같이 계산할 수 있다.

$$\Bbb{E}_\pi\lbrack X\rbrack \approx \dfrac{1}{n}\sum_{i=1}^n \rho(x_i) x_i = \dfrac{1}{3}\big( (1\times \dfrac{0.7}{0.167}) + (2\times \dfrac{0.1}{0.167}) + (1\times \dfrac{0.7}{0.167})\big) = 3.2$$

### Off-policy Prediction using Importance sampling 

Policy $\pi$ 하에서 어떤 trajectory(어떤 시나리오 시퀀스)의 likelihood와 importance weight $\rho$는 다음과 같다.

$$\begin{aligned}
Pr\lbrace A_t, S_{t+1}, A_{t+1}, \dots, S_T \vert S_t, A_{t:T-1} \backsim \pi \rbrace = \prod_{k=t}^{T-1}\pi(A_k \vert S_k) \cdot p(S_{k+1 \vert S_k, A_k}) \\
\rho_{t:T-1} = \dfrac{\prod_{k=t}^{T-1}\pi(A_k \vert S_k) \cdot p(S_{k+1 \vert S_k, A_k})}{\prod_{k=t}^{T-1}b(A_k \vert S_k) \cdot p(S_{k+1 \vert S_k, A_k})} = \prod_{k=t}^{T-1}\dfrac{\pi(A_k \vert S_k)}{b(A_k \vert S_k)} 
\end{aligned}$$

그리고 value function는 두 가지 방법으로 계산 할 수 있다.

=== "ordinary"

    $$V(S) = \dfrac{\sum_{t\in \mathcal{J}(s)} \rho_{t:T(t)-1} G_t}{\vert \mathcal{J}(s) \vert}$$

=== "weighted"

    $$V(S) = \dfrac{\sum_{t\in \mathcal{J}(s)} \rho_{t:T(t)-1} G_t}{\sum_{t\in \mathcal{J}(s)} \rho_{t:T(t)-1}}$$


!!! info "Off-policy MC prediction and control"

    === "Off-policy MC prediction (policy evaluation)"
        
        ![HeadImg](https://drive.google.com/uc?id=18y-Ddxcy20MULt0ualdtc6iZby6A5WWo){ class="skipglightbox" width="100%" }

    === "Off-policy MC control"
        
        ![HeadImg](https://drive.google.com/uc?id=18zGTpS0DVK7Jphd_4QwVmRwRgYtL8lka){ class="skipglightbox" width="100%" }