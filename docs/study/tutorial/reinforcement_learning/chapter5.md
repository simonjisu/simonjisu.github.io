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

예를 들어 $A, B, T$ 세 개의 state($T$는 종료)가 존재하고, Agent가 경험한 두 개의 episode가 관찰되었다. 

| Episode | $s_0$ | $r_0$ | $s_1$ | $r_1$ | $s_2$ | $r_2$ | $s_3$ | $r_3$ | $s_4$ |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
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
G^{e_1}_1 &= 3 + 2 - 4 + 4 - 3 = 2 \\
G^{e_1}_2 &= 2 - 4 + 4 - 3 = -1 \\
G^{e_1}_3 &= 4 - 3 = 1 \\
G^{e_2} &= 3 - 3 = 0 \\
V(A) &= \dfrac{G^{e_1}_1 + G^{e_1}_2 + G^{e_1}_3 + G^{e_2}}{4} = \dfrac{2 -1 + 1 + 0}{4} = 0.5
\end{aligned}$$

마찬가지로 B의 가치를 계산하면 다음과 같다.

$$\begin{aligned} 
G^{e_1}_1 &= -4 + 4 - 3 = -3 \\
G^{e_1}_2 &= -3 \\
G^{e_2}_1 &= 3 - 3 = 0 \\
G^{e_2}_1 &= 3 \\
V(B) &= \dfrac{G^{e_1}_1 + G^{e_1}_2 + G^{e_2}_1 + G^{e_2}}{4} = \dfrac{-3 -3 -2 - 3}{4} = -2.75
\end{aligned}$$
