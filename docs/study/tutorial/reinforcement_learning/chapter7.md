---
title: "7. n-step Bootstrapping"
hide:
  - tags
tags:
  - "reinforcement learning"
  - "bootstrapping"
---

> 데이터사이언스 대학원 강화학습 수업을 듣고 정리한 내용입니다.

![HeadImg](https://drive.google.com/uc?id=1AxWcB2C1jSotrHuh5PkU6B6u5zVDRyBo){ class="skipglightbox" width="80%" }

**n-step TD** 방법은 MC와 one-step TD 사이의 있는 방법이다. n-step TD는 n-step 만큼 bootstrapping 한다. Bootstrap이란 추정된 가치 혹은 수익을 기반으로 value function을 업데이트 한다는 뜻이다.

|name | n | equation |
| --- | --- | --- |
|TD(0) | 1 | $G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})$ |
|TD(1) | 2 | $G_t^{(2)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2})$ |
| $\vdots$ | $\vdots$ | $\vdots$ |
| MC | $\infty$ | $G_t^{\infty} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{t-1} R_T$ |

## n-step TD Target

$$G_t^{(n)} = G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n})$$

n-step return 계산시 $t$에서 $t+1$으로 transition될 때 접근할 수 없는 future reward 항이 포함되어 있다 $( R_{t+1}, \cdots, R_{t+n} )$. 따라서 $t+n$ 시점에서 업데이트가 진행된다. 

$$V_{t+n}(S_t) := V_{t+n-1}(S_t) + \alpha \lbrack G_t^{(n)} - V_{t+n-1}(S_t) \rbrack, \quad 0 \leq t \lt T$$

!!! info "n-step TD"

    === "pseudo code"

        ![HeadImg](https://drive.google.com/uc?id=1B0fVo-qOjWvIqDiq391FULmJnvtiOGjZ){ class="skipglightbox" width="100%" }

    $\tau$는 $t \geq n-1$를 체크하려고 하는 것이다. 즉, $t$가 $n$ 이후에 업데이트를 시작한다. $R_{t+n}$ 이후의 값은 value function $V_{t+n-1}(S_{t+n})$의 값으로 계산된다. 최악의 경우에도 n-step return의 기댓값이 $V_{t+n-1}(s)$에서 추정정되는 값보다 작거나 같다는 특성을 가지고 있으며 이를 **error reduction property**라고 한다. 

    $$\underset{s}{\max} \vert \Bbb{E}_\pi \lbrack G_t^{(n)} \vert S_t = s \rbrack - v_\pi(s) \vert \leq \gamma^n \underset{s}{\max} \vert V_{t+n-1}(s) - v_\pi(s) \vert $$

## n-step SARSA

n-step return 계산시 state value 대신 action value를 사용한다.

$$G_t^{(n)} = G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n Q_{t+n-1}(S_{t+n}, A_{t+n})$$

그리고 GPI update에도 마찬가지로 q-value function으로 업데이트 한다.

$$Q_{t+n}(S_t, A_t) := Q_{t+n-1}(S_t, A_t) + \alpha \lbrack G_t^{(n)} - Q_{t+n-1}(S_t, A_t) \rbrack, \quad 0 \leq t \lt T$$ 

다른 states에서는 $Q_{t+n}(s, a) = Q_{t+n-1}(s, a)$로 업데이트 되지 않는다.


!!! info "n-step SARSA"

    === "pseudo code"

        ![HeadImg](https://drive.google.com/uc?id=1B2QfrizFauDMcnt4VzUDSEod8UyP29T3){ class="skipglightbox" width="100%" }

Expected SARSA의 경우에는 $\bar{V}_{t+n-1}(S_{t+n}) := \sum_a \pi(a\vert s) Q_t(s, a)$를 사용한다.

## n-step Off-Policy Learning

n-step importance sampling ratio를 통해 off-policy learning을 할 수 있다.

$$\rho_{t:h} = \prod_{k=t}^{\min(h, T-1)} \frac{\pi(A_k \Vert S_k)}{b(A_k \Vert S_k)}$$

이를 업데이트 규칙에 대입하면 다음과 같다.

$$V_{t+n}(S_t) \leftarrow V_{t+n-1}(S_t) + \alpha \rho_{t:t+n-1} \lbrack G_{t:t+n} - V_{t+n-1}(S_t) \rbrack, \quad 0 \leq t \lt T$$

n-step SARSA의 경우에는 다음과 같다.

$$Q_{t+n}(S_t, A_t) \leftarrow Q_{t+n-1}(S_t, A_t) + \alpha \rho_{t+1:t+n-1} \lbrack G_{t:t+n} - Q_{t+n-1}(S_t, A_t) \rbrack, \quad 0 \leq t \lt T$$

!!! info "n-step SARSA off-policy learning"

    === "pseudo code"

        ![HeadImg](https://drive.google.com/uc?id=1B4TKtCqEDa8ljUcwsyAX68aV4n3akSVt){ class="skipglightbox" width="100%" }

그렇다면 Q-learning은 off-policy 알고리즘인데 왜 importance sampling ratio를 사용하지 않는가[^1]? 두 policy 간의 차이를 줄이기 위해서 importance sampling을 하는데, Q-Learning은 해당 상태에서 주어진 모든 action을 확률적으로 사용하는 것이 아니라 greedy 하게 선택하기 때문에 필요가 없다. 즉, 선택되는 action $a'$의 $\pi(a' \vert s') = 1$이 되고 나머지는 $0$이 되기 때문이다. 


[^1]: [Q-Learning: off-policy TD control](https://talkingaboutme.tistory.com/entry/RL-Q-learning-Off-policy-TD-Control)