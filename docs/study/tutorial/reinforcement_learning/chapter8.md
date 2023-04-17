---
title: "8. Planning and Learning with Tabular Methods"
hide:
  - tags
tags:
  - "reinforcement learning"
  - "planning"
---

> 데이터사이언스 대학원 강화학습 수업을 듣고 정리한 내용입니다.

지금까지는 model-based와 model-free 방법들을 다루어 보았다. 

* Model-based (e.g., Dynamic Programming)은 planning에 초점이 맞춰져있다.
* Model-free (e.g., Monte Carlo, Temporal Difference)은 learning에 초점이 맞춰져있다.

## Planning

**planning**은 model을 입력으로하고 policy를 출력으로하는 어떤 계산 프로세스로 정의된다. 

$$\text{model} \xrightarrow[]{\text{planning}} \text{policy}$$

* State-space planning: state space에서 optimal policy를 찾는 과정. Actions는 state과 state 사이의 transition을 야기하며, value function는 state로 계산된다.
* Plan-space planning: plan space를 찾는 과정. 하나의 plan에서 다른 plan으로 변경하는 연산(operators)을 찾는 것이다.

우선 같은 structure을 공유하는 state-space planning만 알아본다. Policy를 개선하기 위해서 value function을 계산하고, simulated experiences가 value function을 계산하는데 근거가 되어준다.

$$\text{model} \xrightarrow[]{} \text{simulated experience} \xrightarrow[]{\text{backups}} \text{values} \xrightarrow[]{} \text{policy}$$

## Planning & Learning

Planning과 Learning의 차이점은 학습에 사용되는 경험 종류다. Planning은 model에서 생성된 simulated experience를 사용하고, Learning은 환경에서 생성된 real experience를 사용한다. 보통 두 방법을 결합해서 사용하는데 대표적인 예로 "the planning method based on Q-learning"이 있다.

!!! info "random-sample one-step tabular Q-planning"

    === "pseudo code"
    
        ![HeadImg](https://drive.google.com/uc?id=1B5ouTXjMSO0A2wJxkFeHP2UPL67MYe-E){ class="skipglightbox" width="100%" }

        * 2번에서 보통 real experience를 쓰지만, sample method가 사용되었다.

## Dyna-Q: Integrated Planning and Learning

|Dyna |General Dyna| 
|:-:|:-:|
| ![HeadImg](https://drive.google.com/uc?id=1BHRiar1J0oJMe0ZrHF9pKFO54I_7jPPp){ class="skipglightbox" width="100%" } | ![HeadImg](https://drive.google.com/uc?id=1BMxCdEq2dTAnaD6dOgg1JwLWVJjSE0Qi){ class="skipglightbox" width="100%" } |

**Dyna-Q**는 online planning agent의 주요 functions을 포함하는 간단한 아키텍쳐다. Online 상황에서 planning, acting, learning을 결합한 방법이다. 

* **Model-Learning**: real experience로 model을 학습하는 것으로, 환경을 조금 더 잘 정확하게 따라할 수 있도록 한다.
* **Direct RL**: 강화학습으로 value function과 policy 을 개선하는 과정.
* **Indirect RL**: model을 통해 value function과 policy를 개선하는 과정.
* **Planning**: model을 통해 simulated experience를 생성하고, 이를 통해 value function과 policy을 개선한다.
* **Search Control**: simulated experience를 생성하기 위해 starting state와 action을 선택해서 model에 입력하는 과정.


!!! info "Tabular Dyna-Q"

    === "pseudo code"
    
        ![HeadImg](https://drive.google.com/uc?id=1BOyzRDso-Lw8HrBjBIlxIhbH0YJL1Pb4){ class="skipglightbox" width="100%" }

        * 여기서 (e) 와 (f) 과정이 없으면 one-step tabular Q-learning과 동일하다.


!!! note "Dyna Maze"

    ![HeadImg](https://drive.google.com/uc?id=1BXGoERyTg6R1Q1YURb2rVxhCXFRIowd7){ class="skipglightbox" width="100%" }

    미로 문제에서 Dyna-Q를 적용한 결과다. Dyna-Q는 planning을 통해 더 많은 경험을 얻어서 더 빠르게 optimal policy를 찾을 수 있다.

    * 모든 episode는 $S$에서 시작한다. 미로의 밖같과 검은 장애물은 지나갈 수 없다.
    * Actions = 상, 하, 좌, 우
    * Reward = $G$에 도착시 $+1$, 나머지는 $0$.

    ![HeadImg](https://drive.google.com/uc?id=1BXQ7GQ__7ov2V_7RAP48kf6RRMRUOCfc){ class="skipglightbox" width="100%" }

    위 그림은 두번째 episode에서 Planning 유무에 다른 policy 차이를 보여준다. Planning의 유무에 따라서 optimal policy를 찾는 속도가 달라진다.
    
### When the Model is Wrong

Model이 항상 옳은 것은 아니다. 예를 들어, (1) 환경이 stochastic하고, 경험의 갯수가 충분치 않을 때, (2) 일반화가 잘 되지 않을 때, (3) 환경이 변했는데 model이 관찰하지 못하는 경우 등이 있다. 다음 예제를 한 번 보자. Dyna-Q+는 조금 있다 다룬다.

!!! note "Example: Blocking and Shorcut Maze"

    === "Blocking Maze" 

        ![HeadImg](https://drive.google.com/uc?id=1BZ_NWxjrRoukE0kaZ1X7vKl3OgksSsAO){ class="skipglightbox" width="80%" }

        간단한 미로이지만 1000 time steps 이후에 장벽으로 생긴 블록이 오른쪽으로 한칸 이동한다. 처음에 최적의 경로는 오른쪽으로 탐색하는 것이고 블록이 옮겨지면 왼쪽으로 찾는 것이 최적의 경로다. 

    === "Shortcut Maze" 

        ![HeadImg](https://drive.google.com/uc?id=1BgPlwrHxWAv-Eni7UvGx0qO2Nrpk9tf3){ class="skipglightbox" width="80%" }

        간단한 미로이지만 3000 time steps 이후에 장벽의 오른쪽이 뚫리며 숏컷이 생긴다. 처음에 최적의 경로는 왼쪽으로  탐색하는 것이고 블록이 사라지면 오른쪽으로 찾는 것이 최적의 경로다. 그러나 Dyna-Q는 블록이 사라지고 나서 최적의 경로를 찾지 못하고 이전의 경로에 만족하는 모습을 보인다(cumulateive reward의 기울기가 급격하게 상승하지 않는다). Exploration-exploitation trade-off가 존재한다.

### Dyna-Q+

Dyna-Q+는 각 state-action 쌍이 지난번 시도 이후로 얼만큼 오랫동안 경과했는지 기록하는 휴리스틱한 방법이다. 시간이 오래되었을 수록 해당 state-action 쌍에 대한 모델의 지식이 잘못 되었다는 것을 알려주기 위함이다. 그래서 보상에 경과된 시간을 추가하여 업데이트 한다. 

$$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ (R + \kappa \sqrt{\tau}) + \gamma \max_{a} Q(S', a) - Q(S, A) \right]$$

* $\kappa$는 경과된 시간에 대한 보상의 가중치이다. $\kappa$가 크면 경과된 시간이 길어질수록 보상이 커지고, $\kappa$가 작으면 경과된 시간이 길어질수록 보상이 작아진다.

## Prioritized Sweeping

Dyan-Q 에서 Planning