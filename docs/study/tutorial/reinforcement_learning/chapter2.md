---
title: "2. Multi-armed Bandits"
hide:
  - tags
tags:
  - "reinforcement learning"
---


<figure markdown>
  ![HeadImg](https://drive.google.com/uc?id=1pptc4MA_slesmfXtkRZJjjFeuAkGWO2r){ class="skipglightbox" width="100%" }
  <figcaption>Reference: Pixabay</figcaption>
</figure>

> 데이터사이언스 대학원 강화학습 수업을 듣고 정리한 내용입니다.

John이라는 외국인이 카지도 슬롯머신(한국에서는 사적으로 하면 불법이지만, 일본에서는 길거리에 빠칭코 가게들이 많다)에 들어가서 슬롯 머신으로 게임을 진행하려고 한다. 수 많은 슬롯 머신 기기들이 있는데 어떤 기기에서 칩을 벌 수 있는 확률이 높을까?

**Multi-armed Bandits** 은 이러한 여러 대(multi)의 레버(armed)를 가진 슬롯 머신(bandits)를 이야기한다. 이 문제는 State 간에 전환(transit)이 없기 때문에 Reinforcement Learning의 제일 간단한 상황이다. 

## K-armed Bandit Problem

<figure markdown>
  ![HeadImg](https://drive.google.com/uc?id=1T8eAgwwVHfEzMWtpkVk0sQFx5ak3rZ7B){ class="skipglightbox" width="100%" }
  <figcaption>Reference: Pixabay</figcaption>
</figure>

만약 여러분이 친구랑 두 개의 코인(확률이 공정하고 0.5가 아님)을 가지고 내기를 하는데, 여러분이 선택한 코인이 윗면(head)이 나오면 $1를 여러분이 가지고 아랫면(tail)이 나오면 $1를 친구가 가져간다.

이러한 문제를 **K-armed bandit** 문제라고 한다. 여기서 K는 두 개의 코인 중 선택 해야하는 행동, armed bandit는 코인의 앞면, 뒷면이 나올 수 있는 상황이다. 

Formal한 정의를 하자면, 각 time-step $t$에서 $k$ 개 행동(Actions)중에 선택된 행동을 $A_t$라고 하며, 선택에 따라서 기대되는(혹은 평균) 보상을 $R_t$라고 한다. 따라서 어떤 행동 $a$의 가치(Value) $q_{*}(a)$는 선택된 행동의 보상 기댓값과 같으며 수식으로 다음과 같다.

$$q_{*}(a) \coloneqq \Bbb{E} \lbrack R_t \vert A_t = a \rbrack $$

다만, 우리는 선택된 행동의 가치는 모른다(만약에 알면 모든 사람은 복권에 당첨되었을 것이다). 따라서 선택한 행동의 가치를 추정해야하며, time-step $t$에서 선택한 행동 $a$의 **가치 추정값(estimated value of action)**을 $Q_t(a)$라고 한다. 우리의 목적은 $Q_t(a)$가 $q_{*}(a)$에 가장 근접하게만 만들면 최적의 결과를 얻을 수 있다.

그렇다면 $Q_t(a)$는 어떻게 계산할 수 있을까? 가장 간단한 방법은 현재 실행 했 던 기록을 가지고 계산하는 **sample-average** 방법이다.

$$\begin{aligned}Q_t(a) = \coloneqq \end{aligned}$$