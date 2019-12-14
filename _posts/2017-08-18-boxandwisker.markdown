---
layout: post
title: "박스앤위스커-박장시님 강연 후기"
date: "2017-08-18 13:34:35 +0900"
author: "Soo"
categories: others
---
**박스앤위스커 - 박장시님 강연 후기**
===

## 주제: 데이터 사이언스에 대한 몇 가지 실제 사례 소개
---
### 사례들

#### 1. 문제: 데이터로 무엇을 분석해야하는지 모르는 경우
데이터는 많은데, 분석을 어떻게 시작하는 모르는 경우, 우선 평소에 궁금한 것이 무엇인지 알아보기

streaming data의 검정: A/B test(시간 변수를 통제하기 위함)

* 온라인에서의 t-test, large-scale hypothesis testing시 effect size도 고려할 것(p-value가 얼만큼 변했는지)

#### 2. 문제: 인터넷 쇼핑몰에서 어떤 상품 배치가 최적?
MD vs 기계의 상품 배치 대결

웹에서의 '땅값' 개념 도입

A/B test

* 소규모 실험을 우선시 해서, 성과를 보여준 다음에 계속 확대하는 방향으로 가라

#### 3. 문제: MAB 테스트엔진
A/B test의 단점:

1) 테스트가 끝나기 전까지(결과를 얻기전까지) 몇 초간 손해를 볼 수가 있음(기회비용)

2) exploration vs exploitation
> Discovering new possibilities, conducting research, varying product lines, risk taking, innovation all fall under the realm of exploration. On the other hand, exploitation involves the refinement of current procedures: efficiency, production, execution, and so forth.
>
> 출처: http://www.indigosim.com/tutorials/exploration/t1s1.htm

3) 항상 변하는 세상: 언제나 옳은 진리는 없음, 상황에 맞춰서 정답도 변하는 세상

MAB: Multi-armed bandit - [개념링크](https://en.wikipedia.org/wiki/Multi-armed_bandit)

* 장점: 다양한 테스트가 가능하다

#### 4. 문제: 전시장 데이터 시각화
비콘 data(거리만 나타남)의 처리: 비콘 로그를 이용한 이동범위 추정

데이터의 편향 가능성: 관심있는 사람만 참가하기 때문에 보편적이지 않을 수도?

데이터 이상치(outlier)의 처리: 분석후, 현장 전문가 모셔서 이상치를 검증하고 제거

* Force-Directed graph
* color scheme: 히트맵 그릴시 https://colorbrewer2.org 참고할것
* numpy for grid: 큰 지도 데이터 경우, 지도를 하나의 큰 matrix로 볼 것

#### 5. 문제: 탱시 운행 정보 시각화
택시 미터기 데이터 처리: 미터기를 안누르는 경우도 있고해서 빈 데이터가 가끔 씩 존재했음(택시를 탔는 기록이었는데 다음 데이터에도 다시 타는 기록이 남은 경우), 이때 Finite-state machine 설계를 해서 부족한 데이터를 깔끔하게 보충함(택시를 탔으면 다음에는 무조건 내려야하는 것).

#### 6. 문제: 스킬 트리 분석, EDA - 탐색적 자료분석의 중요성
특별한 과정은 없음, 우선 데이터의 분포를 그려보는 것이 중요, 그래서 이 데이터를 어떻게 처리할 지 고민할 것

* ggplot

#### 7. 문제: 던전 이탈률 분석
회귀분석, 의사결정나무를 쓰는 이유는 사람들이 이해하기 쉽기 때문이다. 퍼포먼스는 약간 떨어지지만, 사람들을 설득하는데 도움이 됨. 우선 적용하는게 좋음

Validation을 어떻게 할지 처음부터 같이 고민할 것.

#### 8. 다차원큐브탐색
파이콘 2017 세션 강의 참고하기!

<p><br /></p>
<p><br /></p>

---
### 느낀점
강연이 좋았던게 다양한 사례를 통해서 어떻게 데이터를 접근할지 알려주고, 실제 고객들에게(혹은 다른 사람들에게) 설득하는 방법을 터득할 수가 있었음.

그리고 항상 프로젝트를 작게 시작하는 법도 배움, 조금씩 해서 성공하면 확장하는 방식으로 사고해야겠음.

"부산으로 가는데, 모든 신호등이 한번에 초록불로 변할 수는 없는 법" 이말도 인상 깊었다.
