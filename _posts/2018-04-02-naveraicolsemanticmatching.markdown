---
layout: post
title: "Semantic Matching Model"
date: "2018-04-02 16:46:07 +0900"
categories: "NaverAI2018"
author: "Soo"
comments: true
---
# Semantic Matching Model

본 내용은 Naver AI Colloquium 에서 김선훈 님이 발표한 "시멘틱 매칭 모델"을 제 기억을 토대로 재구성한 것입니다.

## Semantic Matching 의 필요성

우리가 이해하는 언어와 기계가 이해하는 언어는 확연히 다르다. 기계는 0과 1로 모든 것을 입력받는대로 인식하고, 조금이라도 틀리면 오류를 뱉어낸다. 여기서 인간의 이해와 기계의 이해 간의 차이를 **Semantic Gap** 이라고 하는데, 이 Gap을 줄여나가는게, 기계가 언어를 이해할 수 있게 만드는 일이다.

우리는 Semantic Matching를 통해, 기존의 Text Matching 보다 조금 더 다양한 표현을 잡을 수 있고, 오타의 문제도 조금 해결 할 수 있다.

예를 들어, 어떤 문장과 유사한 뜻의 문장을 찾고 싶은 일이 생길 수 있다.

아래의 말과 비슷한 말은 아래 후보중에 어떤 것일까?

> <span style="color: #e87d7d">Q: 오른쪽 아랫배가 아픈.. 왜 그런거죠?</span>
>
> <span style="color: #7d7ee8">1. 오른쪽 아랫무릎이 너무 아픈데.. 왜 그런거죠?</span>  
>
> <span style="color: #7d7ee8">2. 우측 하복부 통증 원인 알려줘</span>

기존의 단어기반 매칭(Text Matching)은 의미적으로 비슷한 2번이 아닌, "오른쪽", "아랫", "아픈", "왜 그런거죠?" 가 공통적으로 들어간 **1번** 을 선택할 것이다.

또 다른 예를 들어보자.

> <span style="color: #e87d7d">Q: 인구수 재일 많은 나라가 어디야?</span>
>
> <span style="color: #7d7ee8">1. 재일교포는 어느 나라 사람이야?</span>  
>
> <span style="color: #7d7ee8">2. 사람이 가장 많은 국가가 어디지?</span>

의미적으로 2번이 정답이겠지만, "제일" 이 오타가 나서 **1번** 과 유사하게 매칭 될 수도 있다.

위에 예시만 봐도, 사람의 말을 알아 듣는 음성 스피커를 만들게 된다면, 당연히 Semantic Matching 이 필요하다고 느낄 것이다.

## Semantic Matching Approach

### Encoding Based Methods
* sentence representation 추출
* 실시간 처리 가능(approximate nearest neighbor)

### Joint Methods
* 상호간 feature 이용이 가능(attention mechanism, exact match feature)
* 성능 향상 가능
