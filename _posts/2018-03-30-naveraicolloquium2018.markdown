---
layout: post
title: "Naver AI Colloquium 2018"
date: "2018-03-30 20:14:21 +0900"
categories: "NaverAI2018"
author: "Soo"
comments: true
---
# Naver AI Colloquium 2018 참석 후기
~~\+ 클로바스피커 후기~~
## 개요

<ul id="light-slider1">
  <li><img src="/assets/ML/naveraicol/logo.png"></li>
  <li><img src="/assets/ML/naveraicol/ment1.png"></li>
  <li><img src="/assets/ML/naveraicol/ment2.png"></li>
  <li><img src="/assets/ML/naveraicol/ment3.png"></li>
</ul>

출처: [<span style="color: #7d7ee8">네이버 AI 콜로키움</span>](http://naveraiconf.naver.com/)

운 좋게 기회가 되서 [**Naver AI Colloquium 2018**](http://naveraiconf.naver.com/) 에 다녀왔다.
싸이트에 접속해서 프로그램 표를 자세히 보면 알겠지만, 이번 콜로키움에서는 크게 4가지 주제를 다뤘다.

* 언어 분야(Search & Natural Language)
* 비전 분야(Computer Vision, Mobility&Location Intelligence)
* 추천 분야(Recommendation)
* 데이터 엔지니어 분야(AI Algorithm, System&Platform)

나는 언어 분야, 즉 자연어을 다루는 쪽에 관심이 많아서 Track A 만 거의 듣고, 추천 분야 하나 정도 들었다. 각 강의에서 다루는 내용은 차후 하나씩 정리해서 올릴 예정이다.

## 후기

<img src="/assets/ML/naveraicol/professorsungkim.JPG">

"모두의 딥러닝"으로 유명하신 Sung Kim 교수님!! (문제가 되면 사진 내리겠습니다~ 댓글 달아주세요)

인상 깊었던 세션과 그 이유를 몇개 꼽자면, ~~어쩌다보니 다 네이버 직원분들이 발표하신거네ㅎㅎ~~

1. Semantic Matching Model - 김선훈 (NAVER)
시멘틱 매칭의 필요성:
  - **키워드(단어기반)** 매칭보다는 **시멘틱(의미기반)** 매칭이 다양한 표현과 오타를 커버할 수 있는 가능성이 높다.
  - Sementic Gap: 인간이해와 기계이해의 차이, 이것을 줄이는게 큰 과제
2. Neural Speed Reading - 서민준 (NAVER)
  - Skim-RNN: "속독"에서 나온 아이디어, 중요하지 않은 단어는 적게 업데이트!
  - Big RNN 과 Small RNN 의 결정 짓는 Decision Function
  - Layer를 쌓으면 중요한 정보를 캐치 (마치 글을 두번째 읽을 때는 주요 단어만 보게 되는 것과 같음)
3. Hybrid Natural Language Understanding 모델 - 김경덕 (NAVER)
  - 문제 정의의 중요성: 잘해야 문제를 해결하기 쉽고 명확하다.
  - 팬턴 기반 검색 NLU + 데이터 통계 기반 NLU, 뭐든지 하나만 고르는 것은 아니다.
4. 자기학습 기반 챗봇(발표세션은 아님)
  - 챗봇의 전체 과정:
  Query $\rightarrow$ 언어적 특징 추출 $\rightarrow$ 쿼리 분류기(대화여부 및 도메인 인지) $\rightarrow$ 여러 모델로 부터 답변 생성 $\rightarrow$ Answer
  - N-hot representation: 토큰원형 + 품사태깅 + 어미

이 정도인 것 같다.

## 클로바 스피커(프렌즈) 후기

<img src="/assets/ML/naveraicol/speaker.jpeg">

어쩌다 운좋게 경품에 당첨 되서 받았다. ㅎㅎㅎ 감사합니다.
이놈...생각보다 귀엽다. 아직까지 일본의 프렌즈보다 기능이 덜 있는 것 같다. 라인으로 메세지 보내는 기능 시도해보았는데, 안되드라...

영상링크: [<span style="color: #7d7ee8">【公式】Clova Friendsができること </span>](https://youtu.be/lK-9yDoHsZ8)

아무튼 아직 개선할 사항이 많다. 예를 들어, "레미제라블 Do you hear the people sing? 노래틀어줘"라고 말하면, 말씀하신 사항을 찾지 못했다고 대답한다.
어떤 세션에서 들었던것 같은데, 내 생각에는 레미제라블, 노래틀어줘는 노래틀어주는 분야로 의도로 분류되고, 나머지 영어는 번역하는 의도로 분류된 것 같다. (스피커에 말한 내용을 볼 수 있는데, 음성인식을 진짜 제대로 잘 된다. 괜히 1위라고 말한게 아닌듯 ㅋㅋ)

---
내년에 또 하게되면 참가하고싶다~

관심 있었던 세션들을 정리하면 바로 올리겠다.
