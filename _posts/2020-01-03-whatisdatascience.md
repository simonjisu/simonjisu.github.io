---
layout: post
title: "What is Data Science?"
date: "2020-01-03 14:19:38 +0900"
categories: datascience
author: "Soo"
comments: true
toc: true
---

# 데이터 사이언스란?

`Joma Tech` 라는 분의 유튜브에서 간결하고 명료하게 데이터 사이언스의 유래 및 역할에 대해 설명하여, 이를 정리하고 현재 내 상황, 그리고 나아가야할 방향에 대해 분석해보려고 한다.

<iframe width="560" height="315" src="https://www.youtube.com/embed/xC-c7E5PK0Y" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

* 2020.02.02 추가: 데이터관련 직군에 대한 [Report](https://d2wahc834rj2un.cloudfront.net/Workera%20Report.pdf?fbclid=IwAR0IEBfU-7w231SNnaJFM_DYPEqJQDgOdf5_eCVs0aGsazO9XBWaVxzrbF0)

---

## 데이터 사이언스와 데이터 사이언티스트의 역할

동영상에서 소개한 데이터 사이언티스트의 근본은 **"Problem Solver"**다. 현실에 존재하는 풀기 어려운 문제를 데이터를 사용해서 좋은 방향으로 이끌어가는 것이 데이터 사이언티스트의 역할이다.

{% include image.html id="1Wm5n8IMK5ruCcgu-7ccUxPIH8XOuGx2N" desc="출처: https://hackernoon.com/the-ai-hierarchy-of-needs-18f111fcc007" width="auto" height="auto" %}

위 사진은 영상에서 소개한 데이터 사이언스의 계층적 요구에 대해서 사진이다. 피라미드 꼭대기부터 정리해보면 다음과 같다.

| 번호 | 카테고리 | 업무 | 
|--|--|--|
| 1 | LEARN/OPTIMIZE | AI, Deep Learning|
| 2 | LEARN/OPTIMIZE | A/B Testing, Experimentation, Simple ML Algorithm| 
| 3 | AGGREGATE/LABEL | Analytics, Metrics, Segments, Aggregates, Features, Training Data | 
| 4 | EXPLORE/TRANSFORM | Cleaning, Anomaly Detection, Prep |
| 5 | MOVE/STORE | Reliable Data Flow, Infrastructure, Pipelines, ETL(extract, transform, load), Structured and unstructured data storage |
| 6 | COLLECT | Instrumentation, Logging, Sensors, External Data, User generated content | 

동영상에 따르면 스타트업 같은 경우, 리소스가 부족하기 때문에 데이터 사이언티스트는 1~6의 업무를 다 맡게 된다. 중견 기업의 경우 약간의 리소스를 더 사용하여, 데이터의 수집(6)은 소프트웨어 엔지니어가, 데이터의 보관 및 정제 준비(4, 5)는 데이터 엔지니어가, 나머지는 데이터 사이언티스트가 하게 된다. 조금더 큰 기업이라면 탑 3개의 분야를 한번 더 나눠서 데이터 분석, 평가 방법 설정 실험 등(2, 3)은 데이터 사이언스 애널리틱스, AI, Deep Learning 부분(1)의 업무는 리서치 사이언티스트 혹은 코어 데이터 사이언스가 맡게 된다.

정리하면 이 분야의 직군 분류는 다음과 같다.
* 소프트웨어 엔지니어(풀스택) - Software Engineer(Full Stack)
* 데이터 엔지니어 - Data Engineer
* 데이터 사이언스 애널릭틱스 - Data Science Analytics
* 리서치 사이언티스트 / 코어 데이터 사이언스 - Research Data Scientist / Core Data Science

이쯤에서 ~~차후에 가고 싶은~~ 페이스북(Facebook)의 채용 공고를 몇개 살펴보면 그 역할이 다 다르다는 것을 확인 할 수 있다.

> **Research Scientist, Artificial Intelligence (PhD)**

{% include image.html id="1D83TD6ZtJrSNwsSYNZVl86IynC0SWnuI" desc="출처: https://www.facebook.com/careers/jobs/985225105171427" width="auto" height="auto" %}

<br>

> **Data Scientist, Analytics (PhD)**

{% include image.html id="1c_DpRBhg9yAwQYE59swch4X2OFhSt4qv" desc="출처: https://www.facebook.com/careers/jobs/387405225294114" width="auto" height="auto" %}

<br>

> **Data Engineer, Machine Learning**

{% include image.html id="1mabjOctjAExkzV2SLpbv7UPo6zMpu_Uu" desc="출처: https://www.facebook.com/careers/jobs/997860117229177" width="auto" height="auto" %}

--

## 상황분석 및 나아가야할 방향

최근 언론에서 이야기하는 부분은 극 소수인 1번 분야이고 지금까지 내가 공부한 방향은 대부분 딥러닝 쪽이었다. 그러나 더 현실적인 문제는 4, 5 번인 데이터 엔지니어 분야, 3번인 데이터 분석 쪽에 있다고 생각한다. 특정 문제를 해결하기 위해 데이터를 수집 및 정제하고, 어플리케이션으로 과정을 경험해보면 좋은 포트폴리오가 될거라고 생각했다.

그렇다면 앞으로 대학원 2년간 어떤 전략을 짤 것인가? 

1. 기존의 딥러닝 분야의 공부는 계속하되, XAI, NLP 두 분야만 집중적으로 공부한다(+ 수학 및 통계 공부는 지속).
2. 데이터 기반 Product를 분석 및 연구
    * 각 산업별로 관련 회사 리스트업
    * 어떤 문제들이 있었고, 어떻게 해결했는지 스터디하기
    * 가능하면 업계사람들 많이 만나보기(묻고 싶은 질문지 준비하기)
3. 간단한 서비스, 웹 어플리케이션을 만들고 개선함으로써 데이터 수집 및 가공하는 연습하기(팀단위로 진행 목표)
    * 실존하는 해결하지 못한 문제를 찾아보고 괜찮은 서비스 기획해보기
    * SQL 기반의 데이터 파이프라인 설계(ETL 프로세스 설계 및 구현)
