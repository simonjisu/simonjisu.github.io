---
layout: post
title: "Background: Text2SQL"
date: "2021-10-24 21:00:01 +0900"
categories: nlp
author: "Soo"
comments: true
toc: true
---

{% include image.html id="1GcfpUQEg_all-HDuvJmrIlxEgo_yq9oX" desc="Reference: Pixabay" width="100%" height="auto" %}

# Text-to-SQL

관계형 데이터 베이스(Relational Database)는 현재 가장 많이 쓰이고 있는 데이터 베이스다. 관계형 데이터 베이스는 여러개의 테이블(table)로 이루어져 있으며, 우리가 평소에 많이 접하는 엑셀이 테이블의 가장 흔한 예시라고 할 수 있다. 

당신이 만약에 스타벅스 한국지사 관리자라고 상상을 해보자. 한국지사의 데이터 베이스에는 각 지역별 매장의 위치, 매출, 비용 등등 많은 양의 정보(테이블)가 들어 있다. 보통 회사의 데이터 분석가들이 DB에서 SQL을 통해 데이터를 추출하여, 분석을 하고 분석한 내용을 당신에게 보고서를 만들 것이다. 예를 들어 보고서에서 다음과 같은 테이블을 보았다.

### RegionData

|region|n_store|sales|
|---|---|---|
|서울특별시|568|537,727,860,697|
|경기도|361|341,760,136,816|
|부산광역시|127|120,231,405,473|
|대구광역시|71|67,215,982,587|
|인천광역시|63|59,642,350,746|
|경상남도|59|55,855,534,826|
|광주광역시|59|55,855,534,826|
|대전광역시|56|53,015,422,886|
|경상북도|47|44,495,087,065|
|충청남도|33|31,241,231,343|
|울산광역시|30|28,401,119,403|
|전라북도|27|25,561,007,463|
|충청북도|25|23,667,599,502|
|강원도|25|23,667,599,502|
|제주특별자치도|23|21,774,191,542|
|전라남도|23|21,774,191,542|
|세종특별자치시|11|10,413,743,781|

스타벅스 매장의 개수에 따라서 정렬된 테이블을 보고 당신은 이런 궁금증이 생겼다. "지역별로 매출이 매장의 개수에 따라 많아지는데, 비용은 어떻게 될까?", "효율적으로 매출을 내는 지역은 어디일까?" 

이러한 궁금증들이 당장 답변을 받고 싶지만, 다시 직원에게 보고서를 만들어 오라고 하기에는 너무 시간이 오래 걸린다. 만약, 당신이 자연어로 질문을 했을 때 필요한 SQL문을 인공지능이 알아서 작성하게 된다면 얼마나 좋을까?

이렇게 자연어 발화를 SQL로 바꾸는 분야를 **Text-to-SQL(혹은 NL-to-SQL)**이라고 한다. SQL을 배우지 않아도 자연어로 쉽게 데이터를 추출가능하게 하고, 데이터의 접근성을 향상 시킬 수 있다. 

사실, 이러한 분야는 오래전부터 연구가 시작되었다. 이러한 문제를 보통 Natural Language Interfaces to Databases(NLIDBs, [Androutsopoulos et al, 1995](https://arxiv.org/abs/cmp-lg/9503016))라는 이름으로 데이터 베이스 분야에서 존재 했었다. 이 당시에는 전통적인 자연어 처리 기법으로 문제를 해결하려고 노력했다. 그러나 최근 Transfomer 및 Pre-trained Language Model의 급 부상으로 딥러닝 기반의 방법론이 많이 적용이 되었다.

추가로 말하자면, 이러한 분야를 Semantic Parsing 이라고도 하며, 보통 자연어 발화를 유의미한 표현으로 전환하는 문제를 말한다.

---

# 데이터 이야기

사실 2017년 전까지는 이 분야에는 ImageNet처럼 방대하게 실험해 볼 수 있는 데이터 세트가 없었다. 2017년 이후 부터 WikiSQL, SPIDER 등 다양한 벤치마크 데이터 세트가 등장하면서 활발한 연구가 시작되었다.

## WikiSQL

2017년에 Salesforce에서 크라우드 소싱하여 상대적으로 큰 데이터 세트인 WikiSQL를 만들었다. 지금까지도 데이터 세트는 표준 데이터처럼 사용한다. WikiSQL은 총 25,683개의 테이블과 80,654 쌍의 자연어 질의와 SQL문이 있다. 문제의 단순화를 위해서, 오직 독립적인 하나의 테이블의 국한하여 질의를 하고, SELECT, WHERE 및 간단한 AGGREGATION 정도의 단순한 구문으로 구성되어 있다. 

위의 스타벅스 테이블을 예로 들면 다음과 같다. 

```
질의: 서울특별시의 스타벅스 매장 개수가 어떻게 되?
SQL: SELECT n_store FROM RegionData WHERE region = "서울특별시"
```

## SPIDER

WikiSQL의 문제점은 심플한 쿼리에다 자주 쓰이는 JOIN, GROUP BY, ORDER BY 등이 없다는 점이다. 이러한 단점을 보완하기 위해서 Yale 대학교를 주축으로 11개의 대학에서 대략 1,000시간을 소요해서 200개의 데이터베이스, 10,181개의 자연어 질문, 5,693개의 난이도가 상이한 쿼리문으로 구성된 [SPIDER](https://yale-lily.github.io/spider) 데이터 세트를 제작했다. 기존의 비해 양도 많고, 훈련/테스트 데이터 세트를 데이터 베이스 기준으로 잘라서 실험했다는 점에서 신선한 관점을 줬었다.

{% include image.html id="1FugtDbQb5TINzAmaknIyS5s-HPbt9kfz" desc="다른 데이터 세트와 Spider 1.0 데이터 세트의 비교" width="100%" height="auto" %}

물론 여기도 여러가지 제약이 있는데, 자세한 내용은 해당 논문 리뷰([링크](https://github.com/simonjisu/Text2SQL/blob/main/MD/01.md)) 를 참고하길 바란다.

---

# 접근법

결국 Text2SQL의 목적은 자연어 발화가 주어지면 올바른 SQL을 생성하는 문제로, 어떻게 보면 Machine Translation과 비슷한 Task라고 볼 수 있다. 그런 점에서 초기에 Seq-to-Seq로 접근 하는 부분이 많았지만, 이러한 접근 법은 순서가 어느 정도 중요한 SQL 구문 특성상 잘 작동하지 않았다. 

다른 법근 법으로 SQL문을 각기 다른 파트인 Sub-Tasks로 나눠서, 이 문제를 풀려고 했고, 네이버의 SQLova, MS의 HydraNet등 다양한 모델들이 좋은 성능을 내기 시작했다. 

이 후에도 다양한 접근 법들이 나오고 있는데, Domain Ontology, Linking, 타입(Type) 임베딩, Graph적인 접근 등등 다양한 방법들이 시도되고 있다. 차후 글들에서 이러한 접근법을 하나씩 읽어보고 소개하려고 한다.

---

# 문제점

이 분야를 연구하면서 교수님으로부터 제일 많이 들었던 소리는 실제 기업의 데이터 베이스가 주어진 데이터 세트처럼 아름답지가 않다고 한다. 무슨 말인가 했는데, Text2SQL의 핵심중 하나는 사실 테이블의 칼럼 이름과 레코드 값들이 자연어와 "언어"라는 도메인을 공유하고 있다는 점이다. 그러나 실제 기업의 데이터 베이스의 칼럼 이름은 간단하게 짓고, 나중에 mapping table로 해당 칼럼 이름을 확인한다는 말을 많이 들었다(물론 이런 곳이 없는 곳이 있을 것 같지만).

다른 문제점은 인간에게 당연한 상식이나 추론을 적용하기 어렵다는 점이다. 즉, 명확한 질문에 해당하는 SQL문만 가능하다는 것인데, 예를 들어, "Display the employee id for the employees who report to John"의 질문을 쿼리로 바꾸면 다음과 같다. 

```sql
SELECT employee_id
FROM employees
WHERE manager_id = (
	SELECT employee_id
	FROM employees
	WHERE first_name = 'John'
)
```

사람은 "X reports to Y"구문에서 "John"이 "employee manager"라는 상식을 유추할 수 있지만, 이는 데이터베이스로부터 알 수 없는 사실이다. 

이러한 문제점 들을 고려하여 정말 사람들에게 유용하게 적용 될 수 있는 Text2SQL 연구가 필요할 것이다.