---
title: "Learning MongoDB - 1"
hide:
  - tags
tags:
  - mongodb
---

> MongoDB is a document database designed for ease of application development and scaling.[^1] 

[^1]: MongoDB - Docs: [https://www.mongodb.com/docs/manual/](https://www.mongodb.com/docs/manual/)

MongoDB는 유명한 NoSQL, 도큐먼트 지향(document-oriented) 데이터베이스 시스템중 하나다. 

## NoSQL

NoSQL이란 Not only SQL의 약자이며, 단순히 비관계형(Non-relational)을 지향하는 것이 아니라 기존의 관계형 DBMS가 가지고 있는 특성에 다른 특성을 부가적으로 지원한다는 것을 의미한다. 기존에 많은 데이터들이 테이블의 형태로 존재했지만, 빅데이터 시대가 되면서 비정형 데이터를 유연하게 저장하고 처리하고 싶어서 이러한 NoSQL 데이터베이스가 각광을 받기 시작했다.

테이블 간의 JOIN 기능 없고 데이터의 스키마와 속성들을 다양하게 정의할 수 있다. 또한 데이터 처리의 완결성을 미보장하지만 폭 넓은 확장성을 제공한다는 특징이 있다. 

## Document-oriented

MongoDB에서는 레코드(record)를 도큐먼트(document)라고 부른다. 도큐먼트는 **필드(field)** 와 **값(value)** 쌍으로 이루어진 데이터 구조를 가지고 있으며, 이는 JSON(JavaScript Object Notation)과 매우 유사하다. 필드의 값은 다른 도큐먼트, 배열 혹은 도큐먼트 배열을 포함할 수도 있다.

```javascript
{
  _id: ObjectID("35n2lkjald438"),  // _id 필드가 primary key 가 된다.
  name: "soo",
  age: 34,
  grade: "A+"
  hobby: ["game", "golf"],
  image: {
    url: "https://simonjisu.github.io",
    caption: "", 
    type: "jpg"
  }
}
```

이런 형태는 많은 다른 프로그래밍 언어의 기본 데이터 유형에 해당하여 매우 친숙하다(e.g., Python - `dict`). 그리고 다양한 스키마를 유연하게 정의할 수 있다.

## Collections / Views / On-Demand Materialized Views

모든 도큐먼트(document)는 콜렉션(collections)에 저장된다. RDBMS에서 Table에 해당한다. 뿐만 아니라 standard views와 on-demand materialized view를 제공한다. 둘의 차이점은 다음과 같다.[^2]

* standard views는 view를 읽을때 계산되며, 디스크에 저장되지 않는다.
* on-demand materialized views는 디스트에 저장어 읽는다. `$merge` 혹은 `$out` 스테이지를 사용하여 저장된 데이터를 업데이트 한다.

[^2]: [https://www.mongodb.com/docs/manual/core/materialized-views/](https://www.mongodb.com/docs/manual/core/materialized-views/)

## 참고하면 좋은 자료

- [오래된 좋은 MongoDB 강좌](https://velopert.com/436)
- [MongoDB란 - 역사, 설계 목표, 핵심 기능, 몽고DB를 사용하는 이유](https://hoing.io/archives/1379)