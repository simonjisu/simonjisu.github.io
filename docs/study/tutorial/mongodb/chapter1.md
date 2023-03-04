---
title: "1. Introduction"
hide:
  - tags
tags:
  - mongodb
---

<figure markdown>
  ![HeadImg](https://drive.google.com/uc?export=view&id=1d6DGjVD44lyiJnvoMD3uRFPLwuOQwibW){ class="skipglightbox" width="100%" }
  <figcaption>Reference: MongoDB</figcaption>
</figure>

> MongoDB is a document database designed for ease of application development and scaling.[^1] 

[^1]: [MongoDB - Docs](https://www.mongodb.com/docs/manual/)

MongoDB는 유명한 NoSQL, 도큐먼트 지향(document-oriented) 데이터베이스 시스템 중 하나다. 

## NoSQL

NoSQL이란 Not only SQL의 약자이며, 단순히 비관계형(Non-relational)을 지향하는 것이 아니라 기존의 관계형 DBMS가 가지고 있는 특성에 다른 특성을 부가적으로 지원한다는 것을 의미한다. 기존에 많은 데이터들이 테이블의 형태로 존재했지만, 빅데이터 시대가 되면서 비정형 데이터를 유연하게 저장하고 처리하고 싶어서 이러한 NoSQL 데이터베이스가 각광을 받기 시작했다.

테이블 간의 JOIN 기능 없고 데이터의 스키마와 속성들을 다양하게 정의할 수 있다. 또한 데이터 처리의 완결성을 미보장하지만 폭 넓은 확장성을 제공한다는 특징이 있다. 

## Document-oriented

MongoDB에서는 레코드(record)를 도큐먼트(document)라고 부른다. 도큐먼트는 **필드(field)** 와 **값(value)** 쌍으로 이루어진 데이터 구조를 가지고 있으며, 이는 JSON(JavaScript Object Notation)과 매우 유사하다. 필드는 **키(key)** 라고도 불리우며, 키값은 다른 도큐먼트, 배열 혹은 도큐먼트 배열을 포함할 수도 있다. MongoDB는 JSON Schema[^10]를 확장하여 BSON(binary JSON)를 만들어서 스키마로 사용하고 있다. 덧붙이면 스키마(Schema)는 데이터 오브젝트의 타입, 자료구조 등을 설명하는 명세서라고 생각하면 된다. [^11]

[^10]: [JSON Schema](https://json-schema.org/)
[^11]: [MongoDB - Schemas](https://www.mongodb.com/docs/atlas/app-services/schemas/)

``` javascript title="document object"
{
  _id: ObjectID("35n2lkjald438"),  // (1)
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

1.  :man_raising_hand: `_id` 필드가 primary key 가 된다.

이런 형태는 많은 다른 프로그래밍 언어의 기본 데이터 유형에 해당하여 매우 친숙하다(e.g., Python - `dict`). 그리고 다양한 스키마를 유연하게 정의할 수 있다.

## Collections / Views / On-Demand Materialized Views

콜렉션(collections)은 도큐먼트들(documents)의 그룹이다. RDBMS에서 Table에 해당한다. 뿐만 아니라 standard views와 on-demand materialized view를 제공한다. 둘의 차이점은 다음과 같다.[^2]

* standard views는 view를 읽을때 계산되며, 디스크에 저장되지 않는다.
* on-demand materialized views는 디스트에 저장어 읽는다. `$merge` 혹은 `$out` 스테이지를 사용하여 저장된 데이터를 업데이트 한다.

[^2]: [Mongo-DB - Materialzed Views](https://www.mongodb.com/docs/manual/core/materialized-views/)

## Features

### High Performance

* Embedded Data Modeling 지원하여 I/O(Input/Output) activity을 줄인다. 아래 예제 처럼 하나의 도큐먼트에 담을 수 있는 내용을 일반적인 Normalized Data Modeling에서 읽고 쓰려면 2~3 개의 도큐먼트를 사용해야 한다.

!!! Example

    === "Embedded Data Model"

        ``` javascript title="document"
        {
            _id: <ObjectId1>,
            name: "soopace",
            contact: {  // (1)
                phone: "010-1234-1234",
                email: "simonjisu@gmail.com"
            },
            register: {  // (2)
                class_names: ["Big Data", "Machine Learning"]
            }
        }
        ```

        1.  :man_raising_hand: Embedded `contact` sub-document
        2.  :man_raising_hand: Embedded `register` sub-document
  
    === "Normalized Data Model"

        ``` javascript title="`user` document"
        {
            _id: <ObjectId1>,
            name: "soopace"
        }
        ```

        ``` javascript title="`contact` document"
        {
            _id: <ObjectId2>,
            user_id: <ObjectId1>,  // (1)
            phone: "010-1234-1234",
            email: "simonjisu@gmail.com"
        }
        ```

        1.  :man_raising_hand: foreign key of `user` document

        ``` javascript title="`register` document"
        {
            _id: <ObjectId3>,
            user_id: <ObjectId1>,  // (1)
            class_names: ["Big Data", "Machine Learning"]
        }
        ```

        1.  :man_raising_hand: foreign key of `user` document

* MongoDB는 다양한 Indexes를 지원하여 더 빠르게 검색을 수행할 수 있게 도와준다.[^3] Indexes의 데이터 구조는 B-Tree로 구현되어 있다.

[^3]: [Mongo-DB - Indexes](https://www.mongodb.com/docs/manual/indexes/)

### Query API

CRUD(create, read, update, and delete)[^4] Operation 을 지원한다. 뿐만아니라 Data Aggregation[^5], Text Search[^6] 그리고 Geospatial Queries[^7] 도 수행할 수 있다.

[^4]: [Mongo-DB - CRUD](https://www.mongodb.com/docs/manual/crud/)
[^5]: [Mongo-DB - Aggregation](https://www.mongodb.com/docs/manual/core/aggregation-pipeline/)
[^6]: [Mongo-DB - Text Search](https://www.mongodb.com/docs/manual/text-search/)
[^7]: [Mongo-DB - Geospatial Tutorial](https://www.mongodb.com/docs/manual/tutorial/geospatial-tutorial/)

### High Availability

MongoDB의 레플리케이션(복제, Replication)은 "replica set"[^8]라는 것을 통해서 이루어 진다. 보통 서버, 네트워크 장애가 발생할 때를 대비해 데이터를 중복하여 저장하는 방식을 레플리케이션이라고 한다. 예를 들어 서비스 사용자가 많아져 Database의 부하가 커질 때를 대비해, 복사본에서만 Select작업을 수행하고, 다른 명령어는 기존 Master 노드에서 처리하게 할 수도 있다. Replica Set을 통해서 MongoDB 또한 여타 다른 데이터 베이스와 마찬가지로 자동 장애 조치(automatic failover), 데이터 중복성(data redundancy) 특징을 제공한다.

[^8]: [Mongo-DB - Replication](https://www.mongodb.com/docs/manual/replication/)

### Horizontal Scalability

* 샤딩(Sharding)[^9]을 통해 데이터를 클러스터에 분산시킬 수 있다. 

[^9]: [Mongo-DB - Sharding](https://www.mongodb.com/docs/manual/sharding/)

## 참고하면 좋은 자료

- [오래된 좋은 MongoDB 강좌](https://velopert.com/436)
- [MongoDB란 - 역사, 설계 목표, 핵심 기능, 몽고DB를 사용하는 이유](https://hoing.io/archives/1379)
- [MongoDB Data Modeling](https://hevodata.com/learn/mongodb-data-modeling/)
- [Database의 샤딩(Sharding)이란?](https://nesoy.github.io/articles/2018-05/Database-Shard)