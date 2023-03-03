---
title: "2. Create Database / Collection / Document"
hide:
  - tags
tags:
  - mongodb
---

<figure markdown>
  ![HeadImg](https://drive.google.com/uc?export=view&id=1d6DGjVD44lyiJnvoMD3uRFPLwuOQwibW){ width="100%" }
  <figcaption>Reference: MongoDB</figcaption>
</figure>

이번 시간에는 데이터베이스, 컬렉션, 도큐먼트를 생성하는 방법을 알아본다.

=== "1. Database"

    ### Create and Switch Database

    현재 데이터베이스 표시

    ```bash
    test> db
    test
    ```

    데이터베이스 사용, 만약에 없으면 첫 레코드가 삽입될 때 데이터베이스가 생성된다.

    ```bash
    test> use visiting
    'switched to db visiting'
    visiting>
    ```

    ### Drop Database

    우선 데이터베이스를 선택하고 삭제해야 한다.

    ```bash
    test> use visiting
    visiting> db.dropDatabase();
    { ok: 1, dropped: 'visiting' }
    ```

=== "2. Collection"

    ### Create Collection

    Collection[^1]을 명시적으로 생성하려면 다음과 같이 한다.

    [^1]: [MongoDB - Collections](https://www.mongodb.com/docs/manual/core/databases-and-collections/#collections)

    ```bash
    > db.createCollection('test1')
    ```

    도큐먼트를 입력함으로써 명시적으로 만들지 않아도 자동으로 생성된다.

    ```bash
    > db.test2.insertOne({ x: 1 })
    ```

    현재 collections 보기

    ```bash
    > show collections
    ```

    ### Drop Collection

    ```bash
    > db.test1.drop()
    ```

=== "3. Document"

    ### Insert Document

    하나 혹은 여러 개의 도큐먼트 삽입할 수 있다.

    ```bash
    > db.visit.insert({"name": "A", "visit": ["Paris", "London"], "age": 33})
    > db.visit.insert([  # (1)
        {"name": "B", "visit": ["London"], "age": 56},
        {"name": "C", "visit": ["Seoul", "London"], "age": 45}
    ])
    ```

    1.  :man_raising_hand: `Shift` + `Enter`로 line break 가능.

    ### Filter Document 

    `find` 메소드를 사용하여 특정 조건으로 필터링 할 수 있다. 

    ```bash
    > db.visit.find()  # (1)
    > db.visit.find( {"name": "A"} )
    > db.visit.find( {"age": { $gt: 40 } } )
    > db.visit.find( {"visit": { $in: ["Seoul", "Paris"] } } )
    ```

    1.  :man_raising_hand: 전체 데이터 조회, `db.visit.find({})` 와 같다. `.pretty()` 를 붙이면 이쁘게 출력한다.

    두번째 인자에 projection 을 추가하여 특정 필요한 필드만 반환 할 수 있다. `0`을 주면 해당 필드를 제외하고, `1`을 주면 해당 필드만 반환한다. `_id` 필드만 제외하고, 포함과 제외를 동시에 **혼용**할 수 없다. 예를 들어, `"A"` 라는 사람의 레코드를 조회할 때 `visit` 필드만 반환하고 싶다면, `visit` 필드만 `1` 로 두던지, 아니면 다른 두 필드(`name`, `age`)를 `0`으로 두는 것이다.

    ```bash
    > db.visit.find( {"name": "A"}, {"_id": 0, "visit": 1} )
    > db.visit.find( {"name": "A"}, {"_id": 0, "name": 0, "age": 0} )  # (1)    
    ```

    1.  :man_raising_hand: `_id` 필드만 혼용이 가능하다. 예, `db.visit.find( {"name": "A"}, {"_id": 1, "name": 0, "age": 0} )`

    projection에 여러 연산자를 사용하여 조회를 수행할 수 있다. 예를 들어, $slice 연산자를 사용하여 배열의 첫번째 데이터만 조회할 수 있다.

    ```bash
    > db.visit.find( {"name": "A"}, {"_id": 0, "visit": {$slice: 1} } )
    ```

    ### Remove Document

    `remove` 메소드를 사용하여 특정 조건을 포함한 데이터 제거

    ```bash
    > db.visit.remove( {"name": "A"} )
    ```


=== "4. Aggregation"

    ### Aggregate Pipeline

    여러 도큐먼트에서 그룹 값을 뽑아내려고 하는 작업을 집계(aggregation)라고 하며, MongoDB에서는 aggregation pipeline[^2]이 집계 작업을 수행한다.
    
    [^2]: [MongoDB - Aggregation Pipeline](https://www.mongodb.com/docs/manual/core/aggregation-pipeline/#std-label-aggregation-pipeline)

    - 참고: [Practical MongoDB Aggregations](https://www.practical-mongodb-aggregations.com/front-cover.html)

    예제1. 각 사람이 방문한 도시의 개수를 세어보고, 개수가 많은 순서대로 정렬해보기

    ```bash
    > db.visit.aggregate([
        {  # Pipeline 1: (1)
            $project: {
                _id: "$name",
                countVisit: { $size: "$visit" }
            }
        },
        {  # Pipeline 2: (2)
            $sort: { countVisit: -1 }
        }
    ])
    ```

    1.  :man_raising_hand: `_id` 는 `name` 필드를 따르고, `countVisit` 는 `visit` 필드의 길이를 구한다.
    2.  :man_raising_hand: `countVisit` 필드를 기준으로 내림차순 정렬한다. 

    예제2. 도시별로 방문했던 사람의 나이를 합산하기
    
    ```bash
    > db.visit.aggregate([
        {  # Pipeline 1: (1)
            $unwind: "$visit"
        },
        {  # Pipeline 2: (2)
            $group: {
                _id: "$visit",
                totalAge: { $sum: "$age" }
            }
        }
    ])
    ```

    1.  :man_raising_hand: `visit` 필드를 배열로 간주하고, 각 배열의 요소를 도큐먼트로 만든다.
    2.  :man_raising_hand: `visit` 필드를 기준으로 그룹을 만들고, `totalAge` 필드에 `age` 필드의 값을 더한다.
