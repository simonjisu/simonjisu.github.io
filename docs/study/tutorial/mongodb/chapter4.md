---
title: "4. PyMongo"
hide:
  - tags
tags:
  - mongodb
  - pymongo
---

<figure markdown>
  ![HeadImg](https://drive.google.com/uc?id=1d6DGjVD44lyiJnvoMD3uRFPLwuOQwibW){ class="skipglightbox" width="100%" }
  <figcaption>Reference: MongoDB</figcaption>
</figure>

MongoDB에서는 다양한 언어로 개발을 지원하기위해 여러 드라이버 커넥션을 지원하고 있는데, 이번시간에는 PyMongo를 사용해본다. [^1]

[^1]: [MongoDB - PyMongo](https://www.mongodb.com/docs/drivers/pymongo/)

## Installation and Connection 

터미널에서 `pymongo`를 설치한다.

```bash
$ pip install pymongo
```

이후 다음과 같이 패키지를 불러오고 MongoDB와 연결한다.

```python
import pymongo
import pprint
from pymongo import MongoClient
print(pymongo.__version__)
# 4.3.3

client = MongoClient(host='localhost', port=27017)
# or use uri format
# client = MongoClient('mongodb://localhost:27017/')
try:
    print(client.server_info())
except Exception:
    print("Unable to connect to the server.")
# show current database names
print(client.list_database_names())
```

## CRUD Operation

기존에 mongodb shell에서 사용법이 비슷하다. 데이터베이스 생성, 컬렉션 생성 등 자유롭게 파이썬 문법으로 실행할 수 있다.  

```python
# Select database
db = client['new_db']  # or `db = client.new_db`

# Show current collection names
print('Collection names:', db.list_collection_names())

# Insert a document into the collection
db.test.insert_one({'name': 'test', 'age': 20})

# Check the collection names again
print('Collection names:', db.list_collection_names()) 

# Drop collections
db.test.drop()

# Check the collection names again
print('Collection names:', db.list_collection_names())
```

가짜 데이터를 만들고 Query를 수행해본다. 이름과 나이가 있는 데이터에서 20살 이상인 사람의 이름만 뽑아내보자.

```python
# Let's insert a few more documents
db.test.insert_many([
    {'name': 'joe', 'age': 21},
    {'name': 'sam', 'age': 20},
    {'name': 'john', 'age': 19},
    {'name': 'amy', 'age': 23}
])

# Filter the documents by age > 20 and project the name field
for res in db.test.find(filter={'age': {'$gt': 20}}, projection={'_id': 0, 'name': 1}):
    pprint.pprint(res)
```

## 대량의 데이터로 실습해보기

### Sample Analytics Dataset

이번 시간에는 금융 계정 및 거래 데이터인 Sample Analytics Dataset[^1]로 데이터를 불러오는 방법을 알아본다. 실습 코드는 [여기](https://github.com/simonjisu/bkms2_2023spring/tree/main/00_mongodb)에서 확인할 수 있다.

[^1]: [Sample Analytics Dataset(GitHub)](https://github.com/mcampo2/mongodb-sample-databases/tree/master/sample_analytics)

해당 데이터는 다음과 같이 3개의 컬렉션으로 정의되어 있다.

| Collection Name | Description
|---|---
| `accounts` | Contains details on customer accounts.
| `customers` | Contains details on customers.
| `transactions` | Contains customer transactions.

데이터 스키마 및 샘플 데이터는 다음과 같다. MongoDB Extented JSON[^2] 형태로 구성되어 있는데 JSON, BSON의 확장형태라고 보면 된다. `$` 표기로 되어 있는 필드는 명시적으로 데이터 타입을 표기해주겠다는 뜻이다. `accounts` 의 예시로 보면, `$oid`는 `ObjectId`, `$numberInt`는 `Int32`를 가르킨다.

[^2]: [MongoDB - Extended-json](https://www.mongodb.com/docs/manual/reference/mongodb-extended-json/)


=== "`accounts` sample data"

    ``` json
    {
        "_id": {"$oid": "5ca4bbc7a2dd94ee5816238c" },  // ObjectID
        "account_id": { "$numberInt": "371138" },  // Integer
        "limit": { "$numberInt": "9000" },  // Integer
        "products": [ "Derivatives", "InvestmentStock"]  // Array
    }
    ```

=== "`customers` sample data"

    ``` json
    {
        "_id": { "$oid": "5ca4bbcea2dd94ee58162a68" },  // ObjectID
        "username": "fmiller",  // String
        "name": "Elizabeth Ray",  // String
        "address": "9286 Bethany Glens\nVasqueztown, CO 22939",  // String
        "birthdate":{ "$date": { "$numberLong": "226117231000" } },  // Embedded Document
        "email": "arroyocolton@gmail.com",  // String
        "active": true,  // Boolean
        "accounts":[
            {"$numberInt":"371138"}, {"$numberInt":"324287"}, {"$numberInt":"276528"}, 
            {"$numberInt":"332179"}, {"$numberInt":"422649"}, {"$numberInt":"387979"}
        ], // Array (1)
        "tier_and_details": {
            "0df078f33aa74a2e9696e0520c1a828a": { 
                "tier": "Bronze", 
                "id": "0df078f33aa74a2e9696e0520c1a828a",
                "active": true,
                "benefits": ["sports tickets"]
            },
            "699456451cc24f028d2aa99d7534c219": {
                "tier":"Bronze",
                "benefits": ["24 hour dedicated line", "concierge services"],
                "active":true, 
                "id": "699456451cc24f028d2aa99d7534c219"
            }
        }
    }
    ```

    1.  :man_raising_hand: refer to `account_id` in `accounts`

=== "`transactions` sample data"

    ``` json
    {
        "_id": { "$oid": "5ca4bbc1a2dd94ee58161cb1" },  // ObjectID
        "account_id": { "$numberInt": "443178" },  // Integer (1)
        "transaction_count": { "$numberInt": "66" },  // Integer
        "bucket_start_date": { "$date": { "$numberLong": "-28598400000" } },  // Date
        "bucket_end_date": { "$date": { "$numberLong": "1483401600000" } },  // Date
        "transactions": [
            {
                "date": { "$date": { "$numberLong": "1063065600000" } },  // Date
                "amount": { "$numberInt": "7514" },  // Integer
                "transaction_code": "buy",  // String
                "symbol": "adbe",  // String
                "price": "19.1072802650074180519368383102118968963623046875",
                "total": "143572.1039112657392422534031"
            },  
            // ...
            {
                "date": { "$date": { "$numberLong": "1120694400000" } },
                "amount": { "$numberInt": "2881" },
                "transaction_code":"buy",
                "symbol": "msft",
                "price": "20.6769287918292690164889791049063205718994140625",
                "total": "59570.23184926012403650474880"
            }
        ]
    }
    ```

    1.  :man_raising_hand: refer to `account_id` in `accounts`

### Import data

데이터를 데이터베이스에 불러오자. 현재 데이터는 JSON-Line 형태로 각각 `.json` 확장자에 저장 되어있다. `bson` 패키지에서 `json_util`을 같이 불러와서 데이터 타입 관련 필드(e.g., `$oid`)를 처리해준다.

```python
from pathlib import Path
import json
from bson import json_util

data_path = Path('./datasets/sample_analytics/')
db = client['analytics']  # select database

def jsonl_to_bson(path: str|Path):
    with open(path) as file:
        data = [json.loads(x, object_hook=json_util.object_hook) for x in file.readlines()]
    return data

for file_name in ['accounts.json', 'customers.json', 'transactions.json']:
    collection_name = file_name.split('.')[0]
    collection = db[collection_name]  # select collection
    collection.insert_many(jsonl_to_bson(data_path / file_name))
```

`insert_many` 메서드는 자동으로 batch 데이터를 MongoDB가 받아들일 수 있는 최대의 메세지 크키 만큼의 작은 sub-batch로 쪼개서 데이터 삽입 작업을 수행한다 

### Query data

입력이 완료되면 간단한 쿼리를 날려보자. `fmiller`라는 유저의 정보를 조회해본다. 단, `username`, `active`, `name`, `accounts` 이외에 다른 정보는 반환하지 않기로 한다. 

```python
# search user 'fmiller' in the accounts collection
res = db.customers.find(
    filter={'username': 'fmiller'}, 
    projection={'_id': 0, 'username': 1, 'active': 1, 'name': 1, 'accounts': 1}
)
for doc in res:
    pprint.pprint(doc)
```

결과: 

``` python
{
    'accounts': [371138, 324287, 276528, 332179, 422649, 387979],
    'active': True,
    'name': 'Elizabeth Ray',
    'username': 'fmiller'
}
```

다음으로 조금 어려운 쿼리를 날려보자. `fmiller`라는 유저가 가지고 있는 모든 상품(`accounts.products`)을 알아보자(중복 없이), 단 원래 이름(`accounts.name`)과 상품 정보(`products_type`, 새로 정의)를 반환해야한다. 한 명의 유저는 여러 겨의 계좌를 가질 수 있다. `customers` 컬렉션에서는 유저의 보유 계좌를 배열의 형태로 저장되어 있으며, 각 배열 값은 `accounts.account_id`를 참조하고 있다.

솔루션으로 다음과 같이 `aggregation-pipeline`을 이용하여 수행 할 수 있다. 

``` python
res = db.customers.aggregate([
    {'$match': {'username': 'fmiller'}},  # Pipe1: (1)
    {'$project': {'_id': 0, 'name': 1, 'accounts': 1}},  # Pipe2: (2) 
    {'$lookup': {  # Pipe3: (3) 
        'from': 'accounts',
        'localField': 'accounts',
        'foreignField': 'account_id',
        'as': 'products_type',
    }},
    {'$project': {'products_type': '$products_type.products', 'name': 1}},  # Pipe4: (4)
    {'$addFields': {  # Pipe5: (5)
        'products_type': {
            '$reduce': {
                'input': '$products_type',  # --> $$this
                'initialValue': [],   # --> $$value
                'in': {'$setUnion': ['$$value', '$$this']}
            }
        }
    }},
])

for doc in res:
    pprint.pprint(doc)
```

1.  :man_raising_hand: `$match` 연산자로 `fmiller` 유저를 필터링 한다.
2.  :man_raising_hand: `$project` 연산자로 필요한 정보인 `name` 과 `accounts`를 필터링 한다.
3.  :man_raising_hand: `$lookup` 연산자[^3]로 `accounts` 콜렉션에서 `account_id`를 참조하여 계좌 정보를 가져온다(RDBMS에서 JOIN과 비슷하다). `from`은 불러올 콜렉션, `localField`는 JOIN을 수행 할 필드, `foreignField`는 `from` 콜렉션에서 JOIN할 필드, 그리고 `as`는 이름을 명명한다.
4.  :man_raising_hand: `$project` 연산자로 필요한 정보인 `name` 필터링 하고 `products_type` 필드를 새로 정의한다.
5.  :man_raising_hand: `$addFields` 연산자로 새로운 필드를 생성하는데, `$reduce` 연산자[^4]를 사용하여 MapReduce를 수행한다. 여기서는 `SetUnion` 연산[^5]을 수행하게 된다. 

[^3]: [MongoDB - Lookup](https://www.mongodb.com/docs/manual/reference/operator/aggregation/lookup/)
[^4]: [MongoDB - Reduce](https://www.mongodb.com/docs/manual/reference/operator/aggregation/reduce/)
[^5]: [MongoDB - Set Union](https://www.mongodb.com/docs/manual/reference/operator/aggregation/setUnion/)


결과: 

``` python
{
    'name': 'Elizabeth Ray',
    'products_type': [
        'Brokerage', 'Commodity', 'CurrencyService', 
        'Derivatives', 'InvestmentFund', 'InvestmentStock'
    ]
}
```

마지막으로, `fmiller`의 각 거래 심볼(`symbol`)별 계좌수익을 보려고 한다. `transactions.transaction_code` 가 `'buy'` 인 경우 마이너스, 그렇지 않으면 플러스로 만들어야 하며, `transactions.total` 자체가 `string` 형태로 처음 저장되었기 때문에, 타입변환도 같이 해줘야한다.

```python
res = db.customers.aggregate([
    {'$match': {'username': 'fmiller'}},
    {'$project': {'_id': 0, 'accounts': 1}},
    {'$lookup': {
        'from': 'transactions',
        'localField': 'accounts',
        'foreignField': 'account_id',
        'as': 'transactions_info',
    }},
    {'$unwind': '$transactions_info'},
    {'$project': {
        'transactions': '$transactions_info.transactions'
    }},
    {'$unwind': '$transactions'},
    {'$project': {
        'symbol': '$transactions.symbol',
        'total': {'$cond': [
            {'$eq': ['$transactions.transaction_code', 'buy']}, 
            {'$multiply': [{'$toDouble': '$transactions.total'}, -1]}, 
            {'$toDouble': '$transactions.total'}
            ]
        },
    }},
    {'$group': {
        '_id': '$symbol',
        'total': {'$sum': '$total'}
    }},
])

for doc in res:
    pprint.pprint(doc)
```

물론, 조금 더 쉽게 하는 방법은 각 컬렉션 별로 필요한 조건을 찾아서(`find`) python 코드로 구현하면 그만이지만, 이번 예제에서는 한 번의 `aggergation-pipeline` 으로 만들어 보았다. 

## 참고하면 좋은 자료

- [PyMongo Tutorials](https://pymongo.readthedocs.io/en/stable/tutorial.html)
