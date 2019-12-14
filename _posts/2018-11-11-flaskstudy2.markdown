---
layout: post
title: "[비전공자의 Flask-2] 본격 앱 만들기 1"
date: "2018-11-11 17:39:48 +0900"
categories: programming
author: "Soo"
comments: true
---

# [비전공자의 Flask-2] 본격 앱 만들기 1

## 폴더 생성부터 데이터베이스 만들기

내가 만드려고 하는 앱은 사용자가 어떤 query 를 날리면 이를 모델을 거쳐서 결과를 웹에서 보여주는 간단한 앱이다. 예를 들어, 번역기 같은 앱의 경우를 생각해보자.

1. 사용자 (웹페이지 방문자)는 번역하고자 하는 글을 쓴고 "번역" 버튼을 클릭한다.
2. 번역 클릭 후, 해당 string이 flask를 통해 전송 받으면 이를 모델에 넣어서 결과를 뱉는다.
3. 결과를 다시 flask 앱을 통해 사용자에게 보여준다. 

구체적으로 [End to End Memory Network](https://simonjisu.github.io/datascience/2017/08/04/E2EMN.html) 모델을 활용해서 스토리 내용을 사용자가 선택해서 질문을 던지면 그에 대한 결과를 받는 앱을 만들 것이다.

## Step 0: 폴더 생성하기

어플리케이션 개발을 시작하기전에, 어플리케이션에서 사용할 폴더를 만들자.

```
/e2eapp
    /static
    /templates
```

앞으로 이 **"nmtapp"** 폴더 안에 우리가 사용할 것들을 넣는다. **"static"** 은 사용자들을 위한 폴더, 이 폴더는 css와 javascript 파일들이 저장되는 곳이다. Flasks는 templates 폴더에서 [Jinja2](http://jinja.pocoo.org/) 템플릿을 찾을 것이다.

## Step 1: 데이터베이스 스키마

데이터베이스 스키마를 생성해야 한다. 우리의 어플리케이션은 단지 하나의 테이블만 필요하며 사용이 매우 쉬운 SQLite를 쓸것이다. 다음의 내용을 schema.sql 이라는 이름의 파일로 방금 생성한 nmtapp 폴더에 저장한다. 

```
drop table if exists nmtmain;
create table nmtmain (
  id integer primary key autoincrement,
   question string not null
);
```

해당 테이블의 이름은 "nmtmain" 이며, 간략하게 칼럼을 소개하자면, 아래와같다.

* id: 자동으로 증가되는 정수이며 프라이머리 키(primary key) 이다.
* query: 사용자들이 입력한 질문.

> 데이터베이스 스키마(database schema)란 데이터베이스에서 자료의 구조, 자료의 표현 방법, 자료 간의 관계를 형식 언어로 정의한 구조이다. 

## Step 2: 어플리케이션 셋업 코드

```
/e2eapp
    /static
    /templates
    /schema.sql
    /e2estart.py
    /settings.py
```

### settings.py

추가로 앱을 실행하기 위한 파일들을 만든다. 중요한 정보 혹은 환경변수가 있는 파일은 **"settings.py"** 에 넣기로 한다.

```
## settings.py
# configuration
DATABASE = '../data/e2e.db'
DEBUG = True
SECRET_KEY = 'development key'
USERNAME = 'admin'
PASSWORD = 'password'
```

클라이언트에서의 세션을 안전하게 보장하기 위해서는 secret\_key 가 필요하다. secret\_key는 추측이 어렵도록 가능한 복잡하게 선택하여야 한다. 디버그(DEBUG) 플래그는 인터랙티브 디버거를 활성화 시키거나 비활성화 시키는 일을 한다. 운영시스템에서는 디버그 모드를 절대로 활성화 시키지 말아야 한다. 왜냐하면 디버그 모드에서는 사용자가 서버의 코드를 실행할수가 있기 때문이다.

### nmtstart.py

앱을 실행하는 **"e2estart.py"** 파일을 만든다.

```
# all the imports
import sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash

# create application
app = Flask(__name__)
app.config.from_pyfile("./settings.py", silent=True)

def connect_db():
    return sqlite3.connect(app.config['DATABASE'])

if __name__ == '__main__':
    app.run(host='0.0.0.0')
```

아직 모르는 import 가 많지만, 차차 배울테니 우선 앱을 만들고 실행하는 부분을 알아보자.

**환경설정 불러오기:** 

```
config.from_pyfile("./settings.py", silent=True)
```
만약에 현재 파일 위치에 환경변수 객체(settings.py 의 내용)를 작성하였으면 `from_object(__name__)` 라는 명령어를 쓰면 된다.

**실행**

```
(venv) /e2eapp $ python3 e2estart.py  
...
192.168.6.34 - - [11/Nov/2018 19:08:40] "GET / HTTP/1.1" 404 -
192.168.6.34 - - [11/Nov/2018 19:08:40] "GET /favicon.ico HTTP/1.1" 404 -
```

지금은 어떤 뷰(view)를 만들지 않았기 때문에, 브라우저에서 페이지를 찾을 수 없다는 404에러를 볼 수 있을 것이다. 이건 나중에 살펴보고, 우선 데이터베이스를 만들고 진행하도록 하자.

## Step 3: 데이터베이스 생성하기

현재 만들고자 하는 앱은 관계형 데이터베이스 시스템에 의해 구동되는 어플리케이션이다. 이러한 시스템은 어떻게 데이터를 저장할지에 대한 정보를 가지고 있는 스키마가 필요하다. 그래서 처음으로 서버를 실행하기 전에 스키마를 생성하는 것이 중요하다.

우선 데이터를 어디다 저장할지 정해보자. 데이터는 앱 밖에 있는 **data** 폴더를 생성해서 저장하기로 한다.

```
/demo
	/venv
	/e2eapp
	/data
```

아까 만들어둔 **"schema.sql"** 파일을 이용하여 sqlite3 명령어를 사용하여 다음과 같이 만들 수 있다. 

```
(venv) /e2eapp $ sqlite3 ../data/e2e.db < schema.sql
```

Sqlite3 가 설치 안됐을 수도 있다. 아래 명령어를 쳐서 (가상환경 빠져나와서) sqlite3 를 설치하자.

```
$ sudo apt-get install sqlite3
```

데이터베이스를 초기화하는 함수를 만들고 싶다면 contextlib 의 closing 함수를 import 한다. 

```
# import 에 추가
from contextlib import closing

def init_db():
    with closing(connect_db()) as db:
        with app.open_resource('schema.sql') as f:
            db.cursor().executescript(f.read())
        db.commit()
```

**함수설명**

* **closing:** with 블럭안에서 연결한 커넥션을 유지하도록 도와준다.
* **app객체의 open_resource:** 리소스 경로(nmtapp 의 폴더)의 파일을 열고 그 값을 읽을 수 있다. 우리는 이것을 이용하여 데이터베이스에 연결하는 스크립트를 실행시킬 것이다.

다음 시간에는 데이터베이스와 연결하고, 뷰함수 및 템플릿을 만들어보자.

## Reference

* [flask 한글 튜토리얼](https://flask-docs-kr.readthedocs.io/ko/latest/installation.html)
* [SQLite로 가볍게 배우는 데이터베이스](https://wikidocs.net/book/1530)
