---
layout: post
title: "[비전공자의 Flask-1] 첫 앱 만들어보기"
date: "2018-10-30 21:25:57 +0900"
categories: programming
author: "Soo"
comments: true
toc: true
---

설치가 완료 되었으니 빠르게 첫 앱을 만들어보자.

# Hello World 찍기

컴퓨터 책에 보면 꼭 해보라는 문구가 있다. 그 말을 찍어 볼 것이다.

hello.py 파일을 하나 만들어 아래와 같이 작성하자.

```
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run()
```

hello.py 를 실행하자.

```
(venv) $ python3 hello.py
 * Serving Flask app "hello" (lazy loading)
 * Environment: production
   WARNING: Do not use the development server in a production environment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

일단 **WARNING** 이 뜨는데 어떻게 해결하는지는 나중에 살펴보고 통신이 어떻게 되는지 살펴보자.

특별히 `app.run()` 에 host 를 지정하지 않으면 **http://127.0.0.1:5000/** 내부 주소(localhost) 로 앱이 실행이 된다. 

문제는 라즈베리파이 내부에 인터넷 익스플로어 같은 어플리케이션이(혹은 글쓴이가 찾지를 못한거 일수도, 설령 찾았다 해도 모니터가 없어서 볼수가 없음) 없기 때문에 내부 주소를 통해 해당 페이지에 접속하지 말고 외부 주소로 접속해서 결과를 봐야한다. 

우선 app.run 부분을 아래와 같이 바꿔주고 다시 실행해보자.

```
app.run(host='0.0.0.0')
```

그리고 추가로 라즈베리파이를 현재 어떻게 접속하고 있는지를 이해해야한다. 

현재 글쓴이는 **노트북**을 통해 라즈베리파이의 리눅스 터미널로 접속하고 있는데 이게 어떻게 진행되는 지 알아보자.

## 포트포워딩을 이해보자

개인별로 사정이 다르지만, 집에서 라즈베리파이 서버를 사용한다고 가정하고 진행하겠다.

우선 자신의 집의 외부로 연결되는 IP(Internet Protocol) 를 알아야 한다. 인터넷에 "[IP 확인](http://www.findip.kr/)" 만 쳐봐도 자신이 접속한 컴퓨터의 IP 를 알수 있다. 보통 해당 고유의 IP 를 통해 집안 곳곳 공유기를 통해 통신한다.

공유기나 내부 네트워크를 사용해서 인터넷에 접속할 경우 사설 IP(Private IP)라고 하는 특정 주소 범위(192.168.0.1 ~ 192.168.255.254)가 내부적으로 사용되고, 공인 IP 주소를 찾기 힘든 경우가 있다.

<img src="https://dl.dropbox.com/s/exbbawgg64w0b75/1030_networkmap.png" height="480" width="520">

비전공자라 용어가 정확하지는 않을 수도 있지만, 위 그림처럼 **외부 > 우리집** 으로 오는 신호는 IP 고, 집 **내부 > 내부** 로 이동하는 신호는 사설 IP 라고 생각 하면 될것이다. 라즈베리파이가 외부에서 받는 신호는 빨간색 포트를 통과하게 된다. 이정도만 이해하고 넘어가자.

어쨋든 공유기 홈페이지에서 포트포워딩 작업을 진행해야한다. 공유기 홈페이지를 접속해서 "**포트포워드**" 라는 단어가 들어간 항목을 찾아가보자. 그리고 자신의 라즈베리 파이가 연결된 **내부 IP 주소** 를 찾아서 포트를 열어주자, 테스트를 위해 5000 번을 포트로 쓴다. 위 그림의 예시로 들자면 아래와 같다. (물론 예시로 든거기 때문에 똑같이 따라하면 안된다.)

> 공유기(벽장) > 거실공유기 : 192.168.54.245, TCP 포트번호(내부/외부)를 5000 으로 설정
>  
> 거실공유기 > 라즈베리파이 : 192.168.0.64, TCP 포트번호(내부/외부)를 5000 으로 설정

위와 같이 설정시, 내 노트북 크롬에 `[외부 IP 주소]:5000` 라고 치면, 해당 통신이 외부신호를 거쳐고, 내부에서 192.168.54.245 > 192.168.0.64 를 거쳐서 라즈베리파이에게 닿게 되고, 아까 실행한 flask app의 결과인 "Hello World!"를 받을 수 있게 된다.

자세한 포트포워딩 설정 방법은 인터넷에 많으니 잘 찾아보시길 바란다. [예시](http://studyforus.tistory.com/35)

---

# 실행하기

자, 설정을 마쳤다면 실행하고 있는 노트북의 인터넷 창에서 `[외부 IP 주소]:5000` 를 쳐보자. 그리고 라즈베리파이 터미널을 확인하면 

```
192.168.6.34 - - [30/Oct/2018 21:23:18] "GET / HTTP/1.1" 200 -
```

표시가 뜰텐데, **"정상적으로 신호를 주고 받아서 'Hello World!' 를 보냈어!"** 라는 뜻이다.

인터넷 창을 확인해보면 Hello World! 문구가 떠있을 것이다. 야호!

기본 어플리케이션인 "Hello World!" 를 성공시켰으니 충분히 고생했다고 생각한다. 다음 시간에는 본격적으로 튜토리얼을 따라서 시작해보도록 한다.

# Reference

* [flask 한글 튜토리얼](https://flask-docs-kr.readthedocs.io/ko/latest/installation.html)