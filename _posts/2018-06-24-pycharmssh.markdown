---
layout: post
title: "PyCharm SSH 연결하기"
date: "2018-06-24 18:41:07 +0900"
categories: programming
author: "Soo"
comments: true
---

# PyCharm SSH 연결하기

최근 딥러닝서버를 만들고 나서 Jupyter Notebook 만 사용했다. 그런데 오늘 파이참에서는 학생들을 위해 매 1년 마다 프로 버젼을 제공해주고 있다는 것을 들었다. 이를 사용해 보기로 했다. 

> PyCharm For Student: [https://www.jetbrains.com/student/](https://www.jetbrains.com/student/)

PyCharm Pro를 쓰면 **Run On Remote Server** 기능을 쓸 수 가 있는데, 현재 운영체제인 Mac OS 에서 코드를 실행하면, 서버 Ubuntu 18.04 LTS 에서 실행 된다는 것이다. 게다가 PyCharm의 디버깅 툴도 사용할수 있다.

그러나 이를 실행하는 과정이 쉬운 것은 아니었다. 아래 3개의 과정으로 설명하려고 한다.

1. **Interpreter 설정**
2. **Deployment 설정**
3. **프로젝트 연결 설정**

---

## Interpreter

많은 블로그에서 우선 Interpreter 설정을 진행하라고 해서 나도 따라했다.

<img src="https://dl.dropbox.com/s/4kqy5xwpdz7qe26/0625_interpreter.png">


파이참 시작화면에서 오른쪽 아래 `Configure > Preferences` 혹은 새로운 프로젝트를 만든 뒤, `⌘,`를 누르자. 그러면 위와 같은 화면이 나오는데, `Project Interpreter` 를 선택하자. 

처음에는 `<No interpreter>` 라고 나올텐데, 옆에 `톱니바퀴 > add` 를 눌러주자.

<img src="https://dl.dropbox.com/s/mewttyzbf7btzqs/0625_add_interpreter.png">

> `Host` : IP 
>  
> `Port` : 포트번호
> 
> `Username` : 사용자 이름

위 세가지 사항을 차례대로 입력한다.

<img src="https://dl.dropbox.com/s/r29aktz21vy5e9g/0625_sshkey.png">

> `Private Key file` : SSH 에서 Private key 의 위치를 써준다. 보통은 `~/.ssh` 폴더 안에 있다.
>  
> `Passphrase` : Private Key 의 비밀번호를 넣는다.

이렇게 쉽게 성공했으면 얼마나 좋았을까?

그런데 여기서 아무리 연결을 하려고해도 <span style="color: #e87d7d">Authentication Fail</span> 이라는 빨간 글씨만 보였다.

구글링 결과 나와 같은 오류를 가진 사람들이 올린글이 하나 있었는데([링크](https://bit.ly/2Im44VD)), 요약하면, **"아무리 패스워드를 정확하게 입력해도 연결이 안된다. 내가 뭘 놓치고 있는거냐?"** 라는 글이었다.

나는 아래 댓글의 방안으로 해결 했다.

<img src="https://dl.dropbox.com/s/hu4h1mlmsuaxezh/0625_solution.png">

요약하면, **deployment server** 를 먼저 설정한 후에 하라는 말이었다.

---

## Deployment

`Deployment` 는 `Build, Execution, Deployment` 속성에 있었다. 아무것도 없다면 `+` 눌러서 새로 만들자.

<img src="https://dl.dropbox.com/s/u59u4f4qcte59dv/0625_deployment.png">

> `SFTP` 를 선택한 후
>  
> `SFTP host` : IP
>  
> `Port` : 포트번호
>  
> `Root path` : 루트 패스인데 $HOME 위치를 설정한다.
> 
> `User name` : 사용자 이름
>  
> `Auth type` : 패스워드 혹은 Key Pair(나는 ssh 키를 쓰기 때문에 이것을 선택)
>  
> `Private key` : 키위치
>  
> `Key passphrase` : 키 비밀번호

위 정보를 다 입력하면 `Test SFTP connection` 을 눌러봐서 테스트 해본다. 만약 통과가 되면 `Apply` 를 누르자! 

그리고 `Interpreter` 에 돌아가서 다시 설정해준다. 

접속후 파이썬을 연결 해야하는데, 따로 python을 설치한게 없다면, 기본적으로 2.7 버전인 `/usr/bin/python` 패스가 설정 될 것이다. 

만약에 자신의 서버에서 Python3 을 따로 설치했다면 서버 terminal 에서 아래와 같이 쳐준다.

```
$ which python3
/usr/local/bin/python3
```
이 위치를 복사해서 쓰자. 

모든게 정상적으로 작동하면, **Connection Sucessfully** 를 확인 할 수 가 있다.

아까 `<No interpreter>` 칸에 `remote Python 3.6.5 (sftp://[유저이름]@[IP]:[포트]/[파이썬위치])` 가 뜨면 성공한 것이다.

---

## 프로젝트 연결 설정 확인

자신의 프로젝트와 잘 연결 되었는지 확인 해보는 작업을 한다.

<img src="https://dl.dropbox.com/s/k04w9sa9qplxr1m/0625_openproject.png">

Pycharm에서 새로운 프로젝트를 시작하거나 이미 존재하는 프로젝트를 오픈한다.

<img src="https://dl.dropbox.com/s/h8ac2hnryxs9t23/0625_mappings.png">

첫째, `⌘,` 를 눌러서 `Deployment` 설정에 들어가서 아래의 설정을 해준다.

> `Local Path` : 로컬 PC 의 프로젝트 위치
>  
> `Deployment path on Server ***` : 서버의 프로젝트 위치, 이때 앞단에 설정했던 홈 디렉토리 `Root Path` 를 빼고 설정해줘야 한다. $HOME/프로젝트위치

둘째, 다시 `Project Interpreter` 에 접속해서 서버에 있는 파이썬과 연결 됐는지 확인한다.

셋째, 파일 실행을위해 주 실행파일 `main.py`을 선택 후 (없다면 실행할 파일을 선택), 메뉴에서 `Run > Edit Configuration` 을 선택한다.

<img src="https://dl.dropbox.com/s/gl4kjtjep5qjgjg/0625_remotepython.png">

위에 `+` 버튼을 눌러서 새로운 파이썬 실행파일을 연결하자.

> `Script path`: 스크립트 실행 파일 루트
>  
> `Python Interpreter` : Remote Python 으로 설정 됐는지 확인
>  
> `Environment Variables` : 딥러닝에서 GPU를 쓰려면 환경을 인식해줘야한다. ㅠㅠ 
> 
> 아래 그림과 같이 설정해주자. (그전에 cuda 설치를 못했다면? [링크](https://simonjisu.github.io/datascience/2018/06/03/gpuserver3.html))
> 
> <img src="https://dl.dropbox.com/s/ere9ckvmt23x343/0625_env.png" style="width: 400px;">

## 꿀팁

1. SSH 터미널을 사용하려면 `Tools > Start SSH session...`
2. 로컬에서 변경사항을 자동 업로드 하려면 `Tools > Deployment > Automatic uploads(always)` 