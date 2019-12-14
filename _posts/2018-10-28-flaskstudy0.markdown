---
layout: post
title: "[비전공자의 Flask-0] 설치하기"
date: "2018-10-28 19:22:40 +0900"
categories: programming
author: "Soo"
comments: true
---

# [비전공자의 Flask-0] 설치하기

python 으로 딥러닝 공부도 열심히 하지만, 라즈베리파이 서버에 웹 데모 프로그램 만드려고 한다. 나중에 app 만들때 활용할 수 있을 것 같다.

우선 Flask 가 무엇인지는 공식 튜토리얼 문서를 한 번 쓰윽 훑고 오자. 머리말을 읽는 것은 중요하다.

> [공식 튜토리얼 - 머리말](https://flask-docs-kr.readthedocs.io/ko/latest/foreword.html)

읽고 오면 이제 본격적으로 시작해보자.

컴퓨터를 다룰때 제일 짜증나는 부분이 설치다. 내 마음대로 안되는 것도 컴퓨터다. 시행착오도 겪어야 하고 ... 아주 오냐오냐 해줘야 말을 듣기 때문에, 같이 한번 잘 다뤄줘보자.

---

## 가상환경 설치

가상환경으로 작업하는 이유는 구글한테 물어보면 잘 대답해준다. 개인적으로는 **"지우기 편하다"** 가 제일 큰 장점인것 같다.

virtualenv 패키지를 받는다.

```
$ sudo pip install virtualenv
```

자신의 폴더로 들어가서 가상환경을 만든다.

```
$ mkdir demo
$ cd demo
$ virtualenv venv
```

가상환경으로 접속한다

```
$ . venv/bin/activate
```

앞에 괄호에 만든 가상환경이 뜨면, 성공한것이다. 만약에 가상환경을 빠져나오려면 아래와 같이 실행한다.

```
(venv) $ deactivate
```

---

## 필요한 패키지 설치하기

### Flask

```
(venv) $ pip install Flask
```


### PyTorch (11.15 수정)

PyTorch 를 설치하는 이유는 단순히 훈련된 모델을 실행하여 결과값을 얻으려고 하는 것이다. 필요없다면 이 과정을 건너뛰어도 좋다.

설치전 필요한 패키지, `pip list` 를 통해서 없는 패키지라면 설치해주자.

```
future
numpy
pyyaml
setuptools
six
typing
```

만약에 당신의 서버가 라즈베리파이가 아니라면 그냥 `pip install torch torchvision torchtext` 을 써준다.

하지만 우리의 귀엽고 작은 라즈베리파이는 파이토치를 pip 로 바로 설치를 못한다. "[라즈베리파이에 파이토치 설치하기](https://gist.github.com/fgolemo/b973a3fa1aaa67ac61c480ae8440e754)" 를 확인하고 따라해보자. 우선 필수 패키지를 설치해준다.

```
$ sudo apt-get install libopenblas-dev cython libatlas-dev \
m4 libblas-dev python3-dev cmake
# 필수로 먼저 설치해주자
(venv) $ pip install pyyaml numpy 
```

pytorch 설치전 리눅스 환경변수를 만들어준다. `.profile` 밑단에 환경변수를 설정하자. 우리의 작은 라즈베리파이는 GPU를 지원할 CUDA 가 필요 있을 리가 없다. "NO\_CUDA" 변수를 1로 설정한다. "NO\_DISTRIBUTED" 는 뭔지 모르겠다. (알려주세요)


```
$ vi ~/.profile
# 아래 내용을 밑단에 추가한다.
export NO_CUDA=1
export NO_DISTRIBUTED=1

# 설정후, 터미널 재시작한다.
$ source ~/.bashrc
```

download 폴더를 만들어서 안에 PyTorch 를 clone 한다.

```
(venv) $ mkdir downloads && cd downloads
(venv) $ git clone --recursive https://github.com/pytorch/pytorch
(venv) $ cd pytorch
(venv) $ git checkout tags/v0.4.1 -b build
(venv) $ git submodule update --init --recursive
```

PyTorch 를 빌드한다. 한숨 자고 오는게 마음 편하다..

```
(venv) $ python3 setup.py build
```

시간이 오래걸려서 백그라운드로 돌려놓고싶다면 아래와 같이 해라. 단, 터미널을 종료하면 안된다.

```
(venv) $ date && python3 setup.py build && date && python3 setup.py install &> message-build &
(venv) $ 2018. 11. 14. (수) 17:46:27 KST
(venv) $ ...설치내용 주르륵...
```

아래 명령어를 쳐서 date 함수가 잘 출력됐으면, 빌드는 성공적으로 진행된 것이다.

```
(venv) $ tail -l message-build
(venv) $ 2018. 11. 15. (목) 05:54:24 KST
```

거의 열두시간 걸렸다... ㅋㅋ 에러가 없을 경우 아래를 계속 진행한다.

```
(venv) $ python3 setup.py install
```

위에 방법도 좋지만 향후에 재설치를 할수 있게 아래처럼 해준다.

```
(venv) $ export NO_CUDA=1
(venv) $ export NO_DISTRIBUTED=1
(venv) $ pip install wheel
(venv) $ python3 setup.py bdist_wheel
(venv) $ cd dist
(venv) $ pip install [dist 폴더 안에 있는 wheel 파일]
```

<img src="https://dl.dropbox.com/s/0sm9i9ajhp5y5kw/1115_installtorch.png">

해당 wheel 파일을 어딘가 다른 곳에 저장 해두었다가 나중에 설치할 일이 생기면 다시 아래처럼만 하면 된다.

```
(venv) $ pip install [pytorch wheel 파일]
```

설치 종료후 잘 설치가 됐는지, 확인해보자

<img src="https://dl.dropbox.com/s/s5h0s5lc187a2ek/1115_testtorch.png">


다음 시간에는 빠르게 앱을 만들어보고 잘 작동하는지 테스트를 해볼 예정이다.

## Reference

* [flask 한글 튜토리얼](https://flask-docs-kr.readthedocs.io/ko/latest/installation.html)
* [리눅스 환경변수 확인하기](http://onecellboy.tistory.com/220)
* [라즈베리파이에 파이토치 설치하기](https://wormtooth.com/20180617-pytorch-on-raspberrypi/)
