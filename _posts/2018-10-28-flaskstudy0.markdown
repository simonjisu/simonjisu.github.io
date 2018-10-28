---
layout: post
title: "[Flask] 0 설치하기"
date: "2018-10-28 19:22:40 +0900"
categories: "Flask"
author: "Soo"
comments: true
---

# [Flask] 0 설치하기

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
$ mkdir NMT_demo
$ cd NMT_demo
$ virtualenv nmt
```

가상환경으로 접속한다

```
$ . nmt/bin/activate
```

앞에 괄호에 만든 가상환경이 뜨면, 성공한것이다. 만약에 가상환경을 빠져나오려면 아래와 같이 실행한다.

```
(nmt) $ deactivate
```
 
---

## 필요한 패키지 설치하기

### Flask

```
(nmt) $ pip install Flask
```

### PyTorch

만약에 당신의 서버가 라즈베리파이가 아니라면 그냥 `pip install torch torchvision torchtext` 을 써준다.

하지만 우리의 귀엽고 작은 라즈베리파이는 파이토치를 pip 로 바로 설치를 못한다. "[라즈베리파이에 파이토치 설치하기](https://gist.github.com/fgolemo/b973a3fa1aaa67ac61c480ae8440e754)" 를 확인하고 따라해보자. 우선 필수 패키지를 설치해준다.

```
(nmt) $ sudo apt-get install libopenblas-dev cython libatlas-dev \
m4 libblas-dev python3-dev cmake python3-yaml
(nmt) $ pip install pyyaml  # 설치가 안됐을 수도 있으니까 따로 한번 설치해준다.
```

pytorch 설치전 리눅스 환경변수를 만들어준다. 우리의 작은 라즈베리파이는 GPU를 지원할 CUDA 가 필요 있을 리가 없다. "NO\_CUDA" 변수를 1로 설정한다. "NO\_DISTRIBUTED" 는 뭔지 모르겠다. (알려주세요)

```
(nmt) $ export NO_CUDA=1
(nmt) $ export NO_DISTRIBUTED=1
```

download 폴더를 만들어서 안에 PyTorch 를 clone 한다.

```
(nmt) $ mkdir downloads
(nmt) $ cd downloads
(nmt) $ git clone --recursive https://github.com/pytorch/pytorch
(nmt) $ cd pytorch
```

PyTorch 를 빌드한다. 약 2~3시간 걸린다. [영화](https://ko.wikipedia.org/wiki/%EC%9D%B8%EC%85%89%EC%85%98) 한 편을 보고 오면 딱이다.

```
(nmt) $ python3 setup.py build
```

에러가 없을 경우 아래를 계속 진행한다. "-E" 는 여기서 중요한데, 아까 설정한 환경변수를 포함해서 실행하게 만드는 것이다.

```
(nmt) $ sudo -E python3 setup.py install
```

다음 시간에는 빠르게 앱을 만들어보고 잘 작동하는지 테스트를 해볼 예정이다.

## Reference

* [flask 한글 튜토리얼](https://flask-docs-kr.readthedocs.io/ko/latest/installation.html)
* [리눅스 환경변수 확인하기](http://onecellboy.tistory.com/220)
* [라즈베리파이에 파이토치 설치하기](https://gist.github.com/fgolemo/b973a3fa1aaa67ac61c480ae8440e754)