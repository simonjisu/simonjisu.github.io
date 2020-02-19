---
layout: post
title: "Github pages 로 프로젝트 문서화"
date: "2018-12-04 22:04:38 +0900"
categories: programming
author: "Soo"
comments: true
toc: true
---

잘 만든 프로젝트를 jupyter notebook 으로 보여주기에는 난잡해 보일 수가 있다. 프로젝트를 정리하고 싶다면 프로젝트 폴더에 docs 를 만들어서 github가 제공하는 웹호스팅 기능을 이용해서 프로젝트 홈페이지를 만들수 있다. 

* 실습환경: Ubuntu Server 18.04 LTS

---

# 준비과정

## Install nodejs & npm

사용하기 위해서 우선 nodejs 와 노드 패키지 매니저(npm) 를 설치해야한다.

```
$ curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -
$ sudo apt-get install -y nodejs
```

패키지 버전을 체크해보자.

```
$ nodejs --version
v10.14.1

$ npm --version
6.4.1
```

## Install gitbook client

gitbook client 를 설치해야한다.

```
$ npm install gitbook-cli -g
```

## Initialize gitbook

이제 본젹적으로 gitbook 을 사용해보자. 처음 프로젝트를 시작하면서 문서화를 생각하는 것은 좋다. 만약에 이미 작업한 프로젝트가 있다면, 중간에 `posts` 를 만드는 작업만 유심히 살펴봐도 된다. 프로젝트 디렉토리 구조는 아래와 같다.

```
.
\README.md
\SUMMARY.md
\book.json
\notebooks  # 내 프로젝트 쥬피터 노트북
\posts  # 내 프로젝트 관련 포스터
	\chapter-1
		\README.md
		\01_helloworld.md
\docs  # github page 로 가는 html 파일들 등
	\...
		
```

그럼 우선 프로젝트 디렉토리를 만들어보자.

```
$ mkdir [프로젝트 디렉토리] && cd [프로젝트 디렉토리]
```

그리고 `posts` 디렉토리를 만들자. 이 `posts` 안에 있는 모든 내용이 향후에 웹페이지로 갈것이다. 만들고 아래 명령어를 시작해보자.

```
[.] $ gitbook init
Installing GitBook 3.2.3
...
warn: no summary file in this book 
info: create README.md 
info: create SUMMARY.md 
info: initialization is finished 
``` 

```
[.] $ ls
README.md  SUMMARY.md
```

**README.md** 파일은 gitbook 의 첫번째 페이지, **SUMMARY.md** 는 gitbook 의 목차 역할을 한다. `posts` 폴더 안에 첫번째 챕터를 만들어보자.

```
[.] $ mkdir posts && cd posts
[./posts] $ mkdir chapter-1 && cd chapter-1
[./posts/chapter-1] $ vi README.md  # 아무거나 쓰고 저장하자
[./posts/chapter-1] $ vi 01_helloworld.md  # 아무거나 쓰고 저장하자
```

이제 SUMMARY.md 에서 목차를 수정해보자. 

```
[./posts/chapter-1] $ cd ../..
[.] $ vi SUMMARY.md
```

```
# Summary
  
* [Introduction](README.md)
* [Chapter-1](post/chapter-1/README.md)
	* [01 hello world](post/chapter-1/01_helloworld.md)
```

여기까지 왔으면 기본적인 설정은 완료된것이다. `gitbook serve` 명령어를 통해 한번 살펴보자.

```
[.] $ gitbook serve
Live reload server started on port: 35729
Press CTRL+C to quit ...

info: 7 plugins are installed 
info: loading plugin "livereload"... OK 
info: loading plugin "highlight"... OK 
info: loading plugin "search"... OK 
info: loading plugin "lunr"... OK 
info: loading plugin "sharing"... OK 
info: loading plugin "fontsettings"... OK 
info: loading plugin "theme-default"... OK 
info: found 3 pages 
info: found 0 asset files 
info: >> generation finished with success in 0.3s ! 

Starting server ...
Serving book on http://localhost:4000
```

**http://localhost:4000** 로 접속을 시도해보자. 아래와 같은 화면이 나오면 성공이다.

<img src="https://dl.dropbox.com/s/77u9dksoz4tio2x/1204_gitbook.png">

**CTRL+C** 를 눌러서 꺼주자.


## Initialize git

테스트 후, 프로젝트 디렉토리에 `_book` 라는 폴더가 생성됐을 것이다. 이 폴더를 통햇 github 페이지를 만든다. 이 폴더는 나중에 github 저장소에 올릴 필요가 없기 때문에 .gitignore 에 추가해줘야한다. (밑에 한번 지우는 과정을 거치지만 혹시 모르는 상태에 대비해서 작성해준다.)

```
[.] $ vi .gitignore
```

아래 코드를 넣어주자. **node_modules** 는 plugin 에 필요한 패키지들을 설치하는 디렉토리인데 올리지 않는다.

```
# Book build output
_book
# Dependency packages
node_modules
# Jupyter Notebook checkpoint
.ipynb_checkpoints
```

이제 github 에 올려보도록 한다. 우선 자신의 github 에 [프로젝트 디렉토리]와 같은 이름의 github 저장소를 생성하자. 그리고 git 을 사용하기 위해, 다시 돌아와서 해당 프로젝트 디렉토리를 git 저장소로 만들어준다. (애당초에 repository 를 만들어서 clone 하는 상태에서 시작해도 좋다.)

```
[.] $ git init
[.](master) $ git remote add origin git@github.com:[사용자이름]/[프로젝트 디렉토리].git
```

완료되었으면 저장소에 올려보도록 한다. `publish-gitbook.sh` 라는 쉘 스크립트를 만들어준다.

```
[.](master) $ vi publish-gitbook.sh
```

```
#!/bin/bash

# remove gitbook old things
rm -rf _book
rm -rf docs

# gitbook init
gitbook install && gitbook build

# build pages
mkdir docs
cp -R _book/* docs/

# delete things
git clean -fx _book

# upload
git add .
git commit -a -m "update docs"
git push -u origin master
```

## 저장소에서 활성화 하기

자신의 github 저장소의 **Settings** 에 가서 **Github Pages** 항목의 **master branch /docs folder** 를 누르고 **Save** 를 누르자.

<img src="https://dl.dropbox.com/s/n4kcz94j5z77ia4/1204_repo1.png">

<img src="https://dl.dropbox.com/s/4s8rsdyl71ph1hd/1204_repo2.png">

이제 인터넷 주소창에 **https://[사용자이름].github.io/[프로젝트 디렉토리]** 에 접속하면 아까 보았던 gitbook 모습을 볼 수 있다.

---

# Customizing

## book.json

프로젝트 디렉토리에서 `book.json` 파일을 생성하여 커스터마이징이 가능하다. 사실 쓰는건 플러그인과 변수 정도이지만, book.json의 기본 설정을 [여기](https://toolchain.gitbook.com/config.html)서 참고할 수 있다.

## plugins

gitbook 에 다양한 플러그인을 설치 할 수 있는데, [https://plugins.gitbook.com/](https://plugins.gitbook.com/) 사이트에서 확인 할 수 있다.

주로 사용하는것은 수식편집이 가능한 "katex", 방문자 분석을 위한 구글 애널릭틱스 "ga", 댓글을 달수 있는 "disqus" (회원가입 필요함) 정도다.

```
{
	"plugins": [
		"katex",
		"disqus",
		"ga"
		],
	"pluginsConfig": {
        "disqus": {
            "shortName": "XXXXXXX"
        },
        "ga": {
            "token": "UA-XXXX-Y"
        }
}
```

---

# References

* [GitBook Toolchain Documentation for Multi-Languages](https://tinydew4.gitbooks.io/gitbook/ko/structure.html)
* [윈도우에서 깃북 제작 및 깃헙 페이지로 호스팅하기](https://blog.psangwoo.com/coding/2018/01/31/gitbook-on-windows.html)
* [GitBook完整教程](https://book.zhlzzz.com/gitbook/)
* [깃헙 Pages에 깃북 배포하기](https://beomi.github.io/2017/11/20/Deploy-Gitbook-to-Github-Pages/)
* [github plugin 설명(중국어)](https://gitbook.zhangjikai.com/plugins.html)