---
layout: post
title: "Visual Studio Code Remote Development"
date: "2020-03-24 14:19:38 +0900"
categories: programming
author: "Soo"
comments: true
toc: true
---

# 계기

대학원 수업 중 VMware의 [SAP HANA](https://www.sap.com/korea/products/hana.html)를 사용할 일이 있었는데, VM에 있는 Container CMD가 너무 불편해서 VS Code와 연결하는 작업을 진행했다.

---

# 환경 및 준비물

- OS 환경: Windows 10
- 준비물: 
    - [Visual Stuido Code](https://code.visualstudio.com/)
    - VMware의 가상머신의 Host IP

## VMware란?

[VMware](https://www.vmware.com/kr.html)는 하드웨어 가상 머신 기능을 제공하는 소프트웨어다. 내 컴퓨터의 일부 리소스를 사용하여 내부에 가상의 컴퓨터를 하나 더 만드는 것이다.

여기서는 VMware에 SAP HANA를 설치해서 사용한다(소개링크: [SAP HANA](https://www.sap.com/korea/products/hana.html)). 사실 이게 뭔지 잘 몰라도 된다. 중요한 것은 가상머신과 내 컴퓨터와 통신하는 IP 주소를 알아내는 것이다. 가상머신을 키게되면 다음 그림과 같이 IP 주소(`IP address`)를 보여주는데 이것을 꼭 기억해두자.

{% include image.html id="1etP3nFKJvKEcai35qowatBJWPQgv3g28" desc="가성머신 Power ON 화면" width="100%" height="auto" %}

## VS Code Remote Development란?

쉽게 말해서 외부 컴퓨터 혹은 가상환경(Remote OS)을 현재 내 컴퓨터(Local OS)에서 원격으로 조종하는 것이다.

{% include image.html id="1fpWgA8YuFAj4q94fqCaau0y3LeCALlfZ" desc="Remote Development Package" width="100%" height="auto" %}

이 패키지는 다음 3가지 패키지를 통합한 것이다.

* **Remote-SSH**: Work with source code in any location by opening folders on a remote machine/VM using SSH. Supports x86_64, ARMv7l (AArch32), and ARMv8l (AArch64) glibc-based Linux, Windows 10/Server (1803+), and macOS 10.14+ (Mojave) SSH hosts.
* **Remote-Containers**: Work with a sandboxed toolchain or container based application by opening any folder mounted into or inside a container.
* **Remote-WSL**: Get a Linux-powered development experience from the comfort of Windows by opening any folder in the Windows Subsystem for Linux.

실제로 사용할 것은 `Remote - SSH`다. 이제 Visual Stuido Code 설치를 완료했으면 이제 시작해보자.

---

# 과정

## 1. VS Code에서 Remote Development 설치

* VS Code 키고 Extension으로 가기(단축키: `Ctrl + Shift + X`)
    {% include image.html id="10lbaK-XGUrvSaQbqS1XE6BAnEsWA8_tV" desc="Extension 에서 검색" width="60%" height="auto" %}

* 검색창에 **remote development** 검색후 설치
    {% include image.html id="1Q1HQpUEH3X5sFq29qJ1WGLpwCPofdMjT" desc="1. Remote Development 설치" width="100%" height="auto" %}

## 2. 연결설정 세팅

* 설치가 완료되면 VS Code 좌측 하단에 초록색 버튼이 생기는데 이걸 누른다.
    {% include image.html id="1i5253NbIp0WG6eoFM0m6R-wc_emJkFGp" desc="파란 상태라인 옆 초록색 버튼 누르기" width="75%" height="auto" %}

* Remote로 연결 할 수 있는 Command Palette가 뜬다. 여기서 `Remote-SSH: Open Configuration File...` 를 클릭한다.
    {% include image.html id="1ZWVM6W8RRb9Ge9-ZZeXTAkl2Ual59Nrn" desc="Remote-SSH 선택" width="75%" height="auto" %}

* 원하는 경로에 config 파일 만든다(이미 있으면 해당 파일에 작성).
    {% include image.html id="1RHenESTELTmwV50OU_-K5Zb815v5zNht" desc="Config 경로 선택" width="75%" height="auto" %}

* 다음과 같이 config 파일을 만들고 저장한다.
    {% include image.html id="1n3NJ0phct1qOjNXakXUPdO45eIFEktQM" desc="Remote-SSH 선택" width="75%" height="auto" %}
    * **Host**: 간편하게 지정하는 호스트 이름이다. 만약 CMD 에서 ssh를 이용해 접속하려면 `ssh [Host]` 만 쓰면 밑에 있는 세팅이 자동으로 적용된다.
        ```bash
        $ ssh hxehost  
        # 다음 명령어와 같다.
        $ ssh hxeadm@192.168.153.128:22 
        ```
    * **HostName**: 실제 호스트 이름, 보통 접속하려는 IP 주소거나 도메인 이름이다.
    * **User**: 외부 컴퓨터 혹은 가상환경 로그인 하려는 이름이다.
    * **Port**: 접속하려는 포트, 22번은 SSH(Secure Shell)에 사용되는 기본 포트이며, 만약 개인 서버라면 웬만하면 바꾸는게 좋다(여기서는 VMware에 접속하는 것이기 때문에 그냥 두었다.)

## 3. 연결하기

* 다시 좌측 하단의 초록버튼을 누른 후, 이번에는 `Remote-SSH: Connect to Host...`를 누른다.
* 2번에서 설정한 `hxehost`가 생기고, 이를 누르면 연결을 시작한다. 당연히 VMware의 가상머신(`hxehost`)은 켜둬야한다.
    {% include image.html id="1ipIXZSH3M26yv9GkkNrDaZ_MIo3B5Oaj" desc="Remote-SSH: Connect to Host" width="75%" height="auto" %}

* 연결을 시작하면 파란색 상태라인이 보라색으로 바뀌면서 연결을 시도한다.
    {% include image.html id="1qj7fv6e7CrlnAvaQnLG-cuApKHCnz8xp" desc="가상머신과 연결하기" width="75%" height="auto" %}

* 만약 가상머신에 로그인 password가 있다면 입력하라는 창이 뜬다.
    {% include image.html id="1Dl6Hfop-CL1jUVtrLw8uiEZ4frRmwbEN" desc="비밀번호 입력" width="100%" height="auto" %}

* 연결이 완료되면 상태창에 어떤 호스트와 연결됐는지 뜬다.
    {% include image.html id="1xz0jPstm8SlKOifxazBXgu3nlXAyT7__" desc="연결 완료후 상태라인" width="75%" height="auto" %}

이제 자유롭게 관련 스크립트를 작성하고 파일을 실행할 수 있다! 또한 단축키로 `Ctrl + Shift + ~`를 누르면 가상머신의 CMD를 활용할 수 있는데, 여기서는 복사 붙여넣기가 되서 너무 편하다 ㅎㅎ.
