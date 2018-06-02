---
layout: post
title: "개인 딥러닝용 서버 설치 과정기 - 1 사양 및 우분투 서버 설치"
date: "2018-06-02 22:26:58 +0900"
categories: "DataScience"
author: "Soo"
comments: true
---
# Install Ubuntu 18.04 GPU Server For DeepLearning - 1

개인 딥러닝용 서버 설치 과정과 삽질을 담은 글입니다.

## 컴퓨터 사양 상세

|항목|상품코드|제품명|금액|수량|최종금액|
|:-:|:-:|-|:-:|:-:|:-:|
|CPU|399920|[INTEL] 코어7세대 i5-7600 정품박스 (카비레이크/3.5GHz/6MB/쿨러포함)|258,000원|1|258,000원|
|MAIN<br>BOARD|408703|[GIGABYTE] GA-H110M-M.2 듀러블에디션 피씨디렉트 (인텔H110/M-ATX)|71,100원|1|71,100원|
|메모리|390790|[삼성전자] 삼성 DDR4 16GB PC4-19200|183,000원|2|366,000원|
|HDD|347917|[WD] BLUE 2TB WD20EZRZ (3.5HDD/SATA3/5400rpm/64M)|67,730원|1|67,730원|
|GPU|373864|[MSI] GeForce GTX1060 OC D5 6GB 윈드스톰|389,000원|1|389,000원|
|POWER|420859|[CORSAIR] CX750 NEW 80PLUS BRONZE (ATX/750W)|94,370원|1|94,370원|
|BOX|365393|[COX] RC 170T USB3.0 (미들타워)|13,500원|1|13,500원
|OTHER|3877|[컴퓨존] 일반조립비 (하드웨어조립/OS는 설치되지않습니다)|20,000원|1|20,000원|

<ul id="light-slider1">
  <li><img src="/assets/ds/gpuserver/1.jpeg"></li>
  <li><img src="/assets/ds/gpuserver/2.jpeg"></li>
  <li><img src="/assets/ds/gpuserver/3.jpeg"></li>
  <li><img src="/assets/ds/gpuserver/4.jpeg"></li>
  <li><img src="/assets/ds/gpuserver/5.jpeg"></li>
  <li><img src="/assets/ds/gpuserver/6.jpeg"></li>
  <li><img src="/assets/ds/gpuserver/7.jpeg"></li>
  <li><img src="/assets/ds/gpuserver/8.jpeg"></li>
  <li><img src="/assets/ds/gpuserver/9.jpeg"></li>
</ul>


2018년 5월 28일 컴퓨존에서 주문해서, 5월 30일 수요일 도착했다. 총비용은 대략 130만원 정도 ㅎㅎ 언른 GPU를 쓰고 싶은 생각에 그날밤 바로 설치를 진행하였다.

## Ubuntu Server 설치

5월 30일 저녁, 우분투를 설치하려고 하니 버전이 마음에 걸렸다. NVIDIA CUDA TOOLKIT을 보니 리눅스 17.04 버전 까지 지원하는듯 했기 때문이다. 16.04를 설치해야하나? 싶은 찰나에 그냥 최신으로 한번 도전해보기로 했다. 안되면 다시 갈지머..

### Making a bootable Ubuntu USB disk Tutorial at Mac OS

맥에서 부팅 디스크 만들기, 정말 간단하다. [tutorials.ubuntu.com](https://tutorials.ubuntu.com/tutorial/tutorial-create-a-usb-stick-on-macos#0) 튜토리얼을 따라가면 된다.

**준비물**

> * 2GB 이상의 USB Driver
> * Mac OS 컴퓨터
> * 우분투 서버 ISO 다운로드 [우분투 서버 다운로드](https://www.ubuntu.com/download/server)

#### 구동 디스크 FORMAT

1. 응용프로그램 > 유틸리티 > 디스크 유틸리티 선택
2. MAC OS에 꼽은 USB 를 선택한 다음에 `지우기` 를 선택한다.
3. 이름을 짓고, `MS-DOS(FAT)` 선택한다. (그림에는 Scheme가 있지만 Serria 이후에는 없다는 말이 있음)
4. 포맷한다. 지운다.

<img src="/assets/ds/gpuserver/format_disk.png">

#### Etcher 를 사용한 시동 디스크 생성

[Etcher](https://etcher.io/) 먼저 받는다. 그후에는 정말 간단하다.

1. `Select image` 에 다운 받은 `Ubuntu Server 18.04 ISO` 를 고른다.
2. `Select drive` 에 포맷한 디스크를 선택
3. `Flash!`

<img src="/assets/ds/gpuserver/etcher.png">

이제 설치 준비 완료되었다.

### Install Ubuntu Server 18.04

이제 설치릃 해보자. 설치를 하려면, 최소 한번은 모니터에 연결해서 설치해야한다. 나는 정말 서버만을 생각해서 모니터를 않샀기에... HDMI 케이블로 티비화면으로 연결했다... 덕분에 고생이 두배!

사실 간단하다. 아까 구운 `시동 디스크`를 꼽아주고 부팅을 하면 된다.

<ul id="light-slider2">
  <li><img src="/assets/ds/gpuserver/u1.png"></li>
  <li><img src="/assets/ds/gpuserver/u2.png"></li>
  <li><img src="/assets/ds/gpuserver/u3.png"></li>
  <li><img src="/assets/ds/gpuserver/u4.png"></li>
  <li><img src="/assets/ds/gpuserver/u5.png"></li>
  <li><img src="/assets/ds/gpuserver/u6.png"></li>
  <li><img src="/assets/ds/gpuserver/u7.png"></li>
  <li><img src="/assets/ds/gpuserver/u8.png"></li>
  <li><img src="/assets/ds/gpuserver/u9.png"></li>
  <li><img src="/assets/ds/gpuserver/u10.png"></li>
  <li><img src="/assets/ds/gpuserver/u11.png"></li>
  <li><img src="/assets/ds/gpuserver/u12.png"></li>
</ul>

1. 언어선택: 왠만하면 영어로 하자
2. 키보드선택: 왠만하면 영어로 가자
3. `Install Ubuntu` 선택
4. 다음
5. 특별한 주소가 있으면 작성 아니면, 다음
6. 디스크 포맷: 디스크 통째로 포맷한 후에 설치할 것이니 1번
7. 디스크 선택
8. 마지막 확인
9. 정말루?
10. 당신의 이름 / 서버 이름 / 유저이름(로그인용) / 패스워드(로그인용) 등
11. 설치중... 리붓!
12. 완료

다음 장에는 설치후에 내가 했던 작업들: `원격 부팅과 접속` 을 주로 이야기 하겠다.
