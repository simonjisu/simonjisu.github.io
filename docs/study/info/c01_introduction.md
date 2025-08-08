---
title: "Chapter 1. Introduction to Information Theory"
hide:
  - tags
tags:
  - information theory
  - theory
  - information
---

!!! info inline "Claude Shannon, 1948"

    The fundamental problem of communication is that of reproducing at one point either exactly or approximately a message selected at another point.

이 책의 첫 절반은 정보 컨텐츠를 어떻게 측정하는지에 대한 내용이며, 데이터를 어떻게 압축하는지 그리고 불완전한 커뮤니케이션 채널에서 어떻게 완벽하게 커뮤니케이션 하는 지를 다룬다.

## 1.1 불완전하고 잡음이 많은 커뮤니케이션 채널에서 완벽한 커뮤니케이션을 달성할 수 있을까?

**커뮤니케이션 채널(communication channel)**이란 **송신사(Sender)**와 **수신자(Receiver)** 사이에서 정보를 전달하는 **물리적 또는 논리적** 경로다. 보통의 채널에는 잡음(Noise)이 있음. 예를 들어, 모뎀을 통한 전화통신, 하드디스크(HDD)에서 데이터를 읽을 때 생기는 오류가 이에 해당한다. 완벽한 커뮤니케이션 과정이란 소스 데이터($s$)가 채널을 통과하여 수신되었을 때 반대편에서 그 데이터를 정확하게 복구하는 프로세스다.

예를 들어, 데이터를 하드 디스크에 입력시 $1-f$의 확률로 제대로 각 비트(bit, 0 혹은 1)를 기록하고, 반대로 $f$의 확률로 잘못 기록한다. 이때 확률은 다음과 같다.

<figure markdown="span">
  ![HeadImg](https://lh3.googleusercontent.com/d/1_yVeByCLoQFce2pSICQAa5BqMahRZY5T){ class="skipglightbox" width="50%"}
  <figcaption>Binary symmetric Channel: 하나의 비트를 송수신 하는 채널. </figcaption>
</figure>

$$ \begin{aligned}
P(y=0 \vert x=0) &= 1 - f & 
P(y=0 \vert x=1) &= f \\
P(y=1 \vert x=0) &= f & 
P(y=1 \vert x=1) &= 1- f
\end{aligned}$$

만약에 확률적으로 10개 중에 하나의 비트가 잘못되는 경우, 즉 $f=0.1$이라고 가정해보자. 유용한 하드 디스크를 만들기 위해서 적어도 $10^{-15}$의 에러 정도만 용납한다. 물리적 솔루션으로는 에러가 더 적은 회로(Circuity)를 사용하거나, 열을 낮추거나 등등이 있을 수 있다. 그러나 이는 커뮤니케이션 채널의 비용을 야기한다.

그렇다면 **'시스템'**적인 솔루션은 무엇인가? **정보 이론(Information Theory)**와 **코딩 이론(Coding Theory)**에서는 물리적 솔루션 보다 다른 접근을 취한다. 주어진 잡음이 많은 환경을 수긍하고 커뮤니케이션 시스템을 추가함으로써 이를 해결하려고 한다.

<figure markdown="span">
  ![HeadImg](https://lh3.googleusercontent.com/d/1eAtNmPENLmN7yEWQsLuCk2lueKZu4MLn){ class="skipglightbox" width="100%"}
  <figcaption>시스템 솔루션</figcaption>
</figure>

위 그림에서 앞 뒤로 **encoder**와 **decoder**를 추가한다. 인코더(Encoder)는 소스 메세지(source message, $s$)를 전송된 메세지(transmitted message, $t$)로 인코딩한다. 이 메세지는 잡음이 있는 채널을 통과하여 전송 받은 메세지($r$)이 되며, 디코더(Decoder)는 이를 다시 해독하여 최종 메세지($\hat{s}$)를 얻는다. 만약 잡음이 완벽하게 없는 채널이라면 $s=\hat{s}$ 일 것이다. 따라서 정보 이론(Information Theory)은 제한적 시스템하에서 에러를 얼마만큼 수정할 수 있는 퍼포먼스를 측정하는 도구이며, 코딩 이론(Coding Theory)은 인코딩과 디코딩 시스템을 어떻게 설계할 것인가에 대한 문제라고 볼 수 있다.

## 1.2 이진 대칭 채널(binary symmetric channel)에서의 오류 정정 코드(Error-correcting codes)

잡음이 있는 채널하에서 제일 간단하게 오류를 체크 할 수 있는 방법은 중복이다. 인코딩할 때 원본을 3개로 복사하고, 전송된 메세지를 디코딩할 때 다수결 투표 방식(majority vote)으로 이들을 확인하는 것이다. 

!!! note "Example"

    $s = \begin{bmatrix} 0 & 0 & 1 & 0 & 1 & 1 & 0\end{bmatrix}$ 인 소스 메세지가 있다. 이를 3개로 복사하여 인코딩하고, 전송된 메세지를 디코딩하는 과정은 아래 표와 같으며, 이진 계산은 이진 연산 혹은 XOR[^1] 연산과 같다.

    [^1]: [XOR](https://ko.wikipedia.org/wiki/%EB%B0%B0%ED%83%80%EC%A0%81_%EB%85%BC%EB%A6%AC%ED%95%A9) 연산은 두 비트가 같으면 0, 다르면 1을 반환한다.

    | $s$ | 0 | 0 | 1 | 0 | 1 | 1 | 0 |
    |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
    | $t$ | 000 | 000 | 111 | 000 | 111 | 111 | 000 | 
    | $n$ | 000 | 001 | 000 | 000 | 101 | 000 | 000 |
    | $r$ | {==000==} | {==00==}1 | {==111==} | {==000==} | {==0==}1{==0==} | {==111==} | {==000==} |
    | $\hat{s}$ | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
    | corrected |   | :material-check: |  |  |  |  |  |
    | undetected |   |  |  |  | :material-check: |  |  |

    위 표에서 $n$은 어떤 랜덤하게 부여된 노이즈 벡터를 나타내며 $r$은 전송된 메세지를 나타낸다. 노란색으로 칠한 부분은 다수결 투표에서 다수를 차지한 숫자다. $t$의 숫자가  0인 경우 $n$이 0이여만 원래 메세지를 보존 할 수 있으며, 1인 경우에는 $n$이 0이여만 원래 메세지를 보존 할 수 있다. 하지만 $n$은 우리가 컨트롤 할 수 없기에 $r$에서 되도록 원본 숫자가 다수가 될 수 있도록 에러를 줄이는 시스템을 만드는 것이 중요하다. 두 번째 숫자는 정상적으로 메세지가 복구 되었지만, 다섯 번째 숫자는 에러를 잡아내지 못했다.

이러한 방식을 통해서 