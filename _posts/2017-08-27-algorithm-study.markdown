---
layout: post
title: "algorithm study"
date: "2017-08-27 15:43:39 +0900"
categories: programming
author: Soo
comments: true
toc: true
---
**알고리즘 공부**
===

**\<Hello Coding 그림으로 개념을 이해하는 알고리즘\>** 책을 요약 정리한 것입니다.

## Chapter 1. 알고리즘의 소개
---
#### 단순탐색 simple search
단순히 순서대로 추측을 하는 것

#### 이진탐색 binary search
<span style="color: #e87d7d">정렬된</span> 원소 리스트를 입력으로 받고, 리스트에 원하는 원소가 있으면 그 원소의 위치를 반환, 아니면 null 반환
<p><br /></p>
#### running time 그리고 빅오 표기법(Big O notation)
알고리즘의 시간은 어떻게 증가하는가로 측정함

* 단순 탐색: <span style="color: #7d7ee8">선형 시간(linear time)</span> 만큼 걸림, $O(n)$

* 이진 탐색: <span style="color: #7d7ee8">로그 시간(logarithmic time)</span> 만큼 걸림, $O(\log_{2}{n})$

**빅오 표기법(Big O notation)**: 연산 횟수를 나타냄

| 많이 사용하는 빅오(빠른순) | 설명 / 예시 |
|:--|:--|
| $O(1)$ | 고정 시간 |
| $O(\log{n})$ | 이진 탐색 |
| $O(n)$ | 단순 탐색 |
| $O(n*\log{n})$ | 퀵 정렬 |
| $O(n^2)$ | 선택 정렬 |
| $O(n!)$ | 외판원 문제 |

<p><br /></p>
<p><br /></p>

## Chapter 2. 선택 정렬
---
#### 배열과 연결 리스트

**리스트**: 원소를 메모리 아무 곳에다 저장해두고, 각 원소에는 목록의 다음 원소에 대한 주소가 저장되어있음, 특정 원소의 위치를 알려면 앞단의 원소의 위치를 알아야함, 그러나 메모리 공간을 예약 요청해서 저장할 필요가 없음

**배열**: 원소들을 메모리에 차례대로 저장함, 특정 원소의 위치를 알기 쉬움 대신 필요한 만큼 미리 일정한 메모리 공간을 요청함, 즉 원소를 추가할 일이 없으면 쓸데 없이 낭비하거나, 추가할 목록이 더 많이 커져 새로 다시 메모리 공간을 요청해야하는 단점이 있음.

**배열**과 **리스트**에서 읽기와 쓰기 연산을 하는 데 걸리는 시간:

|  | 배열 | 리스트 |
|:--:|:--:|:--:|
| 읽기 | $O(1)$ | $O(n)$ |
| 삽입 |  $O(n)$ | $O(1)$ |
| 삭제 |  $O(n)$ | $O(1)$ |

**자료에 접근하는 방식:**

* 임의 접근(random access): 임이의 원소에 접근 가능, ex) 배열
* 순차 접근(sequential access): 원소를 첫 번째 부터 하나씩 읽는 것, ex) 연결리스트

#### 선택 정렬
리스트에에서 모든 항목을 살펴보고 최대값을 찾아서 새로운 리스트에 정렬

걸리는 시간: $O(n^2)$

> * 왜 $O(n^2)$ 시간인가?
>
> 매번 실행할 때마다 점검횟수는 $n-1, n-2, \cdots , 2, 1$ 로 줄어들고, 평균적으로 $n/2$ 만큼 점검한다. 따라서 실제 시간은 평균적으로 $O(n \times 1/2 \times n)$ 인데, 빅오 표기법에서는 상수항은 무시하기 때문에 $O(n \times n)$이 되는 것이다.


<p><br /></p>
<p><br /></p>

## Chapter 3. 재귀 Recursion
---
### 재귀
함수가 자기 자신을 호출하는 것

* 기본 단계(base case): 함수가 자기 자신을 다시 호출하지 않는 경우, 즉 무한 반복으로 빠져들지 않게 하는 부분

* 재귀 단계(recursive case): 함수가 자기 자신을 호출하는 부분

```
def countdown(i):
    print i
    if i <= 1:  # 기본 단계
        return
    else:  # 재귀 단계
        countdown(i-1)
```
### 스택
아래와 같은 자료구조를 **스택**이라고 함

* 푸시(push): 맨 위에 새로운 항목 삽입

* 팝(pop): 맨 위에 항목을 읽기

**호출 스택(call stack)**: 여러 개의 함수를 호출하면서 함수에 사용되는 변수를 저장하는 스택을 호출 스택이라함

```
def fact(x):
    if x == 1:
        return 1
    else:
        return x * fact(x-1)
```
호출 할때 마치 레고블럭을 쌓고 떼듯이 호출함,

예를 들어, factorial 함수를 들자면

| 진행 순서 | 함수 | 설명 | 스택(제일 앞에 것이 최근 스택)|
|:--:|:--|:--:|:--:|
| 1 | fact(3) | fact(3) 호출 | ``fact(3)안의 x = 3`` |
| 2 | if x == 1 | False | |
| 3 | else: |  |  |
| 4 | return x * fact(x-1) | fact(2) 호출 | ``fact(2)안의 x = 2``, ``fact(3)안의 x = 3`` |
| 5 | if x == 1 | False | |
| 6 | else: |  |  |
| 7 | return x * fact(x-1) | fact(1) 호출 | ``fact(1)안의 x = 1``, ``fact(2)안의 x = 2``, ``fact(3)안의 x = 3`` |
| 8 | if x == 1 | True | |
| 9 | return 1 | 아직 함수 호출이 끝난건 아니다. 이제 스택에서 ``fact(1)안의 x=1``를 pop하면 된다. 즉, 반환해야하는 첫 번째 호출인 것이다. 1을 반환| ``fact(1)안의 x = 1``, ``fact(2)안의 x = 2``, ``fact(3)안의 x = 3`` |
| 10 | return x * fact(x-1) | 2 * fact(1) 반환하기 | ``fact(2)안의 x = 2``, ``fact(3)안의 x = 3`` |
| 11 | return x * fact(x-1) | 3 * fact(2) 반환하기 | ``fact(3)안의 x = 3`` |
| 12 | 6 |  |  |

* 단점: 모든 정보를 저장해두어야 하기 때문에 메모리를 많이 소비한다.

## Chapter 4. 분할 정복
---
### 분할 정복(divide-and-conquer)
단계:

1. 가장 간단한 경우로 기본 단계를 찾음.
2. 문제가 기본 단계가 될 때까지 나누거나 작게 만듬.

예제: summation
```
def sum(arr):
    total = 0
    for x in arr:
        total += x
    return total
```
분할 정복 전략으로 재귀 함수를 이요해서 합계를 구하려면?
1. 기본 단계: 간단하게 원소 갯수가 0 혹은 1이면 기본 단계가 됨
2. 재귀 함수 호출: sum([2,3,4]) 가 아닌 2 + sum([3,4]) 로 문제를 줄임, 즉 이 함수는 리스트를 받으면, 리스트가 비어 있을 시 0을 반환하고, 아니면 총합 = 리스트의 첫번째 숫자와 나머지 리스트의 총합을 더한 값으로 출력한다.

```
def newsum(lst):
    summ = 0
    if not lst:
        return 0
    elif len(lst) == 1:  
        return lst[0]
    else:
        summ = lst[0] + newsum(lst[1:])
        return summ

print(newsum([2,3,4]))
```

### 퀵 정렬(quick sort)
기준을 정해서 그것보다 작은 것과 큰 것을 나눠서 재귀함수를 호출하는 방법

```
# ascending
def quicksort(lst):
    if len(lst) < 2:
        return lst
    else:
        pivot = lst[0]
        less = [i for i in lst[1:] if i <= pivot]
        greater = [i for i in lst[1:] if i > pivot]
        return quicksort(less) + [pivot] + quicksort(greater)
```
