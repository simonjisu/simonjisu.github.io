---
layout: post
title: "[Algorithms] Bubble Sort & Insertion Sort"
date: "2020-05-02 14:19:38 +0900"
categories: python
author: "Soo"
comments: true
toc: true
---

# Sorting Problem

[**정렬(Sorting) 문제**](https://en.wikipedia.org/wiki/Sorting_algorithm)는 입력 시퀀스 $(x_1, x_2, \cdots, x_n)$를 오름차순의 순열(permutation)으로 만드는 문제를 말한다. 예를 들면 다음과 같이 `input_seq`를 `output_seq`로 바꾸는 형태다. 

```python
>>> input_seq = [5, 7, 2, 3, 1]  # Before Sorting
>>> output_seq = sorted(input_seq)
>>> output_seq  # After Sorting
[1, 2, 3, 5, 7]
```
가장 많이 사용되는 순서의 종류는 숫자 순서(numerical order), [사전 순서(lexicographical order)](https://en.wikipedia.org/wiki/Lexicographical_order) 다.

## Inplace Algorithms

[알고리즘 Introduction 글](https://simonjisu.github.io/python/2020/04/20/algorithmintro.html)에서 알고리즘의 효율을 따지기 위해 시간 이외에 중요한 요소가 **메모리 공간** 이라고 언급했었다. 이와 연관된 용어가 바로 알고리즘의 Inplace 여부다. 

**Inplace Algorithms** 이란, 실행하는데 추가로 공간이 필요하지 않는(혹은 거의 필요하지 않는) 알고리즘을 말한다. 앞으로 다뤄볼 정렬 문제에 사용되는 알고리즘은 대부분은 Inplace 알고리즘들이다.

## Stable or Unstable

이외에도 정렬 문제에서 자주 나오는 용어는 **안정성(stability)**다. Stable 과 Unstable 두 가지로 나누는데, 정렬 후에도 기존 입력 시퀀스의 특성 또한 그 순서을 유지하는 것이 stable sort, 그렇지 않은 것을 unstable sort라고 한다. 

다음 그림 처럼, 숫자 순서로 정렬한 포커 그림을 보자. 정렬 후에도 하트, 스페이드라는 특성이 기존의 "하트5 > 스페이드5" 순서로 유지되는 것이 stable sort, 밑에 그림 처럼 그 순서가 유지되지 않는 것이 unstable sort다.

{% include image.html id="1JAW-0E7H5Dh2C_BMfI6vP9r4q5czRFAf" desc="[출처] Wikipedia: Sorting algorithm" width="50%" height="auto" %}

이러한 특성을 유지하는 정렬 알고리즘이 있고 그렇지 않은 것들이 있다. 어떤 것이 Stable 하고 아닌지는 마지막에 한번에 정리하고, 지금부터 각 알고리즘을 하나씩 알아보기로 한다.

# Bubble Sort

**버블 정렬(Bubble Sort)**은 각 스텝에서 서로 인접한(adjacent) 두 원소를 크기를 비교하여 바르지 않은 순서일 경우 두 원소를 교환(swap)하는 알고리즘이다. `그림1`과 같이 알고리즘의 각 스텝을 표현할 수 있다. 

{% include image.html id="1JoOMFOFarMnqXIUpKmRQDnWbCEkPsfUI" desc="[그림1] Bubble Sort" width="100%" height="auto" %}

각 스텝별로 Index **j**가 가르키는 원소와 **j+1**번째 원소와 비교하여 **j**번째 원소가 더 크면 **j+1**번째 원소와 swap하게 된다. 스텝이 지날수록 **j**가 가질수 있는 최대 크기는 점점 줄어드며, 최대값이 0이 되었을 때 비로소 멈추게 된다. 코드로 구현하면 다음과 같다.

```python
def bubble_sort(l: list):
    r"""
    Bubble Sort
    Args: 
        l: input list
    Return:
        sorted list by ascending
    """
    def swap(p, q):
        r"""swap p-th element and q-th element"""
        t = l[p]
        l[p] = l[q]
        l[q] = t
    n = len(l)
    for i in range(n):  
        for j in range(n-1-i):
            if l[j] > l[j+1]:
                swap(j, j+1)
    return l 
```

버블 정렬의 특징은 구현이 간단하며, 추가 공간이 필요하지 않는 inplace 알고리즘이다.

## 알고리즘 복잡도

```python
n = len(l)                  # 1
for i in range(n):          # n
    for j in range(n-1-i):  # (n-1) + (n-2) + ... + 1
        if l[j] > l[j+1]:   # 비교: (n-1) + (n-2) + ... + 1
            swap(j, j+1)    # 스왑: (n-1) + (n-2) + ... + 1
```

최악의 경우만 생각해보면 모든 원소가 역순으로 정렬되어 있을 때, Index **j**가 순환하면서 모든 원소와 비교와 교환하게 되는데, 각각의 스텝 횟수는 $N-1, N-2 \cdots, 1$라서 총 $T(N) = c \times \dfrac{N(N-1)}{2} + \alpha$이 된다($c$는 상수). 따라서 $T(N) = O(N^2)$ 이다.

# Insertion Sort

**삽입 정렬(Insertion Sort)**은 정렬된 부분과 정렬되지 않는 부분을 따로 두어 정렬되지 않은 부분의 첫 원소부터 차례대로 정렬된 부분으로 넣는 알고리즘이다. 정렬된 부분은 보통 리스트의 첫번째 원소를 택한다. `그림2`과 같이 알고리즘의 각 스텝을 표현할 수 있다. 

{% include image.html id="1uN_BaCpFNpFS1ZD3HtkqigMI9scolVEJ" desc="[그림2] Insertion Sort" width="75%" height="auto" %}

각 스텝별로 Index **j**가 가르키는 원소와 파란색으로 표시된 정렬된 부분의 원소와 비교하여 삽입할 index를 찾는 것이다. 실제로는 삽입할 index를 찾게 될때까지 비교가 완료된 원소를 오른쪽으로 미는 작업(Shift)을 한다. 코드로 구현하면 다음과 같다.

```python
def insertion_sort(l):
    r"""
    Insertion Sort
    Args: 
        l: input list
    Return:
        sorted list by ascending
    """
    n = len(l)
    for j in range(1, n):
        value = l[j]
        i = j
        while i > 0 and value < l[i-1]:
            l[i] = l[i-1]
            i -= 1
        l[i] = value

    return l
```

삽입 정렬도 상대적으로 구현이 간단하며, 추가 공간이 필요하지 않는 inplace 알고리즘이다.

## 알고리즘 복잡도

```python
n = len(l)                          # 1
for j in range(1, n):               # n-1
    value = l[j]                    # n-1
    i = j                           # n-1
    while i > 0 and value < l[i-1]: # 비교: 1 + 2 + ... + (n-1)
        l[i] = l[i-1]               # Shift: 1 + 2 + ... + (n-1)
        i -= 1                      # 1 + 2 + ... + (n-1)
    l[i] = value                    # 삽입: n-1
```

최악의 경우만 생각해보면 모든 원소가 역순으로 정렬되어 있을 때, Index **j**가 순환하면서 앞의 정렬된 부분과 모두 비교하고 Shift하는 횟수는 $1, 2, \cdots, N-1$라서 총 $T(N) = c \dfrac{N(N-1)}{2} + \alpha$이 된다($c$는 상수). 따라서 $T(N) = O(N^2)$ 이다.

# References:

* 본 글은 기본적으로 서울대학교 이재진 교수님의 강의를 듣고 제가 공부한 것을 정리한 글입니다.