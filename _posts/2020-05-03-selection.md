---
layout: post
title: "[Algorithms] Selection Sort"
date: "2020-05-02 14:38:38 +0900"
categories: python
author: "Soo"
comments: true
toc: true
---

이전 포스팅: [Bubble Sort & Insertion Sort](https://simonjisu.github.io/python/2020/05/02/bubbleinsertion.html)

# Selection Sort

**선택 정렬(Selection Sort)**의 아이디어는 정말 간단하다. 원소들중 가장 작은 원소를 찾아 첫번째 자리부터 채워넣는 것이다. 마지막 한자리가 남을 때까지 알고리즘은 계속된다. `그림1`과 같이 알고리즘의 각 스텝을 표현할 수 있다. 

{% include image.html id="1kvETAdY2928P8c1-5gBcCaaTcRkDz_ZE" desc="[그림1] Selection Sort" width="75%" height="auto" %}

각 스텝별로 Index **j**가 가르키는 원소를 기준으로 나머지 원소들중 가장 작은 값을 찾아내서 그 값과 교환한다. 코드로 구현하면 다음과 같다.

```python
def selection_sort(l):
    r"""
    Selection Sort
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
    for j in range(n-1):
        smallest_idx = j
        for i in range(j+1, n):
            if l[smallest_idx] >= l[i]:
                smallest_idx = i
        swap(j, smallest_idx)
    return l
```

선택 정렬 또한 간단한 아이디어로써 구현이 쉬운 편이며, 추가 공간이 필요하지 않는 inplace 알고리즘이다.

## 알고리즘 복잡도

```python
n = len(l)                          # 1
for j in range(n-1):                # n-1
    smallest_idx = j                # n-1
    for i in range(j+1, n):         # (n-1) + (n-2) + ... + 1
        if l[smallest_idx] >= l[i]: # 비교: (n-1) + (n-2) + ... + 1
            smallest_idx = i        # (n-1) + (n-2) + ... + 1
    swap(j, smallest_idx)           # 스왑: n-1
```

최악의 경우에, 제일 큰 원소가 제일 앞으로 나와 있고, 나머지 모든 원소가 정렬되어 있을 때를 생각할 수 있다. Index **j**가 순환하면서 나머지 모든 원소와는 한번씩 비교해야하는데, 횟수는 $N-1, N-2 \cdots, 1$이다 index를 찾으면 교환은 한번씩만 하면 되기 때문에 $n-1$번이다. 따라서, 총 $T(N) = c \times \dfrac{N(N-1)}{2} + \alpha$이 된다($c$는 상수). 즉, $T(N) = O(N^2)$ 이다.

# References:

* 본 글은 기본적으로 서울대학교 이재진 교수님의 강의를 듣고 제가 공부한 것을 정리한 글입니다.
