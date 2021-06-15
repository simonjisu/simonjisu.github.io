---
layout: post
title: "[Algorithms] Comparison Sort"
date: "2020-06-06 11:38:38 +0900"
categories: algorithms
author: "Soo"
comments: true
toc: true
---

# Comparison Sort

지금까지 봐온 정렬 알고리즘들은 원소들간의 어떤 추상적인 비교연산을 통해 순서를 정하기 때문에 [비교 정렬(comparison sort)](https://en.wikipedia.org/wiki/Comparison_sort)이라고 한다. 이번 포스팅에서는 각 정렬방법들의 비교를 해보려고 한다.

## 각 정렬 방법들간 비교

|Sort| Worst $T(N)$| Best $T(N)$| Stability | Inplace|
|--|--|--|--|--|--|
|Bubble| $O(N^2)$| $O(N^2)$| Yes | Yes |
|Insertion| $O(N^2)$| $O(N^2)$| Yes | Yes |
|Selection| $O(N^2)$| $O(N^2)$| No | Yes |
|Merge| $O(N\log N)$| $O(N\log N)$| Yes | No |
|Quick| $O(N^2)$| $O(N\log N)$| No | Yes |

## 각 정렬 방법들의 시간 복잡도 비교

각 정렬 방법들의 시간 복잡도를 비교하기 위해서 다음과 같은 실험을 하였다. 

1. 입력 리스트의 크기 n 은 2000 부터 시작하여 17000 까지 1000 개씩 추가하여, 총 16 개 크기로 진행한다.
2. 각 입력 크기마다 10회 실험을 진행하고, 리스트는 0 ~ 2000 사이의 숫자로 랜덤 샘플링을 하여 구성한다(2000 포함). 최종 수치는 10회 실험의 평균 값으로 결정한다.
3. 각 알고리즘이 실행되는 실제 시간(t로 표기)은 time 패키지로 시작시간과 끝나는 시간의 차이로 측정한다.
4. 입력크기에 따라 차이나 너무 크게 나서 기존 수치과 log로 변환한 수치를 같이 본다. 각 복잡도 수치를 log로 치환하면 `그림1`과 같다.

{% include image.html id="129gdSPce6z6nLokn6uJPZszfBt9Qa4G-" desc="[그림1] Sorting Experiment" width="110%" height="auto" %}

실험진행의 결과는 `그림2`와 같다. selection sort와 insertion sort는 비슷한 시간을 가진다는 것을 알 수있다. 반면 bubble sort는 두 번의 for문을 모두 꼭 돌아야하기 때문에 이들 보다 실행히간이 많이 걸린다.

{% include image.html id="1dPN8TOjaW3wzEEz62i80fVOj0WOjmJQc" desc="[그림2] 모든 알고리즘의 Time Complexity" width="110%" height="auto" %}

merge sort와 quick sort는 비슷한 실행시간을 가진다.

{% include image.html id="1HQsKgKLFKRSrK8hyRzIKtlyAK9gFP3fk" desc="[그림3] merge sort와 quick sort의 Time Complexity" width="110%" height="auto" %}

## Stability 

Stability는 Stable 과 Unstable 두 가지로 나뉘는데, 정렬 후에도 기존 입력 시퀀스의 특성 또한 그 순서을 유지하는 것이 stable sort, 그렇지 않은 것을 unstable sort라고 한다. 다음 `그림4`처럼 카드를 숫자의 순서대로 정렬하려고 한다. 각 카드는 고유의 문양이 같이 있다. 기존의 문양 순서대로 정렬되면 stable sort라고 한다.

{% include image.html id="1fPY3iG4szY1UxiB6D6c_yn4x7aL_yFfr" desc="[그림4] 카드 정렬하기" width="110%" height="auto" %}

각 정렬 알고리즘의 stability를 확인해보면 Bubble, Insertion, Merge는 stable sort고, Selection과 Quick은 unstable sort다.

```python
>>> l = [(5, "colver"), (2, "diamond"), (1, "diamond"), (2, "heart"), (1, "spade"), (5, "spade")]
>>> key = lambda x: x[0] 
>>> bubble_sort_key(l, key=key) 
[(1, 'diamond'), (1, 'spade'), (2, 'diamond'), (2, 'heart'), (5, 'colver'), (5, 'spade')]

>>> insertion_sort_key(l, key=key) 
[(1, 'diamond'), (1, 'spade'), (2, 'diamond'), (2, 'heart'), (5, 'colver'), (5, 'spade')]

>>> selection_sort_key(l, key=key) 
[(1, 'diamond'), (1, 'spade'), (2, 'heart'), (2, 'diamond'), (5, 'colver'), (5, 'spade')]

>>> merge_sort_key(l, key=key) 
[(1, 'diamond'), (1, 'spade'), (2, 'diamond'), (2, 'heart'), (5, 'colver'), (5, 'spade')]

>>> quick_sort_key(l, key=key) 
[(1, 'spade'), (1, 'diamond'), (2, 'diamond'), (2, 'heart'), (5, 'colver'), (5, 'spade')]
```

# References:

* 본 글은 기본적으로 서울대학교 이재진 교수님의 강의를 듣고 제가 공부한 것을 정리한 글입니다.

# 관련 포스팅: 

* [Bubble Sort & Insertion Sort](https://simonjisu.github.io/algorithms/2020/05/02/bubbleinsertion.html)
* [Selection Sort](https://simonjisu.github.io/algorithms/2020/05/02/selection.html)
* [Merge Sort](https://simonjisu.github.io/algorithms/2020/05/03/merge.html)
* [Quick Sort](https://simonjisu.github.io/algorithms/2020/05/04/quick.html)
* (현재글)[Comparsion Sort](https://simonjisu.github.io/algorithms/2020/06/06/comparisonsort.html)