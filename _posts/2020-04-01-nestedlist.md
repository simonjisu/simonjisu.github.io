---
layout: post
title: "[Algorithms] flatten nested list"
date: "2020-04-01 14:19:38 +0900"
categories: algorithms
author: "Soo"
comments: true
toc: true
---

# Nestsed list란?

**Nested list** 란 list 안에 list 혹은 기타 다른 타입의 원소를 가지는 구조다. 예를 들자면 다음과 같다. 

```python
[1, [2, 3], [[4], [5], [6]]]
```

실제 세상에서 우리가 자주 보는 nested list의 단계(level)는 2단계 정도다. 친숙한 Excel형태의 matrix, 혹은 자연어 처리에서 문장을 토큰으로 나눈 형태가 그 예시다.

```python
# Excel
[[  4, 12,  3,  4], 
 [  4,  3, 65,  3],
 [  3, 33, 22,  1],
 [  2, 11,  8,  2]]

# Processed Natural Language
[["오늘", "아침", "글", "을", "쓴다", "."], 
 ["파이썬", "관련", "글", "을", "작성", "했다", "." ]] 
```

nested list의 모든 원소들을 하나씩 해체하여, 원소가 list인 경우, 그 내부값을 모두 꺼내서 오직 하나의 list안에 담아내는 과정을 **flatten**이라고 한다. 

flatten을 하는 이유는 여러가지가 있다. 자연어 처리를 예로 들자면, 단어의 개수를 파악하고 번호를 부여하기 위해, 유니크한 토큰(token) 혹은 단어(word)들의 집합(set)을 구할 필요가 있다. 코드로 다음과 같이 할 수 있다.

```python
>>> x = [["오늘", "아침", "글", "을", "쓴다", "."], 
         ["파이썬", "관련", "글", "을", "작성", "했다", "." ]] 
>>> flatten = lambda nested_li: [ele for li in nested_li for ele in li]
>>> set(flatten(x))
{'파이썬', '글', '을', '쓴다', '아침', '오늘', '했다', '관련', '작성', '.'}
```

2단계 nested list의 경우 2번의 for문을 사용하면 해결할 수 있다. 그러나 이보다 더 깊은 경우는 어떻게 할까? 앞으로 소개할 Generator를 활용해서 이를 해결한다.

# Iterator & Generator

Python에서 set, list등은 모두 `__iter__()` method를 내장하고 있다. 이를 python 내장함수 `iter`와 함께 사용하면 [**Iterator**](https://docs.python.org/ko/3.7/c-api/iterator.html) 객체를 만들 수 있다. 그리고 `next`를 사용하면 원소를 하나씩 뽑아 낼 수 있다. 

```python
a = {1, 2, 3}
b = [1, 2, 3]
c = {1: "a", 2: "b", 3: "c"}
for x in [a, b, c]:
    print(type(x.__iter__()))

# <class 'set_iterator'>
# <class 'list_iterator'>
# <class 'dict_keyiterator'>

set_iterator = iter(a) # set_iterator = a.__iter__() 와 같다
print(next(set_iterator))
# 1
```

**Generator**는 Iterator를 생성해주는 함수다. 위와 같이 길이가 정해진 일반적인 Iterator와 달리 Generator는 명확한 끝이 없는 Iterator 객체를 만들 수 있다. Generator를 만들기 위해서는 [`yield`](https://docs.python.org/3/reference/expressions.html#yieldexpr) 명령어와 함께 사용하거나, [PEP 289](https://www.python.org/dev/peps/pep-0289/) 에서 정의된 형태의 표현(expression)을 사용하면 된다.

```python
def odd_generator(x):
    """generate number if it is odd, smaller than x"""
    for number in range(x):
        if number % 2 == 0:
            yield number

for i in odd_generator(10):
    print(i)

# same as
x = 10
odd_generator = (number for number in range(x) if number % 2 == 0)
for i in odd_generator:
    print(i)
```

Generator의 시각화된 자세한 과정을 보고 싶으면 [http://www.pythontutor.com](http://www.pythontutor.com/visualize.html#mode=edit)에서 다음 코드를 붙여넣고 실행시켜보자!

Generator를 쓰는 이유는 메모리를 효율적으로 사용할 수 있기 때문에다. 다음 코드를 살펴보면, 리스트는 모든 원소들(1~100)에 해당하는 메모리를 미리 배정하지만, generator는 함수에 접근할 때(`next()`를 호출시) 메모리를 할당한다. 

```python
import sys

# list comprehension
a = [i for i in range(100)]
# generator expression
b = (i for i in range(100))

print("size of a:", sys.getsizeof(a))
# size of a: 912
print("size of b:", sys.getsizeof(b))
# size of b: 120
```

## "yield" & "yield form"

`yield from`은 `from` 뒤에 따라오는 subiterator를 한번 더 `yield`하게 된다. 다음 두 개의 같은 작업을 표현한 예제를 통해 빠르게 이해해보자.  

```python
def normal_generator(x):
    for element in range(x):
        yield element

def from_generator(x): 
    yield from range(x)

# normal generator
print(type(normal_generator(5)))
print(list(normal_generator(5)))
# <class 'generator'>
# [0, 1, 2, 3, 4]

# from generator
print(type(from_generator(5)))
print(list(from_generator(5)))
# <class 'generator'>
# [0, 1, 2, 3, 4]
```

이를 활용하면 리스트 안에 리스트를 원소로 가지는 nested list를 1차원 리스트로 만들 수 있다. 일반적인 list comprehension을 사용하게 되면 2단계 깊이 정도 밖에 1차원으로 만들 수 있지만, 다음과 같이 `yield from`을 이용한 generator를 만든다면 깊은 netsted list도 1차원 리스트롤 만들 수 있다.

```python
def flatten(li):
    for ele in li:
        if isinstance(ele, list):
            yield from flatten(ele)
        else:
            yield ele

x = [[[1], 2], [[[[3]], 4, 5], 6], 7, [[8]], [9], 10]

print(type(flatten(x)))
# <generator object flatten at 0x00000212BF603CC8>
print(list(flatten(x)))
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```
