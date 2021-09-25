---
layout: post
title: "카카오 블라인드 코딩테스트 후기"
date: "2021-09-25 22:00:01 +0900"
categories: others
author: "Soo"
comments: true
toc: true
---

{% include image.html id="1kJshUGkeuqBEeUh8CQ3diK3L7Bqc4yiS" desc="Reference: Pixabay" width="100%" height="auto" %}

지난 18일과 25일 양일간 2022 카카오 블라인드 코딩테스트를 마쳤다. 오늘은 그 후기를 써보고자 한다.

- [2022 KAKAO BLIND RECRUITMENT 링크](https://programmers.co.kr/competitions/1571/2022-kakao-blind-recruitment)

# 1차 테스트

1차 테스트는 여차 다른 코딩테스트와 다른 것이 없었다. 시간은 대략 5시간 정도였고, 작년과 문제는 완전히 다르지만 대략 7문제 정도 출제됐다. 나는 4문제 정도 풀었다. 지금은 문제를 언급하면 문제가 됨으로 나중에 공개되면 하나씩 다시 돌아볼 예정이다. 

코딩 테스트 공부는 나동빈 님의 "[**이것이 취업을 위한 코딩테스트다 with 파이썬**](http://www.yes24.com/Product/Goods/91433923)" 책을 보면서 전체적인 느낌을 훑고, Leetcode 문제를 조금씩 풀어봤다. 책이 정말 많은 도움이 됐는데, 코딩 테스트를 어디서 부터 공부 할 지 모르겠다면 한 번 구매해서 보는 것을 강력히 추천한다. 이 책은 코딩테스트를 어떻게 준비하는지 부터 시작해서 어떤 문제가 있는 지 유형별로 알려준다. 

다만 너무 늦게 시작해서 주요 알고리즘 이론만 공부한게 조금 아쉬웠다. 익숙하지 않으니 변형된 문제에 시간을 많이 소비했다. 내가 느낀 취업에서 코딩 테스트는 기초 체력과 같다. 복싱장에 가면 처음에 줄넘기부터 배워 체력을 키우듯이, 코딩 테스트도 매일 꾸준히 훈련하고 응용해야 수업시간에  배우는 알고리즘이 녹슬지 않는다. 

그래서 새로운 목표를 새웠다. 앞으로 졸업전 남은 기간 동안 적어도 하루에 1시간은 코딩 테스트에 시간을 쏟을 것이다.

- [읽어두면 좋은 알고리즘을 공부하는 방법 관련 글](https://gmlwjd9405.github.io/2018/05/14/how-to-study-algorithms.html)
- [2021년도 1차 코딩 테스트 해설](https://tech.kakao.com/2021/01/25/2021-kakao-recruitment-round-1/)

---

# 2차 테스트

2차 테스트는 CS 테스트와 코딩 테스트가 준비 되어 있었다.

## CS 테스트

CS 테스트는 총 10개의 문제로 20분 정도 주어졌는데, 생각보다 어려웠다. 아무래도 일반 개발자도 아니고, 컴공 기초지식이 부족하다보니 은근히 쉬운 문제도 틀렸다. 비록 지금은 인공지능을 공부하고 있지만, 인공지능은 전체 시스템의 5% 정도만 차지 하고 있다(그림의 논문 참고). 다른 어플리케이션과 융합을 시키려면 적어도 기본적인 지식은 알아둬야 된다고 생각이 드는 하루였다. 이번에 공부하면서 좋은 [repository](https://github.com/JaeYeopHan/Interview_Question_for_Beginner) 하나 발견했는데, 앞으로 하나씩 내걸로 만들어야겠다. 

{% include image.html id="12F8TOfwcA6GRbXprshHzhJeeIBCrrgMs" desc="https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf" width="100%" height="auto" %}

## 코딩 테스트

코딩 테스트는 4시간 45분 동안 REST API를 활용하여 문제를 해결하는 것이다. 지문을 이해하고 구조화 할 줄 알아야 한다. REST API 호출 처리 모듈를 작년의 문제를 참고하여, python으로 Parser 모듈을 만들었다([링크](https://gist.github.com/simonjisu/c63f28d9740f3577a2a51ee2337790b3)).

- [2021 2차 코딩 테스트 해설](https://tech.kakao.com/2021/02/16/2021-kakao-recruitment-round-2/)

```python
import json
import requests

class Parser(object):
    def __init__(self, token: str, base_url: str, verbose=0):
                
        if base_url[-1] == '/':
            base_url = base_url[:-1]
        self.base_url = base_url
        self.content_type = 'application/json'
        self.token = token
        self.verbose = verbose

    def post(self, x, headers=None, data=None):
        if x[0] != '/':
            x = '/' + x
        url = self.base_url + x
        response = requests.post(url, headers=headers, data=data)
        if self.verbose:
            print(response.status_code)
        return response.json()
    
    def get(self, x, headers=None, params=None):
        if x[0] != '/':
            x = '/' + x
        url = self.base_url + x
        response = requests.get(url, headers=headers, params=params)
        if self.verbose:
            print(response.status_code)
        return response.json()
    
    def put(self, x, headers=None, data=None):
        if x[0] != '/':
            x = '/' + x
        url = self.base_url + x
        response = requests.put(url, headers=headers, data=data)
        if self.verbose:
            print(response.status_code)
        return response.json()
```

그럼에도 불구하고 지문을 이해하는데 대략 30분정도 걸렸고, 이를 다시 지문에 맞게 구조화하는데 2시간 가량 걸렸다. 이 부분에서 너무 많은 시간을 쏟아서 뒤에 테스트 해보고 싶은 알고리즘을 구성하는데 시간이 부족했다.

이번의 문제는 매칭 시스템과 관련이 있었는데, 전혀 감을 잡지 못했다. 예전에 자주하던 게임중에 Clash of Clan 이라는 게임이 있었는데, 여기서도 타워레벨과 유저 rating간의 매칭이 불균형하다고 불평이 많았었다. 이런 문제를 커뮤니티에서 보면서 이런 점을 개선 시길 방안은 없을까 생각했었는데, 게임 회사에서는 많이 고민하는 문제 일 것 같다. 다음의 영상을 보면서 영감을 얻어 본다.

- [NDC21-게임기획] 실력점수? 랭킹? 다 비슷한 거 아닌가요: [https://youtu.be/DbRr7X8B-Co](https://youtu.be/DbRr7X8B-Co)