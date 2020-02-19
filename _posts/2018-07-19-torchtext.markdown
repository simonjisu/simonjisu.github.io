---
layout: post
title: "Torchtext Tutorial"
date: "2018-07-19 00:18:29 +0900"
categories: nlp
author: "Soo"
comments: true
toc: true
---

> 튜토리얼 Notebook: [github](https://github.com/simonjisu/pytorch_tutorials/blob/master/00_Basic_Utils/01_TorchText.ipynb), [nbviewer](https://nbviewer.jupyter.org/github/simonjisu/pytorch_tutorials/blob/master/00_Basic_Utils/01_TorchText.ipynb)

자연어 처리에서 전처리시 자주 사용하는 패키지 하나를 소개하려고 한다. 

PyTorch 는 데이터를 불러오는 강력한 [Data Loader](https://pytorch.org/docs/stable/data.html) 라는 유틸이 있는데, TorchText 는 NLP 분야만을 위한 Data Loader 이다.

> **documentation:** [http://torchtext.readthedocs.io/en/latest/index.html](http://torchtext.readthedocs.io/en/latest/index.html)

**설치:**

```
pip install torchtext
``` 

TorchText 는 자연어 처리에서 아래의 과정을 한번에 쉽게 해준다.

* 토크나이징(Tokenization)
* 단어장 생성(Build Vocabulary)
* 토큰의 수치화(Numericalize all tokens)
* 데이터 로더 생성(Create Data Loader)

<br>

# 사용방법

## 1. 필드지정(Create Field)

필드란 텐서로 표현 될 수 있는 텍스트 데이터 타입을 처리한다. 각 토큰을 숫자 인덱으로 맵핑시켜주는 단어장(Vocabulary) 객체가 있다. 또한 토큰화 하는 함수, 전처리 등을 지정할 수 있다.

아래와 같은 문장과 이에 대한 긍정/부정 정도를 분류하는 데이터셋이 있다면,

```
["The Importance of Being Earnest , so thick with wit it plays like a reading from Bartlett 's Familiar Quotations", 
'3']
```

텍스트를 뜻하는 `TEXT`, 해당 문장의 sentiment 를 뜻하는 `LABEL` 필드객체 두 개를 만든다. 

```python
from torchtext.data import Field

TEXT = Field(sequential=True,
             use_vocab=True,
             tokenize=str.split,
             lower=True, 
             batch_first=True)  
LABEL = Field(sequential=False,  
              use_vocab=False,   
              preprocessing = lambda x: int(x),  
              batch_first=True)
```

**Arguments:** 

* **sequential:** TEXT 는 순서가 있는 (sequential) 데이터기 때문에 인자를 True 로 두고, LABEL 데이터는 순서가 필요없기 때문에 False 로 둔다.
* **use_vocab:** 단어장(Vocab) 객체를 사용할지의 여부. 텍스트 데이터있는 TEXT 에만 True 로 인자를 전달한다.
* **tokenize:** 단어의 토크나이징을 맡아줄 함수다. 여기선 "공백"을 기준으로 나누는 함수를 사용했습니다. 한국어의 경우 보통 `konlpy` 의 토크나이징 함수들을 사용한다. 혹은 개인이 만든 함수도 사용할 수 있다.
* **lower:** 소문자 전환 여부 입니다. 보통 True 로 두며, 단어가 많아질수록 나중에 더 많은 차원에 임베딩해야하기 때문에, 왠만하면 영어는 소문자로 만들어준다.
* **batch_first:** 배치를 우선시 하게 되면, tensor 의 크기는 (B, 문장의 최대 길이) 로 만들어진다.
* **preprocessing:** 전처리는 토큰화 후, 수치화하기 전 사이에서 작동한다. 여기서는 Label 데이터가 string 타입이기 때문에 int 타입으로 만들어준다.

더 자세한 것은 문서를 참조하시길 바란다.

## 2. 데이터 세트 만들기(Create Datasets)

데이터 세트는 위에 지정한 필드에 기반하여 데이터를 불러오는 작업을 한다. 보통 Train, Valid, Test 세트가 있으면 `splits` 메서드를 사용해서 아래와 같이 만들어준다.

```python
from torchtext.data import TabularDataset

train_data = TabularDataset.splits(path='./data/',
					train='train_path',
					valid='valid_path',
					test='test_path',
					format='tsv', 
					fields=[('text', TEXT), ('label', LABEL)])
```

만약에 없다면? 아래와 같이 객체에 그냥 넣어준다.

```python
train_data = TabularDataset(path='./data/examples.tsv', 
				format='tsv', 
				fields=[('text', TEXT), ('label', LABEL)])
```

* **fields:** 아까 만들어준 필드는 리스트 형태로 `[('필드이름(임의지정)', 필드객체), ('필드이름(임의지정)', 필드객체)]` 로 넣어준다.

## 3. 단어장 생성(Build vocabulary)

토큰과 Interger index 를 매칭시켜주는 단어장을 생성한다. 단, 기본적으로 `<unk>` 토큰을 0, `<pad>` 토큰을 1 로 만들어준다. 단, 필드지정시, 문장의 시작 토큰(init_token)과, 끝의 토큰(eos_token)을 넣으면 3, 4 번으로 할당된다. 메서드 안에는 생성한 데이터 세트를 넣어준다.

훈련 데이터를 기반으로 단어장을 생성하려면 아래의 명령어를 입력한다.

```python
TEXT.build_vocab(train_data)
```

## 4. 데이터 로더 만들기(Create Data Loader)

마지막으로 배치 사이즈 만큼 데이터를 불러올 데이터 로더를 만든다. 데이터 세트 때와 마찬가지로 데이터 세트가 분리되어 있다면, `splits` 메서드를 사용한다.

```python
from torchtext.data import Iterator

train_loader, valid_loader, test_loader = \
	TabularDataset.splits((train_data, valid_data, test_data), 
				batch_size=3, 
				device=None,  # gpu 사용시 "cuda" 입력
				repeat=False)
```

만약에 없다면? 아래와 같이 `Iterator` 객체에 그냥 넣어준다.

```python
train_loader = Iterator(train_data, 
			batch_size=3, 
			device=None,  # gpu 사용시 "cuda" 입력
			repeat=False)
```

이렇게 하면 매 배치 때마다 최대 길이에 따라 알아서 패딩(padding) 작업도 같이 해준다. 패딩이란 문장의 길이를 같게 만들기 위해서 의미없는 토큰 `<pad>` 를 나머지 길이가 부족한 문장에게 붙여주는 토큰이다. 잘 생각해보면 input 길이를 같게 만들어 주는 과정이다.

## 테스트

```python
for batch in train_loader:
    break
print(batch.text)
print(batch.label)
#=======================================================
#tensor([[   643,    191,      4,     43,   1447,      3,   4384,    485,
#              7,    207,    892,    107,     43,     85,    408,      3,
#            376,     17,      5,   6447,  11035,     37,     98,     43,
#            199,   5859,      2,      1,      1,      1,      1,      1,
#              1,      1,      1],
#        [     3,   4515,     51,    444,      4,   3738,     30,     94,
#            957,   3498,     59,    700,  13967,      6,   2287,   4435,
#              4,    431,     40,      3,   1201,      7,    486,   1134,
#           4120,     59,      5,    166,   1749,    547,      6,   1339,
#            144,  14759,      2],
#        [    29,      7,    195,    568,    192,     63,    229,     60,
#             17,     21,    202,    334,     18,      5,    535,     20,
#              4,     15,    628,    231,     52,      9,    303,    195,
#           6910,      8,  10136,      8,      3,   2204,   4340,      2,
#              1,      1,      1]])
#tensor([ 0,  3,  1])
```

3개의 배치에, 각 토큰에 해당하는 단어의 숫자가 들어가게 되고, 패딩 또한 잘 되었다.

이처럼 거의 5 줄이면 이 모든 과정을 처리해주는 강력크한 도구다.

---

# 만약에 사용하지 않겠다면?

노트북: [github](https://github.com/simonjisu/pytorch_tutorials/blob/master/00_Basic_Utils/01_TorchText.ipynb), [nbviewer](https://nbviewer.jupyter.org/github/simonjisu/pytorch_tutorials/blob/master/00_Basic_Utils/01_TorchText.ipynb) 에 상세하게 나만의 데이터 로더를 커스터마이징 하는 방법 또한 적어 두었다.

1. 순수 파이썬 만 사용한 코드
2. 파이토치의 Custom Dataset 를 활용한 Data Loader 만들기

하지만, 여기서는 소개하지 않겠다. 한 번 TorchText를 사용하게 되면 위 두 가지 방법은 왠만하면 생각도 안날 것이다.

---

# 다양한 데이터 세트

`torchtext.datasets` 안에는 자연어 처리에 많이 사용되는 데이터 세트들이 이미 포함돼있다. 여기서는 소개하지 않겠다.

Documentation 참고: [http://torchtext.readthedocs.io/en/latest/datasets.html#](http://torchtext.readthedocs.io/en/latest/datasets.html#)

* Sentiment Analysis
* Question Classification
* Entailment
* Language Modeling
* Machine Translation