---
layout: post
title: "Resolving problems in Named Entity Recognition with SpaCy and HuggingFace"
date: "2022-01-02 13:15:01 +0900"
categories: nlp
author: "Soo"
comments: true
toc: true
---

{% include image.html id="1pi0oJIu97-PuTIplqR3agq-5xiTN7XAC" desc="Reference: Pixabay" width="100%" height="auto" %}

# Named Entity Recognition

자연어 처리에서 NER(Named Entity Recognition)는 개체명 인식 문제라고 하며, 주로 문장에서 각 단어에 해당하는 개체명(Named Entity)을 예측하는 문제다. 이는 다양한 곳에서 활용되고 있는데, 대표적으로 경제 기사에서 기업명, 사람 이름, 시간 등을 구별해내는데 유용하다. 

{% include image.html id="1IcebJ8TpV8sSCLpyW-Wwk3-GGlE9DDyO" desc="Reference: a NER Example" width="100%" height="auto" %}

개인 혹은 기업 마다 분류하고자 하는 개체명이 있고 이를 위한 커스텀 모델을 훈련 시키고자 할 것이다. 오늘은 SpaCy패키지와 HuggingFace 패키지를 사용해 자신만의 데이터 세트에서 NER을 수행할 때, 생기는 문제점들과 해결 방안에 대해서 이야기하고자 한다. 

# NER data preparation

NER을 해결하기 위해서 가장 중요한 것은 데이터를 준비하는 것이다. NER의 데이터 형태는 문장과 개체명에 문장에 위치하는 좌표가 필요하다. 다음 예시 처럼 `Thrun`은 사람 이름인 `PERSON`이 개체명이 되며, 그 좌표는 `(5, 20)`가 된다(시작이 0번). 이렇게 각 개체명을 표시하는 과정 자체를 보통 **태깅(tagging)**이라고도 한다.

```python
text = ''.join(
"""
When Thrun started working on self-driving cars at Google in 2007, 
few people outside of the company took him
seriously. "I can tell you very senior CEOs of major American
car companies would shake my hand and turn away because I wasn't 
worth talking to," said Thrun, in an interview with Recode earlier this week.
""".split('\n'))

entities = [
    (5, 10, 'PERSON'), (51, 57, 'ORG'), (61, 65, 'DATE'), (162, 173, 'GPE'), 
    (260, 265, 'PERSON'), (288, 294, 'ORG'), (295, 312, 'DATE')
]
```

이번 글에서는 [SpaCy](https://spacy.io/usage) 와 [HuggingFace](https://huggingface.co/)의 BertTokenizerFast를 활용할 것임으로 아래 피키지들을 설치한다.

```
$ pip install -U spacy transformers
$ python -m spacy download en_core_web_sm
```

[expand]summary:Extraction Entities

아래 코드는 SpaCy 패키지를 사용해서 이미 훈련된 모델에서 Entities를 뽑아내는 과정이다. 만약 다른 개체명을 태킹하고 싶다면 사람이 직접 보고 좌표를 만들어야 한다. NER tagging Tool은 직접 써보진 않았지만, 오픈 소스로 사용할 수 있는 것으로 하나 참고로 링크를 걸어 둔다. [doccano](https://github.com/doccano/doccano).
    
```python
import spacy

spacy_nlp = spacy.load("en_core_web_sm")
doc = spacy_nlp(text)
entities = []
for x in doc:
    if x.ent_type_ != '':
        entities.append((x.idx, x.idx + len(x), x.ent_type_))
```

[/expand]


## Tokenization

자연어 처리에서 문장은 더 작은 단위의 **토큰(Token)**으로 분리된다. 먼저 문장을 토큰 단위로 분리해보자. 문장을 분리하는 과정을 **토큰화(Tokenization)**한다 라고 말한다.

SpaCy는 문장을 `spacy.tokens.doc` 의 [`Doc`](https://spacy.io/api/doc) 클래스로 반환한다. 각 토큰은 `[Token](https://spacy.io/api/token)` 클래스로 감싸져 있으며 유용한 정보들이 속성으로 담겨져 있다. 이를 `str`타입이 담김 `list` 형태로 반환하려면 `str`함수로 다시 감싸주면 된다.

```python
spacy_nlp = spacy.load("en_core_web_sm")
doc = spacy_nlp(text)
for x in doc:
    print(f'토큰: {x} | 문장에서의 시작위치: {x.idx}')
spacy_tokens = list(map(str, doc))
print(spacy_tokens)
# 토큰: When | 문장에서의 시작위치: 0
# 토큰: Thrun | 문장에서의 시작위치: 5
# 토큰: started | 문장에서의 시작위치: 11
# 토큰: working | 문장에서의 시작위치: 19
# 토큰: on | 문장에서의 시작위치: 27
# 토큰: self | 문장에서의 시작위치: 30
# 토큰: - | 문장에서의 시작위치: 34
# ['When', 'Thrun', 'started', 'working', 'on', 'self' ... (생략)
```

BertTokenizerFast는 구글에서 개발한 [SentencePiece](https://github.com/google/sentencepiece)에 기반해 토큰을 분리한다. SentencePiece는 **byte-pair-encoding (BPE)**[[Sennrich et al.](http://www.aclweb.org/anthology/P16-1162)]와 **unigram language model**[[Kudo.](https://arxiv.org/abs/1804.10959)]를 구현하여 만든 패키지다. 대량의 데이터를 기반으로 문자열을 압축하는데, 자세한 내용은 [ratsgo 님의 글](https://ratsgo.github.io/nlpbook/docs/preprocess/bpe/)을 참고하길 바란다. `Thrun`같이 자주 나오지 않는 토큰은 3개의 `T`, `##hr`, `##un`, 더 작은 단위 토큰들로 분리된다. 

```python
from transformers import BertTokenizerFast
# 미리 한습된 토큰화 모듈이 필요하다. 보통 다른 큰 기업들이 한 것을 불러온다.
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
bert_tokens = bert_tokenizer.tokenize(text)

# ['When', 'T', '##hr', '##un', 'started', 'working', 'on', 'self' ... (생략)
```

## BILUO tags / IOB tags

각 개체명에 해당하는 텍스트 하나 혹은 그 이상의 토큰으로 분리될 수도 있기 때문에 시작, 중간, 끝 등의 태그(tag)를 같이 달아줘야한다. 이런 방식을 주로 **IOB scheme**라고 하며, 조금 더 정보가 풍부한 **BILUO scheme**도 있다. 

| 태그 | 설명 |
| --- | --- |
| B | 여러 개의 토큰으로 구성된 entity의 시작 |
| I | 여러 개의 토큰으로 구성된 entity의 중간 |
| L | 여러 개의 토큰으로 구성된 entity의 마지막 |
| U | 단일 토큰으로 구성된 entitiy |
| O | entity가 아닌 토큰 |

이러한 정보를 결합하여 개체명의 레이블인 **태그(tag)**를 만들게 된다. 예를 들어 `Thrun`의 태그는 문장에서 하나밖에 없기 때문에   

SpaCy에는 이와 관련된 유용한 모듈들을 제공한다. `offsets_to_biluo_tags`함수는 개체명의 좌표로 부터 tags를 만든다. 

```python
import spacy
from spacy.training import offsets_to_biluo_tags, biluo_to_iob

def get_tags(spacy_nlp, text, ents, tag_type='iob'):
    """
    IOB SCHEME
    I - Token is inside an entity.
    O - Token is outside an entity.
    B - Token is the beginning of an entity.

    BILUO SCHEME
    B - Token is the beginning of a multi-token entity.
    I - Token is inside a multi-token entity.
    L - Token is the last token of a multi-token entity.
    U - Token is a single-token unit entity.
    O - Token is outside an entity.
    """
    doc = spacy_nlp(text) 
    biluo_tags = offsets_to_biluo_tags(doc, ents)
    if tag_type == 'iob':
        return biluo_to_iob(biluo_tags)
    elif tag_type == 'biluo':
        return biluo_tags
    else:
        raise KeyError(f"tag_type must be either `iob` or `biluo`, your is `{tag_type}`")

biluo_tags = get_tags(spacy_nlp, text, ents=entities, tag_type='biluo')
print(biluo_tags)
# ['O', 'U-PERSON', 'O', ...(생략)... , 'O', 'U-ORG', 'B-DATE', 'I-DATE', 'L-DATE', 'O']
iob_tags = get_tags(spacy_nlp, text, ents=entities, tag_type='iob')
print(iob_tags )
# ['O', 'B-PERSON', 'O', ...(생략)... , 'O', 'B-ORG', 'B-DATE', 'I-DATE', 'I-DATE', 'O']
```

# Problems between SpaCy and BERT

문장과 개체명 데이터가 있을 때, SpaCy의 `offsets_to_biluo_tags`함수로 손쉽게 태그를 얻을 수 있다. 따라서 보통 SpaCy로 태그를 처리하게 된다. 그러나 학습은 보통 딥러닝 모델인 BERT를 활용하는데, 여기서 문제가 발생한다.

NER 학습을 위해서 각 토큰에 해당하는 정답을 알려줘야 한다. 

| BERT 토큰 | 정답 |
| --- | --- |
| When | O |
| T | B-PERSON |
| ##hr | B-PERSON |
| ##un | B-PERSON |
| started | O |

그러나 SpaCy로 얻은 태그는 다음과 같다. BERT에서 추가로 분리된 토큰들을 해당하는 태그로 연결해야 한다.

| SpaCy 토큰 | 태그 |
| --- | --- |
| When | O |
| Thrun | B-PERSON |
| started | O |

## 문제화

이 문제는 다음과 같이 재구성 할 수 있다. 길이가 같거나 다른 두 리스트(`longer_tokens`, `shorter_tokens`)에서 str타입의 토큰이 담겨져 있는데, 길이기 긴 리스트의 토큰은 짧은 토큰의 일부일 수가 있다(`shorter_tokens`의 길이는 `longer_tokens`의 길이보다 클 수는 없다). 분리된 토큰들은 시작을 제외하고, `##`이라는 문자열로 연결될 수도 있고 아닐 수도 있다. 짧은 리스트는 각기 해당하는 태그들이 있는데 해당 태그를 긴 리스트에 알맞게 연결해야 한다.

```python
# 예시: 'Thrun' = 'T' + '##hr', '##un'
longer_tokens = ['When', 'T', '##hr', '##un', 'started']
shorter_tokens = ['When', 'Thrun', 'started']
tags = ['O', 'B-PERSON', 'O']

# 결과물
# ['O', 'B-PERSON', 'B-PERSON', 'B-PERSON', 'O']
```

이 문제를 해결하는 것은 의외로 간단하다. 다음 해설을 함께 보자.

### Token Mappings

```python
from collections import defaultdict

def get_token_mappings(longer_tokens, shorter_tokens):
    # 리스트에 해당하는 두 개의 포인터를 정의 한다.
    # i: shorter_tokens / j: longer_tokens
    i, j = 0, 0
    # 짧은 리스트의 토큰은 여러개의 긴 리스트의 토큰들로 구성되어 있을 수 있다.
    # 따라서 다음과 같은 자료구조를 가진다.
    # {s_tkn_index: [l_tkn_index_1, l_tkn_index_2 ...]}
    token_mappings = defaultdict(list) #{shorter_token: [longer_token]}
    spanned = ''
    while i < len(shorter_tokens) and j < len(longer_tokens):
        s_tkn = shorter_tokens[i]
        l_tkn = longer_tokens[j]
        print(f'| {i}: {s_tkn} | {j}: {l_tkn} | ', end='')
        if s_tkn == l_tkn:  # 토큰이 같으면 두 포인터를 동시에 증가시킨다.
            token_mappings[i].append(j)
            i += 1
            j += 1
            spanned = ''
            print('-')
        else:  # 토큰이 다르면 긴 리스트의 포인터를 증가시킨다.
            token_mappings[i].append(j)
            j += 1
            spanned += l_tkn[2:] if l_tkn.startswith('##') else l_tkn
            print(spanned)
            # 기록된 spanned 문자열이 짧은 리스트의 토큰과 같다면 
            # 짧은 리스트의 포인터를 증가시킨다.
            if spanned == s_tkn:
                i += 1 
                spanned = ''
    return token_mappings

token_mappings = get_token_mappings(
    longer_tokens=bert_tokens, 
    shorter_tokens=spacy_tokens
)

spanned_tags = ['-'] * len(bert_tokens)
for i, t in enumerate(tags):
    for k in token_mappings[i]:
        spanned_tags[k] = t
print(token_mappings)
print(spanned_tags)
# defaultdict(list,
#            {0: [0],
#             1: [1, 2, 3],
#             2: [4] ... (생략)
# 
# ['O', 'B-PERSON', 'B-PERSON', 'B-PERSON', 'O', ... (생략)
```

# 결론

여러 유용한 패키지들을 같이 사용할 때 항상 이런 문제들이 생기는데, 두 패키지 기능을 비슷하게 작동하게 만드는 것이 참 어려운 일이다. 데이터 기반으로 학습된 BERT Tokenizer가 규칙 기반으로 분리되는 SpaCy와 다르게 분리 될 수도 있다. 예) `wasn't` → SpaCy: `was + n't` / BERT: `wasn + ' + t`. 

사실 위 문제는 SpaCy를 쓰지 않고, BERT Tokenizer의 출력에서 ‘offset_mapping’ 기능을 반환하여  biluo 태깅을 만드는 것이 가장 빠른 해결책이 될 수가 있다. 그러나 이런 문제도 겪으면서, 내가 맞닿은 문제를 구조화 하고, 어떻게 해결할지 고민하는 것도 좋은 공부가 되리라 믿는다. 이렇게 작은 부분도 알고리즘 연습의 영역이라고 생각한다.