---
title: "All about similarity"
hide:
  - tags
tags:
  - "similarity"
  - "jaccard"
  - "cosine"
  - "euclidean"
  - "levenshtein"
  - "hamming"
---

https://www.youtube.com/playlist?list=PLIUOU7oqGTLhlWpTz4NnuT3FekouIVlqc

유사도 검색(Similarity Search)[^1]은 일반적으로 말해서 광범위한 공간에서 주어진 쿼리(query)에 대해 가장 유사한 객체를 찾는 방법론이라고 할 수 있다. 최근 벡터 데이터베이스에서 많이 사용하고 있는 cosine-similarty가 그 예시 중 하나다. (수학적 정의는 차후에 llm-book 시리즈를 다룰때 다루도록 함). 최근에 RAG(Retrieval Augmented Generation)에서도 유사도 검색을 통해서 쿼리와 적합문 문서를 주입시켜서 환각효과(Hallucination)을 줄이는 효과를 거둘 수 있다고 한다. 이 글에서는 다양한 유사도를 소개하며, 어떤 상황에서 어떤 유사도 기법을 사용해야하는지 알아보자. 

[^1]: [Similarity Search](https://en.wikipedia.org/wiki/Similarity_search)

## Jaccard Similarity

자카드(Jaccard) 유사도[^2]는 두 집합 사이의 유사도를 측정하는 방법이다. 수식으로 표현하면 다음과 같다.

[^2]: [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)

$$J(A, B) = \dfrac{\vert A \cap B \vert}{\vert A \cup B \vert}$$

자카드 유사도는 두 집합의 교집합을 두 집합의 합집합으로 나눈 값이다. 자카드 유사도는 0과 1사이의 값을 가지며, 1에 가까울수록 유사도가 높다고 볼 수 있다. 자카드 유사도는 집합의 크기가 작을 때 유용하게 사용할 수 있으며 고전적인 방법이다. 


??? info "code for Jaccard similarity and Examples"

    === "Jaccard similarity in Python" 

        ```python 
        def jaccard(x: set, y: set):
            shared = x.intersection(y)
            union = x.union(y)
            return len(shared) / len(union)
        ```

    === "Examples" 

        ```python
        import matplotlib.pyplot as plt
        from matplotlib_venn import venn3

        sentence1 = 'The quick brown fox jumps over the lazy dog'.lower()
        sentence2 = 'A fast brown fox leaps over a lazy dog'.lower()
        sentence3 = 'The brown cat walks under the busy dog'.lower()

        set1 = set(sentence1.split())
        set2 = set(sentence2.split())
        set3 = set(sentence3.split())

        print(f'Jaccard similarity for sentences:\ns1: {sentence1}\ns2: {sentence2}')
        print(f'J = {jaccard(set1, set2):.4f}')
        print(f'Jaccard similarity for sentences:\ns1: {sentence1}\ns3: {sentence3}')
        print(f'J = {jaccard(set1, set3):.4f}')
        ```

    === "Figures" 

        ```python
        v = venn3([set1, set2, set3], set_labels=('Sentence 1', 'Sentence 2', 'Sentence3'))

        # S = s_1 \cup s_2 \cup s_3
        s100 = set1-set2-set3  # S \setminus (s_2 \cup s_3)
        s010 = set2-set1-set3  # S \setminus (s_1 \cup s_3)
        s001 = set3-set1-set2  # S \setminus (s_1 \cup s_2)
        s110 = set1&set2-set3  # (s_1 \cap s_2) \setminus s_3
        s101 = set1&set3-set2  # (s_1 \cap s_3) \setminus s_2
        s011 = set2&set3-set1  # (s_2 \cap s_3) \setminus s_1
        s111 = set1&set2&set3  # s_1 \cap s_2 \cap s_3

        for s, s_id in zip([s100, s010, s001, s110, s101, s011, s111], 
                        ['100', '010', '001', '110', '101', '011', '111']):
            v.get_label_by_id(s_id).set_text('\n'.join([str(len(s))]+list(s)))
            v.get_label_by_id(s_id).set_fontsize(9)

        plt.show()
        ```

<figure markdown>
  ![HeadImg](https://drive.google.com/uc?id=1ccJurJEKP7i6ZpFSSGaM4CHiTxUmDicD){ class="skipglightbox" width="100%" }
  <figcaption>Figure: 각 문장 별 집합 S</figcaption>
</figure>

위 예제에서 자카드 유사도를 계산해보자. 전체 집합을 $S = s_1 \cup s_2 \cup s_3$라고 한다.

!!! info "Jaccard similarity for sentences"

    === "Sentence 1 and Sentence 2" 

        $$ J = \dfrac{\vert s_1 \cap s_2 \vert}{\vert s_1 \cup s_2 \vert} = \dfrac{\vert \text{fox, over, lazy, brown, dog} \vert}{\vert \text{quick, jumps, the, } \cdots \text{, fast, leaps, a}\vert} = \dfrac{5}{11} = 0.4545$$

    === "Sentence 1 and Sentence 3" 

        $$ J =  \dfrac{\vert s_1 \cap s_3 \vert}{\vert s_1 \cup s_3 \vert} = \dfrac{\vert \text{the, brown, dog} \vert}{\vert \text{quick, jumps, the, }\cdots \text{, under, busy, cat}\vert} = \dfrac{3}{12} = 0.25$$


그러나 자카드 유사도는 집합의 특성 때문에 빈도수를 반영하지 못한다. 특히 자연어 처리에서 단어의 빈도수가 곧 그 단어의 중요도를 반영하는 경우(TF-IDF)에는 자카드 유사도가 적합하지 않다.

## Levenshtein Distance

tbd
