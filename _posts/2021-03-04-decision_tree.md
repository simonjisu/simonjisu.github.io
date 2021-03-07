---
layout: post
title: "Decision Tree"
date: "2021-03-04 00:00:01 +0900"
categories: machinelearning
author: "Soo"
comments: true
toc: true
---

가장 고전적인 머신러닝 모델이지만 정확하게 알고 넘어가야할 것 같아 정리한다.

# References

- [위키백과: 결정 트리 학습법](https://ko.wikipedia.org/wiki/%EA%B2%B0%EC%A0%95_%ED%8A%B8%EB%A6%AC_%ED%95%99%EC%8A%B5%EB%B2%95)
- [ratsgo blog: 의사결정나무(Decision Tree)](https://ratsgo.github.io/machine%20learning/2017/03/26/tree/)
- [라온피플 blog: Decision Tree](https://m.blog.naver.com/laonple/220861527086)
- [Scikit-Learn: Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
- [Scikit-Learn: Post Pruning](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)

# 의사결정 나무(Decision Tree)

의사결정 나무는 분류 및 회귀에 사용되는 전통적인 **비모수(non-parametric)** 지도학습 방법이다. 이 알고리즘의 목적은 입력 데이터 피처(features)로부터 타겟 데이터를 유추할 수 있는 간단한 규칙을 학습하는 것이다. 위키백과에 나온 타이타닉 탑승객 생존 여부를 나태내는 트리로 예를 들면 다음과 같다.

{% include image.html id="1U_uzhLK2KzUYOyxjCLoHcVjzVneLOQBn" desc="출처: 위키백과" width="50%" height="auto" %}

리프 노드(leaf node)가 아닌 노드들(마름모 모양)은 각 입력 데이터 피처에 대한 규칙을 나타내고 있으며, 이 그림에서는 위에서 부터 
1. 성별
2. 나이
3. 탑승한 배우자와 자녀의 수

이다. 반면 리프 노드(생존/사망)는 각 규칙에 이어지는 경로를 따랐을 때, 타겟 데이터의 값이다. 이 그림에서는 
- 생존 확률 = 해당 리프 노드에 속하는 생존 수 / 해당 클래스에 속하는 전체 승객 수
- 리프노드에 해당할 확률 = 사망 혹은 생존 수 / 전체 탑승객 수

을 나타내고 있다.

---

## 학습의 기준: 불확실성

학습의 기준은 불확실성을 나타내는 **엔트로피(entropy)** 혹은 **불순도(impurity)**가 최대로 감소하는 방향으로 진행된다. 이전 단계와 현재 단계의 불확실성의 차이를 **정보획득(information gain, 이하 IG)**이라고 하며, 정보획득이 많은 방향으로 학습을 진행한다고 말 할 수 있다. 

예를 들어 아래 그림처럼, 특정 영역내에 두 색상의 공을 분류하는 문제가 있고, 사각형의 수평 혹은 수직 변에서 특정 지점을 기준으로 반으로 나누는 것을 규칙이라고 해보자. 우리의 목적은 10개의 공을 잘 나누는 규칙들을 학습하는 것이며, 빨간공을 1, 초록공을 2로 표기한다.

{% include image.html id="17jk7c6NbX0ckSKhZqebV5M7m-a3bHyeV" desc="영역 A" width="50%" height="auto" %}

이를 나누는 기준 엔트로피는 다음과 같으며 특정 영역 $A$의 불확실성을 나타낸다. 여기서 $p_k$는 특정 k 클래스에 속할 확률을 나타낸다($p_k$ = `k-class 공의 개수` / `전체 공의 개수`).

$$Entropy(A)= - \sum_{k=1}^{K} p_k \log_{2}(p_k)$$

현재 상태의 엔트로피를 계산하면, 다음과 같다. 

$$Entropy(A)= - \Big( \dfrac{6}{10}\log_{2}(\dfrac{6}{10}) + \dfrac{4}{10}\log_{2}(\dfrac{4}{10}) \Big) = 0.9710$$

```python
import numpy as np

def calculate_entropy(pks):
    log = np.log2(pks)
    return - (pks * log).sum()

pks = np.array([6/10, 4/10])  # A: p_red, p_greem
entropy = calculate_entropy(pks)
print(f"Entropy is {entropy:.4f}")
# Entropy is 0.9710
```

{% include image.html id="133G4Y4HmuHBGTeSisfZw76h3CoXhD-5B" desc="좌우 영역 A1, A2 나눌 경우" width="50%" height="auto" %}

이제 가로변의 임의로 한 곳을 나눠서 두 개의 영역(좌: $A_1$, 우: $A_2$)으로 나눠본다. 이때의 엔트로피는 두 엔트로피의 가중합으로 계산할 수 있으며, 다음과 같다. 

$$\begin{aligned} Entropy(A) &= \sum_{i=1}^{m} \dfrac{1}{m} Entropy(A_m) = \dfrac{1}{2} Entropy(A_1) + \dfrac{1}{2} Entropy(A_2) \\
&= - \sum_{i=1}^{2} \dfrac{1}{2} \sum_{k=1}^K p_k^{(i)} \log_2 (p_k^{(i)}) \\
&= - \dfrac{1}{2} \Big( \dfrac{5}{6}\log_{2}(\dfrac{5}{6}) + \dfrac{1}{6}\log_{2}(\dfrac{1}{6}) \Big) - \dfrac{1}{2} \Big( \dfrac{1}{4}\log_{2}(\dfrac{1}{4}) + \dfrac{3}{4}\log_{2}(\dfrac{3}{4}) \Big) \\
&= 0.7307
\end{aligned}$$

```python
def calculate_entropy(pks):
    if pks.ndim == 1:
        pks = np.expand_dims(pks, 0)
    log = np.log2(pks + 1e-10)
    entropy = - 1/len(pks) * (pks * log).sum()
    return entropy.round(6)

pks = np.array([
    [5/6, 1/6],  # A_1: p_red, p_greem
    [1/4, 3/4]]  # A_2: p_red, p_greem
)
entropy2 = calculate_entropy(pks)
print(f"Entropy is {entropy2:.4f}")
# Entropy is 0.7307
```

이제 우리는 정보획득(IG)을 계산할 수 있게 된다. 이전 단계과 현재 상태의 불확실성 차이인 $IG_1 = 0.9710 - 0.7307 = 0.2403$가 정보획득량이라고 할 수 있다. 

그렇다면 다른 경우에는 어떨까? 예를 들어 가로로 선을 그어 두 영역(상: $A_1$, 하: $A_2$)을 나눠보고 정보획득량을 계산해보자.

{% include image.html id="1VvzOYsLNMEoM2mOtCi9FbHxIo5e-bUVM" desc="상하 영역 A1, A2 나눌 경우" width="50%" height="auto" %}

```python
pks = np.array([
    [4/5, 1/5],  # A_1: p_red, p_greem
    [2/5, 3/5]]  # A_2: p_red, p_greem
)
entropy3 = calculate_entropy(pks)
print(f"Entropy is {entropy3:.4f}")
# Entropy is 0.8464
```

계산해보니 정보획득량은 $IG_2 = 0.9710 - 0.8464 = 0.1246$이 되기 때문에, 세로로 선을 긋는 방법 보다 선호하지 않는 규칙이 될 것이다. 

---

## 학습 방법: 재귀적 분기 & 가지치기

### 재귀적 분기(recursive partitioning)

예를 들어 대출 심사를 하는 데이터가 있다면, 어떻게 진행되는지 알아본다.

| 자동차 소유 | 소득 | 보유 대출 건수| 대출여부 |
|:-:|:-:|:-:|:-:|
|0|650|1| yes|
|0|200|0| no|
|1|700|3| no|
|0|500|0| no|
|0|425|1| no|
|1|900|1| yes|

먼저 처음 시점의 엔트로피를 구하면 다음과 같다.

$$Entropy(A)= - \Big( \dfrac{4}{6}\log(\dfrac{4}{6}) + \dfrac{2}{6}\log(\dfrac{2}{6}) \Big) = 0.9183$$

```python
pks = np.array([4/6, 2/6])
entropy = calculate_entropy(pks)
print(f"Entropy is {entropy:.4f}")
# Entropy is 0.9183
```

그 후 특정 피처를 선정해서 정렬 후에 각 분기점에서 한번씩 엔트로피를 계산하고 IG를 구한다. 예를 들어 다음 표처럼 `소득 <= 200`을 기준으로 나눠서 계산하고, 다음 기준을 선정해서 계속 계산한다. 

| 자동차 소유 | 소득 | 보유 대출 건수 | 대출여부 |
|:-:|:-:|:-:|:-:|
|<span style="color: #7d7ee8">0</span>|<span style="color: #7d7ee8">200</span>|<span style="color: #7d7ee8">0</span>|<span style="color: #7d7ee8">no</span>|
|0|425|1| no|
|0|500|0| no|
|0|650|1| yes|
|1|700|3| no|
|1|900|1| yes|

---

|Colname|Value|MaxIG|ClsCount|
|---|---|---|---|
|income|700|0.557332| {False: {0: 0, 1: 1}, True: {0: 4, 1: 1}}
|existloan|0|0.432821| {False: {0: 1, 1: 0}, True: {0: 3, 1: 2}}
|car|0|0.012657|{False: {0: 1, 1: 1}, True: {0: 3, 1: 1}}

<details class="collaspe-article">
<summary>전체 코드보기</summary>
<div markdown="1">

```python
import numpy as np
import pandas as pd

def calculate_entropy(pks):
    log = np.log2(pks + 1e-10)
    entropy = - 1/len(pks) * (pks * log).sum()
    return entropy.round(6)

def get_info_gain(df, E_base, x_col):
    df = df.sort_values(x_col).reset_index(drop=True)
    unique_x_values = df[x_col].unique()
    print(f"[Start] Processing column: {x_col}: {list(unique_x_values)}")
    
    IG_candidates = []
    IG_counts = []
    for v in unique_x_values[:-1]:
        df["group"] = (df[x_col] <= v)
        counts = df.groupby(["group", "loan"]).size().unstack(fill_value=0)
        pks = counts.div(counts.sum(1), axis=0).values
        E_x_col = calculate_entropy(pks)
        info_gain = E_base - E_x_col
        IG_candidates.append(info_gain)
        IG_counts.append(counts.T.to_dict())
        print(f"  Testing Value <= {v} | Entropy: {E_x_col:.4f} | IG: {info_gain:.4f}")
        print("  Counts:", counts.T.to_dict())
    IG_candidates = np.array(IG_candidates)
    IG_max = np.max(IG_candidates)
    idx_IG_max = np.argmax(IG_candidates)
    IG_value = df.loc[idx_IG_max, x_col]
#     df["group"] = (df[x_col] <= IG_value)
    print(f"[Result] Max information gain is {IG_max:.4f} value: {IG_value}")
    print()
    del df["group"]
    return (IG_value, IG_max, IG_counts[idx_IG_max])

df = pd.DataFrame(
    dict(
        car=[0, 0, 1, 0, 0, 1],
        income=[650, 200, 700, 500, 425, 900],
        existloan=[1, 0, 3, 0, 1, 1],
        loan=[1, 0, 0, 0, 0, 1]
    )
)
E_base = calculate_entropy((df["loan"].value_counts() / len(df)).values)
IGs = []
for x_col in ["car", "income", "existloan"]:
    IG_value, IG_max, IG_counts = get_info_gain(df, E_base, x_col)
    IGs.append((x_col, IG_value, IG_max, IG_counts))
df_res = pd.DataFrame({col: x for col, x in zip(
    ["Colname", "Value", "MaxIG", "ClsCount"], list(zip(*IGs)))})
df_res.sort_values("MaxIG", ascending=False)

# [Start] Processing column: car: [0, 1]
#   Testing Value <= 0 | Entropy: 0.9056 | IG: 0.0127
#   Counts: {False: {0: 1, 1: 1}, True: {0: 3, 1: 1}}
# [Result] Max information gain is 0.0127 value: 0

# [Start] Processing column: income: [200, 425, 500, 650, 700, 900]
#   Testing Value <= 200 | Entropy: 0.4855 | IG: 0.4328
#   Counts: {False: {0: 3, 1: 2}, True: {0: 1, 1: 0}}
#   Testing Value <= 425 | Entropy: 0.5000 | IG: 0.4183
#   Counts: {False: {0: 2, 1: 2}, True: {0: 2, 1: 0}}
#   Testing Value <= 500 | Entropy: 0.4591 | IG: 0.4591
#   Counts: {False: {0: 1, 1: 2}, True: {0: 3, 1: 0}}
#   Testing Value <= 650 | Entropy: 0.9056 | IG: 0.0127
#   Counts: {False: {0: 1, 1: 1}, True: {0: 3, 1: 1}}
#   Testing Value <= 700 | Entropy: 0.3610 | IG: 0.5573
#   Counts: {False: {0: 0, 1: 1}, True: {0: 4, 1: 1}}
# [Result] Max information gain is 0.5573 value: 700

# [Start] Processing column: existloan: [0, 1, 3]
#   Testing Value <= 0 | Entropy: 0.5000 | IG: 0.4183
#   Counts: {False: {0: 2, 1: 2}, True: {0: 2, 1: 0}}
#   Testing Value <= 1 | Entropy: 0.4855 | IG: 0.4328
#   Counts: {False: {0: 1, 1: 0}, True: {0: 3, 1: 2}}
# [Result] Max information gain is 0.4328 value: 0

```
</div></details>

위 코드와 표는 각 입력 피처로 하나씩 규칙을 찾은 결과다. 살펴보면 최대 정보획등량(MaxIG)은 $0.068056$ 으로 `소득 <= 700` 기준으로 나누는 첫번째 노드의 규칙이 된다. 만약 소득이 규칙인 700보다 작은 값이면, 대출여부가 0인 값은 4개, 1인 값은 1개가 되고, 700 보다 크다면 대출여부가 0인 값은 0개, 1인 값은 1개가 된다. 다시 풀어서 말하면, 만약에 당신의 소득이 700 보다 적다면, 학습된 데이터를 기반으로 보았을 때, 대출가능한 확률은 20%(1/5) 정도가 될것이다. 

이렇게 첫번째 규칙이 정해지면, 사람이 정한 하이퍼파라미터인 나무의 최대 깊이(max depth)에 따라서 계속 찾을 것인지 아니면 멈출지를 결정한다. 그리고 엔트로피가 0이 되면 학습을 최대 깊이가 아니더라도 알아서 멈추게 된다. 다만, 의사결정 나무 모델이 깊게 들어갈 수록 과적합(overfitting)될 가능성이 높다.

scikit-learn 패키지를 활용하면 손쉽게 의사결정 나무를 학습시킬 수 있다. 다만 피처의 정렬하여 특정값을 기준값으로 정하지 않고 약간 다르게 적용한다.

```python
import pandas as pd
from sklearn import tree

df = pd.DataFrame(
    dict(
        car=[0, 0, 1, 0, 0, 1],
        income=[650, 200, 700, 500, 425, 900],
        existloan=[1, 0, 3, 0, 1, 1],
        loan=[1, 0, 0, 0, 0, 1]
    )
)

X, y = df.iloc[:, :-1], df.iloc[:, -1]

clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X, y)
```

만약 graphviz를 설치했다면, 의사결정 나무 모델을 시각화 해볼 수도 있다.

```python
import graphviz
dot_data = tree.export_graphviz(
    clf, out_file=None, 
    feature_names=["car", "income", "existloan"], 
    class_names=["대출 불가능", "대출 가능"], 
    filled=True, rounded=True, 
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph
```

{% include image.html id="1B8ZxPjkeFn_TCkG-ll5JSPu0JBGsLuHQ" desc="Graphviz로 의사결정 나무 모델 시각화" width="90%" height="auto" %}

이 모델 그래프에 따르면 대출이 가능한 사람은 수입(income)이 575 이상이어야 하고, 대출 보유 대출 건수(existloan)가 2개 이하여야 대출이 가능하다. 

<br>

### 가지치기(pruning)

과적합을 피하기 위해 다양한 방법이 있다. scikit-learn을 기준으로 설명하면 다음과 같다.

- `min_samples_leaf`: 리프 노드가 되기 위한 최소 샘플의 개수를 말하며, 이 값이 커질 수록 모델이 간결해지나, 학습 데이터가 부족할 경우, 전체 정확도가 떨어질 수가 있다. 
- `max_depth`: 최대 깊이, 깊이를 작게 만들어 모델을 간결하게 만들 수 있지만 정확도가 떨어질 가능성이 있다. 

여기서 말하는 가지치기(purning)는 **비용복잡도 가지치기(Cost Complexity Pruning)**을 말하며, 복잡도 파라미터라고 불리는 $\alpha$로 조절 할 수 있다. 의사결정 나무 모델 $T$가 주어 졌을 때, 비용복잡도 함수$R_{\alpha}(T)$는 다음과 같이 결정된다.

$$R_{\alpha}(T) = R(T) + \alpha \vert \hat{T} \vert$$

여기서 $\vert \hat{T} \vert$는 리프 노드의 개수, $R(T)$는 전체 리프 노드에서 계산된 오분류율이다. Scikit-learn에서는 $R(T)$를 전체 샘플로 가중치화된 리프 노드의 불순도(impurity)로 대신 계산한다. $\alpha$값이 올라 갈 수록, 훈련 데이터에서 의사 결정 나무 모델의 깊이와 노드의 개수가 점점 떨어진다. 따라서, $\alpha$값을 잘 조절하면, 검증 데이터에서 좋은 성능을 낼 수 있는 최적의 모델을 만들 수 있다.

{% include image.html id="1Q_0g6iCs1sS7dvG1c7OA451PPrDUIMEF" desc="Alpha값에 따른 Train/Test 데이터에서 정확도 변화" width="90%" height="auto" %}

---

# 장단점

어떤 문제를 해결할 때, 왜 이 모델을 사용하려고 하는지 이해하고 쓰는 것이 중요하기에, 의사결정 나무 모델의 장단점을 요약해서 정리해보았다. 

## 장점

- 사람이 해석하고, 이해하기 쉽게 시각화가 가능하다.
- 일반화(normalization), 더미변수(dummy variables), 결측치(missing values)에 대한 전처리가 거의 필요없다.
- 수치형과 범주형 데이터를 다룰 수가 있다. 
- 조절해야할 하이퍼파라미터가 상대적으로 적긱 때문에, 빠르게 실험해 볼 수 있다.

## 단점

- 조절을 못하면 과적합된 모델을 생성할 가능성이 크다. 
- 데이터의 작은 변동으로 인해 완전히 다른 트리가 생성될 수 있기 때문에 의사 결정 트리는 불안정할 수 있다. 이 문제는 앙상블 내에서 의사결정 트리를 사용함으로써 완화할 수 있다.
- 예측값이 매끄럽지도 않고 연속적이지도 않고 단편적으로 일정한 근사치이다. 보외법(extrapolation)을 수행하기가 어렵다. 즉, 일반화가 안될 수 있다.
- 특정 클래스의 값에 지배적으로 편향되는 경우, 편향된 트리를 만든다. 따라서 학습 전에 균형 있는 데이터 세트를 만드는 것이 중요하다.