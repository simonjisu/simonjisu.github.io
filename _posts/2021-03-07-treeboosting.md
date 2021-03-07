---
layout: post
title: "Tree-based Ensemble: Boosting"
date: "2021-03-07 17:13:01 +0900"
categories: machinelearning
author: "Soo"
comments: true
toc: true
---

# References

- [위키백과: 결정 트리 학습법](https://ko.wikipedia.org/wiki/%EA%B2%B0%EC%A0%95_%ED%8A%B8%EB%A6%AC_%ED%95%99%EC%8A%B5%EB%B2%95)
- [Xgboost 사용하기](https://brunch.co.kr/@snobberys/137)
- [Ensemble: bagging, boosting](https://brunch.co.kr/@chris-song/98)
- [Gradient Boosting Algorithm의 직관적 이해](https://3months.tistory.com/368)
- [rpmcruz의 gboost코드](https://github.com/rpmcruz/machine-learning/blob/master/ensemble/boosting/gboost.py)

# Ensemble

**앙상블(Ensemble)**의 사전적 의미는 2인 이상의 노래나 연주를 뜻하는데, 머신러닝에서 앙상블 학습이란 하나의 학습 알고리즘 보다 더 좋은 성능을 내기 위해 다수의 약한 성능을 가진 학습 알고리즘을 합쳐서 사용하는 방법이다. 이때, 약한 성능을 가진 학습 알고리즘을 **약한 학습자(Weak Learner)**라고 부른다. 앙상블 학습은 일반적으로 **배깅(Bagging)** 과 **부스팅(Boosting)** 두 가지의 유형으로 나눌 수 있다.

1. Bagging(Bootstrap Aggregation): 샘플을 여러번 뽑아(Bootstrap)서 여러 개의 약한 학습자를 병렬적으로 학습시켜 결과물을 집계(Aggregration)하는 방법. 결과물을 집계하는 방법으로 회귀 문제의 경우 평균을 내거나, 분류 문제의 경우 가장 많이 나온 클래스로 투표(Hard Voting) 혹은 클래스의 확률을 평균화해서 가장 높은 확률로 도출(Soft Voting)한다. 모델의 variance를 줄이는 방향을 원한다면 배깅방법이 적합하다.
2. Boosting: 훈련 데이터를 샘플링하여 순차적으로 약한 학습자를 하나씩 추가시키면서 학습한다. 각 약한 학습자들은 가중치로 연결된다는 것이 특징이다. 다만 이전 약한 학습자가 틀린 데이터의 샘플링이 더 잘 되게 가중치를 부여하여 훈련 데이터를 생성하고 다시 학습한다. 모델의 bias를 줄이는 방향을 생각하고 있다면, 부스팅 방법이 적합하다.

---

# Boosting Ensemble

앙상블 방법론중 부스팅 방법에는 **적응형 부스팅(Adaptive Boosting)**, **경사 부스팅(Gradient Boosting)**이 있다. 이번 글에서는 Gradient Boosting을 중점적으로 다뤄본다.

## Gradient Boosting

Gradient Boosting을 한마디로 하면 Pseudo-Residual Fitting이라고 할 수 있다. 예를 들어, 약한 학습자 $A$로 데이터($Y, X$)를 학습시킨 결과를 다음 수식으로 표현해본다.

$$Y = A(X) + \epsilon_1$$

여기서 $\epsilon_1$은 **오차(error)** 혹은 **잔차(residual)**이라고 부른다. 그러면 남은 $\epsilon_1$에 대해서 다른 약한 학습자 $B$를 예측을 잘하게 학습시켜서 $Y$를 예측한다.

$$\epsilon_1 = B(X) + \epsilon_2$$

이렇게 계속 잔차를 줄여나가면서 여러 개의 약한 학습자를 연결시키는 것이 Gradient Boosting이다. 

### Negative Gradient

그렇다면 왜 Gradient가 들어가는가? 그 해답은 손실함수(loss function)와 연결된다.

예를 들어, Mean Squared Error를 손실함수로 설정하면, 다음과 같이 예측 값에 대해 경사(gradient)를 구할 수 있다. 

$$\begin{aligned} 
\text{Loss} &= L \big(Y, f(X) \big) = \dfrac{1}{2} \big( Y - f(X) \big)^2 \\
\text{gradient} &= \dfrac{\partial L}{\partial f(X)} = \dfrac{1}{2} \times 2 \big( Y - f(X) \big) \times (-1) = -\big( Y - f(X) \big) \\
\text{residual} &= -\dfrac{\partial L}{\partial f(X)} = - \text{gradient}
\end{aligned}$$

이때의 잔차는 음의 경사값이 되는 것을 알 수 있다. 이는 우리가 데이터를 남은 잔차에 대해서 학습하는 방법이 곧 전체 손실값을 줄이는 것과 같다는 이야기다. 

[expand]summary:부연 설명보기 👈

부연 설명하자면, 두 개의 약한 학습자를 예로 들면 다음과 같다.

$$\begin{aligned}
\text{fitting: } Y &= f_1(X) + \epsilon_1 \\
L_1 &= L\big(Y, f_1(X) \big) = \dfrac{1}{2}\big( Y - f_1(X)\big)^2 \\
\dfrac{\partial L_1}{\partial f_1(X)} &= - \big( Y - f_1(X)\big) = - \epsilon_1 \\
\\
\text{fitting: } \epsilon_1 &= f_2(X) + \epsilon_2 \\
L_2 &= L\big(\epsilon_1, f_2(X) \big) = \dfrac{1}{2}\big( \epsilon_1 - f_2(X)\big)^2 \\
\dfrac{\partial L_2}{\partial f_2(X)} &= - \big( \epsilon_1 - f_2(X)\big) = - \epsilon_2 \\
\end{aligned}$$

여기서 $\epsilon_1$을 $L_2$ 에 넣어보면

$$\begin{aligned}
L_2 &= \dfrac{1}{2}\big( Y - f_1(X) - f_2(X)\big)^2 \\
\dfrac{\partial L_2}{\partial f_2(X)} &= - \big( Y - f_1(X) - f_2(X)\big) = - \epsilon_2
\end{aligned}$$

가 되고, $F(X) = f_1(X) + f_2(X)$ 라고 하면, 약한 학습자를 여러 개를 더한 모델($f_1(X) + f_2(X)$)을 최적화 하는 것과 하나의 잘 예측하는 모델($F(X)$)를 최적화하는 것과 같다는 것을 알 수 있다(말장난 같지만, 이유는 같은 잔차$\epsilon_2$ 가 남기 때문).

$$\begin{aligned}
L &= \dfrac{1}{2}\big( Y - F(X)\big)^2 \\
\dfrac{\partial L}{\partial F(X)} &= - \big( Y - F(X)\big) = - \epsilon_2
\end{aligned}$$

또한, 원래 잔차를 구하려면 학습된 모델에 예측을 하고, 타겟값에 예측값을 빼줘야하는데, 미분으로 잔차를 구할 수 있으니 더 빠르게 학습이 가능하다.

[/expand]

결국에는 negative gradient가 잔차와 일치하기 때문에 Gradient 용어가 들어가는 것이다.

알고리즘은 다음과 같이 진행한다([rpmcruz의 gboost코드](https://github.com/rpmcruz/machine-learning/blob/master/ensemble/boosting/gboost.py)를 빌려 약간의 변형후 해석해본다).

```python
# self.first_estimator = 더미모델
# self.base_estimator = 약한 학습자
# self.loss = residual 함수, 여기서는 y - y_pred
# self.eta = Loss(y, y_pred)를 최소로 하는 eta값을 구하는데,
# 여기서는 생략하고 하나의 값으로 통일

def fit(self, X, y):
    self.fs = [self.first_estimator]
    # step 0
    f = self.first_estimator.fit(X, y)
    # step 1
    for m in range(self.M):
        y_pred = self.predict(X)
        # step 2
        R = self.loss(y, y_pred)
        # step 3
        g = clone(self.base_estimator).fit(X, R)
        # step 4
        self.fs.append(g)

def predict(self, X):
    f0 = self.fs[0].predict(X)
    r = np.sum([self.eta * f.predict(X) for f in self.fs[1:]], 0)
    return f0 + r
```

- **Step 0.** dummy 모델(예를 들어 모든 예측값이 0인 모델)로 초기화 한다. 
- **Step 1.** 모델의 개수 `M` 만큼 진행하며 항상 첫 스텝에 `X`에 대한 예측을 한다. 예측은 현재까지 저장해둔 약한 학습자의 예측값과 가중치(`self.eta`)를 곱해서 합한 값으로 한다.
- **Step 2.** 예측값(`y_pred`)과 타겟값(`y`)을 이용해 잔차(`R`)을 구한다.
- **Step 3.** 약한 학습자를 타겟값이 아닌 잔차값에 대해서 학습한다.

---

# Tree-Based Ensemble & CART

의사결정 나무 계열의 앙상블중 부스팅 방법을 사용하는 패키지들은 많이 있으나, 그중에서 XGBoost를 소개하려고 한다. 그 전에 의사결정 나무의 **CART(Classification And Regression Tree)** 알고리즘을 소개하고 넘어간다.

CART 알고리즘은 이름에서도 알 수 있듯이 분류와 회귀를 둘다 가능케하는 알고리즘이다. 분류의 경우 지니 계수(Gini Index)를 사용하고, 회귀의 경우 실제값과 예측값의 오차(MSE)를 사용해서 분기를 정한다. 일반적인 의사결정 나무와 다르게, 리프 노드(leaf node)에는 실제값 대신 결정값(decision values)을 보존하고 있다.

예를 들어, 다음 그림과 같이 가족원 5명중 컴퓨터 게임을 좋아할 만한 사람을 분류하는 모델을 만들면 최종 리프 노드에는 해당 조건에 부합할 경우 양의 점수가 저장되고 그렇지 않으면 음의 점수가 저장된다. 이러한 결정값을 저장함으로써 더 유연한 비교와 추론을 할 수가 있다.

{% include image.html id="15_gKUfmzAV-oZaJ17uYDajhmGjIQc4Sc" desc="출처: XGBoost Documentation" width="80%" height="auto" %}

보통 하나의 의사결정 나무 모델을 쓰지 않고, 앙상블 학습법을 많이 쓰기 때문에, 예를 들어, 다음과 같이 두 개의 의사 결정나무를 예시로 어떻게 진행되는지 살펴보자. 

{% include image.html id="1APEjS1gHrcfJXR_W7BC29mV6dNIteVe9" desc="출처: XGBoost Documentation" width="80%" height="auto" %}

각 샘플(가족원)의 최종 점수를 계산하려면 샘플이 속한 스코어를 전부 더하면 된다. 파란 옷을 입은 아들의 경우, 첫번째 모델에서 $+2$ 그리고 두번째 모델에서 $+0.9$의 결정값을 가짐으로 최종 결정값은 $+2.9$가 된다. 점수가 높을 수록 게임을 더 좋아한다는 뜻임으로, 아들은 게임을 좋아한다고 분류할 수 있다. 

여기에는 두 의사결정 나무 모델이 같은 가중치를 가지고 있다고 가정하고 있다. 물론 가중치를 다르게 줘서, 나이가 더 중요하다고 생각하면 첫번째 모델에 더 가중치를 줘서 높은 결정값을 가지게 할 수도 있다.

---

# XGBoost

Gradient Boosting 알고리즘에서 가장 유명한 패키지는 [**XGBoost**](https://xgboost.readthedocs.io/en/latest/index.html)이다. XGBoost는 Extreme Gradient Boosting 의 약자로써, 말그대로 엄청나게 빠른 Boosting 학습을 지원한다.

- 설치방법과 사용방법은 따로 다루지 않으며 아래 링크를 참고하길 바란다.
    - [XGBoost 공식문서](https://xgboost.readthedocs.io/en/latest/parameter.html)
    - [브런치: Xgboost 사용하기](https://brunch.co.kr/@snobberys/137)
    - [eagle705님의 블로그: XGBoost-정리](https://eagle705.github.io/articles/2018-06/XGBoost-정리)

## 특징

정규화(regularization)를 도입해서 과적합을 피하고 있다. 목적함수에 의사결정 나무의 복잡도($\Omega$)를 구하여 이를 억제하는 방식으로 훈련한다.

$$\begin{aligned}
\text{Loss}^{(t)} = \sum_{i=1}^n L(y_i, y_i^{(t)}) + \sum_{j=1}^{t} \Omega(f_j)
\end{aligned}$$

의사결정 나무를 (1)식과 같이 정의하면, 복잡도은 (2)식 처럼 정의할 수 있다. 여기서 $w$는 리프노드의 결정값 벡터이고, $q$는 각 데이터를 해당하는 $T$개의 모델의 리프노드에 배분하는 함수다. 모델 복잡도에서 $\gamma$는 정보획득(information gain)에 관한 변수이며, 정보획득량이 $\gamma$보다 낮은 경우 약한 학습자를 더하지 않는다. $\lambda$는 결정값에 관한 변수이며, 해당 값이 높을수록 결정값이 커지지 않게 더 큰 제약을 준다.

$$\begin{aligned}
f_t(x) &= w_{q(x)}, w \in R^T , q: R^d \rightarrow \{ 1, 2, \cdots, T\} & \cdots (1)\\
\Omega(f) &= \gamma T + \dfrac{1}{2} \lambda \sum_{j=1}^T w_j^2 & \cdots (2)
\end{aligned}$$

## XGBoost의 병렬처리

원칙적으로 Boosting방법은 Additive 학습 방법이기 때문에, 학습 방법 자체는 병렬이 불가능한 구조다. 하지만, 의사결정 나무를 만드는 과정을 병렬화(하나의 모델 내에서 openMP로 개별 코어에게 분배해서 하위 브랜치를 만듦)함으로써 더 빠르게 학습을 완료시킬 수 있었다. 아래 레퍼런스를 참고하자. 

- [XGBoost vs GBM](https://m.blog.naver.com/nicolechae0627/221811579005)
- [XGBoost의 병렬처리](http://machinelearningkorea.com/2019/07/25/xgboost-의-병렬처리가-어떻게-가능할까/)
- [How does XGBoost do parallel computation?](https://stackoverflow.com/questions/34151051/how-does-xgboost-do-parallel-computation)