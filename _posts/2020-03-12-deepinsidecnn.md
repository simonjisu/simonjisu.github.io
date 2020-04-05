---
layout: post
title: "[VISION] Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps"
date: "2020-03-12 14:19:38 +0900"
categories: paper
author: "Soo"
comments: true
toc: true
---

Paper Link: [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)

# 0. Abstract

이 논문에서는 입력 이미지에 대한 경사(gradient)를 구함으로써 두 가지 이미지 분류 모델의 시각화 기술을 중점적으로 서술했다. 첫째는 class score(최종 분류층 점수)를 극대화하여, ConvNet에서 포착된 클래스의 개념을 시각화하는 이미지를 생성한다. 둘째는 이미지와 이에 해당하는 클래스의 saliency maps(특징 지도)를 생성해내는 것이다. Saliency maps로 weakly supervised image segmentation에 적용했고, deconvolutional network와 비교도 해보았다.

---

# 1. Introduction

이 논문의 기여는 다음과 같다.

1. 입력 이미지의 수치적 최적화를 통해 CNN 모델에서 이해가능한 수준의 시각화된 이미지를 얻을 수 있다.
2. ConvNet을 통한 분류에서 단일 역전파(back-propagation) 경로를 사용하여 주어진 이미지(이미지별 class saliency map)에서 주어진 클래스의 공간적 지지점(spatial support)을 계산하는 방법을 제안한다. 
3. gradient 기반의 시각화 방법으로 deconvolutional network의 재구성 과정을 일반화했다.

---

# 2. Class Model Visualisation

$S_c(I)$가 주어진 이미지($I$)의 클래스($c$) 점수(score)라고 정의한다. 그러면 다음 수식과 같이 점수$S_c$를 최대화 하는 L2 정규화된 이미지를 찾을 수 있을 것이다($\lambda$는 정규화 하이퍼파라미터).

$$\arg \underset{I}{\max} S_c(I) - \lambda \Vert I\Vert^2_2$$

지역적으로 최적화된 이미지($I$)는 역전파(back-propagation)방법으로 찾을 수 있다. 이는 ConvNet의 훈련 과정중 역전파에서 각 층의 가중치를 최적화 할 때와 연관이 있다. 여기서 다른 점이라면 입력 이미지($I$)에 대한 최적화를 수행하는 것이고, 모델 가중치(weights)는 고정시킨다. 전체 과정은 다음과 같다.

1. 먼저 zero image $I$를 만든다.
2. $I$를 네트워크에 입력으로 해당 타겟에 해당하는 출력 스코어$S_c(I)$를 구한다.
3. 출력 스코어$S_c(I)$에 정규화 계수 $\lambda$와 입력 이미지 $I$의 L2 Norm을 곱한 값을 빼주면 최종 손실값$L$이 된다.
4. 손실값을 입력 이미지 $I$에 대해서 미분하여 업데이트 한다
5. 1~4 과정을 반복한다.

---

# 3. Image-Specific class Saliency Visualization

이번 파트에서는 ConvNet가 주어진 이미지와 클래스에 대한 공간적 지지점(spatial support)을 찾는 과정을 설명한다. 주어진 이미지를 $I_0$, 타겟 클래스를 $c$ 그리고 ConvNet에 이미지를 입력하여 얻은 점수 벡터$S_c(I)$ 라고 해보자. 이제 점수 벡터 $S_c(I_0)$에 근거하여 입력 이미지 $I_0$에 픽셀들의 순위를 정할 것이다.

먼저 제일 간단한 예제인 선형모델로 시작해보면 다음과 같다(이미지 $I$는 벡터화 시켰다). 

$$S_c(I) = w_c^TI+b_c$$

이 경우, 가중치벡터 $w_c$내에 있는 각 원소의 크기가 입력 이미지 $I$에 대한 중요도라고 정의할 수 있다. 그러나 심층 신경망에서 점수$S_c(I)$는 깊게 꼬인 비선형함수다. 따라서 위와 같이 적용이 불가능하다. 그러나 이미지 $I_0$가 주어졌을 때, 테일러 1차 급수로 $S_c(I)$에 대한 선형함수를 근사할 수 있다.

$$\begin{aligned} S_c(I) 
&\approx S_c(I_0) + \dfrac{\partial S_c}{\partial I_0}(I - I_0)  \\
&=w^TI+b \\
&\text{where } w= \dfrac{\partial S_c}{\partial I}\Bigg\vert_{I_0} \cdot
\end{aligned}$$

Image-Specific class Saliency의 다른 해석으로 클래스 점수에 대한 미분값($w$, 모델 가중치 값이 아님)의 크기는 어떤 픽셀들이 가장 적은 변화량으로 클래스 점수에 가장 큰 영향일 미치는지를 가르킨다. 이를 통해 이미지의 사물의 위치를 알아내기를 기대할 수 있다.

## Class Saliency Extraction

흑백이미지의 경우 절대값을 취해주면 그대로 추출할 수 있다. 컬러 이미지같은 경우 절대값에서 각 채널을 기준으로 최대 값을 뽑아내서 Saliency Map을 만든다.

$$M_{ij} = \max_c \vert w_{h(i, j, c)} \vert$$

이 논문에서는 ILSVRC-2013에서 높은 점수를 가진 클래스를 가지고 10장의 이미지를 서브 이미지를 crop 한 후, saliency map들을 산출하여 평균내서 한 장으로 합쳐서 그렸다.

## Weakly Supervised Object Localisation

이러한 saliency map을 물체 위치 탐지 문제에 적용했다. 과정을 요약하면 다음과 같다.

{% include image.html id="1V237wxA35x4oebtlzbOqc3h0-nH44cL6" desc="[그림 1] Geodesic Star Convexity for Interactive Image Segmentation" width="100%" height="auto" %}

1. GraphCut 이라는 것을 사용한다. 관심 가지는 클래스를 foreground, 그외에 배경을 background라고하는데, `그림 1`의 Step 2 처럼, foreground와 background 구분짓기 위해서 특정 색상으로 tagging을 해야한다.
2. saliency map은 특정 색상을 지정할 수 없기 때문에, 가우시안 믹스쳐(Gaussian Mixture) 모델을 활용하여 saliency map의 특정 경계값을 기준으로 foreground와 background의 경계 지도을 만든다.
3. `2`에서 만들어진 태깅된 경계 지도로 GraphCut으로 Segmentation을 진행한다.

자세한 설명은 다음과 같다.

[GraphCut](http://www.csd.uwo.ca/~yuri/Papers/iccv01.pdf)을 사용하게된 계기는 saliency map은 물체를 판별하는 영역만 탐지하지 물체 전체를 잡아내지 않기 때문이다. GraphCut을 사용하기 위해서 물체의 경계 지도를 전달하는게 중요하다. Foreground(관심 가지는 물체 클래스)와 background(물체 이외에 배경) 모델은 가우시안 믹스처(Gaussian Mixture)를 적용했다. Saliency 분포값의 95%를 경계로 이보다 높은 값을 가지는 픽셀들로 foreground를 추정했고, 30%를 경계로 이보다 이하의 값을 가지는 픽셀들은 background로 추정했다. 실제로 적용하면 `[그림 2]`의 3번째 그림처럼 나온다.

{% include image.html id="1Tqqu_QRGqMvOyrvoOVLJdaLOjuxGkGGS" desc="[그림 2] 1: 원본 / 2: saliency map / 3: 경계 지도 / 4: segmentated image" width="100%" height="auto" %}

Weakly supervised 임에도 불구하고, ILSVRC-2013 테스트 데이터에서 46.4%의 Top-5 error 성적을 거두었다(당시 우승자는 29.9%를 기록). GraphCut 프로그램은 [여기](http://www.robots.ox.ac.uk/~vgg/software/iseg/)서 사용할 수 있다(matlab code).

---

# 4. Relation to Deconvolutional Networks

저자는 Deconvolution Network(Zeiler & Fergus, 2013) 구조를 사용해 원래 이미지를 재구성하는 것은 사실상 미분하는 것과 거의 동일하다고 이야기한다. 

Deconvolution과 미분의 관계는 전에 작성한 포스트를 참고하길 바란다.

- [[PyTorch] ConvTranspose2d 와 Conv2d 의 관계](https://simonjisu.github.io/datascience/2019/10/27/convtranspose2d.html)

---

# Appendix: 직접 코딩하여 살펴보기

ILSVRC 2015의 1위 모델인 `ResNet152`을 가져와서 [Pixabay](https://pixabay.com/ko/)에 있는 플라밍고(class: 130) 이미지를 사용해서 Class Model Visualization과 Saliency Map을 생성해보았다.

{% include image.html id="1qyoRulVHIqlqESl0roNMSoLB8s9OB9zN" desc="[그림 3] 플라밍고 Class Model Visualization과 Saliency Map" width="100%" height="auto" %}

이미지는 256x256 크기로 재조정하고 224x224 크기로 center crop을 진행했다. 

**Class model visualization**의 경우, 151스텝동안 backpropagation 진행, L2 정규화에 $\lambda$를 1.0 으로 설정한 결과다. 자세히 보면 플라밍고의 머리와 목 부분이 곳곳에서 보인다(사실 이게 어떤 의미인지는 아직 연구가 필요하다). 

**Saliency Map**의 경우, 딱 1회만 역전파를 한 결과다. 논문에서도 서술했지만, 물체를 직접 탐지하지는 않으며, 물체를 판별하는데 도움이되는 영역이 주로 표시된다.

자세한 코드는 다음 항목들에서 이용할 수 있다.

- [GitHub](https://github.com/simonjisu/pytorch_tutorials/blob/master/02_VISION/03_deep_inside_cnn.ipynb) 에서 보기
- [Jupyter Notebook](https://nbviewer.jupyter.org/github/simonjisu/pytorch_tutorials/blob/master/02_VISION/03_deep_inside_cnn.ipynb) 에서 보기