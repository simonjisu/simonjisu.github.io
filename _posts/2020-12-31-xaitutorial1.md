---
layout: post
title: "[XAI] Explainable Artificial Intelligence (XAI) - 1 "
date: "2020-12-31 11:38:38 +0900"
categories: paper
author: "Soo"
comments: true
toc: true
---

# Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI

Paper Link: [https://arxiv.org/abs/1910.10045](https://arxiv.org/abs/1910.10045)

XAI에 대한 전반적인 소개를 정리한 논문이 나와서 차근 차근 요약 정리해보려고 한다(무려 115페이지, reference만 6페이지). 약간의 번역 어투와 생략된 것도 있으니 영어 원문을 참고하길 바란다.

1. [<span style="color:#e25252">Introduction(이번편)</span>](https://simonjisu.github.io/paper/2020/12/31/xaitutorial1.html)
2. [Explainability: What, why, what for and how?](https://simonjisu.github.io/paper/2021/01/14/xaitutorial2.html)
3. [Transparent machine learning models](https://simonjisu.github.io/paper/2021/01/23/xaitutorial3.html)
4. Post-hoc explainability techniques for machile learning models: Taxonomy, shallow models and deep learning
5. XAI: Opportunities, challenges and future research needs
6. Toward responsible AI: Principles of artificial intelligence, fairness, privacy and data fusion
7. Conclusions and outlook

# 1. Introduction 

[expand]summary:영어원문 👈 

Artificial Intelligence (AI) lies at the core of many activity sectors that have embraced new information technologies [1]. While the roots of AI trace back to several decades ago, there is a clear consensus on the paramount importance featured nowadays by intelligent machines endowed with learning, reasoning and adaptation capabilities. It is by virtue of these capabilities that AI methods are achieving unprecedented levels of performance when learning to solve increasingly complex computational tasks, making them pivotal for the future development of the human society [2]. The sophistication of AI-powered systems has lately increased to such an extent that almost no human intervention is required for their design and deployment. When decisions derived from such systems ultimately affect humans’ lives (as in e.g. medicine, law or defense), there is an emerging need for understanding how such decisions are furnished by AI methods [3].

[/expand]

<span style="color:#e25252">요약:</span> 인공지능이 정교해지면서 계산이 점점 복잡해지는 반면, 궁극적으로 인간의 삶에 영향을 미치는(의학, 법률, 국방) 시스템(기계)의 결정이 어떻게 내려졌는지, 우리는 이해할 필요가 있다.

[expand]summary:영어원문 👈 

While the very first AI systems were easily interpretable, the last years have witnessed the rise of opaque decision systems such as Deep Neural Networks (DNNs). The empirical success of Deep Learning (DL) models such as DNNs stems from a combination of efficient learning algorithms and their huge parametric space. The latter space comprises hundreds of layers and millions of parameters, which makes DNNs be considered as complex black-box models [4]. The opposite of black-box-ness is transparency, i.e., the search for a direct understanding of the mechanism by which a model works [5].

[/expand]

<span style="color:#e25252">요약:</span> 딥러닝 모델은 효율적인 학습 알고리즘과 거대한 파라미터 공간의 결합에서 비롯된다. 그리고 black-box 모델로 간주 된다. 이의 반대는 **투명성(transparency)**이다.

[expand]summary:영어원문 👈 

As black-box Machine Learning (ML) models are increasingly being employed to make important predictions in critical contexts, the demand for transparency is increasing from the various stakeholders in AI [6]. The danger is on creating and using decisions that are not justifiable, legitimate, or that simply do not allow obtaining detailed explanations of their behaviour [7]. Explanations supporting the output of a model are crucial, e.g., in precision medicine, where experts require far more information from the model than a simple binary prediction for supporting their diagnosis [8]. Other examples include autonomous vehicles in transportation, security, and finance, among others.

[/expand]

<span style="color:#e25252">요약:</span> Machine Learning 모델이 점점 많이 활용되면서, 이해관계자들로부터 투명성의 요구가 높아지고 있다. 예를 들어, 의료(진단), 교통(자율주행), 보안, 금융등 이 있다.

[expand]summary:영어원문 👈 

In general, humans are reticent to adopt techniques that are not directly interpretable, tractable and trustworthy [9], given the increasing demand for ethical AI [3]. It is customary to think that by focusing solely on performance, the systems will be increasingly opaque. This is true in the sense that there is a trade-off between the performance of a model and its transparency [10]. However, an improvement in the understanding of a system can lead to the correction of its deficiencies. When developing a ML model, the consideration of interpretability as an additional design driver can improve its implementability for 3 reasons:

* Interpretability helps ensure impartiality in decision-making, i.e. to detect, and consequently, correct from bias in the training dataset.
* Interpretability facilitates the provision of robustness by highlighting potential adversarial perturbations that could change the prediction.

* Interpretability can act as an insurance that only meaningful variables infer the output, i.e., guaranteeing that an underlying truthful causality exists in the model reasoning.

All these means that the interpretation of the system should, in order to be considered practical, provide either an understanding of the model mechanisms and predictions, a visualization of the model’s discrimination rules, or hints on what could perturb the model [11].

[/expand]

<span style="color:#e25252">요약:</span> 통상적으로 성과에만 치중할 수록 시스템은 점점 불투명해질 것이라 생각한다. 모델의 성능과 투명성 사이에 trade-off가 있다는 점은 사실이나, 모델에 대한 이해는 모델의 성능 향상을 이끌어 낼 수도 있다. 추가로 ML모델을 개발할 때, 해석 가능성을 모듈로 넣으면 세 가지 이유로 구현 가능성을 향상 시킬 수 있다.

- 해석가능성은 의사결정에서 공정성을 보장하는데 도움이 된다. 즉, 교육 데이터 집합의 편향성을 탐지하고 결과적으로 수정한다.
- 해석가능성은 예측을 바꿀 수 있는 잠재적 적대적 섭동을 강조함으로써 건전성의 제공을 촉진한다.
- 해석가능성은 유의미한 변수만으로 산출물을 유추하는 보험으로서, 즉 모형 추론에서 근본적인 진실적 인과관계가 존재함을 보증하는 보험으로 작용할 수 있다.

즉, 해석가능한 시스템은 모델 매커니즘과 예측에 대한 이해, 모델의 판결 규칙 시각화, 또는 모델을 방해하는 것에 대한 힌트 등을 제공해야한다.

[expand]summary:영어원문 👈 

In order to avoid limiting the effectiveness of the current generation of AI systems, eXplainable AI (XAI) [7] proposes creating a suite of ML techniques that 1) produce more explainable models while maintaining a high level of learning performance (e.g., prediction accuracy), and 2) enable humans to understand, appropriately trust, and effectively manage the emerging generation of artificially intelligent partners. XAI draws as well insights from the Social Sciences [12] and considers the psychology of explanation.

[/expand]

<span style="color:#e25252">요약:</span> 현재의 효과적인 AI 시스템을 제한시키지 않는 선에서, eXplainable AI(XAI)은 1) 학습 퍼포먼스는 최대한으로 유지하면서 설명가능한 모델을 만들것을 제안 2) 사람이 이해하고, 적절하고 효과적으로 신뢰할 수 있도록 한다.

[expand]summary:영어원문 👈 

Fig. 1 displays the rising trend of contributions on XAI and related concepts. This literature outbreak shares its rationale with the research agendas of national governments and agencies. Although some recent surveys [8], [10], [13], [14], [15], [16], [17] summarize the upsurge of activity in XAI across sectors and disciplines, this overview aims to cover the creation of a complete unified framework of categories and concepts that allow for scrutiny and understanding of the field of XAI methods. Furthermore, we pose intriguing thoughts around the explainability of AI models in data fusion contexts with regards to data privacy and model confidentiality. This, along with other research opportunities and challenges identified throughout our study, serve as the pull factor toward Responsible Artificial Intelligence, term by which we refer to a series of AI principles to be necessarily met when deploying AI in real applications. As we will later show in detail, model explainability is among the most crucial aspects to be ensured within this methodological framework. All in all, the novel contributions of this overview can be summarized as follows:

1. Grounded on a first elaboration of concepts and terms used in XAI-related research, we propose a novel definition of explainability that places audience (Fig. 2) as a key aspect to be considered when explaining a ML model. We also elaborate on the diverse purposes sought when using XAI techniques, from trustworthiness to privacy awareness, which round up the claimed importance of purpose and targeted audience in model explainability.
2. We define and examine the different levels of transparency that a ML model can feature by itself, as well as the diverse approaches to post-hoc explainability, namely, the explanation of ML models that are not transparent by design.
3. We thoroughly analyze the literature on XAI and related concepts published to date, covering approximately 400 contributions arranged into two different taxonomies. The first taxonomy addresses the explainability of ML models using the previously made distinction between transparency and post-hoc explainability, including models that are transparent by themselves, Deep and non-Deep (i.e., shallow) learning models. The second taxonomy deals with XAI methods suited for the explanation of Deep Learning models, using classification criteria closely linked to this family of ML methods (e.g. layerwise explanations, representation vectors, attention).
4. We enumerate a series of challenges of XAI that still remain insufficiently addressed to date. Specifically, we identify research needs around the concepts and metrics to evaluate the explainability of ML models, and outline research directions toward making Deep Learning models more understandable. We further augment the scope of our prospects toward the implications of XAI techniques in regards to confidentiality, robustness in adversarial settings, data diversity, and other areas intersecting with explainability.
5. After the previous prospective discussion, we arrive at the concept of Responsible Artificial Intelligence, a manifold concept that imposes the systematic adoption of several AI principles for AI models to be of practical use. In addition to explainability, the guidelines behind Responsible AI establish that fairness, accountability and privacy should also be considered when implementing AI models in real environments.
6. Since Responsible AI blends together model explainability and privacy/security by design, we call for a profound reflection around the benefits and risks of XAI techniques in scenarios dealing with sensitive information and/or confidential ML models. As we will later show, the regulatory push toward data privacy, quality, integrity and governance demands more efforts to assess the role of XAI in this arena. In this regard, we provide an insight on the implications of XAI in terms of privacy and security under different data fusion paradigms.

[/expand]

{% include image.html id="119QnRBvYV4gHiuKz7kpaOVo_2b2tlhz5" desc="Fig 1. 학계에서 XAI 및 연관된 개념의 기여도 추세" width="100%" height="auto" %}

<span style="color:#e25252">요약:</span> `Fig 1`에서 볼 수 있듯이 국가 정부 및 기관의 연구의제의 키워드 추세를 살펴보면 XAI관련 활동이 최근 급증했지만, 통일된 프레임워크가 없다. 이번 논문에서는 통일된 프레임워크의 작성하고, 개인정보 보호 및 모델 기밀성에 대해서 의견을 제시할 것이다. 

1. 지금까지 XAI 관련 연구에서 사용된 개념과 용어의 기초하여, ML 모델을 설명할 때 청중(audience)을 핵심으로 고려할 것이다(그림 2). 또한 XAI 기법을 사용할 때 추구하는 다양한 목적에 따라 세분화할 것이다. 그리고 설명가능성에서 목적과 타겟 청중의 중요함을 이야기 한다.
2. 다양한 레벨의 투명성을 정의하고 검토한다. 대상에는 사후(post-hoc) 설명이 가능한, 자체 설명가능한 혹은 설계에 의해 설명이 불가능한 모델들 등이 있다.
3. XAI에 관한 문헌과 지금까지 출판된 관련 개념들을 철저하게 분석하여, 대략 400개의 기여를 두 개의 다른 분류법으로 배열하였다. 첫 번째 분류법은 이전에 만든 투명성(transparency)과 사후 설명성(post-hoc explainability) 사이의 구별을 사용하여 ML 모델의 설명가능성을 다루고 있으며, 여기에는 스스로 투명하고 깊지 않은(즉, shallow 얉은) 학습 모델이 포함된다. 두 번째 분류법은 딥러닝 모델의 설명에 적합한 XAI 방법을 다루며, 이 ML 방법 계열과 밀접하게 연계된 분류 기준(예: 계층적 설명 layer-wise explanations, 표현 벡터 representation vectors, 어텐션 attention)을 사용한다.
4. 지금까지도 불충분하게 다루어지지 않고 있는 XAI의 일련의 과제를 열거한다. 구체적으로는 ML 모델의 설명 가능성을 평가하기 위해 개념 및 메트릭스를 중심으로 연구 요구를 파악하고, 딥러닝 모델을 보다 이해할 수 있도록 연구 방향을 정리한다. 기밀성, 적대적 설정의 견고성, 데이터 다양성 및 설명 가능성과 교차하는 기타 영역에 관한 XAI 기법의 함축성을 향한 전망의 범위를 더욱 확대합니다.
5. 앞서의 장래의 논의를 거쳐, AI 모델이 실용화하기 위해 여러 가지 AI 원리를 체계적으로 채택하는 매니폴드 개념인 책임감 있는 인공지능의 개념에 도달한다. 책임 AI를 뒷받침하는 가이드라인은 설명가능성 외에도 실제 환경에서 AI 모델을 구현할 때 공정성, 책임성, 프라이버시 등도 고려해야 한다고 규정하고 있다.
6. 책임 있는 AI는 모델 설명 가능성과 개인 정보 보호/보안성을 설계별로 혼합하므로, 민감한 정보 및/또는 기밀 ML 모델을 다루는 시나리오에서 XAI 기법의 유익성과 위해성에 대해 심오한 반성을 요구한다. 나중에 보여드리겠지만, 데이터 개인 정보 보호, 품질, 무결성 및 거버넌스를 향한 규제는 이 분야에서 XAI의 역할을 평가하기 위한 더 많은 노력을 요구합니다. 이와 관련하여, 우리는 서로 다른 데이터 융합 패러다임 하에서의 프라이버시 및 보안 측면에서 XAI의 의미에 대한 통찰력을 제공한다.

[expand]summary:영어원문 👈 

The remainder of this overview is structured as follows: first, Section 2 and subsections therein open a discussion on the terminology and concepts revolving around explainability and interpretability in AI, ending up with the aforementioned novel definition of interpretability (Section 2.1 and 2.2), and a general criterion to categorize and analyze ML models from the XAI perspective. Sections 3 and 4 proceed by reviewing recent findings on XAI for ML models (on transparent models and post-hoc techniques respectively) that comprise the main division in the aforementioned taxonomy. We also include a review on hybrid approaches among the two, to attain XAI. Benefits and caveats of the synergies among the families of methods are discussed in Section 5, where we present a prospect of general challenges and some consequences to be cautious about. Finally, Section 6 elaborates on the concept of Responsible Artificial Intelligence. Section 7 concludes the survey with an outlook aimed at engaging the community around this vibrant research area, which has the potential to impact society, in particular those sectors that have progressively embraced ML as a core technology of their activity.

[/expand]

<span style="color:#e25252">요약:</span> 나머지 부분은 다음과 같이 구성되어 있다: 

- Section 2: 설명가능성(explainability)와 해석가능성(interpretability)의 새로운 정의, XAI 관점에서 ML 모델 분류 및 분석을 위한 용어 및 개념에 대한 이야기
- Section 3, 4: 최근 연구 결과와 하이브리드 방법
- Section 5: 해당 방법들에 대한 장단점 및 주의해야할 몇 가지 결과들 제시
- Section 6: "책임감 있는 인공지능" 개념에 대한 설명
- Section 7: 사회에 영향을 미칠 가능성이 있는 연구 영역인 만큼, ML 기술을 채택한 사람들을 커뮤니티를 참여시키는 목표로 결론을 내리고자 한다.