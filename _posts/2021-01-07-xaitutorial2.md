---
layout: post
title: "[XAI] Explainable Artificial Intelligence (XAI) - 2 "
date: "2021-01-07 11:38:38 +0900"
categories: paper
author: "Soo"
comments: true
toc: false
---

# Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI

Paper Link: [https://arxiv.org/abs/1910.10045](https://arxiv.org/abs/1910.10045)

XAI에 대한 전반적인 소개를 정리한 논문이 나와서 차근 차근 요약 정리해보려고 한다(무려 115페이지, reference만 6페이지). 약간의 번역 어투와 생략된 것도 있으니 영어 원문을 참고하길 바란다.

1. [Introduction](https://simonjisu.github.io/paper/2020/12/31/xaitutorial1.html)
2. [<span style="color:#e25252">Explainability: What, why, what for and how?(이번편)</span>](https://simonjisu.github.io/paper/2021/01/07/xaitutorial2.html)
3. Transparent machine learning models
4. Post-hoc explainability techniques for machile learning models: Taxonomy, shallow models and deep learning
5. XAI: Opportunities, challenges and future research needs
6. Toward responsible AI: Principles of artificial intelligence, fairness, privacy and data fusion
7. Conclusions and outlook

# 2. Explainability: What, Why, What For and How?

{% include collaspe-block.html summary="영어원문" article="Before proceeding with our literature study, it is convenient to first establish a common point of understanding on what the term explainability stands for in the context of AI and, more specifically, ML. This is indeed the purpose of this section, namely, to pause at the numerous definitions that have been done in regards to this concept (what?), to argue why explainability is an important issue in AI and ML (why? what for?) and to introduce the general classification of XAI approaches that will drive the literature study thereafter (how?)."
%}

<span style="color:#e25252">요약:</span> 시작하기 전에 **설명가능성(explainability)**이라는 용어가 AI 혹은 ML의 맥락에서 무엇을 뜻하는지, 공통의 이해점을 확립해야한다.

- What? 이 개념에 대한 정립된 수 많은 정의들을 정리
- Why? What for? 왜 설명가능성이 AI와 ML에서 중요한 이슈인지
- How? 이후 연구할 XAI 접근 방식의 일반적인 분류방식들 소개

---

## 2.1 Terminology Clarification

{% include collaspe-block.html summary="영어원문" article="One of the issues that hinders the establishment of common grounds is the interchangeable misuse of interpretability and explainability in the literature. There are notable differences among these concepts. To begin with, interpretability refers to a passive characteristic of a model referring to the level at which a given model makes sense for a human observer. This feature is also expressed as transparency. By contrast, explainability can be viewed as an active characteristic of a model, denoting any action or procedure taken by a model with the intent of clarifying or detailing its internal functions.
<br>
To summarize the most commonly used nomenclature, in this section we clarify the distinction and similarities among terms often used in the ethical AI and XAI communities.
<br>
- Understandability (or equivalently, intelligibility) denotes the characteristic of a model to make a human understand its function – how the model works – without any need for explaining its internal structure or the algorithmic means by which the model processes data internally [18].
<br>
- Comprehensibility: When conceived for ML models, comprehensibility refers to the ability of a learning algorithm to represent its learned knowledge in a human understandable fashion [19], [20], [21]. This notion of model comprehensibility stems from the postulates of Michalski [22], which stated that *“the results of computer induction should be symbolic descriptions of given entities, semantically and structurally similar to those a human expert might produce observing the same entities. Components of these descriptions should be comprehensible as single ‘chunks’ of information, directly interpretable in natural language, and should relate quantitative and qualitative concepts in an integrated fashion”*. Given its difficult quantification, comprehensibility is normally tied to the evaluation of the model complexity [17].
<br>
- Interpretability: It is defined as the ability to explain or to provide the meaning in understandable terms to a human.
<br>
- Explainability: Explainability is associated with the notion of explanation as an interface between humans and a decision maker that is, at the same time, both an accurate proxy of the decision maker and comprehensible to humans [17].
<br>
- Transparency: A model is considered to be transparent if by itself it is understandable. Since a model can feature different degrees of understandability, transparent models in Section 3 are divided into three categories: simulatable models, decomposable models and algorithmically transparent models [5]."
%}

<span style="color:#aaa"> 참고: 용어가 한국어로 거의다 비슷해서 최대한 의미를 붙여서 추가함 </span> 

<span style="color:#e25252">요약:</span> 공통의 이해점 확립을 방해하는 요소중 하나는 **설명가능성(explainability)**과 **해석가능성(interpretability)**용어의 혼용이다. 이 둘의 개념읜 차이점이 있다. 결론부터 말하자면

- **해석가능성(Interpretability):** 모델이 인간에게 맞춰서 설명하는 <U>모델의 수동적 특성</U>, 투명성(transparency)과 같은 말

- **설명가능성(Explainability):** 모델의 내부 기능을 명확히 하거나 자세히 설명할 목적으로, 수행된 모든 행동 또는 절차를 나타내는 <U>모델의 능동적 특성</U>

가장 일반적으로 사용되는 명명법을 이야기하고자 ethical AI 및 XAI 커뮤니티에서 자주 사용되는 용어 간의 구별과 그 유사성을 명확히한다.

- **이해가능성(Understandability)** 혹은 **명료성(Intelligibility):** 모델 구조 혹은 내부의 알고리즘 기능의 부가 설명 없이도 인간이 바로 이해할 수 있는 모델의 특성 [[18]](https://www.sciencedirect.com/science/article/pii/S1051200417302385)

- **포괄적 이해가능성(Comprehensibility):** 모델이 학습한 지식을 인간의 이해방식으로 나타내는 능력 [[19]](https://scholar.google.com/scholar?q=Evolutionary%20fuzzy%20systems%20for%20explainable%20artificial%20intelligence:%20Why,%20when,%20what%20for,%20and%20where%20to), [[20]](https://scholar.google.com/scholar_lookup?title=A%20framework%20for%20considering%20comprehensibility%20in%20modeling&publication_year=2016&author=M.%20Gleicher), [[21]](https://scholar.google.com/scholar_lookup?title=Extracting%20comprehensible%20models%20from%20trained%20neural%20networks&publication_year=1996&author=M.W.%20Craven). 이 개념은 Michalski[[22]](https://scholar.google.com/scholar_lookup?title=A%20theory%20and%20methodology%20of%20inductive%20learning&publication_year=1983&author=R.S.%20Michalski)의 가정에서 비롯됐다.

    "컴퓨터 유도 결과는 주어진 실체에 대한 상징적 설명이어야 하며, 인간 전문가가 동일한 실체를 관찰하는 것과 의미론적이고 구조적으로 유사해야 한다. 이러한 설명의 구성요소는 자연어로 직접 해석할 수 있는 정보의 단일 '청크'로 이해할 수 있어야 하며, 통합된 방식으로 양적 및 질적 개념을 연관시켜야 한다." 

    정량화가 어렵다는 점을 고려했을 때, 포괄적 이해가능성은 일반적으로 모델 복잡도 평가와 연관된다[[17]](https://scholar.google.com/scholar_lookup?title=A%20survey%20of%20methods%20for%20explaining%20black%20box%20models&publication_year=2018&author=R.%20Guidotti&author=A.%20Monreale&author=S.%20Ruggieri&author=F.%20Turini&author=F.%20Giannotti&author=D.%20Pedreschi).

- **해석가능성(Interpretability)**: 인간이 이해할 수 있는 용어로 의미를 설명하거나 제공하는 능력

- **설명가능성(Explainability):** 사람과 모델간의 "인터페이스(interface)" 역할로서 설명(explanation)과 연관되어 있다. 설명은 모델이 내린 의사결정의 정확한 대리이자 인간이 이해할 수 있는 것이어야 한다[[17]](https://scholar.google.com/scholar_lookup?title=A%20survey%20of%20methods%20for%20explaining%20black%20box%20models&publication_year=2018&author=R.%20Guidotti&author=A.%20Monreale&author=S.%20Ruggieri&author=F.%20Turini&author=F.%20Giannotti&author=D.%20Pedreschi).

- **투명성(Transparency):** 모델은 자신이 스스로 이해가능하다면 투명한 것으로 간주된다. 모델은 정도에 따라 이해력를 다르게 제공할 수 있기 때문에, 투명성 있는 모델은 section 3에서 3가지 항목(시뮬레이션 가능한 모델, 분해가능한 모델 그리고 알고리즘 자체가 투명한 모델)으로 나눌 것이다.

{% include collaspe-block.html summary="영어원문" article="In all the above definitions, understandability emerges as the most essential concept in XAI. Both transparency and interpretability are strongly tied to this concept: while transparency refers to the characteristic of a model to be, on its own, understandable for a human, understandability measures the degree to which a human can understand a decision made by a model. Comprehensibility is also connected to understandability in that it relies on the capability of the audience to understand the knowledge contained in the model. All in all, understandability is a two-sided matter: model understandability and human understandability. This is the reason why the definition of XAI given in Section 2.2 refers to the concept of audience, as the cognitive skills and pursued goal of the users of the model have to be taken into account jointly with the intelligibility and comprehensibility of the model in use. This prominent role taken by understandability makes the concept of audience the cornerstone of XAI, as we next elaborate in further detail."
%}

<span style="color:#e25252">요약:</span> 위의 모든 정의에서 이해가능성(understandability)은 XAI에서 가장 필수적인 개념이다. 투명성(transparency)과 해석가능성(interpretability)은 모두 이 개념과 강하게 연관되어 있다. 투명성(transparency)은 인간에게는 이해가능해야 하고, 모델 스스로는 이해할 수 있는 특성이지만, 이해가능성(interpretability)은 모델의 결정을 인간이 얼만큼 이해가능한지 측정한다. 

또한 포괄적 이해가능성(comprehensibility)은 **청중**(audience, <span style="color:#aaa">[주] 설명을 듣는 사람</span>)이 얼만큼 모델에 포함된 지식을 이해하는 지를 측정한다는 점에서 이해가능성(understandability)과 연결된다. 

대체로, 이해가능성(understandability)은 "모델"과 "인간"의 이해가능성으로 나눌 수 있다. 모델 이용자의 인지능력 및 추구목표는 모델의 명료성과 포괄적 이해가능성이 함께 고려되어야 하기 때문에, Section 2.2에서는 XAI개념을 정의할 때, **청중**<span style="color:#aaa">([주] 모델을 사용하는 인간)</span>의 개념을 먼저 이야기하려고 한다. 이해가능성은 XAI의 초석이 되는 청중의 개념을 만드는 역할을 한다.

---

## 2.2 What?

{% include collaspe-block.html summary="영어원문" article="Although it might be considered to be beyond the scope of this paper, it is worth noting the discussion held around general theories of explanation in the realm of philosophy [23]. Many proposals have been done in this regard, suggesting the need for a general, unified theory that approximates the structure and intent of an explanation. However, nobody has stood the critique when presenting such a general theory. For the time being, the most agreed-upon thought blends together different approaches to explanation drawn from diverse knowledge disciplines. A similar problem is found when addressing interpretability in AI. It appears from the literature that there is not yet a common point of understanding on what interpretability or explainability are. However, many contributions claim the achievement of interpretable models and techniques that empower explainability."
%}

<span style="color:#e25252">요약:</span> 철학의 영역에서 설명(explanation)에 대해 토의에 주목할 필요가 있다[[23]](https://scholar.google.com/scholar_lookup?title=General%20theories%20of%20explanation%3A%20buyer%20beware&publication_year=2013&author=J.%20D%C3%ADez&author=K.%20Khalifa&author=B.%20Leuridan). 왜냐면 일반적이고 통일된 설명 이론의 구조와 의도를 근사하게나마 제시했기 때문이다. 그렇지만 튼튼한 이론은 아니였다. 그래서 그동안 다양한 지식 분야에서 도출된 설명에 대한 접근 박싱을 혼합한 정의를 사용했다. AI에서 해석가능성(interpretability)을 다룰 때도 비슷한 문제가 발견됐다. 해석가능성(interpretability)이나 설명가능성(explainability)의 공통점을 찾지 못했으나, 많은 연구자들은 해석가능한 모델의 생성과 모델 설명력을 강화했다는 연구성과를 주장해왔다.

{% include collaspe-block.html summary="영어원문" article="To shed some light on this lack of consensus, it might be interesting to place the reference starting point at the definition of the term Explainable Artificial Intelligence (XAI) given by D. Gunning in [7]:
<br>
'XAI will create a suite of machine learning techniques that enables human users to understand, appropriately trust, and effectively manage the emerging generation of artificially intelligent partners.'
<br>
This definition brings together two concepts (understanding and trust) that need to be addressed in advance. However, it misses to consider other purposes motivating the need for interpretable AI models, such as causality, transferability, informativeness, fairness and confidence [5], [24], [25], [26]. We will later delve into these topics, mentioning them here as a supporting example of the incompleteness of the above definition."
%}

<span style="color:#e25252">요약:</span> 부족한 공감대를 어느 정도 형성하기 위해서 D. Gunning이 제시한 Explainable Artificial Intelligence(XAI) 용어의 정의를 기준점으로 시작할 수 있을 것 같다. [[7]](https://scholar.google.com/scholar?q=Explainable%20artificial%20intelligence)

> "인간이 이해할 수 있고, 적절하게 신뢰할 수 있으며, 효과적으로 세로운 세대의 인공지능 파트너를 관리할 수 있는 머신러닝 기법 제품군을 XAI는 만들어 낼 것이다" 

이 정의는 다루어야 할 두 가지 개념(이해와 신뢰)을 담았다. 그러나 해석가능한 AI 모델에 필요한 인과성(causality), 전이성(transferability), 정보성(informativeness), 공정성(fairness)과 확실성(confidence) 등을 담지 않았다[[5]](https://scholar.google.com/scholar_lookup?title=The%20mythos%20of%20model%20interpretability&publication_year=2018&author=Z.C.%20Lipton), [[24]](https://scholar.google.com/scholar?q=D.%20Doran,%20S.%20Schulz,%20T.R.%20Besold,%20What%20does%20explainable%20AI%20really%20mean%20a%20new%20conceptualization%20of%20perspectives,%202017.), [[25]](https://scholar.google.com/scholar?q=F.%20Doshi-Velez,%20B.%20Kim,%20Towards%20a%20rigorous%20science%20of%20interpretable%20machine%20learning,%202017.), [[26]](https://scholar.google.com/scholar_lookup?title=Making%20machine%20learning%20models%20interpretable.&publication_year=2012&author=A.%20Vellido&author=J.D.%20Mart%C3%ADn-Guerrero&author=P.J.%20Lisboa). 언급한 주제들은 D. Gunning의 정의에 대한 불완전성을 뒷받침하는 사례로 여기 언급하면서, 나중에 이 주제들을 파헤칠 것이다.

{% include collaspe-block.html summary="영어원문" article="As exemplified by the definition above, a thorough, complete definition of explainability in AI still slips from our fingers. A broader reformulation of this definition (e.g. 'An explainable Artificial Intelligence is one that produces explanations about its functioning') would fail to fully characterize the term in question, leaving aside important aspects such as its purpose. To build upon the completeness, a definition of explanation is first required."
%}

<span style="color:#e25252">요약:</span> 위 예시와 같이, 설명가능성에 대한 완벽한 정의를 내리기 어렵다. 예를 들어, "설명 가능한 인공지능은 그에 대한 기능을 설명하는 것이다" 경우, 설명가능성의 목적성 측면만 이야기한다. 따라서, 먼저 설명(explanation)에 대한 정의가 필요하다.

{% include collaspe-block.html summary="영어원문" article="As extracted from the Cambridge Dictionary of English Language, an explanation is 'the details or reasons that someone gives to make something clear or easy to understand' [27]. In the context of an ML model, this can be rephrased as: 'the details or reasons a model gives to make its functioning clear or easy to understand'. It is at this point where opinions start to diverge. Inherently stemming from the previous definitions, two ambiguities can be pointed out. First, the details or the reasons used to explain, are completely dependent of the audience to which they are presented. Second, whether the explanation has left the concept clear or easy to understand also depends completely on the audience. Therefore, the definition must be rephrased to reflect explicitly the dependence of the explainability of the model on the audience. To this end, a reworked definition could read as: 'Given a certain audience, explainability refers to the details and reasons a model gives to make its functioning clear or easy to understand.' "
%}

<span style="color:#e25252">요약:</span> 케임브릿지 영어사전을 인용하면, 설명(explanation)은 "어떤 것을 명백하게 혹은 이해하기 쉽게 만들어주기 위해 누군가가 밝혀주는 세부 사항이나 이유"다[[27]](https://scholar.google.com/scholar?q=Cambridge%20advanced%20learners%20dictionary). <span style="color:#aaa">([주] 네이버 국어사전의 경우, "어떤 일이나 대상의 내용을 상대편이 잘 알 수 있도록 밝혀 말함. 또는 그런 말.") </span>

Machine learning의 맥락 상, "모델이 자신의 기능을 명백히 하거나 이해하기 쉽게 세부사항 혹은 이유를 밝히는 것"으로 바꿀수 있다. 여기에서 의견이 갈리기 시작한다. 본질적으로 이전 정의에서 비롯 된것으로 두 가지 애매모호한 점이 있다.

1. 설명에 관한 세부 사항이나 이유는 이를 듣는 청중(audience)과 가장 연관이 있다.
2. 설명이 명료하게 혹은 알기 쉽게 되었는지의 여부도 완전히 청중에게 달려있다.

따라서, 두 가지를 반영해 다시 정의하면 다음과 같다.

> **"특정 청중에게 모델이 자신의 기능을 명백하게 혹은 이해하기 쉽게 밝히는 세부사항/이유를 설명가능성이라고 말한다."**

{% include collaspe-block.html summary="영어원문" article="Since explaining, as argumenting, may involve weighting, comparing or convincing an audience with logic-based formalizations of (counter) arguments [28], explainability might convey us into the realm of cognitive psychology and the psychology of explanations [7], since measuring whether something has been understood or put clearly is a hard task to be gauged objectively. However, measuring to which extent the internals of a model can be explained could be tackled objectively. Any means to reduce the complexity of the model or to simplify its outputs should be considered as an XAI approach. How big this leap is in terms of complexity or simplicity will correspond to how explainable the resulting model is. An underlying problem that remains unsolved is that the interpretability gain provided by such XAI approaches may not be straightforward to quantify: for instance, a model simplification can be evaluated based on the reduction of the number of architectural elements or number of parameters of the model itself (as often made, for instance, for DNNs). On the contrary, the use of visualization methods or natural language for the same purpose does not favor a clear quantification of the improvements gained in terms of interpretability. The derivation of general metrics to assess the quality of XAI approaches remain as an open challenge that should be under the spotlight of the field in forthcoming years. We will further discuss on this research direction in Section 5."
%}

<span style="color:#e25252">요약:</span> 어떤 것이 분명하게 이해되었는지는 객관적으로 측정하기 어렵기 때문에, 설명가능성은 인지 심리학영역을 끌어들일 수도 있다. 

<span style="color:#aaa">([주] 주관적인 해석이 섞여 있습니다.)</span> 그러나 모델의 내부가 어느 정도까지 설명될 수 있는지는 객관적으로 측정가능하다. 모델의 복잡성/결과의 단순화 수단들을 XAI 접근법으로 생각할 수 있다. 단순화 정도를 측정하하여, 그 성과를 설명가능성으로 계량하는 것이다. <span style="color:#aaa">(왜? ... 적절한 예시가 떠오르지 않는다...)</span> 하지만 해결되지 않은 근본적인 문제는 이러한 접근 방식으로 제공하는 해석가능성이 직관적으로 정량화하기가 쉽지 않을 수 있다.

예를들어, 모델의 단순화(simplification)는 아키텍쳐를 간소화 하거나 매개변수(parameters) 수를 줄임으로서 달성할 수 있다. 반면, 시각화 벙법들이나 자연어 설명은 해석가능성을 계량하기에 좋은 방법은 아니다.
XAI 방법들의 퀄리티를 평가하기 위한 일반적인 측정지표의 도출은 향후 몇 년간 과제로 남아있다. 이를 Section 5에서 논의 할 것이다.

{% include collaspe-block.html summary="영어원문" article="Explainability is linked to post-hoc explainability since it covers the techniques used to convert a non-interpretable model into a explainable one. In the remaining of this manuscript, explainability will be considered as the main design objective, since it represents a broader concept. A model can be explained, but the interpretability of the model is something that comes from the design of the model itself. Bearing these observations in mind, explainable AI can be defined as follows:
<br>
'Given an audience, an explainable Artificial Intelligence is one that produces details or reasons to make its functioning clear or easy to understand.'
<br>
This definition is posed here as a first contribution of the present overview, implicitly assumes that the ease of understanding and clarity targeted by XAI techniques for the model at hand reverts on different application purposes, such as a better trustworthiness of the model’s output by the audience."
%}

<span style="color:#e25252">요약:</span> 설명가능성(explainability)는 해석이 불가능한 모델을 가능케한다는 점에서 사후(post-hoc) 설명성과 연관이 있다. 이 후의 논문에서는 더 넓은 의미인 설명가능성을 주요 목표로 생각할 것이다. 그러나 모델의 해석가능성(interpretability)은 모델 자체 설계에서 비롯된다. 이러한 생각을 염두해두고, explainable AI는 다음과 같이 정의할 수 있다.

> **설명 가능한 인공지능(explainable Artificial Intelligence)은 청중에게 자신의 기능을 명백하게 혹은 이해하기 쉬운 세부사항/이유를 생산하는 인공지능을 말한다.**

---

## 2.3 Why?

{% include collaspe-block.html summary="영어원문" article="As stated in the introduction, explainability is one of the main barriers AI is facing nowadays in regards to its practical implementation. The inability to explain or to fully understand the reasons by which state-of-the-art ML algorithms perform as well as they do, is a problem that find its roots in two different causes, which are conceptually illustrated in Fig. 2."
%}

<span style="color:#e25252">요약:</span> 도입부에 기술한 바와 같이, 설명가능성은 AI의 실질적 활용에 직면하고 있는 주요 장벽 중 하나이다. 최첨단 ML 알고리즘이 잘 작동하는 이유를 설명하지 못하거나 완전히 이해할 수 없는 것은 두 가지 다른 원인에 그 뿌리를 찾는 문제이며, 이는 개념적으로 `Fig 2`에 나타나 있다. <span style="color:#aaa">([주] 그 원인은 청중이 누구인가에 따라서 ML 알고리즘의 필요한 설명이 다르기 때문이다)</span>

{% include image.html id="1KcJ-gbJw7a8xSghhYs5uM3eefZ9h2vo7" desc="Fig 2. 각기 다른 청중에 따라 달라지는 설명가능성의 목적" width="100%" height="auto" %}

{% include collaspe-block.html summary="영어원문" article="Without a doubt, the first cause is the gap between the research community and business sectors, impeding the full penetration of the newest ML models in sectors that have traditionally lagged behind in the digital transformation of their processes, such as banking, finances, security and health, among many others. In general this issue occurs in strictly regulated sectors with some reluctance to implement techniques that may put at risk their assets."
%}

<span style="color:#e25252">요약:</span> 의심할 여지 없이, 첫 번째 원인은 연구 커뮤니티와 사업 부문 사이의 격차로 인해 은행, 금융, 보안, 건강 등 전통적으로 프로세스의 디지털 전환에서 뒤처진 분야에서 최신 ML 모델의 완전한 보급에 장애가 되고 있다. 일반적으로 이 문제는 엄격하게 규제되는 부문에서 발생하며 자산의 위험을 초래할 수 있는 기법의 시행을 일부 꺼린다.

{% include collaspe-block.html summary="영어원문" article="The second axis is that of knowledge. AI has helped research across the world with the task of inferring relations that were far beyond the human cognitive reach. Every field dealing with huge amounts of reliable data has largely benefited from the adoption of AI and ML techniques. However, we are entering an era in which results and performance metrics are the only interest shown up in research studies. Although for certain disciplines this might be the fair case, science and society are far from being concerned just by performance. The search for understanding is what opens the door for further model improvement and its practical utility.
<br>
The following section develops these ideas further by analyzing the goals motivating the search for explainable AI models."
%}

<span style="color:#e25252">요약:</span> 두번째는 지식을 추구하는 측면이다. AI는 인간의 인지 범위를 벗어난 관계를 추론하는 연구를 도왔다. 막대한 양의 데이터를 다루는 모든 분야는 AI와 ML 기술을 도입함으로서 큰 혜택을 입었다. 그러나 연구 성과 지표에만 관심을 가지는 시대가 접어들 면서 이는 문제가 된다. 성과 지표로만 과학과 사회를 이야기 하기에는 올바르지 않기 때문이다. 이해를 연구한다는 것은 모델을 개선시키고 그 유용성을 증진시키는 일이다.

---

## 2.4 What for?

{% include collaspe-block.html summary="영어원문" article="The research activity around XAI has so far exposed different goals to draw from the achievement of an explainable model. Almost none of the papers reviewed completely agrees in the goals required to describe what an explainable model should compel. However, all these different goals might help discriminate the purpose for which a given exercise of ML explainability is performed. Unfortunately, scarce contributions have attempted to define such goals from a conceptual perspective [5], [13], [24], [30]. We now synthesize and enumerate definitions for these XAI goals, so as to settle a first classification criteria for the full suit of papers covered in this review:"
%}

<span style="color:#e25252">요약:</span> "설명 가능한 모델이 무엇을 강조해야하는가?"라는 목적(혹은 이에대한 합의제시)을 가진 논문은 거의 없었다. 이제부터 분류기준을 정하고, XAI의 목표에 대한 정의를 종합적으로 열거하려고 한다.

{% include collaspe-block.html summary="영어원문" article="Trustworthiness: Several authors agree upon the search for trustworthiness as the primary aim of an explainable AI model [31], [32]. However, declaring a model as explainable as per its capabilities of inducing trust might not be fully compliant with the requirement of model explainability. Trustworthiness might be considered as the confidence of whether a model will act as intended when facing a given problem. Although it should most certainly be a property of any explainable model, it does not imply that every trustworthy model can be considered explainable on its own, nor is trustworthiness a property easy to quantify. Trust might be far from being the only purpose of an explainable model since the relation among the two, if agreed upon, is not reciprocal. Part of the reviewed papers mention the concept of trust when stating their purpose for achieving explainability. However, as seen in Table 1, they do not amount to a large share of the recent contributions related to XAI."
%}

- <span style="color:#e25252">요약:</span> **신뢰도(Trustworthiness):** 몇몇 연구자들은 신뢰도를 설명 가능한 모델의 우선적 목표를 둬야한다고 주장한다([[31]](https://scholar.google.com/scholar_lookup?title=iBCM%3A%20Interactive%20Bayesian%20case%20model%20empowering%20humans%20via%20intuitive%20interaction&publication_year=2015&author=B.%20Kim&author=E.%20Glassman&author=B.%20Johnson&author=J.%20Shah), [[32]](https://scholar.google.com/scholar?q=Why%20should%20I%20trust%20you:%20Explaining%20the%20predictions%20of%20any%20classifier)). 그러나 이러한 주장은 모델 설명성의 요구조건을 완전히 충족하지 못한다. 신뢰도는 모델이 직면한 어떤 문제에서 설계 의도된 바로 행동하는 것으로 간주 할 수 있다. 신뢰도는 설명 가능한 모델의 속성이 되어야 하지만, 모든 신뢰성있는 모델이 설명 가능하지는 않으며, 이 특성을 계량하기 쉽지도 않다. `표1`에서도 확인 할 수 있지만 최근의 연구기여들 중에서 큰 비중을 차지 하지 않는다.

{% include collaspe-block.html summary="영어원문" article="Causality: Another common goal for explainability is that of finding causality among data variables. Several authors argue that explainable models might ease the task of finding relationships that, should they occur, could be tested further for a stronger causal link between the involved variables [159], [160]. The inference of causal relationships from observational data is a field that has been broadly studied over time [161]. As widely acknowledged by the community working on this topic, causality requires a wide frame of prior knowledge to prove that observed effects are causal. A ML model only discovers correlations among the data it learns from, and therefore might not suffice for unveiling a cause-effect relationship. However, causation involves correlation, so an explainable ML model could validate the results provided by causality inference techniques, or provide a first intuition of possible causal relationships within the available data. Again, Table 1 reveals that causality is not among the most important goals if we attend to the amount of papers that state it explicitly as their goal."
%}

- <span style="color:#e25252">요약:</span> **인과성(Causality):** 설명가능성의 다른 목표로는 변수들 간의 인과성을 찾는 것이다. 몇몇 저자들은 이 과정을 용이하게 할 수 있다고 주장한다([[159]](https://scholar.google.com/scholar?q=Smoking%20and%20the%20occurence%20of%20alzheimers%20disease:%20Cross-sectional%20and%20longitudinal%20data%20in%20a%20population-based%20study), [[160]](https://scholar.google.com/scholar?q=An%20empirical%20study%20of%20machine%20learning%20techniques%20for%20affect%20recognition%20in%20humanrobot%20interaction)). 인과 관계의 추론은 상당히 오랜시간 연구되었다([[161]](https://scholar.google.com/scholar_lookup?title=Causality&publication_year=2009&author=J.%20Pearl)). 우리가 관찰한 영향(effects)이 인과성이 있다는 것을 증명하기 위해서, 인과 관계 분야는 광범위한 사전 지식의 프레임을 필요로 한다. 머신러닝 모델은 데이터의 상관 관계를 찾기만 하지, 인과 관계를 충분하게 밝히지는 않는다. 그러나, 인과성은 상관성을 포함하기 때문에, 다양한 기법을 이용해 설명 가능한 머신러닝 모델이 결과에 대해서 검증하거나, 인과관계를 찾아볼 수는 있다. 하지만 `표1`에서 볼 수 있듯이, 논문의 양을 기준으로 한다면, 메인 목표는 아직 아니다.

{% include collaspe-block.html summary="영어원문" article="Transferability: Models are always bounded by constraints that should allow for their seamless transferability. This is the main reason why a training-testing approach is used when dealing with ML problems [162], [163]. Explainability is also an advocate for transferability, since it may ease the task of elucidating the boundaries that might affect a model, allowing for a better understanding and implementation. Similarly, the mere understanding of the inner relations taking place within a model facilitates the ability of a user to reuse this knowledge in another problem. There are cases in which the lack of a proper understanding of the model might drive the user toward incorrect assumptions and fatal consequences [44], [164]. Transferability should also fall between the resulting properties of an explainable model, but again, not every transferable model should be considered as explainable. As observed in Table 1, the amount of papers stating that the ability of rendering a model explainable is to better understand the concepts needed to reuse it or to improve its performance is the second most used reason for pursuing model explainability."
%}

- <span style="color:#e25252">요약:</span> **전이가능성(Transferability):** 모델은 원활한 전이가능성을 가지기 위해서 일반화가 잘 되어야 한다. 그래서 training-testing 방법을 사용해서 훈련하는 것이다([[162]](https://scholar.google.com/scholar_lookup?title=Applied%20predictive%20modeling&publication_year=2013&author=M.%20Kuhn&author=K.%20Johnson), [[163]](https://scholar.google.com/scholar_lookup?title=An%20introduction%20to%20statistical%20learning&publication_year=2013&author=G.%20James&author=D.%20Witten&author=T.%20Hastie&author=R.%20Tibshirani)). 설명가능성도 전이가능성이 필요한데, 모델에 영향을 미칠 수 있는 경계를 확장 하면서 더 나은 이해와 구현을 가능하게 하기 때문이고, 내부관계의 이해를 바탕으로 사용자가 다른 문제에서 이 지식을 재사용 할 수 있기 때문이다. 다만, 모델에 대한 이해가 부족하여 잘못된 가정과 치명적인 결과를 초래할 경우도 있다([[44]](https://scholar.google.com/scholar_lookup?title=Intelligible%20models%20for%20healthcare%3A%20Predicting%20pneumonia%20risk%20and%20hospital%2030-day%20readmission&publication_year=2015&author=R.%20Caruana&author=Y.%20Lou&author=J.%20Gehrke&author=P.%20Koch&author=M.%20Sturm&author=N.%20Elhadad), [[164]](https://scholar.google.com/scholar?q=C.%20Szegedy,%20W.%20Zaremba,%20I.%20Sutskever,%20J.%20Bruna,%20D.%20Erhan,%20I.%20Goodfellow,%20R.%20Fergus,%20Intriguing%20properties%20of%20neural%20networks,%202013.)). 또한, 전이가능성은 설명 가능한 모델의 특성이지만, 모든 전이가능한 모델이 설명가능하지는 않다. `표1`에서 볼수 있듯이, 모델의 전이가능성은 (연구기여의 양적으로 따졌을 때) 모델의 설명성을 추구하는 2번째 이유가 되며, 설명가능성은 모델을 재사용 하기 위해 필요한 개념을 더 잘 이해하고, 모델 성능을 향상시키기 위해 사용될 수 있다.

{% include collaspe-block.html summary="영어원문" article="Informativeness: ML models are used with the ultimate intention of supporting decision making [92]. However, it should not be forgotten that the problem being solved by the model is not equal to that being faced by its human counterpart. Hence, a great deal of information is needed in order to be able to relate the user's decision to the solution given by the model, and to avoid falling in misconception pitfalls. For this purpose, explainable ML models should give information about the problem being tackled. Most of the reasons found among the papers reviewed is that of extracting information about the inner relations of a model. Almost all rule extraction techniques substantiate their approach on the search for a simpler understanding of what the model internally does, stating that the knowledge (information) can be expressed in these simpler proxies that they consider explaining the antecedent. This is the most used argument found among the reviewed papers to back up what they expect from reaching explainable models."
%}

- <span style="color:#e25252">요약:</span> **정보성(Informativeness):** 머신러닝 모델의 궁극적인 목적은 의사 결정의 지원이다([[92]](https://scholar.google.com/scholar_lookup?title=An%20empirical%20evaluation%20of%20the%20comprehensibility%20of%20decision%20table%2C%20tree%20and%20rule%20based%20predictive%20models&publication_year=2011&author=J.%20Huysmans&author=K.%20Dejaeger&author=C.%20Mues&author=J.%20Vanthienen&author=B.%20Baesens)). 그러나 모델이 풀고 있는 문제는 인간이 직면하고 있는 문제와 항상 같은 것은 아니다. ([주] 아직까지 모델은 더 단순한 문제를 해결하고 있기 때문, 아직 복합적인 정보를 결합하여 문제를 해결하지는 못한다.) 따라서 사용자의 의사결정에 모델이 내놓은 솔루션을 오해하지 않게 잘 연관 시키려면 더 많은 양의 정보가 필요하다. 이를 위해, 설명 가능한 모델은 문제에 태클이 될 만한 정보를 더 많이 제공해야한다. 대부분 논문을 살펴본 결과, 그 이유는 모델의 내부 관계에서 정보를 추출하기 위함이었다. 대부분의 규칙기반 기술을 사용하려는 사람들은 모델 행동의 이해를 더 간단하게 만들기 위해 방법론을 연구하고 있었다. 그들은 모델의 지식이나 정보를 더 간단하게 대체할 수 있을 것이라 주장했다. 이는 설명 가능한 모델에서 가장 많은 기대를 하는 특성이다.

{% include collaspe-block.html summary="영어원문" article="Confidence: As a generalization of robustness and stability, confidence should always be assessed on a model in which reliability is expected. The methods to maintain confidence under control are different depending on the model. As stated in [165], [166], [167], stability is a must-have when drawing interpretations from a certain model. Trustworthy interpretations should not be produced by models that are not stable. Hence, an explainable model should contain information about the confidence of its working regime."
%}

- <span style="color:#e25252">요약:</span> **확신(Confidence):** 건전성(robustness)와 안전성(stability)의 일반화로서, 신뢰성이 기대되는 모델은 그 확신의 정도를 가늠할 수 있어야 한다. 확신을 유지하는 방법은 모델에 따라서 다르다. [[165]](https://scholar.google.com/scholar_lookup?title=Robust%20statistics%3A%20The%20approach%20based%20on%20influence%20functions&publication_year=1987&author=D.%20Ruppert), [[166]](https://scholar.google.com/scholar_lookup?title=Iterative%20random%20forests%20to%20discover%20predictive%20and%20stable%20high-order%20interactions&publication_year=2018&author=S.%20Basu&author=K.%20Kumbier&author=J.B.%20Brown&author=B.%20Yu), [[167]](https://scholar.google.com/scholar_lookup?title=Stability&publication_year=2013&author=B.%20Yu)에서 기술한 바와 같이, 안정성은 어떤 모델에서 해석을 도출하기 위해서 꼭 필요한 특성이다. ([주] 일단 모델이 일반화가 잘 되어 있어야 그 해석 또한 안정적으로 도출 할 수 있다.)

{% include collaspe-block.html summary="영어원문" article="Fairness: From a social standpoint, explainability can be considered as the capacity to reach and guarantee fairness in ML models. In a certain literature strand, an explainable ML model suggests a clear visualization of the relations affecting a result, allowing for a fairness or ethical analysis of the model at hand [3], [100]. Likewise, a related objective of XAI is highlighting bias in the data a model was exposed to [168], [169]. The support of algorithms and models is growing fast in fields that involve human lives, hence explainability should be considered as a bridge to avoid the unfair or unethical use of algorithm’s outputs."
%}

- <span style="color:#e25252">요약:</span> **공정성(Fairness):** 사회적 관점에서 설명가능성은 머신러닝의 공정성을 보장하는 능력으로 볼 수 있다. 특정 문헌에서 설명 가능한 모델은 명백한 결과의 관계 시각화를 통해 공정성 혹은 윤리적 분석을 가능하게 한다([[3]](https://scholar.google.com/scholar?q=European%20union%20regulations%20on%20algorithmic%20decision-making%20and%20a%20right%20to%20explanation), [[100]](https://scholar.google.com/scholar_lookup?title=Fair%20prediction%20with%20disparate%20impact%3A%20A%20study%20of%20bias%20in%20recidivism%20prediction%20instruments&publication_year=2017&author=A.%20Chouldechova)). 이와 관련된 목표로는 [[168]](https://scholar.google.com/scholar?q=K.%20Burns,%20L.A.%20Hendricks,%20K.%20Saenko,%20T.%20Darrell,%20A.%20Rohrbach,%20Women%20also%20Snowboard:%20Overcoming%20Bias%20in%20Captioning%20Models,%202018.), [[169]](https://scholar.google.com/scholar_lookup?title=Towards%20explainable%20neural-symbolic%20visual%20reasoning&publication_year=2019&author=A.%20Bennetot&author=J.-L.%20Laurent&author=R.%20Chatila&author=N.%20D%C3%ADaz-Rodr%C3%ADguez)에서 보여준 바와 같이, 모델이 학습한 데이터의 편향을 강조하는 것이다. 사람들의 삶에서 머신러닝 모델(알고리즘)은 앞으로 계속 빠르게 노출될 수 밖에 없기에, 설명가능성은 이러한 불공평과 비윤리적인 알고리즘의 산출물을 피할 수 있게 하는 가교역할이 되어야 한다. ([주] 최근 [이루다](https://blog.pingpong.us/luda-issue-faq/?fbclid=IwAR15-eiWeIPSnv8lT0WXlO07HBpP0aJaoN36vThaGDmIaBeZpU6jiIy_oJw) 일은 신뢰도에도 속하지만, 이 범주에도 속한다고 할 수 있겠다.)

{% include collaspe-block.html summary="영어원문" article="Accessibility: A minor subset of the reviewed contributions argues for explainability as the property that allows end users to get more involved in the process of improving and developing a certain ML model [37], [86]. It seems clear that explainable models will ease the burden felt by non-technical or non-expert users when having to deal with algorithms that seem incomprehensible at first sight. This concept is expressed as the third most considered goal among the surveyed literature."
%}

- <span style="color:#e25252">요약:</span> **접근성(Accessibility):** 일부 연구자들은 설명가능성은 최종 사용자가 특정 모델을 개발하고 개선하는 프로세스에 더 많이 관여할 수 있는 속성이라고 주장한다([[38]](https://scholar.google.com/scholar_lookup?title=Working%20with%20beliefs%3A%20AI%20transparency%20in%20the%20enterprise.&publication_year=2018&author=A.%20Chander&author=R.%20Srinivasan&author=S.%20Chelian&author=J.%20Wang&author=K.%20Uchino), [[86]](https://scholar.google.com/scholar_lookup?title=Explainable%20AI%3A%20Beware%20of%20inmates%20running%20the%20asylum&publication_year=2017&author=T.%20Miller&author=P.%20Howe&author=L.%20Sonenberg)). 설명 가능한 모델은 비전문가 사용자에게 처음 보는 알 수 없는 알고리즘에 대한 부담을 덜어 줄 수 있어보인다. 이는 이번 조사에서 3번째로 많이 고려된 목표다.

{% include collaspe-block.html summary="영어원문" article="Interactivity: Some contributions [50], [59] include the ability of a model to be interactive with the user as one of the goals targeted by an explainable ML model. Once again, this goal is related to fields in which the end users are of great importance, and their ability to tweak and interact with the models is what ensures success."
%}

- <span style="color:#e25252">요약:</span> **상호작용(Interactivity):** 특정 논문([50], [59])에서는 사용자와 모델이 상호작용하는 능력을 설명 가능한 모델의 목표로 잡았다. 이는 최종 사용자와 더 관련이 있으며, 이들의 모델을 수정하고 상호작용하는 능력이 목표달성의 성공을 보장한다.

{% include collaspe-block.html summary="영어원문" article="Privacy awareness: Almost forgotten in the reviewed literature, one of the byproducts enabled by explainability in ML models is its ability to assess privacy. ML models may have complex representations of their learned patterns. Not being able to understand what has been captured by the model [4] and stored in its internal representation may entail a privacy breach. Contrarily, the ability to explain the inner relations of a trained model by non-authorized third parties may also compromise the differential privacy of the data origin. Due to its criticality in sectors where XAI is foreseen to play a crucial role, confidentiality and privacy issues will be covered further in Sections 5.4 and 6.3, respectively."
%}

- <span style="color:#e25252">요약:</span>**프라이버시 인식(Privacy awareness):** 대부분의 논문에서 언급되지 않았지만 설명 가능한 모델의 부산물 중에 하나는 프라이버시를 평가하는 능력이다. 머신러닝에서 학습한 패턴은 복잡한 표현(representations)([주] 여기서 "표현"이란 데이터 혹은 패턴을 압축하여 나타내는 어떤 상태다.)으로 나타낼 수 있다. 모델 내부에 어떤 것이 포착되어 있는지 알 수 없다면, 개인정보의 침해가 일어날 수 있다. 반대로, 비인가 제3자에 의해 훈련된 모델의 내부를 설명할 수 있다면, 그것 또한 다른 의미로써 데이터 출처에 대한 프라이버시 침해라고 할 수 있다. 사안의 중요성 때문에, 이 기밀성(confidentiality)와 개인정보 문제는 section 5.4와 6.3에서 더 다룰 예정이다.

|  XAI Goal | Main target audience (Fig. 2) | References | 
| --- | --- | --- | 
| Trustworthiness | Domain experts, users of the model affected by decisions | [5], [10], [24], [32], [33], [34], [35], [36], [37]
Causality | Domain experts, managers and executive board members, regulatory entities/agencies | [35], [38], [39], [40], [41], [42], [43] | 
Transferability | Domain experts, data scientists | [5], [21], [26], [30], [32], [37], [38], [39], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63], [64], [65], [66], [67], [68], [69], [70], [71], [72], [73], [74], [75], [76], [77], [78], [79], [80], [81], [82], [83], [84], [85]  | 
| Informativeness | All | [5], [21], [25], [26], [30], [32], [34], [35], [37], [38], [41], [44], [45], [46], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [59], [60], [63], [64], [65], [66], [68], [69], [70], [71], [72], [73], [74], [75], [76], [77], [78], [79], [86], [87], [88], [89], [90], [91], [92], [93], [94], [95], [96], [97], [98], [99], [100], [101], [102], [103], [104], [105], [106], [107], [108], [109], [110], [111], [112], [113], [114], [115], [116], [117], [118], [119], [120], [121], [122], [123], [124], [125], [126], [127], [128], [129], [130], [131], [132], [133], [134], [135], [136], [137], [138], [139], [140], [141], [142], [143], [144], [145], [146], [147], [148], [149], [150], [151], [152], [153], [154] | 
| Confidence | Domain experts, developers, managers, regulatory entities/agencies | [5], [35], [45], [46], [48], [54], [61], [72], [88], [89], [96], [108], [117], [119], [155] | 
| Fairness | Users affected by model decisions, regulatory entities/agencies | [5], [24], [35], [45], [47], [99], [100], [101], [120], [121], [128], [156], [157], [158] | 
| Accessibility | Product owners, managers, users affected by model decisions | [21], [26], [30], [32], [37], [50], [53], [55], [62], [67], [68], [69], [70], [71], [74], [75], [76], [86], [93], [94], [103], [105], [107], [108], [111], [112], [113], [114], [115], [124], [129] | 
| Interactivity | Domain experts, users affected by model decisions | [37], [50], [59], [65], [67], [74], [86], [124] | 
| Privacy awareness | Users affected by model decisions, regulatory entities/agencies | [89] |

`표 1` 설명가능성에 도달하기 위해 검토된 문헌에서 추구된 목표들과 그들의 주요 목표 청중들.

---

## 2.5 How?

{% include collaspe-block.html summary="영어원문" article="The literature makes a clear distinction among models that are interpretable by design, and those that can be explained by means of external XAI techniques. This duality could also be regarded as the difference between interpretable models and model interpretability techniques; a more widely accepted classification is that of transparent models and post-hoc explainability. This same duality also appears in the paper presented in [17] in which the distinction its authors make refers to the methods to solve the transparent box design problem against the problem of explaining the black-box problem. This work, further extends the distinction made among transparent models including the different levels of transparency considered."
%}

<span style="color:#e25252">요약:</span> 본 논문은 해석 가능한 모델을 **해석 가능한 모델**(구조적으로 설명 가능한 부분)과 **해석 가능한 기법**(외부 XAI 기법에 의해 설명될 수 있는 부분)으로 나눈다. 현재 보다 널리 받아들여지고 있는 분류법 용어는 투명한 모델(transparent models)과 사후 설명가능성(post-hoc explainability)이다. [17]에서도 언급하는데, 블랙박스 문제를 설명하려면 투명한 모델의 디자인 문제를 참고해야한다. 본 논문에서는 투명한 모델을 다양한 단계로 세분화해서 그 차이를 알아 볼 것이다.

{% include collaspe-block.html summary="영어원문" article="Within transparency, three levels are contemplated: algorithmic transparency, decomposability and simulatability Among post-hoc techniques we may distinguish among text explanations, visualizations, local explanations, explanations by example, explanations by simplification and feature relevance. In this context, there is a broader distinction proposed by [24] discerning between 1) opaque systems, where the mappings from input to output are invisible to the user; 2) interpretable systems, in which users can mathematically analyze the mappings; and 3) comprehensible systems, in which the models should output symbols or rules along with their specific output to aid in the understanding process of the rationale behind the mappings being made. This last classification criterion could be considered included within the one proposed earlier, hence this paper will attempt at following the more specific one."
%}

<span style="color:#e25252">요약:</span> 투명성(transparency)은 알고리즘 투명성(algorithmic transparency), 분해가능성(decomposability) 그리고 시뮬레이션성(simulatability) <span style="color:#aaa">([주] 시스템 혹은 프로세스의 시뮬레이션 능력, the capacity of a system or process to be simulated, 아직 어떤 느낌이 안온다.)</span>순으로 3 가지 단계를 고려해야한다. 사후 분석(post-hoc) 기법은 텍스트 설명, 시각화, 국지적 설명, 예시 설명, 단순화 및 피처중요도등 방법과 구별되어야 한다. 이러한 맥락에서 더 광범위한 구별법이 [[24]](https://scholar.google.com/scholar?q=D.%20Doran,%20S.%20Schulz,%20T.R.%20Besold,%20What%20does%20explainable%20AI%20really%20mean%20a%20new%20conceptualization%20of%20perspectives,%202017.)에서 제시 되었다. 1) 입력에서 출력까지의 매핑이 사용자에게 보이지 않는 불투명한(opaque) 시스템 2) 사용자가 수학적으로 매핑을 분석할 수 있는 해석 가능한(interpretable) 시스템 3) 모델이 결정한 매핑의 이유를 사람이 이해 가능하게 출력하는 포괄적 이해 가능한(comprehensible) 시스템

### 2.5.1. Levels of transparency in machine learning models

{% include collaspe-block.html summary="영어원문" article="Transparent models convey some degree of interpretability by themselves. Models belonging to this category can be also approached in terms of the domain in which they are interpretable, namely, algorithmic transparency, decomposability and simulatability. As we elaborate next in connection to Fig. 3, each of these classes contains its predecessors, e.g. a simulatable model is at the same time a model that is decomposable and algorithmically transparent:"
%}

<span style="color:#e25252">요약:</span> 투명한 모델은 모델 그 자체러 어느 정도의 해석가능성을 가지고 있다. 위에 언급한 대로 알고리즘 투명성, 분해가능성 그리고 시뮬레이션성 순으로 접근 할 수 있다. Fig 3에서 설명하겠지만, 각 분류는 이전 단계의 구성을 포함한다.. 예를 들어, 시뮬레이션 가능한 모델은 분해 가능하며, 투명한 알고리즘을 포함한다.

{% include collaspe-block.html summary="영어원문" article="Simulatability denotes the ability of a model of being simulated or thought about strictly by a human, hence complexity takes a dominant place in this class. This being said, simple but extensive (i.e., with too large amount of rules) rule based systems fall out of this characteristic, whereas a single perceptron neural network falls within. This aspect aligns with the claim that sparse linear models are more interpretable than dense ones [170], and that an interpretable model is one that can be easily presented to a human by means of text and visualizations [32]. Again, endowing a decomposable model with simulatability requires that the model has to be self-contained enough for a human to think and reason about it as a whole."
%}

- <span style="color:#e25252">요약:</span> **시뮬레이션성(Simulatability):** 사람에 의해 엄격하게 시뮬레이션된 모델의 능력이다. 따라서 복잡성이 가장 중요하다. 단순하지만 광범위한 규칙기반 시스템 보다는 퍼셉트론 신경망이 기준에 더 부합한다. 이러한 관점에서 sparse한 선형모델이 dense 한 것보다 더 해석가능성이 높으며 [[170]](https://scholar.google.com/scholar_lookup?title=Regression%20shrinkage%20and%20selection%20via%20the%20lasso&publication_year=1996&author=R.%20Tibshirani), 설명 가능한 모델의 텍스트와 시각화를 통해 인간이 더 쉽게 설명 할 수 있다라는 주장[[32]](https://scholar.google.com/scholar?q=Why%20should%20I%20trust%20you:%20Explaining%20the%20predictions%20of%20any%20classifier)과 일치한다. 다시 말하지만 시뮬레이션성을 가지는 분해 가능한 모델은 인간에게 생각과 이유를 혼자 설명할 수 있어야 한다.

{% include collaspe-block.html summary="영어원문" article="Decomposability stands for the ability to explain each of the parts of a model (input, parameter and calculation). It can be considered as intelligibility as stated in [171]. This characteristic might empower the ability to understand, interpret or explain the behavior of a model. However, as occurs with algorithmic transparency, not every model can fulfill this property. Decomposability requires every input to be readily interpretable (e.g. cumbersome features will not fit the premise). The added constraint for an algorithmically transparent model to become decomposable is that every part of the model must be understandable by a human without the need for additional tools."
%}

- <span style="color:#e25252">요약:</span> **분해가능성(Decomposability):** 모델의 각 부분을 설명하는 능력인데, [[171]](https://scholar.google.com/scholar_lookup?title=Intelligible%20models%20for%20classification%20and%20regression&publication_year=2012&author=Y.%20Lou&author=R.%20Caruana&author=J.%20Gehrke)에서 언급된 명료성(intelligibility)으로 볼 수 있다. 이 특성은 모델의 행동을 이해하고, 해석하거나 설명하는 능력을 강조한다. 그러나 모든 모델이 이 특성이 있는 것은 아니다. 분해가능성은 모든 입력의 쉽게 해석할 수 있어야 하는데, 알고리즘적 투명성까지 만족하기 위해서는 인간이 모델의 모든 부분에서 추가해석 없이 이해할 수 있어야 한다.

{% include collaspe-block.html summary="영어원문" article="Algorithmic transparency can be seen in different ways. It deals with the ability of the user to understand the process followed by the model to produce any given output from its input data. Put it differently, a linear model is deemed transparent because its error surface can be understood and reasoned about, allowing the user to understand how the model will act in every situation it may face [163]. Contrarily, it is not possible to understand it in deep architectures as the loss landscape might be opaque [172], [173] since it cannot be fully observed and the solution has to be approximated through heuristic optimization (e.g. through stochastic gradient descent). The main constraint for algorithmically transparent models is that the model has to be fully explorable by means of mathematical analysis and methods."
%}

- <span style="color:#e25252">요약:</span> **알고리즘적 투명성(Algorithmic transparency):** 사용자가 모델에 대한 프로세스(모델이 도출한 입력 데이터에 대한 출력) 이해 능력을 나타낸다. 선형모델은 사용자가 모델이 어떻게 행동할 지 예측할 수 있고, error surface를 이해하고 설명할 수 있기 때문에 투명하다고 볼 수 있다[[163]](https://scholar.google.com/scholar_lookup?title=An%20introduction%20to%20statistical%20learning&publication_year=2013&author=G.%20James&author=D.%20Witten&author=T.%20Hastie&author=R.%20Tibshirani). 반대로, 깊은 모델 구조를 가지는 모델은 손실값이 불투명하여([[172]](https://scholar.google.com/scholar_lookup?title=Deep%20learning%20without%20poor%20local%20minima&publication_year=2016&author=K.%20Kawaguchi), [[173]](https://scholar.google.com/scholar_lookup?title=Algorithmic%20transparency%20via%20quantitative%20input%20influence%3A%20Theory%20and%20experiments%20with%20learning%20systems&publication_year=2016&author=A.%20Datta&author=S.%20Sen&author=Y.%20Zick)), 특정 휴리스틱한 최적화(예, stochastic gradient descent)를 통해서 근사치를 구해야한다. <span style="color:#aaa">([주] 수학적으로 명쾌한 solution이 안보이면 불투명하다고 보는 것 같다. 특히 비선형함수의 손실값)</span> 알고리즘적 투명성의 주된 제약 조건은 모델이 수학적 분석과 방법을 통해 완전이 탐구 가능해야 한다는 것이다.

{% include image.html id="1iZXqf9hnwcu-N9If2_R4WWF5RinPIhPX" desc="Fig 3. 다양한 단계의 투명성" width="100%" height="auto" %}

<span style="color:#e25252">요약:</span> 다양한 단계의 투명성을 설명해두었다. 머신러닝 모델은 $M_{\varphi}$, 그에 해당하는 파라미터는 $\varphi$ 로 표기했다. (a) 시뮬레이션성 (b) 분해가능성 (c) 알고리즘적 투명성. 각 예제는 일반적인 손실값 가정을 고려하지 않은 선에서, 모델이 설명 대상에 따라서 얼마나 달라지는지를 보여준다.

<span style="color:#aaa"> [주] 이 파트가 제일 이해하기 어려웠는데, (a) 같은 경우 입력 데이터는 잘 몰라도, 내부의 구조를 그대로 재현 가능하면 만족하는 것 같고, (b) 의 경우 사람이 각 입력 피처까지 이해할 수 있어야 한다. (c)의 경우 사람이 전체 데이터의 특성까지 파악하고 이에 대한 규칙을 이해해야 한다라고 이해했다. </span>

### 2.5.2. Post-hoc explainability techniques for machine learning models

{% include collaspe-block.html summary="영어원문" article="Post-hoc explainability targets models that are not readily interpretable by design by resorting to diverse means to enhance their interpretability, such as text explanations, visual explanations, local explanations, explanations by example, explanations by simplification and feature relevance explanations techniques. Each of these techniques covers one of the most common ways humans explain systems and processes by themselves.
<br>
Further along this river, actual techniques, or better put, actual group of techniques are specified to ease the future work of any researcher that intends to look up for an specific technique that suits its knowledge. Not ending there, the classification also includes the type of data in which the techniques has been applied. Note that many techniques might be suitable for many different types of data, although the categorization only considers the type used by the authors that proposed such technique. Overall, post-hoc explainability techniques are divided first by the intention of the author (explanation technique e.g. Explanation by simplification), then, by the method utilized (actual technique e.g. sensitivity analysis) and finally by the type of data in which it was applied (e.g. images)."
%}

<span style="color:#e25252">요약:</span> **사후 설명가능성(post-hoc explainability)**은 쉽게 해석할 수 없는 모델을 위해 고안된 방법이다. 텍스트, 시각화, 부분, 예시, 단순화 그리고 피처 연관 설명 등 방법들이 있다. 

각 방법들에 대해 특정 기술 뿐만 아니라 적용되는 데이터 유형까지 소개한다. 여기서는 인용한 저자들이 적용한 데이터에 의거해 분류를 했지만, 이 중에 어떤 방법들은 다른 분야(데이터)에도 충분히 적용할 수 있다.

{% include collaspe-block.html summary="영어원문" article="Text explanations deal with the problem of bringing explainability for a model by means of learning to generate text explanations that help explaining the results from the model [169]. Text explanations also include every method generating symbols that represent the functioning of the model. These symbols may portrait the rationale of the algorithm by means of a semantic mapping from model to symbols."
%}

- <span style="color:#e25252">요약:</span> **텍스트 설명(Text explanation)**은 모델이 자신의 결과를 설명하는 "텍스트 생성 학습" 문제다. 이 방법은 모델의 기능을 나타내는 심볼을 생성([주] 인간의 언어가 될 수도 있고, 수식일 수도)하는 방식인데, 이 심볼들은 의미론적(semantic)으로 알고리즘의 작동 방식을 매핑한다.

{% include collaspe-block.html summary="영어원문" article="Visual explanation techniques for post-hoc explainability aim at visualizing the model's behavior. Many of the visualization methods existing in the literature come along with dimensionality reduction techniques that allow for a human interpretable simple visualization. Visualizations may be coupled with other techniques to improve their understanding, and are considered as the most suitable way to introduce complex interactions within the variables involved in the model to users not acquainted to ML modeling."
%}

- <span style="color:#e25252">요약:</span> **시각적 설명(Visual explanation)**의 목표는 모델의 행동을 시각적으로 설명한는 것이다. 많은 방법들 중에서 대부분 인간이 해석하기 쉽게 차원감소(dimension reduction) 기법과 함께 사용된다. 이 방법은 다른 방법들과 함께 사용되서 이해를 향상시킬 수 있다. 머신러닝 모델링에 익숙하지 않은 사용자에게 모델과 관련된 변수의 복잡한 상호작용을 알리는데 있어 가장 적합한 도구다.

{% include collaspe-block.html summary="영어원문" article="Local explanations tackle explainability by segmenting the solution space and giving explanations to less complex solution subspaces that are relevant for the whole model. These explanations can be formed by means of techniques with the differentiating property that these only explain part of the whole system’s functioning."
%}

- <span style="color:#e25252">요약:</span> **부분 설명(Local explanation)**은 전체모델의 일부분을 쉽게 설명하는데 집중한다.

{% include collaspe-block.html summary="영어원문" article="Explanations by example consider the extraction of data examples that relate to the result generated by a certain model, enabling to get a better understanding of the model itself. Similarly to how humans behave when attempting to explain a given process, explanations by example are mainly centered in extracting representative examples that grasp the inner relationships and correlations found by the model being analyzed."
%}

- <span style="color:#e25252">요약:</span> **예시 설명(Example explanation)**은 데이터 샘플에서 결과와 관련된 예제를 추출하는 방법이다. 주로 모델 결과의 내적 관계 혹은 상관관계에 관련된 예제를 추출하게 된다. 이 방법은 사람이 어떤 프로세스를 설명하려고 할 때랑 비슷하게 행동한다.

{% include collaspe-block.html summary="영어원문" article="Explanations by simplification collectively denote those techniques in which a whole new system is rebuilt based on the trained model to be explained. This new, simplified model usually attempts at optimizing its resemblance to its antecedent functioning, while reducing its complexity, and keeping a similar performance score. An interesting byproduct of this family of post-hoc techniques is that the simplified model is, in general, easier to be implemented due to its reduced complexity with respect to the model it represents."
%}

- <span style="color:#e25252">요약:</span> **단순화 설명(Simplification explanation)**은 훈련된 모델에 기초하여 설명을 위한 새로운 시스템을 만들어내는 방법이다. 이 새로운 시스템은 복잡성을 최대한 줄이고, 유사한 기능 및 성능을 유지하는 것이 중요하다. 이 방법은 기존에 복잡한 모델에 반해, 더 쉽게 구현할 수 있다는 것이 장점이 있다.

{% include collaspe-block.html summary="영어원문" article="Finally, feature relevance explanation methods for post-hoc explainability clarify the inner functioning of a model by computing a relevance score for its managed variables. These scores quantify the affection (sensitivity) a feature has upon the output of the model. A comparison of the scores among different variables unveils the importance granted by the model to each of such variables when producing its output. Feature relevance methods can be thought to be an indirect method to explain a model."
%}

- <span style="color:#e25252">요약:</span> **피처 연관 설명(Feature relevance explanation)**은 모델의 변수와 관련된 점수를 계산하는 방식으로 구현된다. 이 점수는 피처의 영향력(affection 혹은 sensitivity)를 계량화한다. 출력에 대한 점수가 각기 다르기 때문에 각 피처의 중요도를 확인할 수 있다. 이는 모델을 설명하는 간접적인 방법이다.

{% include collaspe-block.html summary="영어원문" article="The above classification (portrayed graphically in Fig. 4) will be used when reviewing specific/agnostic XAI techniques for ML models in the following sections (Table 2). For each ML model, a distinction of the propositions to each of these categories is presented in order to pose an overall image of the field’s trends."
%}

<span style="color:#e25252">요약:</span> 위 분류를 Fig 4로 표현했으며, 표 2에서도 정리했다. 

{% include image.html id="1CDEwm6jcM8SKvE3YT21n9HlTTPq2Orol" desc="Fig 4. 사후 설명가능성 방법에 대한 컨셉 다이어그램" width="100%" height="auto" %}

---

| | Transparent ML Models | Transparent ML Models |Transparent ML Models | Post-hoc analysis | 
| Model | Simulatability | Decomposability | Algorithmic Transparency | Post-hoc analysis | 
| --- | --- | --- | --- | --- |
| Linear/Logistic Regression | 예측 변수는 사람이 판독할 수 있으며 예측 변수 간의 상호 작용은 최소한으로 유지됨 | 변수는 여전히 읽을 수 있지만, 변수와 관련된 상호작용과 예측 변수의 수는 분해를 강요하는 수준으로 증가했다. | 변수와 교호작용이 너무 복잡하여 수학적 도구가 없으면 분석할 수 없음 | 필요 없음 |
| Decision Trees | 사람은 어떤 수학적 배경도 요구하지 않고 스스로 의사결정 나무의 예측을 시뮬레이션하고 얻을 수 있다. | 데이터가 어떻든 모델은 규칙을 전혀 변경하지 않고 가독성을 유지함 | 데이터로부터 학습한 지식을 사람이 이해 할 만한 규칙으로 설명하고, 예측 프로세스를 직관적으로 알 수 있음 | 필요 없음 |
| K-Nearest Neighbors | 인간의 나이브한 능력에 따라서 모델 복잡도가 결정된다(변수의 개수, 변수간 유사성의 이해도) | 변수의 양이 너무 많거나 유사성 측도가 너무 복잡하여 모형을 완전히 시뮬레이션할 수 없지만 유사성 측도와 변수 집합을 별도로 분해하여 분석할 수 있다.	 | 유사성 측정은 분해될 수 없으며/또는 변수 수가 너무 많아서 사용자는 모델을 분석하기 위해 수학적 및 통계적 도구에 의존해야함 | 필요 없음 |
| Rule Based Learners | 규칙에 포함된 변수는 읽을 수 있으며 규칙 집합의 크기는 외부 도움 없이 사용자가 관리할 수 있음 | 규칙 집합의 크기가 너무 커서 작은 규칙 청크로 분해하지 않고 분석할 수 없음 | 규칙이 너무 복잡해져서(그리고 규칙 집합 크기가 너무 커져서) 모델 동작을 검사하는 데 수학적 도구가 필요함 | 필요 없음 |
| General Additive Models | 모델에 포함된 원활한 기능에 따라 변수 및 변수 간의 상호 작용은 인간이 이해할 수 있는 범위에 제한되어야 함 | 교호작용이 너무 복잡해져서 시뮬레이션할 수 없으므로 모델을 분석하려면 분해 기법이 필요함 | 복잡성 때문에, 변수와 상호작용은 수학적, 통계적 도구를 적용하지 않고는 분석할 수 없음 | 필요 없음 |
| Bayesian Models | 변수 및 변수 자체의 통계적 관계는 청중들이 직접 이해할 수 있어야 함 | 통계적 관계가 너무 많이 포함되어 있어서, 분해를 해야 분석이 용이함 | 통계적 관계는 이미 분해되어도 해석할 수 없는 수준이고 복잡한 수학적 도구를 사용해야 모델을 분석할 수 있음 | 필요 없음 |
| Tree Ensembles | ✗ | ✗ | ✗ | 필요: 모델 단순화 혹은 피처 연관 설명 |
| Support Vector Machines | ✗ | ✗ | ✗ | 필요: 모델 단순화 혹은 부분 설명 |
| Multi–layer Neural Network | ✗ | ✗ | ✗ | 필요: 모델 단순화, 피처 연관 설명 혹은 시각화 설명 |
| Convolutional Neural Network | ✗ | ✗ | ✗ | 필요: 모델 단순화, 피처 연관 설명 혹은 시각화 설명 |
| Recurrent Neural Network | ✗ | ✗ | ✗ | 필요: 피처 연관 설명 |

`표 2` 설명가능성 수준에 따른 ML 모델의 분류에 대한 전체적인 그림.