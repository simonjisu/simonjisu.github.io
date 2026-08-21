---
title: "Information Theory"
description: "정보이론의 기초부터 언어 모델의 손실 함수까지 공부하는 한글 학습 노트"
hide:
  - toc
---

# Information Theory

정보이론(information theory)을 처음 접하는 독자가 확률(probability), 정보량(information), 엔트로피(entropy), 압축(compression)과 통신(communication)을 거쳐 언어 모델의 학습 목적까지 이해할 수 있도록 정리한 한글 학습 노트다.

수학 정의부터 외우기보다 생활 속 질문, 작은 손계산, Python 실험, LLM 예시의 순서로 공부한다. 전체 학습 순서와 주차별 결과물은 커리큘럼에서 확인할 수 있다.

## 학습 순서

| 순서 | 내용 | 문서 |
|---:|---|---|
| 0 | 12주 핵심 과정과 4주 LLM 심화 과정 | [Information Theory 학습 커리큘럼](curriculum.md) |
| 1 | 스무고개, 자기정보량(self-information), 비트(bit)와 내트(nat), 토큰 손실(token loss) | [Chapter 1. 정보란 무엇인가](c01_introduction.md) |
| 2 | 확률변수(random variable), 결합·주변·조건부확률, 독립(independence), 베이즈 정리(Bayes' theorem) | [Chapter 2. 확률은 불확실성을 표현하는 언어다](c02_probability.md) |
| 3 | 샤논 엔트로피(Shannon entropy), 베르누이 엔트로피, 최대 엔트로피, 다음 토큰 엔트로피 | [Chapter 3. 엔트로피: 평균적으로 얼마나 놀라운가](c03_entropy.md) |

## 참고 자료

전체 구성에는 David J. C. MacKay의 *Information Theory, Inference, and Learning Algorithms*를 참고하되, 한글 원고의 설명과 예시는 학습 목표에 맞게 새로 구성한다.[^1]

[^1]: [Information Theory, Inference, and Learning Algorithms](https://www.inference.org.uk/itprnn/book.pdf)
