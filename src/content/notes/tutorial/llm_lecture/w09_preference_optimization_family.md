---
title: "9주차. DPO 계열 비교"
description: "DPO, IPO, KTO, ORPO가 어떤 선호 데이터를 받고 무엇을 최적화하는지 비교하고 데이터에 맞는 방법을 고른다."
tags:
  - LLM
  - DPO
  - IPO
  - KTO
  - ORPO
  - preference optimization
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 16주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

8주차에는 같은 질문의 좋은 답과 나쁜 답을 한 쌍으로 묶어 DPO를 학습했다. 현실의 feedback은 늘 이렇게 반듯하지 않다. 어떤 서비스에는 좋아요와 싫어요만 남고, 어떤 팀은 reference model을 하나 더 올릴 메모리가 부족하다. 이번 주에는 데이터와 자원에 맞춰 DPO, IPO, KTO, ORPO를 고르는 법을 배운다.[^1][^2][^3]

## 이번 주에 배울 것

- DPO, IPO, KTO, ORPO가 받는 데이터 모양
- 네 방법의 loss가 policy를 움직이는 방식
- Preference pair가 없고 binary feedback만 있을 때의 선택 기준
- DPO와 IPO를 같은 데이터로 비교하는 방법
- 승률, 답변 길이, KL, 학습 시간과 메모리를 함께 읽는 법

선수 지식은 6주차의 preference pair, 8주차의 policy·reference log probability와 DPO loss다.

!!! note "알고리즘 이름보다 먼저 데이터 모양을 본다"

    좋은 답과 나쁜 답이 한 쌍인지, 답 하나에 좋아요·싫어요만 붙었는지부터 확인한다. 그다음 reference model을 둘 메모리와 SFT를 따로 할 계획이 있는지 살핀다.

## 1. 네 방법은 같은 문제를 조금씩 다르게 푼다

선호 최적화의 목표는 좋은 답의 확률을 높이고 좋지 않은 답의 확률을 낮추는 것이다. 차이는 “좋다”는 정보를 어떤 모양으로 받으며, 기준점에서 얼마나 멀어졌는지를 어떻게 제한하는가에 있다.

| 방법 | 한 행의 핵심 입력 | 답을 직접 비교하는가 | Reference model | SFT 단계 |
| --- | --- | --- | --- | --- |
| DPO | `prompt`, `chosen`, `rejected` | 예 | 필요 | 보통 먼저 수행 |
| IPO | `prompt`, `chosen`, `rejected` | 예 | 필요 | 보통 먼저 수행 |
| KTO | `prompt`, `completion`, `label` | 아니오 | 필요 | 보통 먼저 수행 |
| ORPO | `prompt`, `chosen`, `rejected` | 예 | 필요 없음 | Preference 학습과 결합 |

이 표에서 “필요”는 원 논문과 기본 구현을 기준으로 한 말이다. PEFT에서는 adapter를 끈 같은 model을 reference처럼 써서 메모리를 아낄 수 있다. 구현에 따라 reference log probability를 미리 계산하기도 한다.[^4][^5]

## 2. DPO는 두 답의 상대적인 변화를 비교한다

Policy의 chosen·rejected log probability 차이를 `policy gap`, reference model의 차이를 `reference gap`이라고 하자. 두 차이를 다시 뺀 값 $m=(\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x))-(\log\pi_{ref}(y_w|x)-\log\pi_{ref}(y_l|x))$이 relative margin이다.

DPO loss는 $L_{DPO}=-\log\sigma(\beta m)$이다.[^6] Margin이 커질수록 loss는 0에 가까워지면서 chosen의 확률을 rejected보다 높이는 쪽으로 policy를 민다.

```python
import torch

margin = torch.tensor([0.8])
beta = 0.1
dpo_loss = -torch.nn.functional.logsigmoid(beta * margin)
print(dpo_loss.item())
```

`beta`는 선호 신호의 세기와 reference에서 벗어나는 정도에 관여한다. 값 하나만 보고 정하지 말고 생성 품질과 KL을 같이 확인해야 한다.[^4]

## 3. IPO는 무한히 큰 margin을 목표로 삼지 않는다

IPO는 preference probability를 사람이 준 답의 참된 효용과 곧바로 같다고 보는 데 문제가 있다고 지적한다. Preference가 완벽하지 않거나 dataset이 작을 때 DPO가 pair를 지나치게 외울 수 있다는 문제의식에서 출발했다.[^1]

IPO의 핵심은 margin을 끝없이 키우는 대신 정해진 목표 가까이에 두는 제곱 loss다. 간단히 쓰면 $L_{IPO}=(m-1/(2\tau))^2$이다. Margin이 목표보다 작아도 loss가 생기고, 지나치게 커도 다시 loss가 커진다.[^1]

TRL에서는 DPOTrainer의 `loss_type="ipo"`를 선택한다. 이때 설정의 `beta`가 IPO 논문 수식의 $\tau$ 역할을 한다.[^4]

```python
from trl import DPOConfig

args = DPOConfig(
    output_dir="outputs/w09_ipo",
    loss_type="ipo",
    beta=0.1,
)
```

!!! warning "DPO의 beta와 IPO의 tau를 숫자만 보고 같다고 해석하지 않는다"

    TRL 설정 이름은 둘 다 `beta`지만 loss 안에서 맡는 역할은 다르다. 같은 0.1을 넣었다고 똑같은 세기의 제약이 되는 것은 아니다.

## 4. KTO는 답 두 개를 한 쌍으로 묶지 않아도 된다

사용자가 답 하나에 좋아요나 싫어요를 눌렀다고 하자. 어느 답과 비교해서 눌렀는지는 알 수 없다. KTO는 이런 unpaired binary feedback을 받는다.[^2]

```json
{
  "prompt": "Name one reason to record a random seed.",
  "completion": "It makes random choices reproducible.",
  "label": true
}
```

KTO는 prospect theory에서 아이디어를 가져왔다. 사람은 같은 크기의 이득과 손실을 똑같이 느끼지 않으며, 기준점보다 나빠진 손실을 더 크게 느끼는 경향이 있다는 설명이다. KTO는 desirable 답과 undesirable 답을 기준점의 양쪽에 놓고 서로 다른 가중치를 줄 수 있다.[^2]

TRL의 KTOTrainer는 `prompt`, `completion`, `label` 형식을 받는다. 이론상 dataset에는 desirable과 undesirable 예시가 모두 있어야 한다. 공식 문서는 한쪽 label만으로도 실행할 수 있다고 설명하지만, 특히 rejected만 있을 때는 작은 learning rate를 권한다.[^5]

## 5. ORPO는 SFT와 preference 학습을 한 단계로 합친다

![RLHF, DPO, ORPO의 학습 구성 비교](/notes/tutorial/llm_lecture/images/w09_orpo_alignment_comparison.png)

*그림 1. RLHF와 DPO는 SFT 뒤에 별도의 정렬 단계를 두지만, ORPO는 chosen 답의 SFT 신호와 chosen·rejected의 odds ratio를 한 단계에서 학습한다. 출처: Hong et al. (2024), Figure 2에서 발췌.[^3]*

ORPO loss는 chosen 답을 배우는 negative log-likelihood와 preference penalty를 더한다. 형태는 $L_{ORPO}=L_{SFT}+\lambda L_{OR}$로 볼 수 있다. Odds는 확률 $p$를 $p/(1-p)$로 바꾼 값이며, ORPO는 chosen의 odds가 rejected보다 커지도록 만든다.[^3]

Reference model이 없으므로 model 사본이나 reference forward가 필요하지 않다. 대신 ORPO는 SFT와 preference optimization을 함께 하는 방법이다. 이미 잘 만든 SFT checkpoint 뒤에 preference 단계만 공정하게 비교하려는 실험이라면 DPO·IPO와 조건이 다르다는 점을 기록해야 한다.

현재 TRL의 ORPOTrainer는 experimental API 아래에 있다. 설치한 TRL version에 따라 import 경로와 인자가 바뀔 수 있으므로 공식 문서를 먼저 확인한다.[^7]

## 6. 어떤 방법을 고를지 데이터부터 묻는다

| 가지고 있는 것 | 먼저 시험할 방법 | 까닭 |
| --- | --- | --- |
| 같은 prompt의 chosen·rejected pair | DPO | 구현과 해석이 단순한 기준선이다. |
| Pair가 적고 label noise나 과적합이 걱정됨 | IPO | Margin에 유한한 목표를 둔다. |
| 답 하나마다 좋아요·싫어요만 있음 | KTO | Completion끼리 짝을 만들 필요가 없다. |
| SFT와 preference를 한 번에 하고 reference 메모리를 줄이고 싶음 | ORPO | Chosen NLL과 preference loss를 한 단계에서 계산한다. |
| Positive와 negative 중 한쪽만 거의 있음 | 먼저 수집을 보완 | 어느 방법이든 비교 기준이 약해진다. |

!!! note "선택 순서"

    1. Feedback이 paired인지 unpaired인지 확인한다.
    2. 시작 model이 이미 SFT를 마쳤는지 확인한다.
    3. Reference model을 둘 메모리가 있는지 확인한다.
    4. 가장 단순한 기준선을 먼저 돌린다.
    5. 같은 test prompt로 품질과 비용을 함께 비교한다.

## 7. 승률 하나로 비교를 끝내면 안 된다

두 model의 답을 사람이 고르면 win rate를 만들 수 있다. 자동 판정 task에서는 한쪽만 통과한 prompt를 그 방법의 승리로 세고, 둘 다 통과하거나 둘 다 실패하면 tie로 둘 수 있다.

| 지표 | 확인하는 질문 | 주의할 점 |
| --- | --- | --- |
| Win rate·통과율 | 실제 지시를 더 잘 따르는가? | Judge나 verifier의 오류를 함께 점검한다. |
| 답변 길이 | 길기만 한 답을 선호했는가? | Task가 요구한 길이와 함께 본다. |
| Policy–reference KL | 시작 model에서 얼마나 멀어졌는가? | Token, prompt, 추정법을 같게 둔다. |
| Peak memory | 내 장치에서 실행할 수 있는가? | 측정 도구와 batch를 같게 둔다. |
| 학습 시간 | 같은 예산에서 얼마나 빠른가? | Evaluation 시간을 포함했는지 적는다. |

KL이 작다고 무조건 좋은 model은 아니다. 학습이 거의 되지 않아도 KL은 작다. 반대로 task 통과율이 올라도 KL이 크게 튀고 다른 능력이 나빠졌다면 안전한 개선이라고 보기 어렵다.

## 8. DPO와 IPO를 같은 조건으로 비교한다

이 실험은 8주차와 같은 `Qwen/Qwen2.5-0.5B-Instruct`에서 시작한다.[^8] 영어 지시 준수 task 10종으로 만든 train 400쌍, validation 100쌍, test prompt 100개를 그대로 썼다. LoRA rank 16, learning rate `1e-5`, effective batch size 8, 3 epoch, seed 42도 같게 두고 loss만 DPO와 IPO로 바꿨다.[^9]

![같은 데이터로 학습한 DPO와 IPO 비교](/notes/tutorial/llm_lecture/images/w09_dpo_ipo_comparison.png)

*그림 2. 왼쪽은 task별 test 통과율, 가운데는 평균 생성 token 수, 오른쪽은 각 policy가 만든 20개 답변 문맥에서 계산한 base model과의 평균 token-distribution KL이다. Qwen2.5-0.5B-Instruct와 macOS에서 직접 실행했다.[^9]*

| 지표 | DPO | IPO |
| --- | ---: | ---: |
| Test 통과율 | 49% | 52% |
| 한쪽만 통과한 prompt | 10개 | 13개 |
| 평균 생성 길이 | 7.89 token | 7.62 token |
| 중앙 생성 길이 | 5 token | 5 token |
| 평균 token KL | 1.124 | 1.082 |
| 학습 장치 | macOS MPS | macOS CPU |
| 학습 시간 | 396.23초 | 769.91초 |
| Process peak RSS | 3,375MB | 4,153MB |

IPO가 통과율에서 3%p 앞섰지만, 한쪽만 맞힌 prompt 차이는 3개다. `uppercase_exact`는 DPO 0%, IPO 90%였고 `number_only`도 IPO가 20%p 높았다. 반대로 `two_bullets`는 DPO 100%, IPO 40%였다. 평균 하나로 모든 지시에서 IPO가 낫다고 말할 수 없다.

두 방법의 평균과 중앙 길이는 비슷했다. 답을 통과 여부로 나누면 DPO는 성공 7.67 token, 실패 8.10 token이었다. IPO는 성공 5.94 token, 실패 9.44 token이었다. 이 dataset에서는 IPO의 긴 답이 오히려 형식 검사를 더 자주 어겼다. 길이가 품질을 대신하는 지표가 아님을 보여준다.

KL은 전체 vocab의 policy 분포와 base model 분포를 response 위치마다 비교한 뒤 평균했다. 계산량을 줄이려고 test prompt 중 앞의 20개만 사용했다. DPO 1.124, IPO 1.082로 비슷했으며 이 차이만으로 어느 방법이 reference에 더 안정적으로 머문다고 결론 내리기 어렵다.

IPO 실행은 DPO보다 약 1.94배 오래 걸렸고 peak RSS도 778MB 높았다. 하지만 DPO는 MPS, IPO는 CPU에서 실행돼 속도를 공정하게 비교할 수 없다. Peak RSS 역시 서로 다른 시점에 한 번씩 잰 process 수준 값이라 운영체제 상태와 library cache가 섞여 있다. 이 표는 각 실행에 필요한 자원의 기록이지 알고리즘 자체의 비용 순위가 아니다. 같은 장치에서 여러 번 측정해야 정확히 비교할 수 있다.

!!! warning "이 비교가 알고리즘 전체의 순위를 정하지는 않는다"

    Model 하나, 작은 자동 판정 dataset 하나, seed 하나로 얻은 결과다. 다른 언어와 열린 질문에서도 같은 순서가 나온다고 일반화할 수 없다.

## 9. Preference optimization 선택 가이드를 남긴다

```text
Dataset
├─ chosen/rejected가 같은 prompt에 묶여 있음
│  ├─ 이미 SFT model이 있음 → DPO를 기준선으로, IPO를 과적합 대안으로 비교
│  └─ SFT와 preference를 합치고 싶음 → ORPO 검토
└─ 답 하나에 좋아요/싫어요만 있음
   ├─ 두 label이 충분함 → KTO 검토
   └─ 한 label만 거의 있음 → feedback 수집부터 보완
```

결정 기록에는 알고리즘 이름만 쓰지 않는다. 데이터 schema, label 비율, reference model 유무, loss 설정, 생성 평가와 메모리 측정법을 함께 남긴다.

## 확인 문제

1. DPO와 IPO가 같은 preference pair를 받아도 margin을 다루는 방식은 어떻게 다른가?
2. 좋아요·싫어요만 있고 chosen·rejected pair가 없다면 어떤 방법을 먼저 검토해야 하는가?
3. ORPO가 reference model을 쓰지 않는 대신 한 loss에 합친 두 신호는 무엇인가?
4. 통과율이 올랐을 때 답변 길이와 KL도 함께 확인해야 하는 이유는 무엇인가?
5. DPO와 ORPO를 비교할 때 학습 단계를 똑같다고 가정하면 안 되는 이유는 무엇인가?

## 완료 체크

- [x] DPO, IPO, KTO, ORPO의 데이터와 loss를 비교했다.
- [x] Paired preference와 binary feedback에 따른 선택 기준을 적었다.
- [x] DPO와 IPO를 같은 작은 데이터와 설정으로 실행했다.
- [x] 통과율, 길이, KL, 시간과 peak memory를 기록했다.
- [x] 결과물로 `Preference optimization 선택 가이드`를 만들었다.

---

[^1]: Azar, M. G. et al. (2023). [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036). Preference optimization의 이론적 틀과 IPO loss를 참고했다.
[^2]: Ethayarajh, K. et al. (2024). [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306). Unpaired binary feedback과 prospect theory 기반 loss를 참고했다.
[^3]: Hong, J. et al. (2024). [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691). Figure 2와 ORPO objective를 참고했다.
[^4]: Hugging Face. [TRL: DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer). DPO·IPO loss type, `beta`, dataset과 PEFT reference 처리를 참고했다. 확인일: 2026-08-02.
[^5]: Hugging Face. [TRL: KTO Trainer](https://huggingface.co/docs/trl/kto_trainer). Unpaired preference dataset, label 조건과 설정을 참고했다. 확인일: 2026-08-02.
[^6]: Rafailov, R. et al. (2023). [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290). DPO objective와 reference model의 역할을 참고했다.
[^7]: Hugging Face. [TRL: ORPO Trainer](https://huggingface.co/docs/trl/orpo_trainer). Experimental API, preference format과 학습 설정을 참고했다. 확인일: 2026-08-02.
[^8]: Qwen Team. [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). Model과 chat template 정보를 참고했다. 확인일: 2026-08-02.
[^9]: 직접 실행한 `llm_lecture/week09_preference_family_demo.py`와 8주차 DPO 결과다. Qwen2.5-0.5B-Instruct, train 400쌍, validation 100쌍, test 100개, TRL 1.9.2, Transformers 5.14.1, PyTorch 2.13.0을 사용했다. DPO는 macOS MPS, IPO는 macOS CPU에서 실행했다. 원본 CSV와 adapter는 Git에서 제외했다. 실행일: 2026-08-02.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
카테고리별 탐지/수정: A-8 1→0, C-11 0→0, D-1 0→0, H-1 0→0, I-1 0→0
정량 점검: humanize-korean metrics v2.0 risk band low
자체검증: 고유명사·수식·실측값 보존 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음 / 변경률 30% 이하
등급: B — 자체검증 6/6을 통과했고 기술 용어와 실험 수치를 그대로 보존함
주요 변경: 결론 접속어를 덜어 내고 DPO loss와 policy 변화의 관계를 한 문장으로 연결함
-->
