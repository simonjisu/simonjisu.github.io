---
title: "21주차. VLM 과제와 실패 분석"
description: "Caption, VQA, 문서 읽기, 공간 관계, grounding, object hallucination을 과제에 맞는 지표로 평가하고 이미지 대조 실험으로 실패 원인을 나눈다."
tags:
  - VLM
  - multimodal evaluation
  - VQA
  - DocVQA
  - hallucination
  - grounding
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 26주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

20주차에는 사진과 질문을 chat template에 넣고 작은 VLM을 instruction tuning했다. 학습 loss와 정답률이 좋아졌다고 해서 평가가 끝나지는 않는다. 자연스러운 문장을 만들면서 사진에 없는 물체를 덧붙이거나 질문만 보고 답을 찍기도 한다. 이번 주에는 과제마다 무엇을 재야 하는지 정하고 틀린 답을 원인별로 나눈다.

## 이번 주에 배울 것

- Caption, VQA, 문서 VQA, OCR, 공간 관계, grounding의 차이
- Exact match, ANLS, IoU, object hallucination 지표의 쓰임
- 지식이 필요한 문제와 이미지를 정확히 봐야 하는 문제의 구분
- 원본, 저해상도, 이미지 교체, 이미지 가림 대조 실험
- POPE 방식의 물체 존재 질문과 Accuracy, Precision, Recall, F1, Yes rate
- 틀린 답을 인식, OCR, 지식, 추론, 지시 위반, hallucination으로 분류하는 방법
- 전체 평균에 가려진 task별 실패를 읽는 방법

선수 지식은 12주차의 평가 설계, 17주차의 해상도와 patch, 20주차의 multimodal chat template이다. 실습에서는 20주차와 같은 `HuggingFaceTB/SmolVLM-256M-Instruct`를 쓰되 adapter는 붙이지 않았다. Base checkpoint의 여러 능력을 한꺼번에 살펴보려는 진단 실험이다.

!!! note "말을 잘하는 것과 사진을 잘 보는 것은 다르다"

    사진에 고양이가 없는데도 “창가에 앉은 고양이”라고 자연스럽게 말한다. 문장만 읽으면 그럴듯하지만 사진과 맞지 않는다. VLM 평가는 답의 문법, 이미지 근거, 정답 형식을 따로 살펴야 한다.

## 1. VLM이라는 이름 아래 여러 과제가 있다

한 점수로 VLM의 모든 능력을 나타내기는 어렵다. 필요한 입력과 정답의 모양부터 다르다.

| 과제 | Model이 해야 하는 일 | 정답 형태 | 대표 지표 |
| --- | --- | --- | --- |
| Image captioning | 장면의 중요한 내용을 문장으로 요약 | 한 개 이상의 자연어 문장 | CIDEr, SPICE, 속성 coverage, 사람 평가 |
| VQA | 질문과 관련된 시각 정보만 골라 답함 | 짧은 단어, 숫자, 선택지 | Accuracy, soft VQA accuracy |
| Document VQA | 문서의 글자와 배치를 읽고 질문에 답함 | OCR 문자열, 날짜, 금액 | ANLS, exact match |
| Spatial reasoning | 물체 사이의 방향이나 관계를 판단 | left, right, above 같은 관계 | Accuracy |
| Grounding | 글이 가리키는 물체의 위치를 찾음 | Bounding box나 mask | IoU, Acc@IoU threshold |
| Object hallucination | 없는 물체를 있다고 말하는지 확인 | Caption 속 object 또는 Yes/No | CHAIR, POPE Accuracy·F1·Yes rate |

MMMU는 대학 시험과 교재에서 모은 11.5K 문제로 평가 범위를 넓혔다. 6개 큰 분야, 30개 과목, 183개 세부 분야와 30종의 이미지가 들어간다. 차트, 지도, 악보, 화학 구조처럼 이미지 종류도 서로 다르다.[^1] 하나의 단순 도형 점수는 이 범위를 대신하지 못한다.

## 2. 지표는 정답의 모양에 맞춰 고른다

### 짧은 VQA는 Accuracy부터 본다

색, 개수, 선택지처럼 정답이 하나라면 exact match가 이해하기 쉽다. 정답이 `blue`일 때 `Blue.`를 맞게 처리하려면 대소문자와 마침표를 정규화한다. 숫자 단위나 동의어를 허용한다면 결과를 보기 전에 규칙을 고정한다.

```python
def normalize_short_answer(text: str) -> str:
    return text.lower().strip().rstrip(".")

correct = normalize_short_answer(prediction) == normalize_short_answer(target)
```

정규화가 너무 넓으면 오답을 정답으로 바꾼다. 예를 들어 정답 `2`를 찾으려고 문장 안의 모든 숫자를 허용하면 `12`도 잘못 통과한다. Parser와 prompt를 함께 공개해야 하는 이유다.

### 문서 답변은 작은 OCR 오류를 따로 다룬다

DocVQA 논문은 exact accuracy와 ANLS를 함께 사용한다. Exact match는 글자 하나만 달라도 0점이다. ANLS는 Levenshtein distance를 문자열 길이로 나눈 뒤 similarity로 바꾸므로 작은 OCR 차이에 부분 점수를 준다. 차이가 너무 크면 threshold 0.5에서 0점으로 처리한다.[^2]

정답이 `invoice`이고 예측이 `invo1ce`라면 한 글자만 다르다. 반대로 정답 `42`에 예측 `4`는 두 글자 중 하나가 빠졌다. 두 경우를 똑같은 exact 0점으로만 기록하면 OCR 오류의 크기를 알기 어렵다.

!!! warning "ANLS가 내용 검증을 대신하지 않는다"

    문자열이 가깝다는 뜻일 뿐이다. 금액 `100`과 `900`은 한 글자만 달라도 실제 업무에서는 큰 오류다. 과제의 위험도에 따라 exact match, 숫자 허용 오차, 사람이 확인할 조건을 따로 정한다.

### Grounding은 겹친 영역을 잰다

Grounding model은 물체를 설명하는 문장과 함께 bounding box를 낸다. 예측 box와 정답 box의 교집합을 합집합으로 나눈 값이 IoU다. 식은 $IoU = |B_{pred} \cap B_{gt}| / |B_{pred} \cup B_{gt}|$다.[^3]

작은 숫자 예시에서 정답 box를 `[40, 60, 140, 160]`, 예측 box를 `[60, 40, 160, 140]`으로 두었다. 각 box 넓이는 10,000, 교집합은 6,400, 합집합은 13,600이다. IoU는 약 0.471이므로 IoU 0.5 이상을 정답으로 보는 평가에서는 실패한다.[^4]

```python
intersection = 80 * 80
union = 100 * 100 + 100 * 100 - intersection
iou = intersection / union

print(iou)
# 0.47058823529411764
```

Box 좌표 형식이 `xyxy`, `xywh`, `cxcywh` 중 무엇인지 먼저 확인한다. 형식을 뒤섞으면 눈으로는 같은 box여도 IoU가 틀어진다.

## 3. Caption은 한 문장 점수만 믿기 어렵다

같은 사진도 여러 문장으로 올바르게 설명한다. “A red circle is beside a blue square”와 “There is a blue square next to a red circle”은 단어 순서가 다르지만 내용은 같다. Exact match는 둘 중 하나를 오답으로 만든다.

Caption metric은 각기 다른 부분을 본다. N-gram 계열은 참고 문장과 겹치는 표현을, SPICE는 물체·속성·관계 구조를 살핀다. 자유로운 설명에서는 사람 평가나 강한 VLM judge를 보조로 붙이기도 한다. Judge를 쓴다면 model 이름, prompt, 순서 효과, 재평가 일치율을 기록한다.

이번 작은 실험에서는 정답 문장을 하나로 고정하지 않았다. 사진에 있어야 할 색과 도형 네 단어가 답에 얼마나 들어갔는지 attribute coverage를 계산했다.

```text
필수 속성: red, circle, blue, square
예측: A red circle and a blue square.
점수: 4 / 4 = 1.0
```

이 지표에도 빈틈이 있다. 사진을 다른 장면으로 바꿔도 두 장면이 `circle`이나 `square`를 공유하면 부분 점수가 남는다. 실제 실험에서 shuffled caption 점수가 75%였던 이유다. Caption coverage가 높아도 원래 사진을 정확히 설명했다고 바로 결론 내리지 않는다.

## 4. Object hallucination은 없는 물체를 물어본다

Object hallucination은 model이 이미지에 없거나 정답 annotation과 맞지 않는 물체를 생성하는 현상이다. 긴 caption에서 물체 이름을 뽑아 재는 방법은 문장 길이와 parser에 영향을 받는다. POPE는 이를 Yes/No 분류 문제로 바꾼다.[^5]

![POPE의 ground-truth object, negative sampling, polling question 구성](/notes/tutorial/llm_lecture/images/w21_pope_pipeline.png)

*그림 1. 이미지에서 존재하는 물체를 찾고 존재하지 않는 물체를 random, popular, adversarial 방식으로 고른 뒤 Yes/No 질문을 만드는 POPE pipeline. 출처: Li et al. (2023), Figure 3에서 발췌.[^5]*

POPE는 존재하는 물체와 존재하지 않는 물체를 1:1로 맞춘다. Negative object를 고르는 방식은 세 가지다.[^5]

| 방식 | 존재하지 않는 물체를 고르는 법 | 확인하려는 실패 |
| --- | --- | --- |
| Random | 전체 후보에서 무작위로 선택 | 기본적인 물체 존재 판정 |
| Popular | 데이터에 자주 나오는 물체를 선택 | 자주 본 물체를 습관처럼 답하는지 확인 |
| Adversarial | 사진 속 물체와 자주 함께 나오는 물체를 선택 | 장면의 전형적인 조합을 근거 없이 덧붙이는지 확인 |

질문은 `Is there a chair in the image?`처럼 짧다. 정답은 Yes 또는 No다. Accuracy와 F1을 함께 보고 Yes rate도 남긴다. Yes rate가 100%라면 모든 물체가 있다고 답하는 편향, 0%라면 모든 물체를 부정하는 편향을 의심한다.

## 5. 이미지 대조 실험으로 근거를 확인한다

정답률만으로 model이 이미지를 사용했는지 판단하기 어렵다. 질문 문장이나 데이터의 답 분포만 보고 맞히기도 한다. 같은 질문과 정답을 유지한 채 image input만 바꾸면 시각 정보의 역할이 드러난다.

| 조건 | Image 처리 | 관찰할 점 |
| --- | --- | --- |
| Original | 원본 256×256 이미지 | 기본 성능 |
| Low resolution | 32×32로 줄인 뒤 256×256으로 확대 | 작은 글자와 경계가 사라질 때의 변화 |
| Shuffled | 같은 task의 다른 정답 이미지를 넣음 | 답이 바뀐 이미지를 따라가는가 |
| Covered | 모든 pixel을 회색으로 가림 | 시각 근거 없이 prompt와 prior로 무엇을 답하는가 |

Shuffled 조건에서는 질문과 원래 target을 바꾸지 않는다. 정확도가 내려가는 것이 정상이다. 더 중요한 관찰은 model output이 교체한 사진의 내용으로 바뀌는지다. Covered 조건이 Original과 비슷하다면 질문만으로 풀리는 데이터인지, model이 image를 무시하는지 살핀다.

Image를 완전히 빼면 일부 VLM의 template이나 Processor가 다른 경로로 동작한다. 이번 실습은 입력 구조를 같게 유지하려고 회색 image를 넣었다. 이를 엄밀한 text-only 조건이라고 부르지 않고 covered-image control이라고 적는다.

## 6. 다섯 과제를 같은 작은 VLM로 실행한다

합성 image 21개를 만들었다. Caption 3개, color VQA 3개, count VQA 3개, 공간 관계 3개, receipt document VQA 3개, triangle 존재 질문 6개다. POPE 형태의 존재 질문은 Yes 3개와 No 3개로 맞췄다.[^4]

Model은 `HuggingFaceTB/SmolVLM-256M-Instruct`, revision `7e3e67edbbed1bf9888184d9df282b700a323964`다.[^6] `do_image_splitting=False`, greedy decoding, 최대 32 new token을 사용했다. 네 조건에서 84번 생성했고 macOS MPS에서 model load를 뺀 생성 시간은 약 10.81초였다.[^4]

![원본, 저해상도, 이미지 교체, 이미지 가림 조건의 VLM 과제별 결과](/notes/tutorial/llm_lecture/images/w21_vlm_failure_controls.png)

*그림 2. 합성 image 21개를 네 조건에서 평가한 결과. Caption은 네 속성 coverage, 나머지는 제한된 답의 exact score다. 실제 benchmark가 아니라 failure analysis pipeline을 확인한 진단 실험이다.[^4]*

| Task | Original | 32px 후 확대 | Shuffled | Covered |
| --- | ---: | ---: | ---: | ---: |
| Caption attribute coverage | 100.0% | 91.7% | 75.0% | 0.0% |
| Color VQA | 66.7% | 33.3% | 0.0% | 33.3% |
| Count VQA | 33.3% | 33.3% | 33.3% | 33.3% |
| Document VQA | 0.0% | 0.0% | 0.0% | 0.0% |
| Object presence | 50.0% | 50.0% | 50.0% | 50.0% |
| Spatial relation | 66.7% | 33.3% | 0.0% | 33.3% |
| 21문제 평균 | 52.4% | 41.7% | 29.8% | 28.6% |

Caption은 점수 범위가 0부터 1인 coverage이고 나머지는 문항별 0 또는 1이다. 이를 평균 낸 52.4%는 서로 다른 metric을 섞은 교육용 요약이다. Model benchmark나 다른 model과의 순위표에 쓰면 안 된다.

원본에서 image를 바꾸자 color와 spatial score가 0%로 떨어졌다. Model 답이 사진에 의존했다는 신호다. Count는 모든 조건에서 1이라고 답해 33.3%가 유지됐다. 이 task에서는 image를 제대로 세지 않고 첫 번째 선택지를 반복했다.

## 7. 실패한 답을 직접 읽는다

점수표 아래에는 원문 output을 남긴다. 이번 실험에서 task마다 다른 실패가 보였다.

### Caption은 맞았지만 counting은 한 값에 고정됐다

```text
Original caption image:
Target attributes: red, circle, blue, square
Prediction: A red circle and a blue square.

Original count image with three circles:
Target: 3
Prediction: 1.
```

Caption의 물체와 색은 읽었지만 여러 물체를 세는 질문에서는 모두 `1`을 냈다. VLM이 `circle`을 인식한다는 사실만으로 counting까지 잘한다고 추정하면 안 된다.

### Document VQA는 첫 자리만 읽었다

```text
Image text: TOTAL 17  → Prediction: 1.
Image text: TOTAL 42  → Prediction: 4.
Image text: TOTAL 86  → Prediction: 8.
```

세 답은 첫 숫자와 일치하지만 전체 금액은 모두 틀렸다. 원본과 저해상도 모두 exact 0%였다. 단순히 해상도만 높이는 조치보다 OCR token, Processor의 image splitting, 문서용 checkpoint, 출력 길이와 decoding을 함께 점검한다.

### 같은 50% Accuracy 안에 반대 편향이 숨어 있었다

Triangle 존재 질문에서 Original model은 여섯 문제 모두 `No`라고 답했다. Covered image에서는 여섯 문제 모두 `Yes`라고 답했다.

| 조건 | Accuracy | Precision | Recall | F1 | Yes rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 50.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 32px 후 확대 | 50.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Shuffled | 50.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Covered | 50.0% | 50.0% | 100.0% | 66.7% | 100.0% |

Balanced Yes/No 자료에서는 한쪽 답만 반복해도 Accuracy가 50%다. Original의 Yes rate 0%와 Covered의 100%를 함께 봐야 정반대의 편향이 드러난다. F1만 보아도 Covered가 더 나아 보일 수 있으므로 confusion matrix와 응답 비율을 같이 기록한다.

## 8. 오류 유형을 고치는 담당이 다르다

틀린 sample을 다음 기준으로 나누면 다음 실험이 분명해진다.

| 오류 유형 | 예시 | 먼저 확인할 곳 |
| --- | --- | --- |
| 시각 인식 | 파란색을 초록색이라고 답함 | Image 전처리, vision encoder, 해상도 |
| OCR | `42`를 `4`로 읽음 | Tile, crop, OCR data, 문서용 model |
| 지식 | 표의 기호는 읽었지만 과학 개념을 모름 | LLM 지식, retrieval, 문제의 전제 |
| 공간 추론 | 물체는 찾았지만 left와 above를 혼동 | 좌표 표현, spatial data, reasoning prompt |
| Counting | 물체는 보지만 개수를 한 값으로 반복 | Object separation, count data, decoding |
| 지시 위반 | 정답은 알지만 JSON 대신 문장으로 답함 | Chat template, SFT data, LLM LoRA |
| Hallucination | 없는 table이 있다고 답함 | Negative sample, POPE, image grounding |
| 평가기 오류 | `12` 안의 `2`를 정답으로 처리 | Normalizer, parser, 사람 audit |

한 sample에 오류가 두 개 이상 겹치기도 한다. OCR이 틀려 계산도 틀렸다면 `OCR → reasoning`처럼 첫 실패 지점과 뒤따른 실패를 함께 적는다. 마지막 문장만 보고 reasoning 오류로 묶으면 vision 쪽 문제를 놓친다.

## 9. 평가 보고서에는 평균보다 조건을 먼저 적는다

결과표에는 적어도 다음 정보를 남긴다.

- Model ID와 revision, Processor와 chat template
- Image 원본 크기, resize, crop, tile 설정
- Prompt, decoding, max new token, seed
- Task별 sample 수와 target 분포
- 정규화와 parser code, judge가 있다면 judge 설정
- Original과 image ablation의 task별 점수
- Confusion matrix, Yes rate, 답변 길이 분포
- 대표 성공과 실패 output 원문
- 데이터 중복, text-only shortcut, 오염 여부
- 실행 장치와 시간, 재실행 간 변동

좋은 보고서는 최고 점수를 강조하는 문서가 아니다. 어떤 조건에서 무엇이 깨졌는지 다시 찾을 수 있는 실험 기록이다.

## 확인 문제

1. Caption에 exact match만 쓰면 올바른 다른 표현을 오답으로 만들 수 있는 이유는 무엇인가?
2. DocVQA에서 ANLS와 exact accuracy를 함께 보는 까닭은 무엇인가?
3. 두 bounding box의 IoU가 0.471일 때 IoU 0.5 기준에서 오답이 되는 과정을 설명해보자.
4. Shuffled-image score가 Original보다 크게 떨어지면 model의 image 사용을 어떻게 해석할 수 있는가?
5. Covered image와 완전한 text-only 입력이 같은 조건이 아닌 이유는 무엇인가?
6. Count VQA가 모든 조건에서 33.3%였다는 결과는 어떤 shortcut을 의심하게 하는가?
7. Object presence Accuracy가 두 조건에서 모두 50%인데도 실패 방식이 정반대였던 이유는 무엇인가?
8. POPE에서 Popular와 Adversarial negative sampling이 Random보다 어려울 수 있는 까닭은 무엇인가?
9. `TOTAL 42`를 `4`로 읽은 답을 reasoning 오류로만 분류하면 무엇을 놓치는가?
10. 서로 다른 metric을 평균 낸 52.4%를 다른 VLM의 benchmark 점수와 비교하면 안 되는 이유는 무엇인가?

## 완료 체크

- [x] Caption, VQA, Document VQA, spatial reasoning, grounding을 정답 형태와 metric으로 구분했다.
- [x] Exact match, ANLS, IoU가 서로 다른 오류를 재는 이유를 설명했다.
- [x] POPE의 positive와 negative object 질문과 세 negative sampling 방식을 확인했다.
- [x] 원본, 저해상도, shuffled, covered-image control을 같은 prompt로 실행했다.
- [x] 합성 image 21개의 task별 score와 원문 output을 기록했다.
- [x] Accuracy, F1, Yes rate를 함께 읽어 반대 방향의 응답 편향을 찾았다.
- [x] 오류를 인식, OCR, 지식, 공간 추론, counting, 지시 위반, hallucination, 평가기 오류로 분류했다.
- [x] 작은 진단 실험을 실제 VLM benchmark와 구분했다.

---

[^1]: Yue, X. et al. (2024). [MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI](https://arxiv.org/abs/2311.16502). 11.5K 문제, 6개 분야, 30개 과목, 183개 세부 분야, 30종 이미지 구성을 참고했다.
[^2]: Mathew, M. et al. (2021). [DocVQA: A Dataset for VQA on Document Images](https://openaccess.thecvf.com/content/WACV2021/papers/Mathew_DocVQA_A_Dataset_for_VQA_on_Document_Images_WACV_2021_paper.pdf). §5.1의 exact accuracy와 ANLS 사용 이유를 참고했다.
[^3]: PyTorch. [`torchvision.ops.box_iou`](https://docs.pytorch.org/vision/main/generated/torchvision.ops.box_iou.html). 두 box 집합의 intersection-over-union과 `xyxy`, `xywh`, `cxcywh` 형식을 참고했다. 확인일: 2026-08-04.
[^4]: 직접 실행한 `llm_lecture/week21_vlm_failure_analysis.py`의 결과다. 합성 image, 조건별 image, CSV, JSON과 논문 원본은 Git에서 제외하고 최종 plot만 공개했다. 실행일: 2026-08-04.
[^5]: Li, Y. et al. (2023). [Evaluating Object Hallucination in Large Vision-Language Models](https://aclanthology.org/2023.emnlp-main.20/). Figure 3과 §5의 POPE pipeline, 1:1 positive-negative 구성, Random·Popular·Adversarial sampling, Accuracy·Precision·Recall·F1·Yes ratio를 참고했다.
[^6]: Hugging FaceTB. [SmolVLM-256M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct). Model 규모와 image-text-to-text 사용법을 참고했다. 확인일: 2026-08-04.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
원본/윤문본: 12,576자 / 12,538자, 변경률 1.55%
카테고리별 탐지/수정: A-10 7→0, C-11 2→0, I-4 1→0, D-1 0→0, H-1 0→0
정량 점검: humanize-korean metrics v2.0 risk band low
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 논문 수치, model revision, 실험 조건, 원문 output, task별 지표를 보존함
주요 변경: 반복되던 가능 표현을 직접 서술로 바꾸고 연결어미 뒤의 불필요한 쉼표를 덜어냄
-->
