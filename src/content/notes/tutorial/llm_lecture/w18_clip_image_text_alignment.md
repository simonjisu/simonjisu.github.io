---
title: "18주차. CLIP과 이미지-글 정렬"
description: "CLIP이 이미지와 영어 문장을 같은 embedding 공간에 놓는 원리를 배우고, contrastive loss·검색·zero-shot 분류를 직접 실행한다."
tags:
  - VLM
  - CLIP
  - contrastive learning
  - zero-shot classification
  - image-text retrieval
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 26주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

17주차에서는 이미지를 patch token으로 바꾸는 과정을 살펴봤다. 이제 사진과 문장을 어떻게 연결할지 알아볼 차례다. 이번 주에는 CLIP이 이미지와 영어 문장을 같은 embedding 공간에 놓고, 가까운 쌍을 찾도록 학습하는 과정을 따라간다.

## 이번 주에 배울 것

- Image encoder와 text encoder가 맡는 역할
- 이미지와 문장을 같은 크기의 embedding으로 바꾸는 까닭
- Cosine similarity로 이미지와 글의 가까움을 재는 방법
- Batch 안에서 맞는 image-text 쌍을 찾는 contrastive loss
- Temperature가 similarity 점수와 확률 분포를 바꾸는 방식
- 같은 embedding으로 이미지-글 검색과 zero-shot 분류를 하는 방법
- Class 이름을 문장으로 만드는 prompt가 결과에 미치는 영향

선수 지식은 17주차의 image tensor와 Vision Transformer, vector의 내적, softmax다. Cosine similarity가 낯설다면 두 화살표가 같은 방향을 가리키는 정도를 재는 값이라고 생각해도 좋다.

!!! note "사진과 설명을 같은 지도에 놓기"

    동물 사진과 설명 문장을 하나의 커다란 지도에 점으로 표시한다고 생각해보자. 고양이 사진 근처에는 “a photo of a cat”이 놓이고, 자동차를 설명하는 문장은 멀리 떨어져야 한다. CLIP은 맞는 사진과 글을 가깝게 배치하는 법을 배운다.

## 1. CLIP에는 encoder가 두 개 있다

CLIP은 Contrastive Language-Image Pre-training의 약자다. Image encoder는 사진을 읽고, text encoder는 문장을 읽는다. 둘은 입력 형태가 다르지만 마지막에는 같은 차원의 embedding을 내놓는다.[^1]

```text
image ── image encoder ── projection ── image embedding [batch, 512]
text  ──  text encoder ── projection ──  text embedding [batch, 512]
```

이번 실습의 `openai/clip-vit-base-patch32`는 ViT-B/32를 image encoder로 쓴다. 224×224 이미지를 32×32 patch로 나누므로 patch는 $7 \times 7=49$개다. Text encoder는 Transformer이며, image embedding과 text embedding의 마지막 차원은 모두 512다.[^2][^3]

Projection은 두 encoder가 만든 특징을 공동 embedding 공간으로 옮긴다. 이미지와 문장의 vector 길이가 같아지면 서로 내적해 similarity를 구할 수 있다.

!!! warning "CLIP 자체는 문장을 생성하지 않는다"

    CLIP의 text encoder는 문장을 embedding으로 바꾸지만 다음 token을 생성하는 LLM은 아니다. CLIP은 주어진 이미지와 문장이 얼마나 가까운지 점수를 매긴다. 사진을 보고 긴 답변을 만드는 VLM 연결 구조는 19주차에서 다룬다.

## 2. Cosine similarity는 방향의 비슷함을 잰다

Image embedding을 $v$, text embedding을 $t$라고 하자. Cosine similarity는 $\operatorname{sim}(v,t)=v \cdot t/(\lVert v\rVert \lVert t\rVert)$로 계산한다. 분자의 $v \cdot t$는 두 vector의 내적이고, 분모는 각 vector의 길이를 곱한 값이다.

두 vector를 길이 1로 normalization하면 식은 단순한 내적 $\hat{v} \cdot \hat{t}$가 된다. 방향이 비슷할수록 1에 가까워지고, 직각이면 0, 반대 방향이면 -1에 가까워진다.

```python
import torch.nn.functional as F

image_features = F.normalize(image_features, dim=-1)
text_features = F.normalize(text_features, dim=-1)
cosine_similarity = image_features @ text_features.T

print(cosine_similarity.shape)
# torch.Size([number_of_images, number_of_texts])
```

이미지 2장과 문장 4개를 비교하면 `[2, 4]` similarity matrix가 나온다. 행 하나는 이미지 한 장, 열 하나는 문장 하나를 뜻한다.

## 3. Contrastive learning은 맞는 쌍을 찾는 퀴즈다

![CLIP의 contrastive pre-training과 zero-shot prediction 구조](/notes/tutorial/llm_lecture/images/w18_clip_training_zero_shot.png)

*그림 1. Batch의 image-text 쌍으로 두 encoder를 함께 학습하고, class 이름을 문장으로 바꾸어 zero-shot prediction에 쓰는 과정. 출처: Radford et al. (2021), Figure 1에서 발췌.[^1]*

학습 batch에 image-text 쌍이 $N$개 있다고 하자. 모든 이미지 embedding과 모든 text embedding을 비교하면 $N \times N$ matrix가 만들어진다. $i$번째 이미지와 $i$번째 문장이 원래 한 쌍이라면 대각선이 정답이고, 나머지는 틀린 조합이다.

|  | Text 1: cats | Text 2: parrots |
| --- | ---: | ---: |
| Image 1: cats | 0.281 ✓ | 0.205 |
| Image 2: parrots | 0.203 | 0.297 ✓ |

이 표는 이번에 직접 실행한 cosine similarity다. 각 행에서는 맞는 문장을 찾고, 각 열에서는 맞는 이미지를 찾는다. CLIP 논문의 학습 loss는 image-to-text cross-entropy와 text-to-image cross-entropy의 평균이다.[^1]

Similarity $c_{ij}$를 바로 softmax에 넣지 않고 $s_{ij}=c_{ij}/\tau$로 바꾼다. $\tau$는 temperature다. 정답 index를 $y_i=i$라고 하면 image-to-text loss는 $L_{I\rightarrow T}=\operatorname{CE}(s,y)$, 반대 방향은 $L_{T\rightarrow I}=\operatorname{CE}(s^T,y)$다. 최종 loss는 $L=(L_{I\rightarrow T}+L_{T\rightarrow I})/2$다.

```python
targets = torch.arange(batch_size)
loss_i = cross_entropy(logits_per_image, targets)
loss_t = cross_entropy(logits_per_text, targets)
loss = (loss_i + loss_t) / 2
```

두 쌍만 사용한 이번 예시의 symmetric contrastive loss는 약 0.000286이었다. 이미 학습된 model에 정답이 분명한 이미지 두 장만 넣어 계산했기 때문에 매우 낮다. 이 값을 CLIP의 일반 성능이나 새 model의 학습 결과로 해석하면 안 된다.

!!! warning "Batch 안의 모든 다른 쌍이 정말 오답은 아니다"

    같은 고양이를 다른 말로 설명한 문장이 batch 안에 있을 수 있다. 이때 대각선 밖의 쌍도 의미상 맞지만 학습에서는 negative로 취급될 수 있다. 데이터 중복과 caption 품질, batch 구성까지 함께 살펴야 한다.

## 4. Temperature는 작은 차이를 크게 벌린다

Cosine similarity가 0.299와 0.205라면 첫 번째가 더 가깝지만 차이는 0.094뿐이다. Temperature를 낮추면 이 차이를 softmax가 더 크게 받아들인다.

| Temperature | cats on sofa | two parrots | sports car | bowl of fruit |
| ---: | ---: | ---: | ---: | ---: |
| 1.0 | 26.91% | 24.51% | 24.47% | 24.10% |
| 0.1 | 47.30% | 18.63% | 18.33% | 15.74% |
| 0.01 | 99.98% | 0.01% | 0.01% | 0.00% |

고양이 이미지 한 장에 같은 cosine similarity 네 개를 쓰고 temperature만 바꾼 결과다. 이번 checkpoint의 learned logit scale은 약 100이라서 temperature로 바꾸면 약 0.01이다. Temperature가 낮아져도 문장의 순위는 바뀌지 않는다. Softmax 분포만 더 뾰족해진다.

이 확률은 후보 문장들 사이의 상대 점수다. 후보를 바꾸면 확률도 바뀐다. 99%가 나왔다고 해서 현실에서 99% 확실하다는 뜻은 아니다.

## 5. 같은 embedding으로 검색할 수 있다

Image-to-text retrieval은 이미지 하나와 가장 가까운 문장을 찾는다. Text-to-image retrieval은 반대로 문장 하나와 가장 가까운 이미지를 찾는다. 새 classifier를 학습하지 않아도 normalized embedding과 matrix multiplication으로 두 작업을 할 수 있다.[^2][^4]

```python
inputs = processor(
    images=images,
    text=captions,
    return_tensors="pt",
    padding=True,
)

with torch.inference_mode():
    outputs = model(**inputs)

image_features = F.normalize(outputs.image_embeds, dim=-1)
text_features = F.normalize(outputs.text_embeds, dim=-1)
similarity = image_features @ text_features.T
best_caption = similarity.argmax(dim=1)
```

| 이미지 | cats on sofa | two parrots | sports car | bowl of fruit | 가장 가까운 문장 |
| --- | ---: | ---: | ---: | ---: | --- |
| Cats | 0.299 | 0.205 | 0.204 | 0.189 | cats on sofa |
| Parrots | 0.164 | 0.297 | 0.186 | 0.189 | two parrots |

![두 이미지의 CLIP 검색과 zero-shot 분류 결과](/notes/tutorial/llm_lecture/images/w18_clip_retrieval_zero_shot.png)

*그림 2. 고양이와 앵무새 이미지의 cosine similarity matrix와 zero-shot 분류 결과. macOS CPU, PyTorch 2.13.0, Transformers 5.14.1, Matplotlib 3.11.1에서 직접 실행했다. 이미지 두 장만 사용한 작동 원리 실습이며 benchmark가 아니다.[^5]*

## 6. Class 이름을 문장으로 만들면 zero-shot classifier가 된다

Zero-shot 분류에서는 학습하지 않은 class 이름을 text encoder에 넣는다. 예를 들어 `cats`, `parrots`, `dogs`, `cars`를 각각 문장으로 만든 뒤 이미지와 가장 가까운 문장을 class로 고른다.

```python
classes = ["cats", "parrots", "dogs", "cars"]
prompts = [f"a photo of two {name}" for name in classes]

inputs = processor(
    images=images,
    text=prompts,
    return_tensors="pt",
    padding=True,
)
outputs = model(**inputs)
probabilities = outputs.logits_per_image.softmax(dim=1)
```

| 이미지 | Class 이름만 쓴 정답 확률 | `a photo of two ...` 정답 확률 | 정답 similarity 변화 |
| --- | ---: | ---: | ---: |
| Cats | 98.67% | 99.60% | 0.259 → 0.281 |
| Parrots | 99.97% | 99.98% | 0.283 → 0.297 |

두 prompt 모두 정답을 골랐지만 문장형 prompt에서 정답 similarity가 조금 높아졌다. CLIP 논문도 class 이름만 넣기보다 “a photo of a {class}”처럼 학습 caption에 가까운 문맥을 붙이고, 여러 prompt의 text embedding을 평균내는 방법을 사용했다. 논문의 ImageNet 실험에서는 prompt engineering과 ensembling을 합쳐 context 없는 기준보다 정확도가 약 5%p 높아졌다.[^1]

이번 checkpoint는 영어 자료로 학습되고 영어 이외의 언어를 목적으로 평가하지 않았다. 그래서 실습 prompt는 영어로 작성했다.[^6] 한국어 검색이 필요하다면 한국어 또는 multilingual data로 학습된 model을 고르고 별도 평가를 해야 한다.

## 7. 직접 실행한 환경과 결과를 기록한다

| 항목 | 값 |
| --- | --- |
| Model | `openai/clip-vit-base-patch32` |
| Revision | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| Image encoder | ViT-B/32 |
| Projection dimension | 512 |
| Device | macOS CPU |
| PyTorch | 2.13.0 |
| Transformers | 5.14.1 |
| 입력 | COCO 고양이 이미지 1장, Hugging Face 문서의 앵무새 이미지 1장 |
| Learned logit scale | 약 100.0000 |
| Model load | 약 3.93초 |
| 네 조건의 inference | 약 0.16초 |

시간은 warm-up과 반복 측정이 없는 한 번의 CPU 실행값이다. 속도 비교에는 쓰지 않는다. 이번 실습의 목적은 두 encoder의 출력, similarity matrix, temperature, zero-shot prompt가 이어지는 경로를 확인하는 데 있다.

## 8. 자주 생기는 실수

| 실수 | 생기는 문제 | 확인 방법 |
| --- | --- | --- |
| Embedding을 normalize하지 않고 내적함 | Vector 길이가 similarity에 섞임 | `F.normalize(..., dim=-1)` 확인 |
| Softmax 확률을 절대 신뢰도로 읽음 | 후보 목록에 따라 바뀌는 상대 점수를 과신함 | Cosine similarity와 후보 목록을 함께 기록 |
| Prompt마다 class 뜻이 달라짐 | 공정하지 않은 class 비교가 됨 | 같은 template을 모든 class에 적용 |
| 영어 checkpoint에 한국어만 입력함 | 학습 언어 차이로 검색 품질이 떨어질 수 있음 | Model card의 학습 언어와 평가 범위 확인 |
| 이미지와 text 순서를 섞음 | Contrastive loss의 대각선 정답이 틀어짐 | Pair ID와 target index를 함께 저장 |
| 이미지 두 장의 성공을 accuracy로 보고함 | 표본이 너무 작아 일반화할 수 없음 | 정식 dataset의 test split으로 별도 평가 |
| 사람·민감 속성을 제한 없이 분류함 | 학습 data의 bias와 class 설계 문제가 커짐 | Model card의 out-of-scope와 bias 항목 확인 |

OpenAI model card는 CLIP의 성능과 bias가 class 설계에 따라 달라지며, 배포 전에 task별 검증이 필요하다고 설명한다. 감시나 얼굴 인식 같은 용도는 사용 범위 밖으로 둔다.[^6]

## 확인 문제

1. CLIP이 image encoder와 text encoder를 따로 두는 이유는 무엇인가?
2. 두 encoder의 마지막 embedding 차원을 같게 만드는 까닭은 무엇인가?
3. Embedding을 길이 1로 normalize하면 cosine similarity 계산이 어떻게 단순해지는가?
4. $N$개의 image-text 쌍으로 만든 similarity matrix의 shape는 무엇이며, 대각선은 무엇을 뜻하는가?
5. CLIP이 image-to-text와 text-to-image loss를 함께 쓰는 이유를 설명해보자.
6. Temperature를 1.0에서 0.01로 낮추면 similarity 순위와 softmax 분포는 각각 어떻게 변하는가?
7. Zero-shot 분류에서 class 이름을 문장형 prompt로 바꾸는 이유는 무엇인가?
8. 후보가 `cats`, `parrots` 두 개일 때의 99%와 후보가 1,000개일 때의 99%를 그대로 비교하면 안 되는 이유는 무엇인가?
9. 이번 실습 결과를 CLIP의 일반 accuracy라고 부를 수 없는 이유는 무엇인가?

## 완료 체크

- [x] Image encoder와 text encoder의 출력을 같은 embedding 공간에서 비교했다.
- [x] Cosine similarity와 normalization을 코드로 계산했다.
- [x] $N \times N$ similarity matrix에서 positive와 negative 쌍을 구분했다.
- [x] Symmetric contrastive loss와 temperature의 역할을 설명했다.
- [x] 이미지-글 검색과 zero-shot 분류를 직접 실행했다.
- [x] Class 이름과 문장형 prompt의 결과를 비교했다.
- [x] 작은 예제 결과를 benchmark와 구분했다.

---

[^1]: Radford, A. et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020). Figure 1, §2.3의 contrastive objective, §3.1의 zero-shot classifier와 prompt engineering을 참고했다.
[^2]: OpenAI. [CLIP 공식 구현](https://github.com/openai/CLIP). Image와 text encoding, normalized feature의 similarity, zero-shot prediction 예제를 참고했다. 확인일: 2026-08-04.
[^3]: Hugging Face. [`openai/clip-vit-base-patch32` model card](https://huggingface.co/openai/clip-vit-base-patch32). ViT-B/32 구조, 입력 처리와 checkpoint 사용법을 참고했다. 실습에서는 revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`을 사용했다. 확인일: 2026-08-04.
[^4]: Hugging Face. [Transformers: CLIP](https://huggingface.co/docs/transformers/model_doc/clip). Image feature와 text feature를 같은 latent space에서 비교하는 API를 참고했다. 확인일: 2026-08-04.
[^5]: 직접 실행한 `llm_lecture/week18_clip_demo.py`의 결과다. 고양이 이미지는 CLIP model card가 예제로 연결한 [COCO 이미지](http://images.cocodataset.org/val2017/000000039769.jpg), 앵무새 이미지는 Hugging Face의 [documentation image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png)를 사용했다. 원본 이미지, CSV, JSON, PDF와 실행 코드는 Git에서 제외했다. 실행일: 2026-08-04.
[^6]: OpenAI. [CLIP Model Card](https://github.com/openai/CLIP/blob/main/model-card.md). 영어 이외 언어의 평가 범위, task별 검증 필요성, bias와 out-of-scope use를 참고했다. 확인일: 2026-08-04.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
원본/윤문본: 11,485자 / 11,241자, 변경률 2.12%
카테고리별 탐지/수정: A-10 2→0, A-18 2→0, C-11 0→0, D-1 0→0, H-1 0→0
정량 점검: humanize-korean metrics v2.0 risk band low
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 논문 구조, model 설정, 직접 실행 수치를 그대로 보존함
주요 변경: 긴 문장을 나누고 similarity, temperature, contrastive loss를 작은 표와 코드로 풀어씀
-->
