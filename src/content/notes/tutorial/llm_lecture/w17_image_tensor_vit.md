---
title: "17주차. 이미지 tensor와 Vision Transformer"
description: "RGB 이미지가 pixel_values와 patch embedding으로 바뀌어 Vision Transformer에 들어가는 과정을 실제 tensor shape로 따라간다."
tags:
  - VLM
  - Vision Transformer
  - image preprocessing
  - patch embedding
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 26주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

VLM은 이미지와 글을 함께 읽는다. 글을 token으로 나누는 과정은 앞에서 배웠지만, 사진은 어떻게 token이 될까? 이번 주에는 RGB 이미지 한 장이 tensor로 바뀌고, 작은 patch로 나뉘어 Transformer에 들어가는 과정을 따라간다.

## 이번 주에 배울 것

- RGB 이미지에서 channel, height, width가 뜻하는 것
- Resize, center crop, rescale, normalization의 역할
- `pixel_values`의 `[batch, channel, height, width]` shape
- 이미지를 patch로 나누고 embedding으로 바꾸는 과정
- CLS token과 position embedding이 필요한 이유
- 해상도와 patch 크기가 visual token 수를 바꾸는 방식
- 사전 학습된 작은 Vision Transformer의 실제 tensor shape

선수 지식은 Python list와 PyTorch tensor의 shape, 1주차에서 배운 embedding과 self-attention이다. CNN의 세부 구조는 몰라도 된다.

!!! note "사진을 작은 낱말로 나누기"

    언어 모델은 긴 문장을 token으로 나눈다. Vision Transformer는 사진을 작은 정사각형 patch로 자른다. Patch 하나를 글의 token 하나처럼 생각하면, 1주차에서 배운 Transformer 구조를 이미지에도 적용할 수 있다.

## 1. 이미지도 숫자로 이루어진 tensor다

컴퓨터에서 RGB 이미지는 빨강, 초록, 파랑의 세 channel로 표현한다. 각 pixel은 channel마다 밝기 값을 하나씩 갖는다. 일반적인 8-bit 이미지는 0부터 255까지의 정수를 쓴다.

| RGB 값 | 화면에서 보이는 색 |
| --- | --- |
| `[255, 0, 0]` | 빨강 |
| `[0, 255, 0]` | 초록 |
| `[0, 0, 255]` | 파랑 |
| `[255, 255, 255]` | 흰색 |
| `[0, 0, 0]` | 검은색 |

이미지 library와 deep learning framework가 channel을 놓는 순서는 다를 수 있다.

```text
PIL·NumPy 이미지  [height, width, channel]
PyTorch 입력      [batch, channel, height, width]
```

세로 240px, 가로 360px인 RGB 이미지를 NumPy 배열로 보면 `[240, 360, 3]`이다. PyTorch model에 한 장만 넣을 때는 batch 차원까지 붙어 `[1, 3, 240, 360]`이 된다.

!!! warning "shape만 보고 순서를 짐작하지 않는다"

    `[224, 224, 3]`과 `[3, 224, 224]`는 같은 숫자를 담을 수 있어도 뜻이 다르다. Model이 channel-first 입력을 기대하는데 channel-last tensor를 넣으면 오류가 나거나 잘못된 축을 이미지로 읽는다.

## 2. Processor는 model이 배웠던 입력 조건을 맞춘다

사진마다 크기와 밝기 범위가 다르다. 사전 학습된 model은 훈련할 때 정해진 resize, crop, rescale, normalization 규칙으로 이미지를 보았다. 추론할 때도 같은 규칙을 따라야 한다. Hugging Face의 Image Processor는 이 전처리를 model 설정과 함께 불러온다.[^2]

| 처리 | 하는 일 | 필요한 까닭 |
| --- | --- | --- |
| Resize | 이미지의 가로와 세로를 model 입력 크기에 맞춤 | Batch 안의 tensor shape를 같게 만듦 |
| Center crop | 가운데 영역을 정해진 크기로 자름 | 훈련 때 사용한 화면 범위를 맞춤 |
| Rescale | 0-255 값을 주로 0-1 범위로 바꿈 | 계산하기 알맞은 값의 크기로 줄임 |
| Normalize | Channel별 평균을 빼고 표준편차로 나눔 | 훈련 때 model이 본 값의 분포와 맞춤 |

Pixel 값이 $x$, rescale 뒤 값이 $x'=x/255$, 평균이 $\mu$, 표준편차가 $\sigma$라면 normalization은 $\hat{x}=(x'-\mu)/\sigma$로 쓸 수 있다. 이번 실습 model은 RGB 세 channel 모두 $\mu=0.5$, $\sigma=0.5$를 쓴다. 0은 -1, 0.5는 0, 1은 1이 된다.

```python
from PIL import Image
from transformers import DeiTImageProcessorPil

model_id = "facebook/deit-tiny-patch16-224"
revision = "25f8de47268ea80b6f2227b1c6075084095c8131"

processor = DeiTImageProcessorPil.from_pretrained(
    model_id,
    revision=revision,
)
image = Image.open("sample.png").convert("RGB")
batch = processor(images=image, return_tensors="pt")

print(batch["pixel_values"].shape)
# torch.Size([1, 3, 224, 224])
```

`DeiTImageProcessorPil`은 이번 실행 환경에 `torchvision`이 없어 사용한 PIL backend다. `torchvision`이 설치된 일반 환경에서는 `AutoImageProcessor`를 써도 된다. 중요한 것은 class 이름보다 checkpoint에 저장된 전처리 설정을 함께 불러오는 일이다.[^2][^3]

이번 processor는 360×240 이미지를 224×224로 바꿨다. 원본의 원은 전처리 뒤 세로로 긴 타원처럼 보인다. 가로세로 비율이 다른 이미지를 정사각형으로 바꾸면 모양이 달라질 수 있다는 뜻이다. Model마다 짧은 변을 기준으로 resize한 뒤 crop하거나, 이미지를 여러 tile로 나누는 등 규칙이 다르므로 현재 processor 설정을 확인해야 한다.

## 3. Vision Transformer는 이미지를 patch sequence로 바꾼다

![Vision Transformer의 patch embedding과 encoder 구조](/notes/tutorial/llm_lecture/images/w17_vit_architecture.png)

*그림 1. 이미지를 고정 크기 patch로 나누고 linear projection, position embedding, Transformer encoder를 거치는 구조. 출처: Dosovitskiy et al. (2020), Figure 1에서 발췌.[^1]*

높이 $H$, 너비 $W$, channel 수 $C$인 이미지를 $P \times P$ patch로 나눈다고 하자. Patch 수는 $N=HW/P^2$이다. 입력이 224×224이고 patch가 16×16이면 한 변에서 $224/16=14$개가 나온다. 전체 patch는 $14 \times 14=196$개다.

Patch 하나에는 $16 \times 16 \times 3=768$개의 숫자가 들어 있다. 이 숫자를 한 줄로 펴고 linear projection을 적용하면 model의 hidden size $D$에 맞는 embedding이 된다. 이번 model의 $D$는 192다.

```text
pixel_values         [1, 3, 224, 224]
Conv2d projection    [1, 192, 14, 14]
patch embeddings     [1, 196, 192]
```

구현에서는 kernel size와 stride가 모두 16인 `Conv2d`로 patch 분할과 linear projection을 한꺼번에 처리할 수 있다. 겹치지 않는 16×16 영역마다 같은 projection weight를 적용하기 때문이다. 출력 `[1, 192, 14, 14]`의 공간 축을 한 줄로 펴고 순서를 바꾸면 `[1, 196, 192]`가 된다.[^3]

```python
projection = model.vit.embeddings.patch_embeddings.projection(pixel_values)
patch_embeddings = projection.flatten(2).transpose(1, 2)

print(projection.shape)       # [1, 192, 14, 14]
print(patch_embeddings.shape) # [1, 196, 192]
```

## 4. Patch 순서와 전체 이미지 표시는 따로 더한다

Patch를 펼치면 Transformer는 1차원 sequence를 받는다. 이때 각 embedding만 보고는 왼쪽 위 patch인지 오른쪽 아래 patch인지 알기 어렵다. ViT는 patch embedding에 학습 가능한 position embedding을 더해 위치를 알려준다.[^1]

이미지 분류에는 CLS token도 앞에 하나 붙인다. 모든 patch와 self-attention을 주고받은 뒤, 마지막 CLS 표현을 이미지 전체를 대표하는 vector로 사용한다.

```text
patch token 수             196
CLS token 수                 1
Transformer sequence 길이  197
encoder output       [1, 197, 192]
```

ViT encoder의 self-attention에는 Causal LM처럼 미래를 가리는 causal mask가 없다. 사진의 오른쪽 아래 patch가 왼쪽 위 patch를 봐도 되고, 반대 방향도 가능하다. 이미지는 다음 patch를 순서대로 생성하는 문제가 아니라 전체 장면을 함께 읽는 입력이기 때문이다.

| Causal LM | Vision Transformer encoder |
| --- | --- |
| Text token을 입력으로 사용 | Image patch를 입력으로 사용 |
| 보통 왼쪽 token만 보는 causal mask 사용 | 모든 patch 사이의 attention 허용 |
| 다음 token 예측에 사용 | 이미지 표현과 분류 등에 사용 |
| Token 위치 표현이 필요 | Patch 위치 표현이 필요 |

## 5. 해상도가 커지면 visual token은 빠르게 늘어난다

Patch 크기를 16으로 고정하면 이미지 한 변의 길이가 두 배가 될 때 patch token 수는 네 배가 된다. Attention은 token 쌍을 비교하므로 score 수는 sequence 길이의 제곱에 비례한다. 정사각형 이미지에서는 해상도가 두 배가 될 때 attention score 수가 대략 16배가 되는 셈이다.

| 입력 해상도 | 한 변의 patch | Patch token | CLS 포함 sequence | Head 하나의 attention 쌍 |
| ---: | ---: | ---: | ---: | ---: |
| 224×224 | 14 | 196 | 197 | 38,809 |
| 384×384 | 24 | 576 | 577 | 332,929 |
| 512×512 | 32 | 1,024 | 1,025 | 1,050,625 |
| 1024×1024 | 64 | 4,096 | 4,097 | 16,785,409 |

이 표는 patch size 16에서 $N=(R/16)^2$와 $(N+1)^2$을 계산한 값이다.[^5] 실제 VLM은 입력을 고정 해상도로 줄이거나, 여러 tile로 나누거나, visual token을 압축해 비용을 조절한다. 그 방법은 19주차의 VLM 연결 구조와 22주차의 멀티모달 서빙에서 다시 다룬다.

## 6. 직접 실행한 전처리와 forward 결과

![RGB 이미지 전처리와 해상도별 patch token 증가](/notes/tutorial/llm_lecture/images/w17_preprocessing_patch_growth.png)

*그림 2. 왼쪽은 코드로 그린 360×240 RGB 이미지, 가운데는 DeiT Processor의 224×224 출력과 16×16 patch grid다. 오른쪽은 입력 해상도에 따른 patch token과 head 하나의 attention 쌍을 계산한 결과다. macOS CPU, PyTorch 2.13.0, Transformers 5.14.1, Matplotlib 3.11.1에서 직접 실행했다.[^5]*

사전 학습된 `facebook/deit-tiny-patch16-224`를 고정 revision으로 불러왔다. DeiT는 ViT 구조를 더 적은 데이터로 학습하는 방법을 연구한 model이다. 이번에 사용한 tiny checkpoint는 224×224 입력과 16×16 patch를 쓰며 hidden size 192, encoder 12개, attention head 3개로 구성된다.[^3][^4]

| 단계 | 실제 shape |
| --- | --- |
| 원본 RGB 이미지 | `[240, 360, 3]` |
| `pixel_values` | `[1, 3, 224, 224]` |
| Conv2d projection | `[1, 192, 14, 14]` |
| Patch embedding | `[1, 196, 192]` |
| CLS 포함 encoder 출력 | `[1, 197, 192]` |
| 마지막 layer attention | `[1, 3, 197, 197]` |

전처리된 `pixel_values`의 최솟값은 약 -0.969, 최댓값은 1.000, 평균은 약 0.450이었다. 값이 정확히 -1부터 1까지 모두 채워져야 하는 것은 아니다. 실제 그림에 들어 있는 색이 차지하는 범위에 따라 최솟값과 평균이 달라진다.

CPU forward 한 번은 약 0.015초였지만 warm-up과 반복 측정이 없는 단일 실행이다. Model이나 device의 속도 비교에는 쓰지 않는다. 이번 결과에서 확인할 것은 시간보다 196개 patch가 197개 sequence token으로 바뀌고, head 3개가 각각 197×197 attention matrix를 만든다는 사실이다.

!!! warning "Attention 그림을 곧바로 model의 설명이라고 부르지 않는다"

    Attention weight가 어느 patch를 많이 읽었는지는 살펴볼 수 있다. 하지만 값 하나만 보고 model이 그 물체 때문에 답했다고 단정할 수는 없다. Layer와 head에 따라 값이 다르고, MLP와 residual connection도 출력에 영향을 준다.

## 7. 자주 생기는 실수

| 실수 | 생기는 문제 | 확인 방법 |
| --- | --- | --- |
| RGB 대신 BGR을 넣음 | 빨강과 파랑 channel이 뒤바뀜 | 입력 library와 `convert("RGB")` 확인 |
| HWC와 CHW를 혼동함 | Channel 축을 잘못 읽음 | Model 직전 tensor shape 출력 |
| 0-1 이미지에 rescale을 또 적용함 | 값이 255분의 1로 지나치게 작아짐 | 입력 범위와 `do_rescale` 확인 |
| Checkpoint와 다른 mean·std를 사용함 | 훈련 때와 입력 분포가 달라짐 | Processor config 기록 |
| CLS token을 빼고 sequence 길이를 계산함 | Attention shape가 하나씩 어긋남 | Patch 수에 special token을 더함 |
| 입력 해상도만 임의로 키움 | Position embedding shape 오류나 큰 비용 발생 | Model의 지원 해상도와 interpolation 확인 |

Model ID뿐 아니라 Processor 설정도 실험 기록에 남긴다. 같은 이미지를 넣었는데 결과가 달라졌다면 weight보다 resize, crop, RGB 변환, mean·std가 먼저 달라졌을 수 있다.

## 확인 문제

1. RGB 이미지의 channel 수가 3인 이유는 무엇인가?
2. `[1, 3, 224, 224]`에서 첫 번째 1과 두 번째 3은 각각 무엇을 뜻하는가?
3. 224×224 이미지를 16×16 patch로 나누면 patch가 196개인 이유를 계산해보자.
4. Patch vector의 768차원이 192차원 embedding으로 바뀌는 단계는 무엇인가?
5. Patch token이 196개인데 encoder sequence가 197개인 이유는 무엇인가?
6. ViT encoder가 Causal LM의 causal mask를 사용하지 않는 이유는 무엇인가?
7. 입력 해상도를 두 배로 키웠을 때 patch token과 attention 쌍은 각각 대략 몇 배가 되는가?

## 완료 체크

- [x] RGB 이미지와 `pixel_values`의 축 순서를 구분했다.
- [x] Resize, crop, rescale, normalization의 역할을 설명했다.
- [x] ViT 원 논문의 Figure 1에서 patch와 encoder 흐름을 확인했다.
- [x] 224×224 입력에서 16×16 patch 196개를 계산했다.
- [x] 사전 학습된 작은 model에서 전처리와 forward shape를 실행했다.
- [x] 해상도별 patch token과 attention 쌍을 표와 그림으로 비교했다.

---

[^1]: Dosovitskiy, A. et al. (2020). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). Figure 1과 §3.1의 patch embedding, position embedding, CLS token 구조를 참고했다.
[^2]: Hugging Face. [Transformers: Image processors](https://huggingface.co/docs/transformers/main/image_processors). Resize, crop, rescale, normalization과 `pixel_values` 생성을 참고했다. 확인일: 2026-08-04.
[^3]: Hugging Face. [facebook/deit-tiny-patch16-224 model card](https://huggingface.co/facebook/deit-tiny-patch16-224). Model 입력 해상도, patch 크기, ImageNet-1k 학습 정보와 checkpoint를 참고했다. 실습에서는 revision `25f8de47268ea80b6f2227b1c6075084095c8131`을 사용했다. 확인일: 2026-08-04.
[^4]: Touvron, H. et al. (2020). [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877). DeiT의 ViT 기반 구조와 data-efficient training 방법을 참고했다.
[^5]: 직접 실행한 `llm_lecture/week17_vit_demo.py`의 결과다. 코드로 그린 360×240 RGB 이미지를 사용했고, `facebook/deit-tiny-patch16-224`의 전처리와 CPU forward를 실행했다. 분류 품질을 평가하는 자료가 아니라 tensor shape와 patch 수를 확인하는 실습이다. 원본 PNG, CSV, JSON은 Git에서 제외했다. 실행일: 2026-08-04.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
원본/윤문본: 10,418자 / 10,281자, 변경률 1.31%
카테고리별 탐지/수정: A-10 1→0, A-18 1→0, C-11 0→0, D-1 0→0, H-1 0→0
정량 점검: humanize-korean metrics v2.0 risk band low
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 논문 구조, model 설정, 직접 실행 수치를 그대로 보존함
주요 변경: 긴 문장을 나누고 patch 수식과 tensor shape를 표와 짧은 예제로 풀어씀
-->
