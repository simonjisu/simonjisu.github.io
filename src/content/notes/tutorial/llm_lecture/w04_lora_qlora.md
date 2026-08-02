---
title: "4주차. LoRA와 QLoRA"
description: "전체 weight를 고치는 대신 작은 adapter를 학습하고, 4-bit 양자화로 GPU 메모리를 더 줄이는 방법을 익힌다."
tags:
  - LLM
  - LoRA
  - QLoRA
  - PEFT
  - quantization
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 16주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

3주차에는 모든 weight를 바꾸는 SFT를 실행했다. 모델이 커지면 weight뿐 아니라 gradient와 optimizer state도 GPU 메모리를 차지한다. 이번 주에는 원래 weight를 얼리고 작은 LoRA adapter만 학습한다. 이어서 Base model을 4-bit로 보관하는 QLoRA까지 비교한다.

## 이번 주에 배울 것

- full fine-tuning과 LoRA가 바꾸는 parameter의 차이
- LoRA의 두 작은 행렬 $A$, $B$와 rank $r$의 뜻
- `r`, `lora_alpha`, `target_modules`가 맡는 역할
- QLoRA의 4-bit NF4, double quantization, paged optimizer
- LoRA와 QLoRA의 trainable parameter, GPU memory, 학습 시간을 비교하는 방법

선수 지식은 3주차의 SFT 데이터, assistant-only loss, train/validation 분리다.

!!! note "교과서 대신 얇은 정정 노트를 고친다"

    full fine-tuning은 두꺼운 교과서의 모든 문장을 다시 인쇄하는 일과 비슷하다. LoRA는 교과서는 그대로 두고 작은 정정 노트만 학습한다. QLoRA는 교과서를 더 작은 글자 체계로 압축해 보관하고, 정정 노트는 계산하기 좋은 정밀도로 학습한다.

## 1. LoRA는 weight의 변화량을 작게 나눈다

![LoRA의 pretrained weight와 저랭크 행렬 A, B](/notes/tutorial/llm_lecture/images/w04_lora_reparameterization.png)

*그림 1. pretrained weight $W$는 고정하고 저랭크 행렬 $A$, $B$만 학습하는 LoRA 구조. 출처: Hu et al. (2021), Figure 1에서 발췌.[^1]*

일반적인 linear layer는 입력 $x$에 weight $W_0$를 곱한다. full fine-tuning은 $W_0$의 모든 값을 바꾼다. LoRA는 $W_0$를 고정하고, 변화량 $\Delta W$를 작은 두 행렬의 곱 $BA$로 나타낸다. 전체 식은 $h=W_0x+\frac{\alpha}{r}BAx$다.[^1]

$W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$일 때 두 행렬의 shape는 $A \in \mathbb{R}^{r \times d_{\text{in}}}$와 $B \in \mathbb{R}^{d_{\text{out}} \times r}$이다.

rank $r$는 중간 통로의 폭이다. $r$가 작으면 학습할 값이 적고, 너무 작으면 필요한 변화를 충분히 표현하지 못하기도 한다. 반대로 $r$를 키우면 표현 범위가 넓어지지만 parameter와 메모리도 증가한다.

예를 들어 입력과 출력 차원이 모두 4,096인 layer를 생각해보자.

```text
full weight parameter = 4096 × 4096       = 16,777,216
LoRA parameter (r=16) = 16 × 4096 × 2     =    131,072
비율                                      ≈      0.78%
```

이 계산은 layer 하나의 단순 예시다. 실제 모델의 trainable parameter 비율은 LoRA를 붙이는 layer와 module 수에 따라 달라진다.

## 2. LoRA 설정값은 각각 역할이 다르다

| 설정 | 쉬운 뜻 | 값이 커질 때 |
| --- | --- | --- |
| `r` | 변화량을 표현하는 통로의 폭 | parameter와 표현력이 함께 늘어남 |
| `lora_alpha` | adapter 출력의 크기를 조절하는 값 | 실제 update의 scale이 달라짐 |
| `lora_dropout` | 학습 중 adapter 입력 일부를 가림 | 과적합을 줄일 수 있지만 지나치면 학습이 약해짐 |
| `target_modules` | LoRA를 붙일 layer 목록 | 대상이 많을수록 trainable parameter가 늘어남 |

LoRA 식에서 scale은 보통 `lora_alpha / r`로 적용된다. `r`만 바꾸고 alpha를 그대로 두면 parameter 수뿐 아니라 update의 크기도 함께 달라진다. 실험표에는 두 값을 같이 기록한다.

PEFT는 모델 구조를 알고 있으면 기본 target module을 고른다. QLoRA 방식처럼 Transformer의 linear layer 전반에 adapter를 붙일 때는 `target_modules="all-linear"`를 사용한다.[^3] 어떤 설정이 늘 최고라고 가정하지 말고, trainable parameter 수와 validation 결과를 함께 확인한다.

## 3. QLoRA는 Base model을 4-bit로 보관한다

![Full fine-tuning, LoRA, QLoRA의 메모리 구조 비교](/notes/tutorial/llm_lecture/images/w04_qlora_memory_comparison.png)

*그림 2. full fine-tuning, LoRA, QLoRA가 Base model과 adapter를 보관하고 update하는 방식. 출처: Dettmers et al. (2023), Figure 1에서 발췌.[^2]*

그림의 파란 화살표는 parameter update, 초록 화살표는 gradient 흐름이다. full fine-tuning은 Base model의 weight와 optimizer state를 모두 다룬다. LoRA는 16-bit Base model을 고정하고 adapter만 update한다. QLoRA는 고정된 Base model을 4-bit로 보관해 메모리를 더 줄인다.[^2]

!!! note "4-bit로 저장해도 계산까지 4-bit인 것은 아니다"

    QLoRA는 Base model weight를 4-bit로 보관하지만 행렬 계산에는 `bfloat16` 같은 더 높은 정밀도를 쓴다. gradient는 고정된 Base model을 지나 LoRA adapter로 흐르고, 실제로 학습되는 값은 adapter parameter다.

QLoRA 논문은 메모리를 줄이기 위해 세 가지를 함께 사용했다.[^2]

- **NF4**: 정규분포에 가까운 pretrained weight를 4-bit로 나타내도록 설계한 자료형이다.
- **Double Quantization**: weight를 양자화할 때 쓰는 상수도 다시 양자화한다.
- **Paged Optimizer**: 긴 sequence에서 갑자기 메모리가 치솟을 때 CPU memory를 활용해 out-of-memory 위험을 줄인다.

현재 Transformers 문서도 4-bit Base model 학습에 NF4를 권하고, nested quantization으로 parameter당 약 0.4 bit를 더 줄일 수 있다고 설명한다.[^5] 이 수치는 저장 공간에 대한 설명이지 모델이 네 배 빨라진다는 뜻은 아니다. 속도는 GPU, kernel, sequence length, batch 크기에 따라 달라진다.

## 4. 먼저 LoRA를 실행한다

3주차와 같은 모델, 데이터, seed, `max_length`를 사용해야 학습 방법의 차이를 비교하기 쉽다. 이 실습도 `Qwen/Qwen3-0.6B`를 쓴다.[^6] LoRA와 QLoRA에서는 작은 adapter만 학습하므로 full fine-tuning보다 높은 learning rate를 쓰는 경우가 많다. 현재 TRL 가이드는 SFT 예시로 full fine-tuning의 `2e-5`, LoRA의 `2e-4`를 제시한다.[^4]

```bash
pip install -U "trl[peft]" transformers datasets accelerate
```

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("trl-lib/Capybara", split="train[:1000]")
split = dataset.train_test_split(test_size=0.1, seed=42)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

args = SFTConfig(
    output_dir="outputs/w04_qwen3_lora",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_length=512,
    packing=False,
    assistant_only_loss=True,
    eval_strategy="steps",
    eval_steps=50,
    logging_steps=10,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    model_init_kwargs={
        "dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    },
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    args=args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    peft_config=lora_config,
)

trainer.model.print_trainable_parameters()
trainer.train()
trainer.save_model()
```

출력된 `trainable params`, `all params`, `trainable%`를 기록한다. 모델 크기만 보고 adapter 크기를 짐작하지 말고 실제 값을 남긴다.

## 5. 같은 설정에서 QLoRA로 바꾼다

QLoRA에는 `bitsandbytes`가 더 필요하다. 아래 코드는 현재 TRL의 PEFT integration 방식처럼 `quantization_config`와 `peft_config`를 `SFTTrainer`에 함께 전달한다.[^4]

```bash
pip install -U bitsandbytes
```

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("trl-lib/Capybara", split="train[:1000]")
split = dataset.train_test_split(test_size=0.1, seed=42)

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

args = SFTConfig(
    output_dir="outputs/w04_qwen3_qlora",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_length=512,
    packing=False,
    assistant_only_loss=True,
    eval_strategy="steps",
    eval_steps=50,
    logging_steps=10,
    bf16=compute_dtype == torch.bfloat16,
    fp16=compute_dtype == torch.float16,
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    args=args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    quantization_config=quantization_config,
    peft_config=lora_config,
)

trainer.model.print_trainable_parameters()
trainer.train()
trainer.save_model()
```

`bfloat16`을 지원하지 않는 GPU라면 `bnb_4bit_compute_dtype`와 학습 precision을 장비에 맞게 바꿔야 한다. QLoRA가 메모리를 줄여도 모든 GPU와 운영체제에서 같은 kernel을 지원하는 것은 아니다. 설치 오류가 나면 먼저 공식 bitsandbytes hardware compatibility 표와 CUDA 버전을 확인한다.[^5]

## 6. 세 방법을 같은 표로 비교한다

full fine-tuning, LoRA, QLoRA를 비교할 때 데이터와 생성 설정이 달라지면 원인을 구분하기 어렵다. 아래 표의 빈칸을 실제 측정값으로 채운다.

| 항목 | Full FT | LoRA | QLoRA |
| --- | ---: | ---: | ---: |
| trainable parameter |  |  |  |
| adapter/checkpoint 크기 |  |  |  |
| peak GPU memory |  |  |  |
| 100 step 학습 시간 |  |  |  |
| validation loss |  |  |  |
| 지시 준수 점수 |  |  |  |

`torch.cuda.max_memory_allocated()`는 PyTorch가 할당한 peak memory를 재는 한 방법이다. 실행 직전에 `torch.cuda.reset_peak_memory_stats()`를 호출하고, 학습 뒤 값을 GB로 바꿔 기록한다.

```python
torch.cuda.reset_peak_memory_stats()
trainer.train()
peak_gb = torch.cuda.max_memory_allocated() / 1024**3
print(f"peak allocated: {peak_gb:.2f} GB")
```

adapter를 merge하면 Base model weight에 LoRA 변화량을 합친 새 모델을 만든다. 배포 파일은 단순해지지만, adapter만 바꿔 여러 작업을 전환하는 장점은 사라지고 전체 weight를 저장할 공간이 필요하다. merge 전후에는 같은 prompt와 greedy decoding으로 답변을 비교한다. dtype과 연산 순서가 달라지면 아주 작은 수치 차이가 생기기도 한다.

!!! warning "메모리만 보고 방법을 고르지 않는다"

    QLoRA의 peak memory가 가장 작더라도 학습 속도나 최종 품질이 항상 최고인 것은 아니다. 같은 데이터와 seed에서 메모리, 시간, validation loss, 실제 답변을 함께 본다.

## 7. 실제 비교 결과

`HuggingFaceTB/SmolLM2-135M-Instruct`를 Apple MPS에서 Full FT와 LoRA로 각각 4 step 학습했다. LoRA는 rank 16으로 모든 linear layer에 adapter를 붙였다.[^3][^7]

| 방법 | trainable parameter | 전체 대비 | weight 저장량 추정 | 평균 step 시간 | 마지막 loss | 실행 여부 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full FT | 134,515,008 | 100.00% | 513.1 MB | 0.318초 | 1.699 | 실측 |
| LoRA | 4,884,480 | 3.50% | 531.8 MB | 0.406초 | 1.604 | 실측 |
| QLoRA | 4,884,480 | 3.50% | 82.8 MB | — | — | 저장량만 추정 |

![Full FT, LoRA, QLoRA의 parameter와 저장량 및 학습 시간 비교](/notes/tutorial/llm_lecture/images/w04_peft_comparison_result.png)

*그림 3. Full FT와 LoRA 실측값, QLoRA 저장량 추정값. 출처: SmolLM2-135M-Instruct 직접 실행 결과(2026-08-01, Apple MPS).[^7]*

LoRA로 실제 학습한 parameter는 전체의 3.50%였다. 그래도 표의 저장량이 Full FT보다 큰 까닭은 32-bit Base model과 adapter를 로컬 메모리에 함께 올린 크기를 계산했기 때문이다. adapter checkpoint만 저장한 파일 크기나 학습 중 peak memory와는 다른 값이다.

이 작은 모델에서는 LoRA의 한 step이 Full FT보다 약 0.09초 느렸다. 학습할 parameter가 적다고 매 step이 반드시 빨라지는 것은 아니다. adapter 계산과 작은 모델에서 생기는 고정 비용이 비율상 크게 나타날 수 있다.

QLoRA는 실행하지 못했다. 이번 환경의 Apple MPS에서는 bitsandbytes 4-bit 학습에 필요한 CUDA를 사용할 수 없어서, Base weight를 4-bit로 저장한다고 가정한 크기만 계산했다.[^5] 82.8 MB를 실제 peak memory나 학습 속도로 해석하면 안 된다.

## 확인 문제

1. LoRA가 $W_0$를 고정해도 모델의 출력이 달라지는 이유는 무엇인가?
2. rank $r$를 키우면 trainable parameter 수가 어떻게 달라지는가?
3. `lora_alpha`와 `r`를 함께 기록해야 하는 이유는 무엇인가?
4. QLoRA에서 Base model은 4-bit인데 계산은 더 높은 정밀도로 할 수 있는 이유를 설명해보자.
5. adapter merge가 편리한 경우와 불편한 경우를 하나씩 적어보자.

## 완료 체크

- [ ] LoRA의 $W_0x + (\alpha/r)BAx$ 식과 각 행렬 shape를 설명했다.
- [ ] LoRA와 QLoRA의 trainable parameter 수를 출력했다.
- [ ] 같은 데이터와 seed로 LoRA와 QLoRA를 각각 실행했다.
- [ ] peak GPU memory, 학습 시간, validation loss, 답변을 표로 비교했다.
- [ ] rank를 두 가지 이상 바꿔 parameter 수와 결과 차이를 기록했다.
- [ ] adapter merge 전후 출력을 같은 설정으로 비교했다.
- [ ] 결과물로 `PEFT 비교 보고서`를 완성했다.

---

[^1]: Hu, E. J. et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). Figure 1과 §4를 참고했다.
[^2]: Dettmers, T. et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). Figure 1과 §3을 참고했다.
[^3]: Hugging Face. [PEFT: LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora). `LoraConfig`와 QLoRA-style `target_modules="all-linear"` 설정을 참고했다. 확인일: 2026-07-31.
[^4]: Hugging Face. [TRL: PEFT Integration](https://huggingface.co/docs/trl/main/peft_integration). LoRA learning rate와 QLoRA의 `quantization_config` 사용법을 참고했다. 확인일: 2026-07-31.
[^5]: Hugging Face. [Transformers: bitsandbytes](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes). NF4, compute dtype, nested quantization을 참고했다. 확인일: 2026-07-31.
[^6]: Qwen Team. [Qwen/Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B). 확인일: 2026-07-31.
[^7]: Hugging Face. [HuggingFaceTB/SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct). 실행 모델의 구조와 사용법을 참고했다. 확인일: 2026-08-01.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 절별로 5,000자 이하로 나누어 점검
원본/윤문본: 9748자 / 9882자, 변경률 1.37%
탐지/수정: A-10 7→1, D-1 0→0, H-1 0→0, I-2 0→0, 그 밖의 S1 0→0
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 수식·수치·설정값을 보존하며 반복 표현만 줄임
주요 변경: “쓸 수 있다”→“사용한다”, LoRA 식과 행렬 shape를 inline math 형식으로 변경, Full FT·LoRA 실측표와 QLoRA 미실행 사유 추가
-->
