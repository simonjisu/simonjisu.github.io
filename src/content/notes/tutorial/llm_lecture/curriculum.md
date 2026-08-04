---
title: "LLM과 VLM은 어떻게 배우고, Agent는 어떻게 기억할까: 26주 학습 커리큘럼"
description: "Transformer의 기초부터 모델 학습과 서빙, 이미지와 언어를 함께 다루는 VLM, 경험을 저장하고 다시 쓰는 Agent Memory까지 직접 실험하며 익히는 과정"
tags:
  - LLM
  - VLM
  - Agent Memory
  - Transformer
  - multimodal
  - post-training
  - inference
  - serving
---

거대한 언어 모델을 무작정 외우기보다, 모델이 배우고 답하는 순서를 차근차근 따라가 본다. 글을 익히는 단계에서 출발해 사람의 지시와 선호를 배우고, 실제 서버에서 여러 사람에게 답하기까지 살펴본다. 사진과 문서를 읽는 VLM을 거쳐, 이전 대화와 작업 경험을 다음 행동에 활용하는 Agent Memory까지 범위를 넓힌다.

!!! note "전체 과정을 한 문장으로"

    LLM은 다음 token을 맞히며 언어를 배운다. VLM은 이미지를 작은 조각으로 나누어 token처럼 다루고, 글과 이미지의 관계를 함께 배운다. Agent Memory는 대화와 작업에서 남길 내용을 골라 외부 저장소에 기록하고, 다음 작업에 필요한 기억만 다시 불러온다.

## 이 과정을 공부하는 방법

한 주는 개념, 실습, 확인 문제, 결과물로 구성한다. 새 용어가 나오면 먼저 쉬운 비유로 뜻을 잡고, 그다음 수식과 코드를 살펴본다. 실습 결과에는 실행 환경, 모델, 데이터, 주요 설정값을 함께 적는다. 그래야 몇 달 뒤에도 같은 실험을 다시 해볼 수 있다.

- 시작 전에 Python, PyTorch tensor, 행렬 곱, 확률의 평균을 복습한다.
- 실습할 컴퓨터의 GPU 이름과 메모리 용량을 기록한다. GPU가 없다면 0.5B 안팎의 작은 모델이나 Colab을 쓴다.
- `transformers`, `datasets`, `accelerate`, `peft`, `trl`의 버전을 기록한다.
- 매주 참고 자료를 먼저 읽고, 강의 글 아래에 논문이나 공식 문서 링크를 남긴다.
- 실습 결과는 성공한 값만 남기지 않고 실패 원인과 수정 내용도 적는다.

## 전체 지도

| 단계 | 주차 | 배우는 내용 | 단계 결과물 |
| --- | --- | --- | --- |
| 언어 모델의 뼈대 | 1~2주 | Transformer, Causal LM, 사전 학습, 지시 학습 | forward와 loss 분석 노트 |
| 답변 예시로 학습 | 3~4주 | SFT, LoRA, QLoRA | 작은 instruction model |
| 사람의 선호로 정렬 | 5~9주 | 강화학습, Reward Model, PPO, DPO 계열 | preference-tuned model |
| 추론과 도구 사용 학습 | 10~12주 | GRPO, tool calling, 통합 평가 | process-aware agent 비교 보고서 |
| 빠르고 안정적인 추론 | 13~16주 | KV cache, vLLM, 양자화, 분산 서빙, 관측 | production serving 보고서 |
| 이미지와 언어 함께 이해 | 17~22주 | ViT, CLIP, 연결 구조, 멀티모달 학습, 평가와 서빙 | VLM 비교·서빙 보고서 |
| 경험을 저장하고 다시 사용 | 23~26주 | memory 종류, 저장·검색·갱신, reflection, skill memory, 평가와 보안 | Agent Memory 비교 보고서 |

---

## 1단계. 언어 모델의 뼈대

### 1주차. Transformer와 Causal LM

!!! note "쉬운 비유"

    문장의 마지막 단어를 가린 뒤 무엇이 들어갈지 맞히는 퀴즈를 수없이 푼다고 생각해보자. Causal LM도 왼쪽에 있는 token만 보고 다음 token을 맞힌다.

이번 주에는 token, embedding, positional information, self-attention, MLP, residual connection, causal mask를 배운다. 작은 decoder-only Transformer의 입력이 logits로 바뀌고, 정답 token과 비교해 cross-entropy loss를 계산하는 과정도 직접 추적한다.

- 이 주차에는 attention과 causal mask를 그림으로 이해한다.
- 문장 하나를 token으로 나누고 embedding tensor의 shape를 기록한다.
- 한 번의 forward pass에서 입력, logits, shifted label, loss의 shape를 출력한다.
- “미래 token을 가리지 않으면 왜 정답을 훔쳐보는 셈인가?”에 답한다.
- 결과물로 `Forward/loss 분석 노트`를 완성한다.

참고 자료:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Hugging Face Transformers: Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling)

### 2주차. Pre-training과 Instruction Tuning

!!! note "Base model과 instruction model"

    Base model은 문장을 자연스럽게 이어 쓰는 데 익숙한 학생과 같다. Instruction model은 여기에 질문을 읽고 알맞은 형식으로 답하는 연습을 더 한 모델이다.

사전 학습 데이터가 입력과 정답으로 바뀌는 모습을 살펴본다. `system`, `user`, `assistant` 메시지를 chat template이 하나의 token 열로 바꾸는 과정도 확인하고, Base model과 SFT model의 답변을 비교한다.

- 이 주차에는 Pre-training과 instruction tuning의 목적을 나눠 살펴본다.
- 같은 prompt를 Base model과 Instruct model에 넣어 답변을 비교한다.
- chat template 적용 전후의 문자열과 token ID를 확인한다.
- 데이터 중복, 개인정보, 유해 데이터가 학습에 미치는 문제를 정리한다.
- 결과물로 `Base/SFT 차이 분석표`를 만든다.

참고 자료:

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Hugging Face Transformers: Chat templates](https://huggingface.co/docs/transformers/chat_templating)

---

## 2단계. 답변 예시로 학습

### 3주차. Supervised Fine-Tuning

!!! note "SFT가 하는 일"

    선생님이 질문과 모범 답안을 함께 보여주면 학생은 답하는 방식부터 배운다. SFT도 prompt와 정답 response를 짝으로 주고 정답 token의 확률을 높인다.

데이터를 train, validation, test로 나누고 padding, truncation, packing, assistant-only loss를 배운다. 학습 loss가 내려가도 실제 답변은 나아지지 않을 수 있다. 둘을 따로 평가해야 하는 이유도 확인한다.

- 이 주차에는 데이터 한 건이 SFT loss로 바뀌는 과정을 예시로 따라간다.
- 작은 공개 데이터로 `SFTTrainer` 학습을 실행한다.
- 학습 전후의 답변을 같은 decoding 설정으로 비교한다.
- train loss와 validation loss가 벌어지는 시점을 찾아본다.
- 결과물로 `작은 instruction model과 학습 기록`을 남긴다.

참고 자료:

- [Hugging Face TRL: SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

### 4주차. LoRA와 QLoRA

!!! note "왜 adapter만 학습할까?"

    두꺼운 교과서 전체를 다시 인쇄하는 대신, 바뀐 내용만 얇은 정정 노트로 붙이는 방법과 비슷하다. LoRA는 원래 weight를 고정하고 작은 행렬만 학습한다.

full fine-tuning, LoRA, QLoRA의 trainable parameter 수와 GPU 메모리를 비교한다. rank, alpha, target module이 어떤 뜻인지 실험하고, adapter merge 전후의 출력도 확인한다.

- 이 주차에는 low-rank 행렬을 작은 숫자 예제로 풀어본다.
- 같은 데이터로 LoRA와 QLoRA를 각각 실행한다.
- trainable parameter, peak GPU memory, 학습 시간, 품질을 표로 비교한다.
- rank를 바꾸었을 때 속도와 결과가 어떻게 달라지는지 기록한다.
- 결과물로 `PEFT 비교 보고서`를 만든다.

참고 자료:

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Hugging Face TRL: PEFT Integration](https://huggingface.co/docs/trl/peft_integration)

---

## 3단계. 사람의 선호로 답변 다듬기

### 5주차. 강화학습의 기초

!!! note "LLM을 게임으로 보면"

    지금까지는 정답 문장을 그대로 보여줬다. 강화학습에서는 모델이 먼저 답하고, 그 답에 점수를 준다. 모델은 높은 점수를 받은 행동을 더 자주 하도록 바뀐다.

state, action, policy, trajectory, reward, return, value, advantage를 짧은 게임으로 익힌다. LLM에서는 지금까지의 token이 state, 다음 token이 action이라는 연결도 다룬다.

- 이 주차에는 강화학습 용어를 LLM의 token 생성 과정과 연결한다.
- 두세 개 action만 있는 작은 환경에서 REINFORCE를 구현한다.
- reward가 드문 경우와 잦은 경우의 학습 곡선을 비교한다.
- baseline이 gradient의 흔들림을 줄이는 까닭을 설명한다.
- 결과물로 `Policy gradient 실습 노트`를 완성한다.

참고 자료:

- [Reinforcement Learning: An Introduction, 2nd edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

### 6주차. Reward Model과 RLHF

!!! note "좋은 답을 숫자로 바꾸기"

    사람은 두 답을 보고 어느 쪽이 나은지 고를 수 있다. Reward Model은 이 비교 자료를 배우고, 새 답이 얼마나 좋은지 숫자로 매긴다.

`chosen`과 `rejected` 답변 쌍, Bradley–Terry 형태의 pairwise loss, reward accuracy를 공부한다. labeler가 동의하지 않거나 엉뚱한 지름길을 학습하는 reward hacking도 작은 사례로 확인한다.

- 이 주차에는 SFT에서 Reward Model 학습과 RL로 이어지는 흐름을 살펴본다.
- preference dataset의 한 행이 loss로 바뀌는 과정을 계산한다.
- 작은 Reward Model을 학습하고 pairwise accuracy를 측정한다.
- 길기만 한 답에 높은 점수를 주는 편향이 있는지 확인한다.
- 결과물로 `Preference Reward Model 카드`를 작성한다.

참고 자료:

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Hugging Face TRL: Reward Modeling](https://huggingface.co/docs/trl/reward_trainer)

### 7주차. PPO로 배우는 RLHF

!!! note "너무 멀리 가지 않게 잡아주기"

    높은 점수만 좇으면 모델이 이상한 말투나 편법을 배울 수 있다. PPO와 KL penalty는 새 모델이 기준 모델에서 한 번에 너무 멀리 벗어나지 않도록 제동을 건다.

policy model, reference model, Reward Model, value model의 역할을 나눈다. rollout, advantage, clipped objective, KL, entropy를 로그로 읽는 방법도 익힌다.

- 이 주차에는 PPO 학습에 쓰이는 네 model이 어떤 값을 주고받는지 따라간다.
- 아주 작은 모델과 데이터로 PPO pipeline을 실행한다.
- clip range와 KL coefficient를 바꾸어 학습 안정성을 비교한다.
- reward 상승과 실제 답변 품질이 어긋난 사례를 찾는다.
- 결과물로 `작은 PPO-RLHF pipeline 보고서`를 남긴다.

참고 자료:

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Hugging Face TRL: PPO Trainer](https://huggingface.co/docs/trl/ppo_trainer)

### 8주차. Direct Preference Optimization

!!! note "Reward Model 없이 비교 답안으로 학습"

    DPO는 별도의 채점기를 먼저 만들지 않는다. 좋은 답의 확률은 올리고 나쁜 답의 확률은 내리되, 기준 모델과의 차이도 함께 살핀다.

DPO loss에서 policy model과 reference model의 log probability가 쓰이는 방식을 배운다. beta를 바꾸며 선호를 따르는 정도와 기준 모델에 머무는 정도가 어떻게 달라지는지도 실험한다.

- 이 주차에는 DPO와 PPO 기반 RLHF의 학습 흐름을 나란히 비교한다.
- 같은 preference dataset으로 DPO 학습을 실행한다.
- beta를 바꾸어 chosen/rejected margin과 답변 품질을 비교한다.
- reference model이 필요한 까닭을 설명한다.
- 결과물로 `Preference-tuned model`을 저장한다.

참고 자료:

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [Hugging Face TRL: DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)

### 9주차. DPO 계열 비교

!!! note "방법이 여러 개인 이유"

    모든 데이터가 좋은 답과 나쁜 답의 완벽한 쌍으로 모이지는 않는다. 방법마다 필요한 표시와 기준 모델의 유무가 다르므로, 데이터에 맞는 방식을 골라야 한다.

IPO, KTO, ORPO의 문제의식과 입력 형식을 비교한다. 최신 방법을 많이 나열하기보다 같은 데이터와 평가 기준에서 무엇이 달라지는지 확인한다.

- 이 주차에는 DPO, IPO, KTO, ORPO가 요구하는 데이터와 loss를 비교한다.
- preference pair가 부족하거나 binary feedback만 있을 때의 선택 기준을 적는다.
- DPO와 IPO를 같은 작은 데이터로 실험한다.
- 승률, 길이 편향, KL, 학습 메모리를 함께 비교한다.
- 결과물로 `Preference optimization 선택 가이드`를 만든다.

참고 자료:

- [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)

---

## 4단계. 추론 과정과 도구 사용 학습

### 10주차. GRPO의 원리

!!! note "한 답만 보지 않고 같은 문제의 답끼리 비교하기"

    GRPO는 같은 문제에 여러 답을 만들고 그 묶음 안에서 상대적으로 잘한 답을 찾는다. 별도의 value model 없이 그룹의 점수를 기준선처럼 쓴다는 점이 PPO와 다르다.

group sampling, group-relative advantage, rule-based reward, KL regularization을 배운다. 정답 여부와 출력 형식을 각각 점수로 줄 때 reward scale이 어떤 영향을 주는지도 살펴본다.

- 이 주차에는 PPO와 GRPO의 구성 요소를 나란히 놓고 차이를 찾는다.
- 한 prompt에서 여러 completion을 만들고 상대 advantage를 손으로 계산한다.
- 정확도 reward와 형식 reward를 따로 기록한다.
- 그룹 크기를 바꾸어 reward 분산과 메모리를 비교한다.
- 결과물로 `Group reward 분석 노트`를 완성한다.

참고 자료:

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [Hugging Face TRL: GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)

### 11주차. GRPO로 reasoning과 tool calling 학습

!!! note "결과뿐 아니라 과정도 채점하기"

    계산기나 데이터베이스를 쓰는 모델은 마지막 문장만 맞아서는 부족하다. 맞는 도구를 골랐는지, 인자를 올바르게 넣었는지, 도구 결과를 답에 제대로 썼는지도 살펴야 한다.

도구 schema, structured output, process reward, execution reward를 설계한다. 정답을 문자열로 비교할 수 있는 수학 문제부터 시작한 뒤, 계산기나 작은 데이터베이스를 호출하는 과제로 넓힌다.

- 이 주차에는 하나의 tool trajectory를 message 순서대로 따라간다.
- 형식, tool 선택, 인자, 실행 성공, 최종 정답 reward를 분리해 만든다.
- tool을 쓰지 않아도 되는 문제와 반드시 써야 하는 문제를 섞어 학습한다.
- reward hacking과 무의미한 반복 호출이 있는지 확인한다.
- 결과물로 `Reasoning/tool model과 오류 분석표`를 만든다.

참고 자료:

- [Hugging Face TRL: GRPO Trainer의 Agent Training](https://huggingface.co/docs/trl/grpo_trainer#agent-training)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)

### 12주차. 통합 학습 프로젝트

!!! note "같은 출발점, 다른 학습 방법"

    공정한 비교를 하려면 Base, SFT, DPO, GRPO 모델에 같은 평가 문제와 같은 생성 설정을 써야 한다. 한 모델에만 쉬운 시험을 주면 비교가 되지 않는다.

지금까지 만든 checkpoint를 한자리에 모은다. 일반 대화, 지시 따르기, reasoning, tool calling을 나누어 평가하고, 자동 점수와 사람이 읽은 평가가 다른 사례도 기록한다.

- 이 주차에는 공통 평가 계획과 성공 기준을 결과보다 먼저 정한다.
- Base, SFT, DPO, GRPO 정책을 같은 prompt와 평가 설정으로 비교한다.
- task accuracy, format accuracy, tool execution success, 응답 길이, 추론 시간을 기록한다.
- 20개 답을 직접 읽고 자동 평가의 오류를 찾는다.
- 결과물로 `Base/SFT/DPO/GRPO 비교 보고서`를 완성한다.

참고 자료:

- [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [LightEval 공식 문서](https://huggingface.co/docs/lighteval/index)
- [Hugging Face TRL](https://huggingface.co/docs/trl/index)

---

## 5단계. 빠르고 안정적인 추론

### 13주차. 추론 메모리와 KV cache

!!! note "책과 메모지"

    model weight는 이미 배운 지식이 적힌 책이고, KV cache는 지금 대화에서 앞부분을 다시 계산하지 않으려고 적어두는 메모지다. 책을 4-bit로 줄여도 대화가 길고 사용자가 많으면 메모지가 GPU를 가득 채울 수 있다.

weight, KV cache, activation, temporary buffer, CUDA runtime이 GPU 메모리를 나누어 쓰는 방식을 배운다. MHA, GQA, MQA가 KV head 수를 어떻게 바꾸는지도 계산한다.

- 이 주차에는 학습 메모리와 추론 메모리가 어디에 쓰이는지 구분한다.
- FP32, BF16/FP16, INT8, INT4 weight 크기를 계산하는 도구를 만든다.
- context 2K/8K/32K와 batch 1/8/32에서 KV cache 크기를 계산한다.
- 실제 GPU 사용량과 이론값이 다른 까닭을 기록한다.
- 결과물로 `Weight/KV memory calculator`를 완성한다.

참고 자료:

- [Hugging Face Transformers: Caching](https://huggingface.co/docs/transformers/cache_explanation)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

### 14주차. 추론 최적화와 성능 측정

!!! note "빨리 시작하기와 많이 처리하기"

    대화형 챗봇은 첫 token이 빨리 나오는 일이 중요하다. 문서를 밤새 한꺼번에 만드는 작업은 같은 시간에 더 많은 token을 처리하는 일이 중요하다. 빠르다는 말만으로는 두 상황을 구분할 수 없다.

prefill과 decode, TTFT, TPOT, inter-token latency, end-to-end latency, throughput, goodput을 배운다. continuous batching, PagedAttention, FlashAttention, chunked prefill이 어느 병목을 줄이는지도 실험한다.

- 이 주차에는 prefill과 decode가 GPU를 쓰는 방식을 비교한다.
- prompt 길이, output 길이, concurrency를 바꾼 benchmark를 설계한다.
- Transformers와 vLLM의 공통 측정표를 만들고 CUDA 실측은 `미실행`으로 표시한다.
- 교육용 chunked prefill sweep에서 TTFT와 TPOT의 trade-off를 찾는다.
- 결과물로 `Latency-throughput benchmark` 설계와 synthetic baseline을 만든다.

참고 자료:

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [vLLM: Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [vLLM: Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)

### 15주차. vLLM을 중심으로 추론 서버 익히기

!!! note "서버 엔진을 배우는 순서"

    처음에는 vLLM 하나로 모델을 API 서버로 띄우는 전 과정을 익힌다. 그다음 같은 요청을 SGLang과 TensorRT-LLM에 보내며 차이를 비교한다. 패키지 이름보다 요청이 들어와 token이 나가는 흐름을 아는 것이 먼저다.

OpenAI-compatible API, streaming, scheduling, prefix caching, structured output을 실습한다. SGLang은 shared prefix와 agent workload를, TensorRT-LLM은 NVIDIA 환경의 engine build와 parallelism을 중심으로 살펴본다. TGI는 maintenance mode이므로 역사와 운영 기능을 확인하는 비교 대상으로만 둔다.

- 이 주차에는 client의 요청이 API, scheduler, model runner를 거쳐 응답이 되는 흐름을 따라간다.
- Chat completion과 streaming 요청을 mock server에 보내 전송 형식을 확인하고, 실제 vLLM CUDA 실행은 `미실행`으로 기록한다.
- `max-model-len`, `gpu-memory-utilization`, `max-num-seqs`를 바꾸는 공통 실험표를 만든다.
- 같은 모델로 vLLM, SGLang, TensorRT-LLM을 비교할 기준을 만들고 cross-engine 실측은 `미실행`으로 기록한다.
- 결과물로 `OpenAI-compatible server 실행 및 비교 노트`를 남긴다.

참고 자료:

- [vLLM: Online Serving](https://docs.vllm.ai/en/latest/serving/online_serving/)
- [SGLang: Bench Serving Guide](https://docs.sglang.io/docs/developer_guide/bench_serving)
- [TensorRT-LLM: Overview](https://nvidia.github.io/TensorRT-LLM/overview.html)
- [Hugging Face TGI](https://huggingface.co/docs/text-generation-inference/main/index)

### 16주차. 양자화, 분산 서빙, 운영

!!! note "작게 만들면 무조건 빨라질까?"

    양자화는 weight를 더 적은 bit로 저장한다. 메모리는 줄지만 변환 비용, 지원 kernel, batch 크기에 따라 속도가 기대만큼 오르지 않을 수 있고 답의 품질도 달라질 수 있다. 직접 재는 과정이 필요하다.

BF16, FP8, INT8, AWQ, GPTQ를 비교하고 tensor, pipeline, data, expert parallelism의 쓰임을 구분한다. prefix caching, speculative decoding, prefill-decode 분리도 살펴본다. 마지막에는 metrics, logs, traces로 서버 상태를 관찰한다.

- 이 주차에는 양자화와 parallelism을 고르는 기준을 세운다.
- BF16, FP8, INT8, AWQ, GPTQ의 이상적인 weight 저장량을 비교하고 실제 GPU의 TTFT·TPOT·품질은 `미실행`으로 기록한다.
- Concurrency 1부터 64까지 교육용 latency-throughput 곡선을 그린다.
- Request 수, queue time, TTFT, TPOT, token 수, KV cache 사용률, OOM을 기록하는 표를 만든다.
- 결과물로 채워 쓸 수 있는 `Production serving report` 양식을 완성한다.

참고 자료:

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [vLLM: Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)

---

## 6단계. 이미지와 언어를 함께 이해하기

앞의 16주를 마쳤다면 VLM을 배우는 데 필요한 LLM 기초는 이미 갖춘 셈이다. 여기에 이미지가 tensor로 바뀌는 과정, 이미지와 글을 같은 의미 공간에 놓는 방법, vision encoder와 LLM을 연결하는 구조를 더하면 된다. 이미지 한 장을 이해하고 답하는 VLM의 전체 흐름은 6주면 한 번 완주할 수 있다.

!!! note "왜 6주일까?"

    첫 3주는 이미지와 글이 만나는 구조를 배우고, 다음 2주는 데이터와 평가를 다룬다. 마지막 1주는 실제 서버에 올려 속도와 메모리를 잰다. 영상까지 깊게 다루려면 2주를 더 잡는 편이 좋지만, 먼저 정지 이미지로 기본 원리를 익힌다.

### 17주차. 이미지 tensor와 Vision Transformer

!!! note "사진을 작은 낱말로 나누기"

    언어 모델이 문장을 token으로 나누듯이 Vision Transformer는 사진을 바둑판 같은 patch로 나눈다. 각 patch를 숫자 묶음으로 바꾸면 Transformer가 읽을 수 있는 visual token이 된다.

RGB, channel, height, width, resize, crop, normalization을 익힌다. 한 장의 이미지가 `pixel_values`를 거쳐 patch embedding이 되는 과정을 살펴보고, 해상도와 patch 크기가 visual token 수에 어떤 영향을 주는지 계산한다.

- 이 주차에는 이미지 한 장을 `[channel, height, width]` 모양의 tensor로 바꾸는 과정을 따라간다.
- 같은 사진에 resize, crop, normalization을 적용하고 값과 shape가 어떻게 달라지는지 확인한다.
- 해상도와 patch 크기를 바꾸며 visual token 수를 계산한다.
- 사전 학습된 ViT의 patch embedding과 attention 출력을 살펴본다.
- 결과물로 `이미지 전처리와 patch 흐름도`를 만든다.

참고 자료:

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Hugging Face Transformers: Image classification](https://huggingface.co/docs/transformers/tasks/image_classification)

### 18주차. CLIP과 이미지-글 정렬

!!! note "사진과 설명을 같은 지도에 놓기"

    강아지 사진과 “a photo of a dog”라는 문장이 지도에서 가까운 곳에 놓인다고 생각해보자. CLIP은 맞는 사진과 글은 가깝게, 관계없는 쌍은 멀게 만드는 방법으로 둘의 의미를 맞춘다.

image encoder, text encoder, embedding, cosine similarity, temperature, contrastive loss를 배운다. 같은 image-text embedding을 이용해 zero-shot 분류와 검색을 할 수 있는 까닭도 작은 행렬로 확인한다.

- 이 주차에는 이미지와 caption을 각각 embedding으로 바꾸고 similarity 행렬을 만든다.
- cosine similarity와 temperature가 점수 차이에 미치는 영향을 손으로 계산한다.
- 여러 caption 가운데 사진과 가장 가까운 문장을 찾는 검색 실험을 한다.
- class 이름을 prompt로 바꾸어 zero-shot 분류를 실행하고 prompt에 따른 차이를 기록한다.
- 결과물로 `CLIP 검색·분류 실험 보고서`를 완성한다.

참고 자료:

- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

### 19주차. Vision encoder와 LLM을 잇는 구조

!!! note "서로 다른 두 교실 사이의 통역사"

    Vision encoder는 이미지 특징을 만들고 LLM은 글을 이어 쓴다. 두 model이 주고받는 숫자의 모양과 뜻이 다르므로, projector나 Q-Former 같은 작은 연결 장치가 통역을 맡는다.

LLaVA의 projector, BLIP-2의 Q-Former, Flamingo의 Perceiver Resampler와 cross-attention을 비교한다. Vision encoder와 LLM을 고정할지 함께 학습할지도 나누어 보고, 이미지가 최종적으로 몇 개의 visual token이 되어 LLM에 들어가는지 추적한다.

- 이 주차에는 vision encoder, connector, LLM의 역할을 하나의 구조도에 표시한다.
- LLaVA, BLIP-2, Flamingo가 이미지 특징을 언어 model에 전달하는 방식을 비교한다.
- 각 단계의 batch, token, hidden dimension shape를 기록한다.
- 고정된 부분과 학습되는 부분을 나누고 필요한 메모리의 차이를 설명한다.
- 결과물로 `VLM 구조 비교 지도`를 만든다.

참고 자료:

- [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)

### 20주차. 멀티모달 데이터와 Instruction Tuning

!!! note "질문 안에 사진도 함께 넣기"

    글만 다루는 chat template에는 `user`와 `assistant` 문장이 들어간다. VLM의 한 message에는 text와 image가 순서대로 들어간다. Processor는 이 둘을 각각 token ID와 pixel 값으로 바꾸어 model에 전달한다.

caption, VQA, OCR, 대화형 instruction data의 차이를 살펴본다. 멀티모달 chat template, image placeholder, assistant-only loss를 확인하고, vision encoder·connector·LLM 가운데 어느 부분을 LoRA로 학습할지 계획한다.

- 이 주차에는 text와 image가 섞인 message가 `input_ids`와 `pixel_values`로 바뀌는 과정을 확인한다.
- caption, 짧은 VQA, 자세한 설명, 지시 수행 자료를 train, validation, test로 나눈다.
- 이미지 없이도 답을 맞힐 수 있는 질문과 데이터 중복을 찾아낸다.
- 작은 VLM에서 connector 또는 LoRA를 학습하고 학습 전후 답변을 같은 설정으로 비교한다.
- 결과물로 `멀티모달 데이터 카드와 instruction-tuned adapter`를 만든다.

참고 자료:

- [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- [Hugging Face Transformers: Multimodal chat templates](https://huggingface.co/docs/transformers/chat_templating_multimodal)

### 21주차. VLM 과제와 실패 분석

!!! note "그럴듯한 답과 사진을 본 답은 다르다"

    모델이 사진에 없는 물건을 있다고 말해도 문장은 자연스러울 수 있다. 그래서 VLM은 말투만 읽어서는 평가할 수 없다. 이미지에 실제로 근거한 답인지 따로 확인해야 한다.

image captioning, VQA, 문서·표 읽기, OCR, 공간 관계, grounding을 나누어 평가한다. 일반 지식이 필요한 문제와 이미지를 정확히 봐야 하는 문제도 구분한다. 해상도를 줄였을 때 작은 글자가 사라지는 현상과 object hallucination을 실패 사례로 다룬다.

- 이 주차에는 caption, 일반 VQA, 문서 VQA, 공간 추론 과제를 같은 model로 풀어본다.
- 이미지를 가리거나 바꾼 text-only 대조 실험으로 model이 사진을 실제로 쓰는지 확인한다.
- 정확도, 문서 답변 점수, grounding IoU, hallucination 비율을 과제에 맞게 고른다.
- 틀린 답을 인식, OCR, 지식, 추론, 지시 위반으로 나누어 읽는다.
- 결과물로 `VLM 오류 유형표와 평가 보고서`를 완성한다.

참고 자료:

- [MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark](https://arxiv.org/abs/2311.16502)
- [DocVQA: A Dataset for VQA on Document Images](https://arxiv.org/abs/2007.00398)
- [Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2305.10355)

### 22주차. VLM 추론과 멀티모달 서빙

!!! note "사진이 커지면 기다림도 길어진다"

    글이 길수록 LLM의 계산량이 늘듯이, 사진의 해상도와 장수가 늘면 visual token도 많아진다. 같은 질문이라도 사진 한 장과 열 장을 넣었을 때 필요한 메모리와 첫 token 대기 시간은 달라진다.

Transformers와 vLLM에서 같은 VLM 요청을 실행한다. 이미지 해상도, 이미지 수, 동시 요청 수를 바꾸며 TTFT, TPOT, throughput, GPU memory를 잰다. 외부 이미지 URL을 받는 서버에서는 허용할 주소와 파일 크기를 제한해야 하는 이유도 살펴본다.

- 이 주차에는 model revision, Processor, chat template, decoding 설정을 고정한 공통 benchmark를 만든다.
- 이미지 해상도와 장수를 바꾸며 visual token 수와 peak memory를 기록한다.
- 단일 이미지, 문서 이미지, 여러 이미지 대화의 TTFT와 throughput을 비교한다.
- 허용 media domain, 파일 크기, timeout을 정하고 잘못된 입력의 처리 방법을 점검한다.
- 결과물로 `VLM production serving report`를 작성한다.

참고 자료:

- [Hugging Face Transformers: Image-text-to-text](https://huggingface.co/docs/transformers/main/tasks/image_text_to_text)
- [vLLM: Multimodal Inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)
- [LLaVA-NeXT-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models](https://arxiv.org/abs/2407.07895)

---

## 7단계. 경험을 저장하고 다시 사용하기

앞 과정에서 agent가 reasoning하고 tool을 부르는 방법을 배웠다. 하지만 모델의 weight는 작업을 한 번 마칠 때마다 저절로 바뀌지 않는다. 다음 실행에서도 경험을 쓰려면 무엇을 남기고, 어디에 저장하며, 언제 다시 꺼낼지 정하는 memory system이 필요하다. 이 흐름을 이해하고 작은 구현까지 만드는 데는 4주가 알맞다.

!!! note "Agent Memory가 작동하는 순서"

    Agent는 먼저 대화와 tool 실행 결과를 관찰한다. 그중 다시 쓸 만한 사실, 경험, 규칙을 골라 출처와 시간과 함께 저장한다. 새 작업이 들어오면 질문과 가까운 기억을 검색하고, 일부만 context에 넣어 행동한다. 작업이 끝난 뒤에는 틀린 기억을 고치거나 오래된 기억을 잊는다.

대화 기록을 전부 보관하는 것만으로 memory system이 완성되지는 않는다. Agent Memory에는 무엇을 쓸지 정하는 write policy, 기억을 찾는 retrieval, token 예산에 맞게 줄이는 selection, 충돌한 사실을 고치는 update, 필요 없는 정보를 지우는 forgetting이 함께 들어간다. Multi-agent shared memory와 memory 자체를 학습하는 방법까지 연구하려면 2주를 더 잡을 수 있지만, 먼저 한 agent의 핵심 흐름을 익힌다.

### 23주차. Agent Memory의 종류와 생명주기

!!! note "녹취록, 작업판, 백과사전, 개인 노트"

    Conversation History는 대화를 순서대로 적은 녹취록이다. Agent State는 지금 할 일과 중간 결과가 놓인 작업판이고, Knowledge Base는 여러 사람이 함께 참고하는 백과사전에 가깝다. Memory는 agent가 경험에서 골라 적어둔 개인 노트다.

#### 먼저 구분할 네 가지

| 개념 | 무엇을 담는가 | 범위와 수명 | 예시 |
| --- | --- | --- | --- |
| Memory | Agent가 다음 작업에 다시 쓰려고 고른 사실, 경험, 규칙 | 여러 session과 task에서 불러오며 갱신하거나 삭제할 때까지 남는다 | 사용자 선호, 과거 실패의 교훈, 재사용할 skill |
| Knowledge Base | 특정 agent의 경험과 상관없이 참고하는 외부 지식 | 여러 사용자나 agent가 공유하며 원본 문서를 다시 넣을 때 바뀐다 | 제품 설명서, 사내 규정, 논문 모음 |
| Conversation History | `user`, `assistant`, `tool` message를 시간순으로 쌓은 기록 | 보통 하나의 session이나 thread에 속하며 길어지면 자르거나 요약한다 | 질문, 답변, tool call과 반환값 |
| Agent State | 지금 실행 중인 workflow를 이어가는 데 필요한 현재 상태 | 매 step마다 바뀌며 checkpoint를 남기면 중단한 지점에서 다시 시작할 수 있다 | 현재 목표, plan, 중간 계산값, 승인 대기 상태 |

이 네 가지는 저장 기술이 아니라 역할로 나눈다. Conversation History는 Agent State의 한 항목이 될 수 있고, History와 State에서 중요한 내용을 골라 Memory를 만들기도 한다. Knowledge Base와 Memory가 같은 vector database를 쓰더라도, 외부 자료를 찾아보는지 agent 자신의 경험을 다시 쓰는지에 따라 역할이 달라진다.

모델 weight에 들어 있는 parametric memory, 현재 context와 state에 놓인 working memory, 세션을 넘어 남는 long-term memory를 구분한다. Long-term memory는 다시 사실을 담는 semantic memory, 지나간 작업을 담는 episodic memory, 문제를 푸는 규칙과 skill을 담는 procedural memory로 나눈다.

- 이 주차에는 Memory, Knowledge Base, Conversation History, Agent State를 같은 사례로 비교한다.
- Model weight의 parametric memory, 현재 context의 working memory, 외부 store의 long-term memory를 구분한다.
- 하나의 agent 실행에서 관찰, 저장 판단, 기록, 검색, 행동, 갱신 순서를 그린다.
- 짧은 Conversation History와 Agent State를 checkpoint로 저장하고 같은 thread를 다시 시작한다.
- 사용자 선호, 과거 실패, 계산기 사용법을 분류한 `Agent Memory 종류와 생명주기 지도`를 만든다.

참고 자료:

- [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
- [LangGraph: Memory overview](https://docs.langchain.com/oss/python/concepts/memory)
- [LangGraph: Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [OpenAI Agents SDK: Sessions](https://openai.github.io/openai-agents-python/sessions/)

### 24주차. 기억을 저장하고 찾고 고치기

!!! note "기억 창고에도 사서가 필요하다"

    모든 대화를 한 상자에 던져 넣으면 필요한 기억을 찾기 어렵다. 누가 말했는지, 언제 생긴 정보인지, 무엇에 관한 내용인지 이름표를 붙여야 한다. 새 정보가 들어오면 예전 기록과 충돌하는지도 살펴야 한다.

write, index, retrieve, rerank, read, update, delete pipeline을 만든다. Memory 한 건에는 본문뿐 아니라 user ID, source, timestamp, memory type, confidence를 함께 기록한다. Keyword, embedding similarity, recency, importance를 섞은 검색 점수도 비교한다.

- 이 주차에는 대화에서 장기 보관할 사실만 뽑는 write policy를 만든다.
- 같은 내용을 raw message, summary, 구조화된 JSON으로 저장하고 검색 결과를 비교한다.
- semantic similarity, keyword, recency, importance 점수를 각각 계산한다.
- 새 정보가 과거 정보와 충돌할 때 덮어쓰기, 이력 보존, 확인 요청을 나누어 적용한다.
- 결과물로 `저장·검색·갱신이 가능한 memory pipeline`을 완성한다.

참고 자료:

- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
- [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110)
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413)
- [LangGraph: Add memory](https://docs.langchain.com/oss/python/langgraph/add-memory)

### 25주차. Reflection과 Skill Memory

!!! note "실패 일기와 나만의 공략집"

    실패한 기록을 그대로 다시 읽는 것보다 “다음에는 단위를 먼저 확인한다”처럼 교훈을 짧게 남기는 편이 쓸모 있다. 여러 번 성공한 행동은 순서가 있는 skill로 묶어 공략집처럼 다시 사용할 수 있다.

Episodic memory에서 교훈을 만드는 reflection과 실행 가능한 절차를 모으는 skill library를 배운다. Reflexion처럼 feedback을 글로 남기는 방식과 Voyager처럼 재사용할 code skill을 저장하는 방식을 비교한다. 이 과정은 보통 model weight를 업데이트하지 않고 inference 중에 일어난다.

- 이 주차에는 task, trajectory, 결과, 실패 원인, 다음 규칙을 한 memory schema로 만든다.
- 실패한 tool trajectory를 짧은 reflection으로 바꾸고 다음 시도에 넣는다.
- 성공한 여러 trajectory에서 반복되는 절차를 하나의 skill로 정리한다.
- memory 없음, raw trajectory, reflection, skill library 조건을 같은 과제로 비교한다.
- 결과물로 `경험에서 교훈과 skill을 만드는 agent`를 구현한다.

참고 자료:

- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)

### 26주차. Agent Memory 평가와 안전한 운영

!!! note "틀린 기억은 잊어버린 것보다 위험하다"

    관련 없는 기억을 불러오면 답이 흔들리고, 오래된 주소를 최신 정보처럼 쓰면 실제 행동까지 틀릴 수 있다. 다른 사용자의 기억이 섞이면 개인정보 문제도 생긴다. 얼마나 많이 기억했는지만 재서는 부족하다.

정보 추출, 여러 세션을 잇는 추론, 시간 이해, 정보 갱신, 모를 때 답하지 않는 능력을 나누어 평가한다. Retrieval recall@k와 최종 답변 정확도뿐 아니라 task success, retrieved token 수, latency, 저장 비용도 함께 잰다. User namespace, 접근 제어, provenance, TTL, 삭제, memory poisoning 방어도 점검한다.

- 이 주차에는 no-memory, full-history, summary, vector retrieval, reflective memory를 같은 문제로 비교한다.
- 검색 recall@k와 최종 답변 정확도를 따로 재어 retrieval과 reasoning 오류를 구분한다.
- 정보가 바뀌거나 삭제된 뒤 예전 memory가 다시 나타나는지 시험한다.
- 다른 사용자의 memory와 악성 instruction이 섞이지 않도록 격리와 write-time 검사를 적용한다.
- 결과물로 `Agent Memory 평가·보안·운영 보고서`를 작성한다.

참고 자료:

- [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813)
- [Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions](https://arxiv.org/abs/2507.05257)
- [Memory Poisoning Attack and Defense on Memory Based LLM-Agents](https://arxiv.org/abs/2601.05504)
- [OpenAI Agents SDK: Sessions](https://openai.github.io/openai-agents-python/sessions/)

---

## 최종 프로젝트

주제는 `Process-Aware Multimodal Memory Agent의 평가와 서빙`이다. 하나의 모델을 학습한 뒤 끝내지 않고, 글과 이미지를 이해하는지, 과거 경험을 올바르게 사용하는지, 추론 엔진에 따라 성능이 어떻게 달라지는지 비교한다.

- 공개된 작은 VLM을 정하고 멀티모달 instruction adapter를 만든다.
- 필요하다면 DPO 또는 GRPO를 적용해 지시 따르기와 tool 사용을 다듬는다.
- BF16과 한 가지 이상의 양자화 버전을 준비한다.
- Transformers, vLLM, SGLang 가운데 둘 이상으로 같은 workload를 실행한다.
- 일반 대화, 단일 이미지, 문서·표, 여러 이미지, tool calling을 나누어 평가한다.
- Thread state와 사용자별 long-term memory store를 연결하고, memory가 없는 조건과 비교한다.
- task success, memory recall, hallucination, TTFT, TPOT, throughput, GPU memory를 한 표에 모은다.
- 이미지 해상도, visual token 수, concurrency, cache 설정을 바꾸어 ablation을 수행한다.
- 저장 형식, retrieval top-k, reflection 유무를 바꾸고 오래된 정보와 악성 memory도 시험한다.
- 어떤 설정이 언제 좋은지, 실패한 설정은 왜 실패했는지 보고서로 설명한다.

!!! note "과정을 마치며 답할 네 질문"

    1. 학습 단계에서는 text와 image를 어떤 objective로 연결할 것인가?
    2. 평가 단계에서는 model이 이미지를 실제로 보고 답했는지 어떻게 확인할 것인가?
    3. Memory 단계에서는 무엇을 저장하고, 언제 불러오고, 언제 고치거나 지울 것인가?
    4. 서빙 단계에서는 visual token, latency, throughput, 품질 가운데 무엇을 먼저 지킬 것인가?

<nav class="lecture-navigation" aria-label="강의 시작">
  <a class="lecture-navigation-link next" href="/notes/tutorial/llm_lecture/w01_transformer_causal_lm/" rel="next">
    <span>1주차 시작하기 →</span>
    <strong>Transformer와 Causal LM</strong>
  </a>
</nav>

<!-- HUMANIZE-SUMMARY
장르: 교육용 커리큘럼
검토 단위: 23주차의 네 개념 비교 문단과 표
원본/수정본: 새 문단 초안 1,372자 / 윤문본 1,338자, 이번 후처리 변경률 2.48%
카테고리별 탐지/수정: A-7 0→0, A-8 0→0, C-5 0→0, D-1 0→0, H-1 0→0
정량 점검: humanize-korean metrics v2.0 risk band low
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 의미를 유지한 채 반복되는 연결 표현을 줄임
주요 변경: Memory, Knowledge Base, Conversation History, Agent State의 역할과 겹치는 지점을 표로 구분함
-->
