---
title: "LLM은 어떻게 배우고 대답할까: 16주 학습 커리큘럼"
description: "Transformer의 기초부터 SFT, DPO, GRPO, 추론 최적화와 모델 서빙까지 직접 실험하며 익히는 과정"
tags:
  - LLM
  - Transformer
  - post-training
  - inference
  - serving
---

거대한 언어 모델을 무작정 외우기보다, 모델이 배우고 답하는 순서를 차근차근 따라가 본다. 글을 익히는 단계에서 출발해 사람의 지시와 선호를 배우는 과정을 거친 뒤, 실제 서버에서 여러 사람에게 답하기까지 살펴보는 것이 이 과정의 목표다.

!!! note "전체 과정을 한 문장으로"

    LLM은 먼저 다음 token을 맞히며 언어를 배우고, 좋은 답의 예시와 사람의 선호를 보며 행동을 다듬은 다음, 제한된 GPU 메모리를 나누어 쓰며 사용자에게 답한다.

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

## 최종 프로젝트

주제는 `SFT·GRPO로 학습한 Process-Aware Agent 모델의 최적화와 서빙`이다. 하나의 모델을 학습한 뒤 끝내지 않고, 학습 방법과 추론 엔진을 함께 비교한다.

- Base model을 정하고 SFT checkpoint를 만든다.
- DPO 또는 GRPO를 적용해 정렬된 checkpoint를 만든다.
- BF16과 한 가지 이상의 양자화 버전을 준비한다.
- Transformers, vLLM, SGLang 가운데 둘 이상으로 같은 workload를 실행한다.
- 짧은 대화, 긴 RAG context, tool calling workload를 따로 평가한다.
- TTFT, TPOT, throughput, GPU memory, task success를 한 표에 모은다.
- continuous batching, prefix caching, chunked prefill 중 두 가지 이상을 끄고 켜며 ablation을 수행한다.
- 어떤 설정이 언제 좋은지, 실패한 설정은 왜 실패했는지 보고서로 설명한다.

!!! note "과정을 마치며 답할 세 질문"

    1. 학습 단계에서는 어떤 objective로 모델의 행동을 바꿀 것인가?
    2. 추론 단계에서는 weight와 KV cache를 제한된 GPU에 어떻게 배치할 것인가?
    3. 서빙 단계에서는 latency, throughput, 품질 가운데 무엇을 먼저 지킬 것인가?

<nav class="lecture-navigation" aria-label="강의 시작">
  <a class="lecture-navigation-link next" href="/notes/tutorial/llm_lecture/w01_transformer_causal_lm/" rel="next">
    <span>1주차 시작하기 →</span>
    <strong>Transformer와 Causal LM</strong>
  </a>
</nav>

<!-- HUMANIZE-SUMMARY
장르: 교육용 커리큘럼
검토 단위: 5개 단계와 최종 프로젝트를 각각 5,000자 이하로 나누어 점검
원본/수정본: 15905자 / 15613자, 이번 후처리 변경률 1.84%
카테고리별 탐지/수정: A-7 0→0, A-8 0→0, C-5 0→0, D-1 0→0, H-1 0→0
정량 점검: humanize-korean metrics v2.0 risk band low
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 학습 내용은 유지한 채 내부 파일명만 독자용 문장으로 바꿈
주요 변경: 내부 파일명을 앞세운 작업 문구를 `이 주차에는 ... 이해한다` 형태의 학습 안내로 고침
-->
