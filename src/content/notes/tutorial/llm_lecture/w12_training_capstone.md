---
title: "12주차. 통합 학습 프로젝트"
description: "Base, SFT, DPO, GRPO를 같은 prompt와 evaluator로 비교하고 자동 점수의 오류를 사람이 읽어 확인하는 평가 보고서를 만든다."
tags:
  - LLM
  - evaluation
  - SFT
  - DPO
  - GRPO
  - capstone
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 16주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

1~11주차에는 model의 구조와 여러 학습 방법을 따로 살펴봤다. 이제 checkpoint를 한 평가표에 올릴 차례다. Model마다 prompt나 decoding 설정이 다르면 점수 차이가 학습 방법 때문인지 시험 조건 때문인지 알 수 없다. 이번 주에는 평가 계획을 먼저 고정하고 Base, SFT, DPO, GRPO를 같은 조건에서 비교한다.[^1][^2]

## 이번 주에 배울 것

- 공통 평가 계획과 성공 기준을 먼저 쓰는 법
- Base→SFT 뒤 DPO·GRPO로 갈라지는 공정한 비교 구조
- Task, format, tool execution, 길이와 시간을 함께 기록하는 방법
- lm-evaluation-harness와 LightEval의 역할
- 자동 verifier가 틀린 답을 통과시키는 사례
- 20개 출력을 직접 읽어 평가 코드를 감사하는 절차

선수 지식은 3주차의 train·validation·test 분리, 8주차의 DPO, 10~11주차의 GRPO와 process reward다.

!!! note "시험지를 먼저 봉인한다"

    답안을 본 뒤 채점 기준을 바꾸면 원하는 model에 유리하게 시험을 고칠 수 있다. Prompt, 정답, metric, decoding과 성공 기준을 학습 결과를 보기 전에 정한다.

## 1. 성공 기준부터 문장으로 적는다

예를 들어 tool agent의 목표를 다음처럼 정한다.

```text
Primary success:
- held-out task accuracy >= 70%
- required-tool execution success >= 80%

Guardrails:
- format accuracy >= 90%
- unnecessary tool-call rate <= 10%
- repeated tool-call rate <= 2%
```

Primary metric만 만족하고 guardrail을 어기면 성공으로 보지 않는다. 최종 답이 올라도 불필요한 호출이 늘거나 형식이 망가질 수 있기 때문이다.

| 항목 | 평가 전에 고정할 값 |
| --- | --- |
| Dataset | 이름, revision, split, sample 수 |
| Prompt | system message, few-shot 예시, chat template |
| Generation | temperature, top-p, 최대 token, stop sequence |
| Metric | normalization, answer extraction, 적용 대상 |
| Runtime | batch, dtype, device, seed |
| 성공 기준 | primary metric과 guardrail |

## 2. 평가 도구는 data에서 score까지 같은 길을 만든다

![lm-evaluation-harness Task의 data-to-metric 흐름](/notes/tutorial/llm_lecture/images/w12_lm_eval_task_pipeline.png)

*그림 1. lm-evaluation-harness의 Task가 data source와 target을 prompt로 만들고 model request를 보낸 뒤 metric을 계산하는 흐름. 출처: Biderman et al. (2024), Figure 3에서 발췌.[^1]*

Evaluation harness는 dataset, prompt formatting, model request, output 후처리와 metric을 한 task 안에 묶는다. 같은 task config를 여러 model에 쓰면 빠뜨린 설정을 줄일 수 있다.[^1][^3]

```yaml
task: my_exact_instruction
dataset_path: my_org/my_eval_data
test_split: test
output_type: generate_until
doc_to_text: "{{prompt}}"
doc_to_target: "{{answer}}"
generation_kwargs:
  do_sample: false
  max_gen_toks: 64
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
```

lm-evaluation-harness는 `doc_to_text`, `doc_to_target`, output type, generation 설정, filter와 metric을 YAML로 정한다. 여러 task는 group config로 묶는다.[^3]

## 3. 네 checkpoint는 같은 계보에서 비교한다

공정한 post-training 비교는 다음처럼 갈라진다.

```text
Base
  └─ SFT
      ├─ DPO
      └─ GRPO
```

DPO와 GRPO는 같은 SFT checkpoint에서 시작해야 한다. 한쪽만 더 큰 Base model이나 다른 SFT dataset을 쓰면 학습 알고리즘의 차이를 분리할 수 없다.

| 조건 | Base | SFT | DPO | GRPO |
| --- | --- | --- | --- | --- |
| 시작 weight 계보 | 동일 Base | Base에서 학습 | 같은 SFT에서 시작 | 같은 SFT에서 시작 |
| Test prompt | 동일 | 동일 | 동일 | 동일 |
| Chat template | 동일한 호환 규칙 | 동일 | 동일 | 동일 |
| Decoding | greedy 또는 같은 sampling | 동일 | 동일 | 동일 |
| Evaluator | 동일 | 동일 | 동일 | 동일 |

Base model과 chat model은 template 자체가 다를 수 있다. 이때 문자열을 억지로 같게 만들기보다 각 model이 요구하는 template을 쓰되, user가 보는 질문 내용과 생성 조건을 같게 둔다. 무엇이 달랐는지 보고서에 적는다.

## 4. 한 점수 대신 task 묶음을 만든다

| 영역 | 예시 | 주 metric | 보조 metric |
| --- | --- | --- | --- |
| 일반 대화 | 짧은 사실 설명 | 사람이 읽은 정확성 | 길이 |
| 지시 따르기 | JSON, bullet, exact string | Strict format pass | loose pass |
| Reasoning | 산술, 규칙 추론 | Final accuracy | 과정 검증 |
| Tool calling | 계산기·검색·no-tool | Full trajectory success | tool 선택·실행 |

전체 평균만 공개하면 어느 능력이 좋아지고 나빠졌는지 숨는다. 영역별 점수와 실패 유형을 함께 낸다. 사용 목적에 중요한 영역에는 평균 가중치를 더 줄 수 있지만, 가중치는 결과를 보기 전에 정한다.

## 5. 같은 작은 정책으로 네 단계를 실행한다

실제 LLM 네 개를 CPU에서 다시 학습하는 대신, 11주차의 두 상태·여덟 trajectory 정책을 같은 시작점에서 학습했다.[^4] Base는 균일한 정책이다. SFT는 올바른 trajectory 예시를 12 update 배웠다. DPO는 같은 SFT에서 chosen과 shortcut을 40 update 비교했다. GRPO도 같은 SFT에서 출발해 process reward로 120 update했다.

모든 단계에 같은 task 상태, 후보 trajectory, evaluator를 썼다. 이 실험은 학습 단계와 평가표를 연결하는 연습이다. 자유로운 문장을 만드는 LLM의 품질 순위로 해석하지 않는다.

![Base, SFT, DPO, GRPO의 공통 평가와 자동 verifier 감사](/notes/tutorial/llm_lecture/images/w12_capstone_evaluation.png)

*그림 2. 왼쪽은 같은 toy policy와 evaluator에서 측정한 단계별 기대 통과율이다. 오른쪽은 20개 문자열 답을 자동 contains verifier와 사람이 strict하게 읽어 비교한 결과다. PyTorch 2.13.0, macOS CPU, seed 42에서 직접 실행했다.[^4]*

| 단계 | Task 정확도 | Format | Tool 실행 | 전체 trajectory | 평균 길이 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 50.00% | 87.50% | 50.00% | 12.50% | 18.88 token |
| SFT | 61.11% | 90.28% | 61.11% | 31.94% | 18.35 token |
| DPO | 80.37% | 95.09% | 84.13% | 69.41% | 17.51 token |
| GRPO | 99.69% | 99.92% | 99.72% | 99.26% | 16.54 token |

SFT 예시를 보고 전체 trajectory 성공률이 12.50%에서 31.94%로 올랐다. DPO는 shortcut보다 chosen을 선호하도록 만들어 69.41%에 도달했다. Process reward를 직접 받은 GRPO는 99.26%였다.

Toy policy의 action은 미리 정한 여덟 전략 중 하나다. GRPO가 자연어 reasoning에서 DPO보다 낫다는 증거가 아니다. 실제 비교에서는 seed를 여러 개 쓰고 confidence interval을 내며, 학습 token과 계산 예산도 맞춰야 한다.[^1]

## 6. 응답 길이와 추론 시간도 같은 방법으로 잰다

길이는 tokenizer가 만든 response token 수로 잰다. Chat template의 prompt token과 model이 새로 생성한 token을 섞지 않는다.

```python
new_tokens = output_ids[input_ids.shape[1]:]
response_length = len(new_tokens)
```

추론 시간은 warm-up 뒤 같은 batch와 device에서 여러 번 잰 중앙값을 쓴다. GPU라면 측정 전후에 synchronize가 필요하다.

```python
torch.cuda.synchronize()
started = time.perf_counter()
outputs = model.generate(**inputs, **generation_kwargs)
torch.cuda.synchronize()
elapsed = time.perf_counter() - started
```

이번 toy policy에서 trajectory 하나를 sampling하는 값은 약 0.003~0.010μs였다. Timer 해상도와 Python overhead에 비해 너무 작아 단계별 속도 순위를 매길 수 없다. 표에는 원본 값을 저장했지만 블로그 비교표에서는 제외했다. 실제 LLM에서는 end-to-end latency와 생성 token 수, batch를 함께 기록한다.

## 7. 자동 점수와 사람이 읽은 결과가 다를 수 있다

정답 문자열이 output 안에 들어가기만 하면 통과시키는 verifier를 생각해보자.

```python
def contains_verifier(expected: str, output: str) -> bool:
    return expected.lower() in output.lower()
```

20개 출력을 직접 읽어 strict 정답과 비교했다.[^4]

| Expected | Output | 자동 판정 | 사람 판정 | 문제 |
| --- | --- | --- | --- | --- |
| `42` | `142` | 통과 | 실패 | 부분 문자열 |
| `yes` | `yesterday` | 통과 | 실패 | 단어 경계 없음 |
| `{"value": 7}` | `Result: {"value": 7}` | 통과 | 실패 | JSON-only 지시 위반 |
| `Paris` | `Paris, Texas` | 통과 | 실패 | 같은 이름의 다른 장소 |
| `9` | `The answer is 9.` | 통과 | 실패 | 숫자-only 지시 위반 |

자동 verifier는 20개 중 5개를 잘못 통과시켰다. 사람 판정을 정답으로 보면 일치율은 75%였다. False negative는 없었지만 false positive가 많아 model 점수를 부풀릴 수 있다.

Strict task에는 full match, JSON parsing 뒤 key·value 검사, 숫자 parsing과 허용 오차처럼 구조에 맞는 verifier를 쓴다. 평가 harness 공식 README도 generative task를 처음 연결할 때 `--limit 10`으로 sample output과 answer extraction을 확인하라고 권한다.[^5]

## 8. lm-evaluation-harness와 LightEval로 옮긴다

현재 lm-evaluation-harness CLI는 subcommand 방식으로 바뀌는 중이다. 설치한 version에서 `lm-eval run -h`를 먼저 확인한다.[^5]

```bash
pip install "lm_eval[hf]"

lm-eval run \
  --model hf \
  --model_args pretrained=/checkpoints/my_model \
  --tasks my_instruction,my_reasoning \
  --batch_size 8 \
  --output_path results/my_model \
  --log_samples
```

`--log_samples`로 실제 prompt와 response를 남기면 자동 점수의 오류를 나중에 감사할 근거가 생긴다. PEFT adapter는 model argument에 base와 adapter 경로를 함께 지정한다. CLI가 빠르게 바뀌므로 실행한 commit이나 package version도 기록한다.[^5]

LightEval은 task를 찾아보고 config와 metric을 검사하는 명령을 제공한다.[^6]

```bash
lighteval tasks list
lighteval tasks inspect truthfulqa:mc

lighteval accelerate \
  "model_name=/checkpoints/my_model" \
  "truthfulqa:mc|0,gsm8k|3" \
  --output-dir results/lighteval \
  --save-details
```

LightEval의 generative task에는 exact match, F1, majority-at-k 같은 metric이 있다. Exact match도 공백·대소문자 normalization과 full string·부분 string 선택에 따라 값이 달라진다. Metric 이름만 기록하지 말고 인자까지 남긴다.[^7]

## 9. 최종 비교 보고서에는 실패를 함께 넣는다

| 구역 | 들어갈 내용 |
| --- | --- |
| 목표 | 사용 scenario와 사전에 정한 성공 기준 |
| 계보 | Base, SFT, DPO, GRPO의 시작 checkpoint |
| 데이터 | Train·validation·test revision과 contamination 점검 |
| 설정 | Template, decoding, dtype, batch, seed |
| 품질 | 영역별 accuracy, format, tool success |
| 효율 | 길이, latency, peak memory, 학습·평가 비용 |
| 감사 | 사람이 읽은 sample 수, verifier false positive·negative |
| 실패 | 좋아진 task, 나빠진 task, reward hacking과 평가 오류 |
| 재현 | 코드 commit, package version, raw sample 경로 |

가장 높은 평균 점수를 얻은 model만 고르지 않는다. 실제 서비스가 요구하는 task와 guardrail을 만족하는 checkpoint를 고른다. 어느 model도 성공 기준을 통과하지 못했다면 “우승자 없음”이 올바른 결론이다.

## 확인 문제

1. DPO와 GRPO가 같은 SFT checkpoint에서 시작해야 하는 이유는 무엇인가?
2. Chat template을 model마다 다르게 써야 할 때도 공정성을 지키려면 무엇을 같게 둬야 하는가?
3. `expected in output` 검사가 `42`와 `142`를 구분하지 못하는 문제를 어떻게 고칠 수 있는가?
4. 평균 정확도 외에 format과 tool execution을 따로 공개해야 하는 이유는 무엇인가?
5. 평가 결과를 본 뒤 성공 기준을 바꾸면 어떤 문제가 생기는가?

## 완료 체크

- [x] 공통 평가 계획과 성공 기준을 먼저 적었다.
- [x] Base, SFT, DPO, GRPO를 같은 prompt와 evaluator로 비교했다.
- [x] Task, format, tool 실행, 응답 길이와 추론 시간을 기록했다.
- [x] 20개 출력을 읽고 자동 verifier의 오류 5개를 찾았다.
- [x] 결과물로 `Base/SFT/DPO/GRPO 비교 보고서`를 완성했다.

---

[^1]: Biderman, S. et al. (2024). [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782). 평가 설정의 민감성, 반복 실행과 lm-evaluation-harness의 Task 구조를 참고했다.
[^2]: Liang, P. et al. (2022). [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110). 같은 scenario에서 정확도 이외의 여러 metric을 함께 보는 원칙을 참고했다.
[^3]: EleutherAI. [lm-evaluation-harness Task Guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md). YAML task, prompt·target, output type, generation 설정, filter와 metric을 참고했다. 확인일: 2026-08-02.
[^4]: 직접 실행한 `llm_lecture/week12_training_capstone_demo.py`의 결과다. 같은 두 상태·여덟 trajectory의 categorical policy에 Base, SFT 12 update, DPO 40 update, GRPO 120 update를 적용했다. PyTorch 2.13.0, macOS CPU, seed 42를 사용했다. 20개 출력의 strict 판정도 CSV로 저장했으며 원본은 Git에서 제외했다. 실행일: 2026-08-02.
[^5]: EleutherAI. [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). 현재 CLI, model backend, sample logging, PEFT와 generative task 점검 권고를 참고했다. 확인일: 2026-08-02.
[^6]: Hugging Face. [LightEval Quick Tour](https://huggingface.co/docs/lighteval/quicktour). Backend별 실행, task 탐색과 여러 task 실행법을 참고했다. 확인일: 2026-08-02.
[^7]: Hugging Face. [LightEval Metric List](https://huggingface.co/docs/lighteval/main/en/metric-list). Multiple-choice, language modeling과 generative task metric을 참고했다. 확인일: 2026-08-02.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
카테고리별 탐지/수정: A-10 3→0, C-11 0→0, D-1 0→0, H-1 0→0, I-1 0→0
정량 점검: humanize-korean metrics v2.0 risk band low
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 평가 설정과 20개 출력 감사 수치를 그대로 보존함
주요 변경: 가능 표현을 줄이고 평가 단계의 주어와 행동을 직접 연결함
-->
