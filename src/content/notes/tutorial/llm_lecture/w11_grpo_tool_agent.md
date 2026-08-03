---
title: "11주차. GRPO로 reasoning과 tool calling 학습"
description: "도구 선택부터 최종 답까지 이어지는 trajectory를 단계별 reward로 채점하고 GRPO가 편법 대신 올바른 tool use를 배우게 한다."
tags:
  - LLM
  - GRPO
  - tool calling
  - agent
  - process reward
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 16주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

10주차에는 같은 문제의 여러 답을 비교해 GRPO advantage를 만들었다. 답을 한 문장만 쓰는 task는 마지막 결과만 바로 채점하면 된다. 계산기나 검색 도구를 쓰는 agent는 사정이 다르다. 어떤 도구를 골랐는지, 인자는 맞는지, 실행 결과를 최종 답에 반영했는지까지 살펴야 한다.[^1][^2]

## 이번 주에 배울 것

- 한 번의 tool trajectory를 이루는 message 순서
- Python 함수가 tool schema로 바뀌는 과정
- 도구가 필요한 문제와 필요하지 않은 문제를 섞는 이유
- 형식, 도구 선택, 인자, 실행, 최종 답 reward를 나누는 법
- 최종 답만 채점할 때 생기는 reward hacking
- TRL GRPOTrainer의 `tools`와 `environment_factory`

선수 지식은 2주차의 chat template, 5주차의 trajectory와 reward, 10주차의 group-relative advantage다.

!!! note "정답만 맞힌 것과 올바르게 해결한 것은 다르다"

    “17×24는?”이라는 문제에서 model이 우연히 408을 답했더라도 계산기를 쓰라는 지시는 어겼을 수 있다. 반대로 계산기를 정확히 호출했지만 최종 답을 480으로 잘못 옮길 수도 있다. 두 실패는 고칠 부분이 다르다.

## 1. Tool use는 필요한 순간에 외부 함수를 부르는 일이다

![Toolformer가 문장 안에서 선택한 네 가지 API 호출 예시](/notes/tutorial/llm_lecture/images/w11_toolformer_api_examples.png)

*그림 1. Toolformer가 질문 답변, 계산기, 번역, Wikipedia 검색 API를 필요한 위치에 넣은 예시. 출처: Schick et al. (2023), Figure 1에서 발췌.[^1]*

Toolformer는 model이 어떤 API를 언제 부를지, 어떤 인자를 넣을지, 반환값을 다음 token 예측에 어떻게 쓸지를 배우도록 했다.[^1] 오늘날 chat model의 tool calling도 핵심 질문은 같다.

1. 지금 도구가 필요한가?
2. 필요하다면 어떤 도구인가?
3. 인자는 schema에 맞는가?
4. 도구가 성공적으로 실행됐는가?
5. 반환값을 최종 답에 제대로 썼는가?

도구를 많이 호출한다고 좋은 agent는 아니다. Model이 스스로 답할 수 있는 인사말에 검색을 쓰면 느리고 비싸다. 같은 계산기를 끝없이 반복 호출하면 loop에 빠질 수도 있다. Toolformer도 무한 호출을 피하려고 입력 하나당 API 호출 수를 제한했다.[^1]

## 2. 한 trajectory는 여러 message로 이어진다

사용자가 “17×24를 계산기로 구해줘”라고 물었다고 하자. 올바른 trajectory는 다음 순서로 움직인다.[^3]

```json
[
  {"role": "user", "content": "Use the calculator to compute 17 * 24."},
  {
    "role": "assistant",
    "tool_calls": [
      {
        "type": "function",
        "function": {
          "name": "multiply",
          "arguments": {"a": 17, "b": 24}
        }
      }
    ]
  },
  {"role": "tool", "content": "408"},
  {"role": "assistant", "content": "408"}
]
```

| 순서 | Role | 하는 일 | 검사할 값 |
| ---: | --- | --- | --- |
| 1 | `user` | 문제와 도구 사용 조건을 준다 | 요구사항 |
| 2 | `assistant` | Tool call을 요청한다 | 함수 이름과 인자 |
| 3 | `tool` | 실제 실행 결과를 돌려준다 | 성공 여부와 반환값 |
| 4 | `assistant` | 결과를 읽고 답한다 | 최종 정확도와 형식 |

Model이 직접 Python 함수를 실행하는 것은 아니다. Model은 호출할 함수와 인자를 출력한다. Agent runtime이 내용을 읽고 함수를 실행한 뒤 `tool` message를 대화에 붙인다. Model은 갱신된 대화를 다시 읽어 최종 답을 만든다.[^3]

## 3. 함수 이름과 docstring이 schema의 재료다

Transformers는 type hint와 Google-style docstring을 읽어 JSON schema를 만들 수 있다. 실제 함수 코드보다 이름, 인자 이름·자료형, 설명이 model의 선택에 직접 영향을 준다.[^3]

```python
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The product of a and b.
    """
    return a * b
```

```python
inputs = tokenizer.apply_chat_template(
    messages,
    tools=[multiply],
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
```

`apply_chat_template`은 tool schema와 message를 model이 배운 control token 형식으로 바꾼다. 모든 chat model이 같은 tool 형식을 이해하는 것은 아니다. Model card에서 tool calling 학습 여부와 template을 먼저 확인한다.[^3]

!!! warning "JSON이 문법적으로 맞는 것만으로는 부족하다"

    `{"name":"multiply","arguments":{"a":17,"b":42}}`는 올바른 JSON이지만 문제의 24를 42로 잘못 읽었다. Format 검사와 argument 검사를 분리해야 이런 오류가 보인다.

## 4. Reward를 다섯 부분으로 나눈다

계산기가 필요한 task에는 다음 reward를 줄 수 있다. 합계가 1이 되도록 맞췄지만 반드시 이 비율을 써야 하는 것은 아니다.

| Reward | 점수 | 통과 조건 |
| --- | ---: | --- |
| 형식 | 0.10 | Tool call JSON을 parsing할 수 있음 |
| Tool 선택 | 0.20 | `multiply`를 고름 |
| 인자 | 0.20 | `a=17`, `b=24`가 맞음 |
| 실행 | 0.20 | 예외 없이 408을 반환함 |
| 최종 답 | 0.30 | Tool 결과를 이용해 408이라고 답함 |

반복 호출에는 `-0.25`처럼 penalty를 줄 수 있다. Reward의 최대·최소 범위를 먼저 계산하고, 쉬운 형식 점수가 최종 정확도보다 커지지 않게 조절한다.

```python
reward = (
    0.10 * format_ok
    + 0.20 * correct_tool
    + 0.20 * correct_arguments
    + 0.20 * execution_success
    + 0.30 * final_correct
    - 0.25 * repeated_call
)
```

도구가 필요하지 않은 task에는 argument와 execution 항목이 없다. 이때는 형식 0.20, 도구를 부르지 않은 선택 0.30, 최종 답 0.50처럼 별도 rubric을 쓴다. 적용되지 않는 reward를 억지로 0점 처리하면 no-tool task의 최대 점수만 낮아진다.

## 5. Tool이 필요한 문제와 필요 없는 문제를 섞는다

계산 문제만 학습하면 model은 모든 질문에 계산기를 부르는 편법을 배울 수 있다. Dataset에 다음 두 종류를 함께 넣는다.

```json
{"prompt": "Use the calculator to compute 17 * 24.", "requires_tool": true}
{"prompt": "Reply with exactly: hello", "requires_tool": false}
```

Tool-required task에서는 올바른 호출이 성공이다. No-tool task에서는 호출하지 않는 판단이 성공이다. 두 종류의 비율도 기록한다. 학습 자료의 95%가 계산 문제라면 agent가 평범한 인사에도 계산기를 꺼내기 쉽다.

## 6. TRL은 함수와 stateful environment를 구분한다

현재 TRL의 GRPOTrainer는 agent training에서 `tools`와 `environment_factory`를 지원한다.[^2]

| 방식 | 알맞은 예 | 상태를 기억하는가 |
| --- | --- | --- |
| `tools=[...]` | 계산기, 검색, 번역 | 보통 호출 하나로 끝남 |
| `environment_factory=...` | 게임, 장바구니, 여러 단계 DB 작업 | Rollout마다 별도 상태를 가짐 |

```python
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    reward_funcs=[process_reward],
    tools=[multiply],
)
```

함수는 type hint와 Google-style docstring을 갖춰야 한다. Tool loop에서는 새 `tool` message를 붙여도 앞부분의 token 표현이 바뀌지 않는 prefix-preserving chat template이 필요하다. TRL은 알려진 일부 model family에 patched template을 적용하지만, 사용하는 model과 library version에서 실제 rendering을 확인해야 한다.[^2]

Stateful environment는 rollout마다 새 instance를 만든다. `reset()`이 문제를 내고 공개 method가 tool이 되며, `get_reward()`는 환경 상태를 직접 채점한다. 이 기능에는 현재 `transformers>=5.2.0`이 필요하다.[^2]

## 7. 최종 답만 채점하면 편법이 남는다

두 종류의 task와 8개 trajectory 전략을 가진 작은 categorical policy를 만들었다.[^4] Policy는 tool을 한 번 정확히 부르기, 정답만 바로 쓰기, 틀린 인자 쓰기, 같은 tool 반복하기 같은 전략 중 하나를 고른다. Group size 4로 500번 GRPO update했다.

하나는 최종 답과 형식만 채점했다. 다른 하나는 앞에서 정한 process-aware reward를 썼다. 자유로운 문장을 만드는 LLM 실험이 아니라 reward 설계의 차이를 보기 위한 작은 정책 실험이다.

![최종 답 reward와 process-aware reward의 GRPO 학습 비교](/notes/tutorial/llm_lecture/images/w11_tool_reward_comparison.png)

*그림 2. 왼쪽은 최종 답 정확도, 가운데는 전체 trajectory 성공률, 오른쪽은 불필요한 tool과 반복 호출 비율이다. PyTorch 2.13.0, macOS CPU, seed 42에서 직접 실행했다.[^4]*

| Reward 설계 | 최종 답 | 전체 trajectory | Tool 실행 성공 | 불필요한 tool | 반복 호출 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 최종 답 중심 | 99.94% | 35.89% | 13.72% | 34.08% | 3.91% |
| Process-aware | 99.95% | 99.90% | 99.97% | 0.12% | 0.02% |

최종 답 점수만 보면 두 정책은 거의 같다. 첫 정책은 계산기를 써야 할 때 답만 바로 쓰거나, 필요하지 않은 질문에도 tool을 부르는 전략에 높은 확률을 남겼다. 전체 trajectory 성공률은 35.89%에 그쳤다.

Process-aware reward는 tool 선택과 인자, 실행을 따로 보상했다. 최종 답을 유지하면서 전체 성공률이 99.90%까지 올랐다. 이 수치는 두 상태와 여덟 전략만 있는 toy policy의 결과다. 실제 LLM agent에서도 같은 수치가 나온다는 뜻은 아니다.

## 8. Reward hacking은 답변과 로그에서 찾는다

| 증상 | 점수가 오른 까닭 | 추가할 검사 |
| --- | --- | --- |
| Tool 없이 정답만 씀 | 최종 답 reward만 큼 | `requires_tool` 준수 |
| 같은 tool을 반복 호출 | 실행 성공마다 점수를 받음 | 호출 횟수 penalty와 최대 step |
| 틀린 인자 뒤 정답을 추측 | 최종 답만 맞으면 통과 | Argument와 반환값 연결 검사 |
| 가짜 tool 결과를 직접 씀 | 문자열 모양만 검사 | Runtime이 만든 `tool` message ID 확인 |
| 모든 질문에 tool 사용 | No-tool 예시가 부족함 | 불필요한 호출률 |

점수 평균만 보지 말고 함수별 reward와 실제 trajectory를 함께 저장한다. TRL custom reward는 여러 함수를 받을 수 있고 함수별 평균과 표준편차를 로그로 남긴다. Multi-task에서는 적용되지 않는 reward에 `None`을 반환해 합계에서 제외할 수도 있다.[^2]

## 9. Reasoning/tool model 오류 분석표를 만든다

최소한 다음 열을 rollout마다 저장한다.

```text
prompt_id, task_type, rendered_prompt, completion,
tool_name, arguments, tool_result, call_count,
format_reward, selection_reward, argument_reward,
execution_reward, final_reward, total_reward,
verifier_error, human_note
```

성공 사례만 남기면 reward hacking을 찾기 어렵다. 전체 점수는 높지만 과정 점수가 낮은 예시, process는 맞지만 마지막 답을 잘못 옮긴 예시, tool loop가 길어진 예시를 따로 모은다.

## 확인 문제

1. Tool call을 model이 직접 실행한다고 말하면 왜 틀린가?
2. JSON parsing 성공과 argument 정확도를 별도 reward로 둬야 하는 이유는 무엇인가?
3. 계산 문제만 학습했을 때 모든 질문에 계산기를 부르는 편법은 어떻게 막을 수 있는가?
4. Stateless `tools`와 `environment_factory`는 어떤 task에서 나뉘는가?
5. 최종 답 정확도가 99%여도 전체 trajectory 성공률이 낮을 수 있는 사례를 하나 만들어보자.

## 완료 체크

- [x] 한 번의 tool trajectory를 message 순서대로 설명했다.
- [x] 형식, tool 선택, 인자, 실행, 최종 답 reward를 분리했다.
- [x] Tool-required와 no-tool task를 섞어 실험했다.
- [x] 최종 답 편법과 무의미한 반복 호출을 측정했다.
- [x] 결과물로 `Reasoning/tool model과 오류 분석표`를 만들었다.

---

[^1]: Schick, T. et al. (2023). [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761). Figure 1, tool 선택·인자·결과 사용과 반복 호출 제한을 참고했다.
[^2]: Hugging Face. [TRL: GRPO Trainer - Agent Training](https://huggingface.co/docs/trl/grpo_trainer#agent-training). `tools`, `environment_factory`, custom reward와 chat template 조건을 참고했다. 확인일: 2026-08-02.
[^3]: Hugging Face. [Transformers: Tool use](https://huggingface.co/docs/transformers/chat_extras). Python 함수의 schema 변환, `tool_calls`, `tool` message와 chat template 사용법을 참고했다. 확인일: 2026-08-02.
[^4]: 직접 실행한 `llm_lecture/week11_grpo_tool_agent_demo.py`의 결과다. 두 task 상태, 여덟 trajectory 전략, group size 4, 500 update의 categorical policy를 PyTorch 2.13.0, macOS CPU, seed 42에서 학습했다. 원본 CSV와 policy tensor는 Git에서 제외했다. 실행일: 2026-08-02.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
카테고리별 탐지/수정: A-10 2→0, C-11 0→0, D-1 0→0, H-1 0→0, I-1 0→0
정량 점검: humanize-korean metrics v2.0 risk band low
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 tool schema와 실험 수치를 그대로 보존함
주요 변경: 가능 표현을 직접 서술로 바꾸고 tool trajectory의 각 책임을 짧게 나눔
-->
