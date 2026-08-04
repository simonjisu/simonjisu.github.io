---
title: "25주차. 실패를 교훈과 Skill로 바꾸기"
description: "Episodic memory의 trajectory와 feedback에서 reflection을 만든다. 여러 작업에 다시 쓰는 procedural skill library도 정리한다."
tags:
  - Agent Memory
  - Reflection
  - Skill Memory
  - Reflexion
  - Voyager
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 26주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

24주차에는 대화와 실행 기록에서 필요한 memory를 저장하고 검색했다. 이번에는 실패 기록을 다음 행동에 도움이 되는 문장으로 바꾼다. 여러 번 성공한 행동은 순서가 있는 skill로 묶는다. Agent가 경험을 쌓는다고 해서 model weight가 저절로 바뀌는 것은 아니다. 저장한 reflection과 skill을 다음 inference의 context에 넣어 행동을 바꾸는 방식부터 배운다.

## 이번 주에 배울 것

- Task, trajectory, observation, feedback, reflection의 차이
- 실패 원인과 다음 행동 규칙을 분리해 적는 방법
- Reflexion의 Actor, Evaluator, Self-Reflection, memory 흐름
- Reflection을 짧고 검증 가능한 episodic memory로 저장하는 schema
- 성공한 trajectory를 parameter가 있는 procedural skill로 만드는 방법
- Voyager의 automatic curriculum, skill library, iterative prompting
- Raw trajectory, reflection, skill library의 장단점
- 비슷한 skill을 검색하고 여러 skill을 이어 붙이는 방법
- 잘못된 reflection과 낡은 skill을 고치고 폐기하는 기준
- 15개 장난감 tool task로 네 가지 memory 조건을 비교하는 방법

선수 지식은 11주차의 tool agent와 23주차의 episodic·procedural memory, 24주차의 retrieval pipeline이다. 실습은 LLM을 부르지 않는 결정론적 simulator다. Model 성능을 재는 대신 경험 표현에 따라 action plan이 어떻게 달라지는지 확인한다.

!!! note "실패 일기와 공략집은 쓰임이 다르다"

    Reflection은 “이번에는 왜 실패했고 다음에는 무엇을 바꿀까?”에 답한다. Skill은 “이 종류의 작업을 어떤 순서로 실행할까?”에 답한다. 한 번의 실패에서 나온 교훈을 바로 모든 상황의 공략법으로 일반화하면 위험하다.

## 1. 한 번의 실행을 episode로 묶는다

Agent가 목표 하나를 받고 끝날 때까지 움직인 구간을 episode라고 하자. Episode 안에는 task, trajectory, outcome이 있다.

| 항목 | 쉬운 설명 | Atlas 동기화 예시 |
| --- | --- | --- |
| Task | 이루려는 목표 | 최신 문서를 동기화한다 |
| Observation | 환경에서 받은 정보 | Tool이 `HTTP 429`를 반환했다 |
| Action | Agent가 고른 행동 | 곧바로 다시 요청했다 |
| Trajectory | Observation과 action의 시간순 기록 | 요청, 429, 즉시 재요청, 다시 429 |
| Outcome | 성공 여부와 최종 결과 | 동기화 실패 |
| Feedback | 결과를 평가한 신호 | 재시도 간격이 너무 짧다 |
| Reflection | 원인과 다음 규칙을 적은 글 | 429면 2초부터 backoff한다 |
| Skill | 다시 실행하는 절차 | 인증, 요청, backoff, 검증, 저장 |

Trajectory는 CCTV 영상처럼 무슨 일이 있었는지 보여준다. Reflection은 그 기록을 읽고 남긴 짧은 메모다. Skill은 다음 작업에서 호출할 실행 절차다. 셋을 같은 field에 넣으면 검색 결과가 뒤섞인다.

```json
{
  "episode_id": "ep-25-001",
  "task": "sync Atlas documents",
  "trajectory": [
    "authenticate: success",
    "fetch: HTTP 429",
    "retry immediately: HTTP 429"
  ],
  "outcome": "failure",
  "feedback": "retry interval was too short",
  "reflection": "On HTTP 429, retry with exponential backoff from 2 seconds.",
  "source": ["trace-71", "tool-call-9"]
}
```

Reflection만 남기고 원본 trajectory를 지우지 않는다. 교훈이 틀렸을 때 원래 observation으로 돌아가 확인해야 한다. 반대로 다음 prompt에 긴 trace를 매번 넣지는 않는다. 짧은 reflection을 먼저 검색하고 중요한 행동을 결정할 때 source trace를 펼친다.

## 2. Feedback은 정답과 같지 않다

Feedback은 여러 곳에서 온다.

- 환경은 성공 여부, error code, test 결과를 돌려준다.
- 사용자는 답이 길다거나 조건을 놓쳤다고 알려준다.
- Rule checker는 JSON schema, 권한, 금지 action을 검사한다.
- 별도의 evaluator model은 결과와 rubric을 비교해 글로 평가한다.
- Agent 자신도 trace를 읽고 원인을 추측한다.

`실패`라는 숫자 하나만으로는 다음 행동을 정하기 어렵다. `HTTP 429 이후 즉시 재시도했다`라는 trace와 `간격이 너무 짧다`라는 feedback이 있어야 구체적인 규칙을 만든다. Self-reflection은 유용하지만 사실 확인 장치는 아니다. Agent가 원인을 잘못 짚을 수 있으므로 tool error, test, 사용자 교정처럼 외부에서 확인한 신호를 함께 보존한다.

!!! warning "성공한 답에서도 나쁜 교훈이 나온다"

    우연히 성공했다고 모든 action이 옳았던 것은 아니다. 같은 결제를 두 번 요청했는데 두 번째 요청이 중복 차단으로 막혔다면 결과만 보고 안전한 절차라고 저장해서는 안 된다. Outcome과 과정 검사를 함께 둔다.

## 3. Reflexion은 weight 대신 글을 남긴다

![Actor, Evaluator, Self-reflection model과 장·단기 memory로 이루어진 Reflexion 구조](/notes/tutorial/llm_lecture/images/w25_reflexion_architecture.png)

*그림 1. Reflexion의 Actor, Evaluator, Self-Reflection, trajectory, experience memory 흐름과 반복 algorithm. 출처: Shinn et al. (2023), Figure 2에서 발췌.[^1]*

Reflexion에는 세 역할이 나온다. Actor는 환경에서 action을 고른다. Evaluator는 trajectory의 결과를 평가한다. Self-Reflection model은 trajectory와 평가를 읽고 다음 시도에 쓸 글을 만든다. 그 글은 experience memory에 쌓이고 Actor가 다음 trajectory를 만들 때 context로 읽는다.[^1]

```text
task
  → Actor가 trajectory 생성
  → Environment가 observation과 결과 반환
  → Evaluator가 성공 여부와 feedback 생성
  → Self-Reflection이 실패 원인과 다음 규칙 작성
  → Experience memory에 저장
  → 다음 시도의 Actor prompt에 주입
```

논문은 이를 verbal reinforcement learning이라고 부른다. 보상이나 feedback을 자연어 reflection으로 바꿔 다음 시도에 사용한다. 일반적인 reinforcement learning처럼 gradient로 policy weight를 업데이트하는 과정은 아니다.[^1] 7주차 PPO와 구분해야 한다.

| 구분 | PPO 같은 weight 학습 | Reflexion의 verbal feedback |
| --- | --- | --- |
| 바뀌는 대상 | Model weight | Prompt에 넣는 experience memory |
| 업데이트 방법 | Gradient와 optimizer | 자연어 reflection 저장과 검색 |
| 적용 시점 | Training step 뒤 | 다음 trial의 inference |
| 장점 | 반복 패턴을 model 자체에 반영 | 빠르게 고치고 사람이 읽기 쉬움 |
| 주의점 | 계산 비용과 reward 설계 | Context 길이, 잘못된 교훈, 검색 오류 |

Reflection을 지우면 행동 개선도 사라진다. 다른 model이나 prompt는 같은 교훈을 다르게 해석하기도 한다. “Agent가 학습했다”라는 말을 쓸 때는 weight가 바뀌었는지, 외부 memory만 바뀌었는지 밝힌다.

## 4. 좋은 reflection은 다음 행동을 바꾼다

나쁜 reflection은 `더 주의하자`처럼 원인이 없거나, `동기화는 항상 위험하다`처럼 범위가 지나치게 넓다. 좋은 reflection에는 다섯 항목이 들어간다.

1. 어떤 task와 조건에서 생긴 일인가?
2. 기대한 결과와 실제 결과는 무엇인가?
3. Trace의 어느 action이 실패와 연결되는가?
4. 다음 시도에서 무엇을 다르게 실행할 것인가?
5. 이 규칙을 적용하지 말아야 할 예외는 무엇인가?

```text
Condition: Atlas fetch returns HTTP 429.
Evidence: Two immediate retries returned the same error.
Cause: The retry interval was too short.
Next rule: Retry with exponential backoff starting at 2 seconds.
Stop rule: Stop after 4 attempts and ask for operator review.
```

`Cause`는 확인된 사실과 추론을 나눠 적는다. `두 번의 즉시 재시도가 429였다`는 trace에서 확인한 사실이다. `간격이 짧아서 실패했다`는 원인 추론이다. API 문서가 backoff를 요구한다고 확인하면 confidence를 높인다.

```json
{
  "memory_type": "reflection",
  "task_family": "document_sync",
  "trigger": "HTTP 429",
  "lesson": "retry with exponential backoff from 2 seconds",
  "stop_condition": "4 failed attempts",
  "evidence_ids": ["tool-call-8", "tool-call-9"],
  "confidence": 0.82,
  "status": "candidate",
  "created_at": "2026-08-04T23:00:00+09:00"
}
```

처음에는 `candidate`로 저장한다. 같은 규칙이 다른 task에서도 성공하면 `verified`로 올린다. 반례가 나오면 범위를 좁히거나 `retired`로 바꾼다. Reflection도 version과 평가 기록이 필요한 memory다.

## 5. Reflection을 너무 많이 넣지 않는다

Experience memory가 길어지면 서로 다른 교훈이 충돌한다. 예전 reflection에는 `실패하면 즉시 다시 시도`라고 적혀 있다. 새 reflection에는 `429면 기다린 뒤 재시도`라고 적혀 있다. 다음 기준으로 몇 개만 고른다.

- 현재 task family와 tool이 같은가?
- Error code와 observation이 비슷한가?
- Reflection의 source가 실제 tool trace인가?
- 최근 실행에서도 성공한 규칙인가?
- 지금의 system policy와 충돌하지 않는가?
- 같은 lesson을 말하는 memory가 중복되어 있는가?

Reflection retrieval의 top-k가 크다고 늘 좋지는 않다. 서로 다른 API의 retry 규칙 세 개를 함께 넣으면 Actor가 잘못 섞는다. Task router로 family를 좁힌 뒤 trigger와 error를 검색한다. 그런 다음 최신 policy와 대조한다.

## 6. 여러 번 성공한 절차를 Skill로 묶는다

Reflection은 한 사건의 교훈이다. 같은 종류의 성공 trajectory가 쌓이면 고정된 서비스 이름과 file path를 parameter로 바꾼다.

```text
Raw trajectory
authenticate:atlas → fetch:atlas → backoff:atlas
→ validate:atlas → save:atlas

Parameterized skill
sync(service, rate_limited)
→ authenticate(service)
→ fetch(service)
→ if rate_limited: retry_with_backoff(service)
→ validate(service)
→ save(service)
```

Skill에는 실행 code만 넣지 않는다. 언제 부르는지와 성공 조건, 부작용, 권한, version을 함께 기록한다.

```json
{
  "skill_id": "sync-documents-v3",
  "description": "Synchronize and validate documents from a rate-limited service",
  "parameters": ["service", "destination"],
  "preconditions": ["read access", "destination is writable"],
  "steps": ["authenticate", "fetch", "backoff", "validate", "save"],
  "success_check": "saved checksum equals fetched checksum",
  "side_effects": ["writes destination files"],
  "required_permissions": ["service.read", "files.write"],
  "version": 3,
  "source_episode_ids": ["ep-18", "ep-27", "ep-31"]
}
```

Parameter가 너무 많으면 사실상 새 program을 매번 만드는 셈이다. 너무 적으면 Atlas에서 만든 skill을 Nova에 재사용하지 못한다. 바뀌는 값과 고정해야 할 안전 규칙을 나눈다. `service`와 `destination`은 parameter지만 checksum 검증은 빼면 안 되는 step이다.

## 7. Voyager는 실행 가능한 Skill Library를 쌓는다

Voyager는 Minecraft 환경에서 automatic curriculum, executable skill library, iterative prompting을 함께 사용한다. 성공한 program을 skill library에 넣고 새 task와 관련된 skill을 검색해 prompt에 제공한다. 실행 error와 environment feedback을 보고 program을 고치며 별도의 model fine-tuning 없이 진행한다.[^2]

Voyager의 skill은 자연어 교훈만이 아니라 실행 가능한 code다. 복잡한 행동을 시간 순서가 있는 program으로 보존하고 다른 skill과 조합한다. 논문은 skill description의 embedding으로 관련 skill을 검색한다.[^2]

| Reflection memory | Skill memory |
| --- | --- |
| 실패 원인과 다음 규칙을 글로 남김 | 성공 절차를 code나 workflow로 남김 |
| 한 episode의 조건을 자세히 보존 | 여러 task에서 쓸 parameter를 정의 |
| Actor가 읽고 새 plan을 만듦 | Agent가 호출하거나 조합해 실행 |
| 적용 범위가 모호함 | Interface와 권한을 검사하기 쉬움 |
| 짧지만 실행 가능성이 보장되지 않음 | 실행 가능하지만 낡은 API와 부작용에 취약 |

Generative Agents의 reflection은 여러 observation을 묶어 더 높은 수준의 생각을 만든다. 최근 memory의 importance 합이 threshold를 넘으면 reflection 질문과 insight를 만들고 근거 memory를 연결한다.[^3] Reflexion이 task 실패를 다음 trial에서 고치는 데 초점을 둔다면, Generative Agents는 많은 일상 observation을 주기적으로 종합한다.

## 8. Skill을 검색하고 조합한다

Skill retrieval에는 task 설명만 쓰지 않는다. 필요한 권한과 입출력 type, environment, version을 filter로 확인한다.

```python
def select_skills(task, runtime):
    candidates = skill_store.search(task.description)
    candidates = [
        skill for skill in candidates
        if skill.environment == runtime.environment
        and skill.version_compatible(runtime.api_version)
        and runtime.permissions.contains(skill.required_permissions)
    ]
    return rerank_by_success_and_cost(candidates)[:3]
```

`sync`와 `publish`를 이어야 한다면 두 skill의 출력과 입력이 맞는지 본다. `sync`가 반환한 artifact path를 `publish`가 받도록 연결한다. 두 skill이 각각 안전해도 조합 과정에서 문제가 생긴다. 예를 들어 동기화가 일부만 성공했는데 publish를 실행하면 불완전한 문서가 공개된다. Skill 사이에도 success check와 승인 gate를 둔다.

Skill을 model prompt 안의 text로만 저장할 수도 있고 실제 function이나 workflow graph로 등록할 수도 있다. 외부 action을 실행한다면 후자가 검사하기 쉽다. Model은 skill을 고르고 parameter를 채우되 권한과 schema 검사는 runtime이 맡는다.

## 9. 네 가지 memory 조건을 비교한다

실습은 `sync`, `publish`, `archive`, `export` 네 family와 `sync_and_publish` 조합 task를 만들었다. 서비스는 Atlas, Nova, Orion 세 가지이며 모두 15개다.[^4]

복잡한 task에는 한 step이 더 필요하다. Rate limit이 있으면 backoff를 넣는다. 게시 전에는 승인을 받는다. Archive는 upload 전에 검증하며 secret이 든 export는 먼저 가린다. 단순 Orion task 네 개는 기본 plan으로도 풀린다.

| 조건 | Context에 넣는 경험 | 동작 방식 |
| --- | --- | --- |
| No memory | 없음 | 빠진 step이 있는 기본 plan 사용 |
| Raw trajectory | Atlas의 성공 trace | Trace의 고정된 서비스 이름까지 복사 |
| Reflection | 조건별 짧은 lesson | 현재 서비스의 기본 plan에 빠진 step 추가 |
| Skill library | Parameterized procedure | 서비스 parameter를 채우고 skill을 조합 |

![Raw trajectory, reflection, skill library 조건의 task 성공 수와 family별 성공률](/notes/tutorial/llm_lecture/images/w25_reflection_skill_results.png)

*그림 2. 15개 결정론적 장난감 tool task에서 memory 표현을 바꾼 결과. LLM benchmark나 model 학습 결과가 아니다.[^4]*

| 방법 | 성공 | 평균 memory 문자 수 | 평균 plan step 수 |
| --- | ---: | ---: | ---: |
| No memory | 4/15 | 0.0 | 3.60 |
| Raw trajectory | 8/15 | 43.7 | 4.13 |
| Reflection | 12/15 | 58.1 | 4.33 |
| Skill library | 15/15 | 91.8 | 5.13 |

Raw trajectory는 Atlas task 네 개를 풀었지만 같은 절차가 필요한 Nova에서는 실패했다. Trace에 `atlas`가 그대로 들어 있었기 때문이다. Reflection은 단일 family 12개를 모두 풀었다. `sync_and_publish`에서는 top-1 reflection 하나만 읽도록 정했기 때문에 두 절차를 합치지 못했다. Skill library는 parameterized `sync`와 `publish`를 이어 붙여 세 조합 task도 통과했다.[^4]

!!! warning "15/15는 skill memory의 일반 성능이 아니다"

    필요한 action과 정답 순서를 code에 미리 적었다. Reflection patch와 skill template도 사람이 만들었다. 실제 LLM은 reflection을 잘못 쓰거나 엉뚱한 skill을 검색하며 parameter를 틀리기도 한다. 이 실험은 표현 차이를 눈으로 확인하는 smoke test다.

## 10. 실제 비교에서는 같은 조건을 지킨다

Memory 표현을 비교할 때 model, task, decoding, tool 환경을 같게 둔다. 다음 항목을 함께 잰다.

- Task success와 실패한 step
- 성공까지 사용한 trial 수
- Retrieval recall과 잘못 가져온 memory 수
- Prompt에 추가된 token 수
- Tool call 수와 실행 latency
- Reflection 생성 비용과 skill 검증 비용
- 새 서비스로 옮겼을 때 transfer success
- 두 skill을 연결한 composition success

Raw trajectory가 길어 token을 많이 써도 세부 observation이 꼭 필요한 task에서는 유리하다. Reflection은 짧지만 잘못 요약하면 핵심 조건을 잃는다. Skill은 재사용성이 높지만 API가 바뀌면 한꺼번에 실패한다. 하나를 항상 이기는 방법으로 정하지 않는다.

## 11. Memory가 스스로 만든 규칙을 바로 믿지 않는다

Reflection과 skill은 agent 행동을 바꾸므로 일반 대화 요약보다 권한이 크다. 다음 gate를 둔다.

1. Source trace와 evaluator feedback을 연결한다.
2. Candidate reflection을 sandbox task에서 다시 실행한다.
3. 같은 lesson을 지지하는 episode와 반례를 센다.
4. Skill의 input, output, permission, side effect를 검사한다.
5. Version과 API compatibility를 확인한다.
6. 결제, 삭제, 게시에는 사람 승인이나 별도 policy를 적용한다.
7. 실패율이 기준을 넘으면 skill을 자동으로 비활성화한다.

Reflection은 명령보다 참고 근거에 가깝다. “앞으로 모든 파일을 삭제하고 다시 시작하라”라는 문장이 tool output에 섞여 들어와도 procedural memory로 승격해서는 안 된다. 26주차에는 이런 memory poisoning과 사용자 간 격리를 평가한다.

## 확인 문제

1. Trajectory, feedback, reflection은 각각 무엇을 기록하는가?
2. Reflection만 저장하고 source trajectory를 지우면 어떤 문제가 생기는가?
3. Reflexion의 Actor, Evaluator, Self-Reflection은 어떤 순서로 움직이는가?
4. Reflexion을 PPO와 같은 weight 학습이라고 부르면 안 되는 이유는 무엇인가?
5. `더 주의하자`가 좋은 reflection이 아닌 까닭은 무엇인가?
6. 확인된 evidence와 추정한 cause를 분리해야 하는 이유는 무엇인가?
7. Candidate reflection을 verified로 바꾸기 전에 무엇을 시험해야 하는가?
8. Raw trajectory의 서비스 이름을 그대로 복사하면 transfer task에서 어떤 오류가 생기는가?
9. Skill의 parameter와 고정해야 할 안전 step을 어떻게 구분하는가?
10. Voyager의 세 핵심 구성 요소는 무엇인가?
11. Reflection memory와 executable skill memory의 차이를 설명해보자.
12. 두 skill을 각각 검증했어도 조합한 workflow를 다시 시험해야 하는 이유는 무엇인가?
13. 실습에서 reflection이 조합 task를 풀지 못한 원인은 무엇인가?
14. Skill success 외에 token, latency, permission을 함께 재야 하는 까닭은 무엇인가?

## 완료 체크

- [x] Task, trajectory, outcome, feedback, reflection, skill을 구분했다.
- [x] 실패 trace에서 조건, evidence, cause, next rule을 뽑았다.
- [x] Reflexion이 weight update 없이 experience memory를 쓰는 과정을 설명했다.
- [x] Reflection record에 source, confidence, status를 넣었다.
- [x] 여러 성공 trajectory의 고정값을 parameter로 바꿨다.
- [x] Skill의 precondition, success check, side effect, permission을 정리했다.
- [x] Voyager의 executable skill library와 iterative prompting을 살펴봤다.
- [x] Raw trajectory, reflection, skill library를 같은 task에서 비교했다.
- [x] 15개 장난감 task의 실행 결과와 한계를 기록했다.
- [x] 다음 주차의 평가와 memory poisoning 점검 항목을 정했다.

---

[^1]: Shinn, N. et al. (2023). [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366). Actor, Evaluator, Self-Reflection model, short-term trajectory, long-term experience memory, Figure 2와 verbal feedback loop를 참고했다.
[^2]: Wang, G. et al. (2023). [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291). Automatic curriculum, executable skill library, environment feedback을 반영하는 iterative prompting과 skill retrieval을 참고했다.
[^3]: Park, J. S. et al. (2023). [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442). Observation의 importance 합을 바탕으로 reflection을 만들고 insight를 근거 memory에 연결하는 §4.2를 참고했다.
[^4]: 직접 실행한 `llm_lecture/week25_reflection_skill_memory.py`의 결과다. 사람이 action 순서와 reflection, skill template을 정한 15개 결정론적 task이며 LLM을 호출하거나 model을 학습하지 않았다. Script, CSV, JSON과 논문 원본은 Git에서 제외하고 최종 plot만 공개했다. 실행일: 2026-08-04.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
원본/윤문본: 14,457자 / 14,443자, metrics v2.0 변경률 0.52%
카테고리별 탐지/수정: C-11 연결어미 뒤 쉼표 4→0, A-10 가능 표현 6→0, D-1 결산 표현 1→0, A-8 이중 피동 0→0, H-1 문두 접속사 남발 0→0
정량 점검: humanize-korean metrics v2.0 risk score 1→1, risk band low 유지
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 논문 개념, task 수, 실행 결과와 reference를 보존함
주요 변경 1: “다시 실행할 수 있는 절차” → “다시 실행하는 절차”
주요 변경 2: “따라서 agent가 학습했다” → “Agent가 학습했다”
주요 변경 3: “검색하고, 최신 policy와” → “검색한다. 그런 다음 최신 policy와”
주요 변경 4: “backoff를 넣고, 게시 전에는” → “backoff를 넣는다. 게시 전에는”
-->
