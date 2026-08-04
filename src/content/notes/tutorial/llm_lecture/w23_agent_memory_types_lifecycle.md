---
title: "23주차. Agent Memory의 종류와 생명주기"
description: "Memory, Knowledge Base, Conversation History, Agent State를 구분하고 semantic·episodic·procedural memory와 저장·검색·갱신 생명주기를 익힌다."
tags:
  - Agent Memory
  - Knowledge Base
  - Conversation History
  - Agent State
  - checkpoint
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 26주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

22주차까지는 VLM이 사진을 읽고 답하는 과정을 배웠다. 이제부터는 agent가 지난 경험을 다음 작업에 활용하는 방법을 살펴본다. 대화 내용을 전부 database에 넣었다고 해서 좋은 memory가 생기지는 않는다. 지금 필요한 기록과 오래 남길 경험을 구분한다. 새 정보가 들어오면 낡은 내용도 고쳐야 한다.

## 이번 주에 배울 것

- Memory, Knowledge Base, Conversation History, Agent State의 차이
- Model weight의 parametric memory와 실행 중 working memory의 구분
- Semantic, episodic, procedural memory에 담기는 내용
- 관찰한 내용을 저장하고 다시 꺼내 쓰는 전체 생명주기
- Thread와 checkpoint로 중단된 workflow를 이어가는 방법
- Conversation History를 그대로 넣을 때 생기는 길이와 충돌 문제
- 출처, 사용자 범위, 시간, 활성 상태를 포함한 memory record
- 최근 대화, 전체 대화, 모든 저장소, 선택된 memory의 context 차이

선수 지식은 1주차의 Transformer context, 11주차의 tool agent, 13주차의 KV cache다. 이번 주 실습은 model을 호출하지 않는다. 무엇을 어디에 저장할지와 어떤 기록을 context에 넣을지 확인하는 저장 구조 smoke test다.

!!! note "기억한다는 말부터 나누어 쓰자"

    Framework 문서에서는 conversation history도 short-term memory라고 부른다. 이 표현이 틀린 것은 아니다. 다만 이번 강의에서는 저장소의 역할을 분명히 보려고 대화 원문인 Conversation History와, 경험에서 골라 여러 session에 재사용하는 Memory를 따로 부른다.

## 1. 네 저장소는 같은 물건이 아니다

여행을 준비하는 agent를 떠올려보자. 사용자가 “나는 채식해”라고 말했다. Agent는 항공권 검색을 마친 뒤 호텔 결제 승인을 기다리고 있다. 회사의 환불 규정도 찾아봐야 한다. 이 정보들은 겉으로는 모두 글이지만 쓰임이 다르다.

| 개념 | 무엇을 담는가 | 범위와 수명 | 여행 agent의 예시 |
| --- | --- | --- | --- |
| Memory | 다음 작업에 다시 쓰려고 고른 사실, 경험, 규칙 | 여러 session에서 불러오며 갱신하거나 삭제할 때까지 유지 | 사용자는 채식함, 지난 예약의 실패 원인 |
| Knowledge Base | Agent 개인 경험과 무관한 외부 참고 지식 | 여러 사용자나 agent가 공유하며 원문이 바뀌면 다시 반영 | 항공사 수하물 규정, 회사 출장 규정 |
| Conversation History | `user`, `assistant`, `tool` message를 시간순으로 쌓은 원문 | 보통 한 session이나 thread에 속함 | 사용자의 질문, agent 답변, tool 반환값 |
| Agent State | 지금 workflow를 이어가는 데 필요한 현재 값 | Step마다 바뀌며 checkpoint로 복구 | 현재 목표, 승인 대기, 다음 node, 중간 파일 |

이 구분은 database 제품 이름이 아니라 역할을 가리킨다. Knowledge Base와 Memory가 같은 vector database에 들어갈 수도 있다. Conversation History가 Agent State의 `messages` 항목에 포함되기도 한다. 중요한 질문은 “어디에 저장했는가?”보다 “누가, 언제, 무슨 목적으로 다시 읽는가?”다.

### 같은 문장도 생긴 경로에 따라 역할이 달라진다

`HTTP 429가 발생하면 2초부터 재시도한다`라는 문장을 보자.

- 제품 API 문서에서 가져왔다면 Knowledge Base다.
- 사용자가 대화 중에 말했다면 그 원문은 Conversation History다.
- Agent가 실패를 겪고 다음 실행 규칙으로 골라 저장했다면 procedural memory다.
- 지금 재시도 2회째라는 값은 Agent State다.

문장만 보고 분류하면 이 차이를 놓친다. `source`, `user_id`, `thread_id`, `created_at`, `memory_type` 같은 metadata를 함께 남기는 까닭이다.

## 2. Model 안의 기억과 밖의 기록을 구분한다

LLM에는 이미 많은 정보가 들어 있다. 그렇다고 model이 어제 나눈 대화를 자동으로 기억하는 것은 아니다.

| 층 | 위치 | 언제 바뀌는가 | 예시 |
| --- | --- | --- | --- |
| Parametric memory | Model weight | Pretraining이나 fine-tuning으로 weight를 갱신할 때 | 언어 문법, 학습 자료의 일반 지식 |
| Working memory | 현재 context와 실행 state | Turn과 step이 진행될 때 | 현재 질문, 읽어온 문서, 계산 중간값 |
| Long-term memory | 외부 store | Memory write·update·delete가 일어날 때 | 사용자 선호, 과거 작업, 재사용 규칙 |

대화 한 번을 마쳤다고 model weight가 바뀌지는 않는다. Memory database에 사용자 선호를 저장해도 fine-tuning을 한 것이 아니다. 다음 요청 때 그 record를 검색해 prompt에 넣어야 model이 읽을 수 있다.

KV cache도 long-term memory와 다르다. KV cache는 이미 읽은 token 계산을 같은 추론 안에서 재사용하는 임시 tensor다. Server를 다시 시작하거나 request가 끝나면 보통 사라진다. 사용자 경험을 골라 저장하고 여러 session에서 꺼내는 기능을 대신하지 않는다.

!!! warning "저장되어 있다는 사실만으로 model이 아는 것은 아니다"

    Database에 1만 개의 memory가 있어도 이번 prompt에 들어오지 않으면 LLM은 읽지 못한다. 반대로 관련 없는 memory를 너무 많이 넣으면 중요한 지시를 찾기 어려워진다. 저장과 검색, context 선택을 한 묶음으로 설계한다.

## 3. Long-term memory에는 세 종류가 있다

CoALA는 language agent의 memory를 working, episodic, semantic, procedural memory로 나눈다. Procedural memory에는 LLM weight와 agent code가 포함된다. Semantic memory는 사실을, episodic memory는 지나간 경험을 담는다.[^1]

| 종류 | 쉬운 설명 | Agent 예시 | 다시 쓰는 순간 |
| --- | --- | --- | --- |
| Semantic memory | 알고 있는 사실 | “지수는 짧은 한국어 답을 선호한다” | 답변 형식을 정할 때 |
| Episodic memory | 겪었던 사건 | “지난 동기화에서 HTTP 429가 두 번 났다” | 비슷한 실패를 진단할 때 |
| Procedural memory | 행동 규칙과 방법 | “429면 2초부터 지수 backoff로 재시도한다” | Tool 실행 순서를 정할 때 |

세 종류가 완전히 떨어져 있지는 않다. 한 번의 실패라는 episodic memory에서 재시도 규칙이라는 procedural memory를 만들 수 있다. “이 사용자는 표보다 짧은 목록을 좋아한다”라는 semantic memory도 여러 대화 경험을 요약해 얻는다.

Procedural memory의 범위를 어디까지 볼지도 framework마다 다르다. CoALA 그림에서는 LLM과 agent code를 함께 procedural memory에 둔다.[^1] 실제 서비스에서는 수정 가능한 prompt·workflow code와 학습으로만 바뀌는 model weight를 따로 관리하는 편이 안전하다.

## 4. Agent는 LLM보다 큰 구조다

![CoALA의 procedural, semantic, episodic, working memory와 반복되는 decision cycle](/notes/tutorial/llm_lecture/images/w23_coala_architecture.png)

*그림 1. CoALA의 memory module, decision procedure, working memory, 외부 환경과 decision cycle. 출처: Sumers et al. (2024), Figure 4에서 발췌.[^1]*

CoALA에서 LLM은 agent 전체가 아니라 한 부품이다. Agent는 long-term memory를 읽는 retrieval, working memory에서 생각을 전개하는 reasoning, long-term memory를 쓰는 learning, 외부 환경에 행동하는 grounding을 반복한다.[^1]

그림 오른쪽의 cycle은 다음처럼 읽을 수 있다.

1. **Observation**: 사용자 message나 tool 결과를 받는다.
2. **Planning**: 필요한 memory를 찾고 행동 후보를 만든다.
3. **Evaluation**: 후보가 목표와 규칙에 맞는지 살핀다.
4. **Selection**: 실행할 행동 하나를 고른다.
5. **Execution**: 답하거나 tool을 호출하고 새 observation을 받는다.

Memory는 이 cycle 앞에 붙는 검색 기능 하나가 아니다. 검색한 경험으로 계획한 뒤 실행 결과를 보고 새 memory를 쓸지 결정한다. 이 과정이 내부에서 계속 순환한다.

## 5. Memory의 생명주기는 쓰기 전부터 시작한다

관찰한 모든 문장을 그대로 장기 보관하면 중복과 오래된 정보가 쌓인다. 다음 생명주기를 차례로 둔다.

```text
관찰
  → 저장 판단
  → 구조화와 index
  → 검색과 rerank
  → context 선택
  → 행동
  → 갱신 또는 삭제
```

### 관찰과 저장 판단

`고마워` 같은 인사는 대화 기록에는 남아도 long-term memory로 보관할 이유가 적다. 반면 `앞으로 답은 한국어 두 줄로 해줘`는 다음 session에도 필요한 사용자 선호다. Write policy는 명시적인 선호, 반복된 습관, 재현 가능한 실패, 다시 쓸 규칙을 후보로 삼는다.

### 구조화와 index

Memory 본문만 저장하지 않는다. 최소한 소유자, 출처 message, 발생 시간, 종류, 활성 상태를 붙인다.

```json
{
  "memory_id": "m4",
  "user_id": "jisu",
  "type": "semantic",
  "text": "한국어로 두 개의 bullet을 사용한다.",
  "source_message_ids": ["h9"],
  "created_at": "2026-08-04T22:00:00+09:00",
  "active": true,
  "supersedes": "m1"
}
```

### 검색과 context 선택

질문과 가까운 record를 찾은 뒤에도 모두 넣지는 않는다. 사용자 범위, 권한, 최신 상태, 출처 신뢰도, token 예산을 확인한다. `active=false`인 예전 선호는 감사 기록에는 남겨도 답변 context에서는 뺀다.

### 갱신과 삭제

사용자가 “세 줄 대신 두 줄로 답해줘”라고 바꾸면 예전 memory와 새 memory를 동시에 활성화하지 않는다. 새 record가 무엇을 대체했는지 연결하고 최신 항목만 사용한다. 개인정보 삭제 요청이나 보존 기간 만료가 오면 원문 History, 파생 summary, embedding, cache까지 함께 지울 경로가 필요하다.

## 6. Conversation History는 녹취록에 가깝다

Conversation History는 순서가 중요하다. `user`의 질문 뒤에 `assistant` 답변과 `tool` 결과가 어떻게 이어졌는지 보여준다.

```python
history = [
    {"role": "user", "content": "Atlas 동기화를 시작해줘."},
    {"role": "assistant", "content": "동기화를 시작할게."},
    {"role": "tool", "content": "HTTP 429"},
]
```

OpenAI Agents SDK의 Session은 실행 전에 저장된 history를 가져와 새 입력 앞에 붙인다. 실행이 끝나면 새 user·assistant·tool item을 저장한다.[^2] LangGraph도 thread 안의 message history를 short-term memory로 다루며 checkpointer로 state와 함께 보존한다.[^3]

이 기능은 여러 turn을 이어갈 때 편리하다. 다만 경험을 골라 구조화한 long-term memory와 같지는 않다. 오래된 대화를 전부 넣으면 context가 길어진다. 사용자가 바꾼 선호의 이전 문장도 함께 들어온다. Session history를 자르거나 요약할 수는 있어도 어떤 경험을 여러 thread에서 재사용할지는 별도의 policy가 정해야 한다.

MemGPT는 제한된 context를 main memory처럼 보고 외부 저장소와 정보를 옮기는 virtual context management를 제안했다.[^4] 운영체제에서 RAM과 disk 사이를 오가는 모습에 빗댄 설계다. 실제 context가 무한해지지는 않는다. 그때 필요한 일부 정보만 제한된 창 안으로 가져온다.

## 7. Agent State는 지금의 작업판이다

Agent State에는 현재 목표, 중간 결과, 다음 node, 승인 대기 여부가 들어간다. 이 값은 오래 기억할 교훈이 아니라 workflow를 이어가는 데 필요한 snapshot이다.

```json
{
  "thread_id": "thread-23",
  "goal": "Atlas 보고서 게시",
  "current_step": "await_approval",
  "next_node": "publish",
  "artifacts": ["reports/atlas.md"],
  "approved": false
}
```

이 state를 checkpoint로 저장하면 process가 중단돼도 `publish` 앞에서 다시 시작한다. LangGraph의 checkpointer는 실행 step마다 graph state snapshot을 thread별로 저장한다. 이를 이용해 사람의 승인 대기, 오류 복구, 이전 state 재생을 구현한다.[^5]

Checkpoint에 모든 외부 자원을 복사할 필요는 없다. 큰 문서는 object storage 경로와 checksum만, database transaction은 안전하게 재시도할 idempotency key를 남긴다. Tool 호출이 이미 끝났는지도 기록해야 재시작하면서 결제가 두 번 되는 사고를 막는다.

!!! note "Checkpoint와 Memory는 서로 대신하지 않는다"

    `next_node=publish`는 workflow가 끝나면 필요 없는 state다. “게시 전에 사용자 승인을 받는다”는 규칙은 다음 작업에도 쓸 procedural memory 후보다. 현재 위치와 재사용할 규칙을 나눠 보관한다.

## 8. Knowledge Base는 외부 자료의 출처를 보존한다

Knowledge Base에는 제품 설명서, 사내 규정, 논문처럼 agent 경험과 상관없이 참고할 자료를 넣는다. 원문 문서 ID, version, 문단 위치, 갱신 시간을 남겨 답의 근거를 다시 확인하도록 한다.

| 항목 | Knowledge Base | Memory |
| --- | --- | --- |
| 만들어지는 경로 | 문서 수집과 parsing | 대화와 실행 경험에서 선택 |
| 대표 소유 범위 | 조직, 제품, project | 사용자, agent, team |
| 갱신 계기 | 원본 문서의 새 version | 새 선호, 새 경험, 교정 요청 |
| 중요한 metadata | 문서 URL, version, page, chunk | Source message, 시간, type, confidence |
| 삭제 기준 | 문서 폐기와 보존 정책 | 사용자 삭제, TTL, 충돌, 낮은 가치 |

두 저장소가 같은 embedding index를 사용해도 namespace와 접근 권한을 분리한다. 다른 사용자의 개인 memory가 검색되거나, 오래된 사내 규정이 최신 Knowledge Base보다 앞에 나오면 안 된다.

## 9. 네 저장소를 작은 코드로 나눠본다

실습에는 Conversation History 12개, Agent State 1개, Knowledge Base 3개, Memory 4개를 넣었다.[^6] Memory 네 개 가운데 예전 “세 개 bullet” 선호는 비활성화하고 새 “두 개 bullet” 선호를 활성화했다.

여섯 질문은 현재 선호, 과거 실패, 재시도 규칙, 환불 정책, 마지막 user message, 재개할 workflow 위치를 묻는다. 다음 네 context 구성 방법을 비교했다.

| 방법 | Context에 넣은 내용 | 기대되는 문제 |
| --- | --- | --- |
| Recent history | 마지막 message 4개 | 오래전 경험과 Knowledge Base를 놓침 |
| Full history | 한 thread의 message 12개 | 외부 지식과 state가 없고 옛 선호가 충돌 |
| Everything raw | 네 저장소의 20개 record 전체 | 길고 낡은 memory까지 섞임 |
| Routed active | 질문에 맞는 현재 record 1개 | Route나 검색이 틀리면 근거를 놓침 |

![네 저장소의 record 수와 context 선택 방법별 근거 포함 결과](/notes/tutorial/llm_lecture/images/w23_agent_memory_map.png)

*그림 2. 네 저장소를 나눈 deterministic smoke test. Answerable은 필요한 근거가 있고 비활성화된 충돌 record가 없다는 뜻이며 LLM의 실제 정답률이 아니다.[^6]*

| Context 방법 | 근거를 온전히 담은 질문 | 질문당 평균 context 문자 수 |
| --- | ---: | ---: |
| Recent history | 2/6 | 168 |
| Full history | 3/6 | 598 |
| Everything raw | 5/6 | 1,131 |
| Routed active | 6/6 | 68 |

Everything raw에는 여섯 질문의 자료가 모두 있었지만 옛 선호와 새 선호도 함께 들어갔다. 실험 규칙은 이런 충돌을 실패로 처리해 5/6이 됐다. Routed active는 질문마다 한 record만 골라 여섯 근거를 모두 보존했고 평균 context는 68자였다.[^6]

이 결과로 특정 retrieval 알고리즘이 더 좋다고 말할 수는 없다. Query route와 정답을 사람이 미리 정한 합성 자료이며 LLM도 호출하지 않았다. 확인한 사실은 저장소를 역할별로 나누고 비활성 record를 제외하는 코드가 의도대로 동작했다는 점이다.

같은 script는 Agent State를 JSON checkpoint로 쓴 뒤 다시 읽어 `next_node=publish`가 남아 있는지도 검사했다. 저장과 복구가 같은 process 안에서 이뤄진 간단한 round trip이므로 database 장애 복구 시험은 아니다.[^6]

## 10. 최소 구현은 record와 policy로 시작한다

처음부터 거대한 memory platform을 만들 필요는 없다. 네 자료 구조와 두 policy로 시작한다.

```python
stores = {
    "history": append_only_messages,
    "state": latest_thread_checkpoint,
    "knowledge_base": versioned_documents,
    "memory": selected_experience_records,
}

write_policy = decide_what_to_remember
read_policy = select_records_for_this_request
```

Write policy는 저장할지, 어느 type인지, 기존 memory를 대체하는지 정한다. Read policy는 현재 user와 thread의 권한을 확인하고 질문에 맞는 최신 record만 context에 넣는다. 두 policy의 판단 결과도 log로 남겨야 잘못 저장한 이유와 잘못 검색한 이유를 나눠 찾을 수 있다.

초기 운영 점검에는 다음 항목을 넣는다.

- User와 조직별 namespace를 분리했는가?
- Memory마다 source와 생성 시간이 남는가?
- 새 사실이 예전 사실을 대체할 때 활성 상태가 하나만 남는가?
- Agent가 만든 추측을 사용자 사실처럼 저장하지 않는가?
- Prompt injection이 들어간 tool output을 procedural memory로 승격하지 않는가?
- 사용자가 자신의 History와 Memory를 조회하고 지울 수 있는가?
- Checkpoint를 재생할 때 외부 action이 중복되지 않는가?

## 11. 자주 생기는 오해

### “대화를 database에 저장했으니 memory가 있다”

저장된 것은 Conversation History다. 여러 session에서 재사용할 사실과 경험을 고르는 과정은 아직 없다.

### “Context window가 길면 external memory는 필요 없다”

긴 history가 들어가도 오래된 정보와 새 정보의 충돌, 개인정보 삭제, 사용자 간 분리, 출처 갱신 문제는 남는다. MemGPT도 긴 정보를 전부 한 번에 넣는 대신 필요한 내용을 context로 옮기는 관리 문제를 다룬다.[^4]

### “Vector database가 곧 memory다”

Vector database는 가까운 record를 찾는 저장 기술이다. 무엇을 쓸지, 누가 읽을지, 충돌을 어떻게 고칠지, 언제 지울지는 application policy가 맡는다.

### “Agent State를 오래 보관하면 long-term memory가 된다”

오래 저장했다는 이유만으로 역할이 바뀌지는 않는다. State는 실행을 재개하는 snapshot이다. Long-term memory는 다음 경험에 재사용할 사실과 규칙이다.

## 확인 문제

1. Conversation History와 Memory를 모두 database에 저장하더라도 역할이 다른 이유는 무엇인가?
2. 같은 `HTTP 429 재시도` 문장이 Knowledge Base, History, Memory, State에 들어가는 예를 각각 만들어보자.
3. Model weight에 든 parametric memory가 어제 사용자와 나눈 대화를 자동으로 기억하지 못하는 이유는 무엇인가?
4. KV cache를 long-term memory로 사용할 수 없는 까닭은 무엇인가?
5. Semantic, episodic, procedural memory에 들어갈 여행 agent 사례를 하나씩 적어보자.
6. CoALA에서 retrieval, reasoning, learning, grounding은 각각 무엇을 읽거나 바꾸는가?
7. 사용자가 답변 선호를 세 줄에서 두 줄로 바꾸면 memory record를 어떻게 갱신해야 하는가?
8. Full history에 옛 선호와 새 선호가 함께 들어갈 때 어떤 문제가 생기는가?
9. Agent State의 `next_node`와 procedural memory의 승인 규칙은 어떻게 다른가?
10. Knowledge Base와 Memory가 같은 vector database를 쓸 때도 namespace를 나눠야 하는 이유는 무엇인가?
11. 이번 실험의 6/6을 LLM 정답률이라고 부를 수 없는 이유는 무엇인가?
12. Checkpoint에서 외부 tool 호출의 idempotency 정보를 남겨야 하는 까닭은 무엇인가?

## 완료 체크

- [x] Memory, Knowledge Base, Conversation History, Agent State를 같은 사례로 구분했다.
- [x] Parametric, working, long-term memory의 위치와 갱신 시점을 확인했다.
- [x] Semantic, episodic, procedural memory의 예시를 만들었다.
- [x] Agent Memory의 관찰, 저장, 검색, 행동, 갱신 생명주기를 그렸다.
- [x] Conversation History와 선택된 long-term memory가 같지 않은 이유를 설명했다.
- [x] JSON checkpoint를 저장하고 같은 thread state를 다시 읽었다.
- [x] 네 context 구성 방법의 근거 포함 여부와 길이를 비교했다.
- [x] 오래된 선호를 비활성화하고 현재 record만 선택했다.
- [x] User namespace, 출처, 삭제, prompt injection, action 중복을 점검했다.

---

[^1]: Sumers, T. R. et al. (2024). [Cognitive Architectures for Language Agents](https://openreview.net/forum?id=1i6ZCvflQJ). TMLR 논문의 Figure 4와 §4.1의 working·procedural·semantic·episodic memory, retrieval·reasoning·learning·grounding 구분을 참고했다.
[^2]: OpenAI. [Agents SDK: Sessions](https://openai.github.io/openai-agents-python/sessions/). 실행 전 history 조회, 실행 후 새 item 저장, session별 대화 유지 방식을 참고했다. 확인일: 2026-08-04.
[^3]: LangChain. [Memory overview](https://docs.langchain.com/oss/python/concepts/memory). Thread 범위의 short-term memory와 namespace 범위의 long-term memory, semantic·episodic·procedural 구분을 참고했다. 확인일: 2026-08-04.
[^4]: Packer, C. et al. (2023). [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560). 제한된 context와 외부 저장소 사이에서 정보를 옮기는 virtual context management 개념을 참고했다.
[^5]: LangChain. [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence). Thread별 graph state checkpoint, human-in-the-loop, replay, fault tolerance를 참고했다. 확인일: 2026-08-04.
[^6]: 직접 실행한 `llm_lecture/week23_agent_memory_lifecycle.py`의 결과다. 20개 합성 record와 6개 query로 저장 역할, 충돌 제외, context 구성, JSON checkpoint round trip을 확인했다. LLM은 호출하지 않았으며 script, CSV, JSON은 Git에서 제외하고 최종 plot만 공개했다. 실행일: 2026-08-04.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
원본/윤문본: 14,154자 / 14,141자, 글자 수 기준 변경률 0.09%
카테고리별 탐지/수정: C-11 연결어미 뒤 쉼표 9→0, A-10 가능 표현 5→0, A-18 긴 복합문 4→0, D-1 관용구 0→0
정량 점검: humanize-korean metrics v2.0 risk score 3→1, risk band low 유지
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 논문 개념, framework 동작, 실험 수치, code와 reference를 보존함
주요 변경 1: “기록과 경험을 구분하고, 새 정보가” → “기록과 경험을 구분한다. 새 정보가”
주요 변경 2: “새 입력 앞에 붙이고, 실행이 끝나면” → “새 입력 앞에 붙인다. 실행이 끝나면”
주요 변경 3: “실제 context가 무한해지는 것은 아니며” → “실제 context가 무한해지지는 않는다”
주요 변경 4: “다시 시작할 수 있다” → “다시 시작한다”
주요 변경 5: “State는 실행을 재개하는 snapshot이고” → “State는 실행을 재개하는 snapshot이다”
-->
