---
title: "24주차. 기억을 저장하고 찾고 고치기"
description: "Agent의 write policy, raw·summary·structured memory 표현, dense·keyword·recency·importance 검색, rerank, 충돌 갱신과 삭제를 구현한다."
tags:
  - Agent Memory
  - retrieval
  - embedding
  - reranking
  - memory update
---

<p class="ai-assisted-disclosure">이 글은 AI의 도움을 받아 작성되었습니다.</p>

[← 26주 커리큘럼](/notes/tutorial/llm_lecture/curriculum/)

23주차에는 Memory, Knowledge Base, Conversation History, Agent State를 역할별로 나눴다. 이번에는 Conversation History에서 무엇을 오래 남길지 고른다. 필요한 memory를 찾아 context에 넣는 pipeline도 만든다. 사용자가 선호를 바꾸거나 삭제를 요청했을 때 예전 record가 다시 나타나지 않도록 갱신 규칙도 정한다.

## 이번 주에 배울 것

- 대화에서 장기 보관할 내용을 고르는 write policy
- Raw message, summary, structured JSON의 장단점
- User namespace, source, timestamp, type, confidence를 포함한 schema
- Keyword score와 dense embedding cosine similarity의 차이
- Recency, importance, confidence를 섞는 hybrid score
- Top-k retrieval 뒤 metadata filter와 rerank를 적용하는 순서
- 중복 memory를 합치고 충돌한 선호를 새 version으로 바꾸는 방법
- Soft delete, tombstone, hard delete의 차이
- Recall@1과 Recall@3로 검색 결과를 평가하는 방법

선수 지식은 18주차의 embedding과 cosine similarity, 23주차의 memory 종류와 생명주기다. 실습에서는 15개 영어 memory와 정답을 아는 query 6개를 사용한다. 영어를 쓴 까닭은 실험의 dense encoder가 영어 text에 더 알맞기 때문이다.

!!! note "Memory store에는 사서가 필요하다"

    책을 창고에 넣기만 하면 도서관이 되지 않는다. 제목과 분류표를 만들고 잘못된 판본을 치운다. 질문에 맞는 책도 골라야 한다. Agent Memory에서도 write, index, retrieve, rerank, update, delete가 함께 움직인다.

## 1. 전체 pipeline부터 그린다

Memory pipeline은 한 방향으로 끝나지 않는다. 새 대화가 들어오면 저장 후보를 만든 뒤 기존 record와 충돌하는지 확인한다. 검색 결과가 나쁘면 query와 score를 고친다. 사용자가 정정하면 과거 record의 상태도 바꾼다.

```text
Conversation History
  → extract candidate
  → validate and deduplicate
  → write or update
  → index
  → retrieve candidates
  → filter and rerank
  → select for context
  → answer or act
  → feedback, update, delete
```

한 단계가 틀리면 뒤 단계도 영향을 받는다. “나는 땅콩을 좋아하지 않아”를 “땅콩 알레르기가 있다”로 과장해서 저장하면 retrieval은 정확히 그 record를 찾아도 최종 답은 틀린다. 검색 오류와 저장 오류를 따로 기록해야 하는 이유다.

## 2. Write policy가 저장할 내용을 고른다

모든 message를 long-term memory로 복사하지 않는다. 다음 session에서 다시 쓸 가치가 있는지, 사용자가 직접 말했는지, 얼마 동안 유효한지를 본다.

| 관찰한 문장 | 저장 판단 | 이유 |
| --- | --- | --- |
| “앞으로 한국어로 짧게 답해줘” | Semantic memory 후보 | 다음 대화에도 적용할 명시적 선호 |
| “고마워” | 보통 저장하지 않음 | 다음 task에서 쓸 정보가 적음 |
| Tool이 HTTP 429로 두 번 실패 | Episodic memory 후보 | 비슷한 장애를 진단할 때 유용 |
| “429면 2초부터 지수 backoff” | Procedural memory 후보 | 재사용할 실행 규칙 |
| “아마 사용자는 밤 비행을 좋아할 것” | 바로 저장하지 않음 | Agent의 추측이며 사용자 확인이 없음 |
| 결제 승인 token | Long-term memory에 저장 금지 | 민감하고 수명이 짧은 실행 정보 |

Write policy는 rule과 LLM을 섞어 만든다. `remember this`, `from now on`, `I prefer`처럼 명시적인 표현은 rule로 잡기 쉽다. 여러 대화에서 반복된 습관은 LLM이 더 잘 요약한다. 이때 source message와 confidence를 남기며 낮은 confidence는 사용자 확인을 거친다.

### Hot path와 background write

응답하기 전에 바로 memory를 쓰는 방식을 hot path라고 부른다. 새 선호를 같은 turn부터 적용하기 쉽지만 응답 latency가 늘어난다. 잘못 뽑은 record가 즉시 사용될 위험도 있다. Background write는 대화가 끝난 뒤 후보를 묶어 처리한다. 응답은 빨라지지만 다음 요청이 너무 일찍 오면 새 memory가 아직 없기도 한다. LangGraph 문서도 long-term memory 갱신을 실행 경로 안에서 할지 background task로 할지 나누어 설명한다.[^1]

!!! warning "Agent의 추측을 사용자 사실로 바꾸지 않는다"

    “사용자가 매운 음식을 주문했다”는 한 번의 사건이다. 여기서 “사용자는 항상 매운 음식을 좋아한다”라고 저장하면 episodic observation을 semantic preference로 과장한 셈이다. 반복 관찰이나 명시적인 확인 없이 범위를 넓히지 않는다.

## 3. 같은 내용을 세 가지 모양으로 저장한다

사용자가 `아침에는 디카페인 커피를 마셔`라고 말했다고 하자. 한 memory를 raw, summary, structured 형태로 나타낼 수 있다.

### Raw message

```text
User: You asked what I drink. I usually start the day with
decaffeinated coffee before work. Assistant: Got it.
```

원래 말투와 주변 맥락이 남아 있어 감사를 하거나 잘못된 추출을 고칠 때 유리하다. 인사와 assistant 답변까지 섞여 길다. 검색과 prompt에 그대로 쓰면 noise도 많다.

### Summary

```text
Jisu drinks decaffeinated coffee every morning.
```

핵심이 짧아 embedding과 context 비용을 줄인다. 요약 과정에서 조건을 빠뜨리거나 없던 내용을 보탤 수 있으므로 raw source를 버리면 안 된다.

### Structured JSON

```json
{
  "memory_id": "m01",
  "user_id": "jisu",
  "type": "semantic",
  "key": "morning_drink",
  "value": "decaffeinated coffee",
  "tags": ["morning", "drink", "coffee", "decaf"],
  "source": "thread-1:message-4",
  "confidence": 0.98,
  "active": true
}
```

정확한 filter, conflict detection, update에 알맞다. Schema에 없는 미묘한 조건은 잃기 쉽다. JSON 전체를 한 문장처럼 embedding하면 field 이름과 괄호가 dense 검색을 방해하기도 한다.

실무에서는 하나만 고집하지 않고 세 표현을 연결한다. Raw message는 source pointer로 보존하고 summary는 embedding용 text, structured field는 filter와 update에 사용한다.

## 4. Metadata가 없는 memory는 고치기 어렵다

Memory 한 건에는 적어도 다음 항목을 둔다.

| Field | 필요한 이유 |
| --- | --- |
| `memory_id` | Update와 delete 대상 식별 |
| `user_id`, `namespace` | 다른 사용자와 조직의 record 격리 |
| `type` | Semantic, episodic, procedural filter |
| `text`, `summary` | 사람이 읽는 본문과 검색 text |
| `source` | 원문 message나 tool trace로 돌아가기 |
| `created_at`, `last_accessed_at` | Recency와 보존 기간 계산 |
| `importance` | 안전, 선호, 사소한 사건의 우선순위 구분 |
| `confidence` | 명시적 사실과 추론한 후보 구분 |
| `active`, `deleted_at` | 낡거나 삭제된 record 제외 |
| `supersedes`, `superseded_by` | 새 version과 이전 version 연결 |

A-MEM은 새 memory를 contextual description, keyword, tag가 담긴 note로 만들고 기존 memory와 관련된 link를 찾는다. 새 note가 들어오면서 예전 note의 맥락과 속성이 바뀌는 memory evolution도 제안한다.[^2] Mem0도 대화에서 중요한 정보를 추출하고 consolidation과 retrieval을 거치는 architecture를 제시한다.[^3]

이런 구조가 있어도 LLM이 만든 metadata를 무조건 믿지는 않는다. `user_id`와 접근 권한은 server가 넣는다. `source`는 실제 event ID에서 가져온다. Model이 자유롭게 보안 경계를 작성하게 두면 다른 사용자의 namespace로 잘못 기록한다.

## 5. Retrieval은 후보를 좁히는 일이다

검색은 보통 hard filter와 score 계산을 함께 쓴다.

1. 요청한 사용자의 namespace만 남긴다.
2. `active=true`이고 삭제되지 않은 record만 남긴다.
3. Keyword와 dense embedding으로 후보를 찾는다.
4. Recency, importance, confidence를 합쳐 rerank한다.
5. 필요하면 memory type과 time range로 다시 거른다.
6. Token 예산 안에서 top-k를 context에 넣는다.

LangGraph의 long-term store도 namespace에 item을 넣고 semantic query로 검색하는 구조를 제공한다.[^1] Production에서는 database filter가 먼저 작동하도록 `user_id`, `active`, `type`에 index를 둔다. 다른 사용자의 memory를 vector search한 뒤 application code에서 버리는 방식은 정보 노출 위험이 있다.

### Keyword score

Keyword 검색은 query와 record가 같은 단어를 쓸 때 강하다. `Atlas`, `429`, `reports/atlas.md` 같은 고유명사와 오류 code, file path를 정확히 찾는다. 표현이 `rate limit`과 `too many requests`처럼 달라지면 놓칠 수 있다.

### Dense similarity

Embedding model은 query와 memory를 vector로 바꾸고 cosine similarity를 계산한다. 단어가 달라도 뜻이 가까운 문장을 찾는 데 유리하다. 반대로 “실패했다”와 “실패를 해결하는 규칙”처럼 같은 주제를 가진 다른 type을 혼동하기도 한다.

### Recency와 importance

최근 record에 높은 recency를 주면 바뀐 선호를 앞에 놓기 쉽다. 그러나 오래된 알레르기처럼 중요한 정보가 밀리면 위험하다. Importance는 안전과 재사용 가치를 보완한다. 두 score 모두 relevance를 대신하지 않고 rerank의 작은 신호로 쓴다.

## 6. Generative Agents의 세 score

![Memory stream에서 recency, importance, relevance로 현재 질문에 필요한 기록을 찾는 과정](/notes/tutorial/llm_lecture/images/w24_generative_agents_retrieval.png)

*그림 1. 많은 observation 중 현재 상황과 관련된 memory를 recency, importance, relevance로 골라 LLM에 넣는 과정. 출처: Park et al. (2023), Figure 6에서 발췌.[^4]*

Generative Agents는 memory stream의 record마다 생성 시간과 최근 접근 시간을 남긴다. Retrieval에서는 recency, importance, relevance를 0부터 1 사이로 정규화하고 가중합한다. 논문 구현의 세 가중치는 모두 1이다.[^4]

간단히 $score = \alpha_r R_{recency} + \alpha_i R_{importance} + \alpha_s R_{relevance}$로 쓸 수 있다. Relevance는 query와 memory embedding의 cosine similarity, recency는 마지막 접근 뒤 시간이 지날수록 작아지는 값이다. Importance는 memory를 만들 때 LLM이 1부터 10 사이로 평가했다.[^4]

이 식은 정답이 정해진 법칙이 아니다. 금융 agent와 게임 character가 중요하게 여길 사건은 다르다. Score별 분포와 top-k 결과를 validation query에서 살펴보고 weight를 정한다.

## 7. 이번 실습의 hybrid score

실습은 `openai/clip-vit-base-patch32`의 text encoder를 dense embedder로 재사용했다. Revision은 `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`로 고정했다.[^5] CLIP은 image-text contrastive model이며 production sentence retrieval 전용 model이 아니다. 여기서는 18주차에 내려받은 checkpoint로 pipeline을 재현하기 위한 교육용 선택이다.

Active 상태인 지수의 memory 11개만 검색 후보로 넣었다. Mina의 record 2개, 영어 답변을 선호한다는 예전 record 1개, 삭제된 Kyoto 여행 계획 1개는 먼저 제외했다.[^5]

Dense similarity를 후보 안에서 0부터 1로 min-max 정규화한 뒤 다음 score를 사용했다.

$S = 0.30S_{semantic} + 0.45S_{keyword} + 0.08S_{recency} + 0.12S_{importance} + 0.05S_{confidence}$

Recency는 $S_{recency} = \exp(-age\_days / 90)$로 계산했다. Weight와 90일 기준은 이 합성 자료를 설명하려고 정한 값이다. 서비스 기본값으로 복사하면 안 된다.

```python
score = (
    0.30 * semantic
    + 0.45 * keyword
    + 0.08 * recency
    + 0.12 * importance
    + 0.05 * confidence
)
```

Query 여섯 개는 아침 음료, Atlas 재시도 규칙, 비행기 좌석, 알레르기 재료, 현재 답변 형식, report file 위치를 물었다. 각 query마다 목표 memory ID를 미리 정했다.[^5]

## 8. 저장 표현과 검색 방법을 비교한다

![Raw, summary, structured memory와 hybrid retrieval의 Recall 및 score 구성](/notes/tutorial/llm_lecture/images/w24_memory_retrieval_results.png)

*그림 2. 정답을 아는 query 6개에서 표현과 검색 방법을 바꾼 결과. 사람이 만든 작은 영어 자료이며 일반 memory benchmark가 아니다.[^5]*

| 방법 | Recall@1 | Recall@3 | Memory당 평균 문자 수 |
| --- | ---: | ---: | ---: |
| Raw dense | 4/6 | 5/6 | 92 |
| Summary dense | 4/6 | 6/6 | 55 |
| Structured dense | 2/6 | 5/6 | 146 |
| Hybrid structured | 5/6 | 6/6 | 146 |
| Hybrid + type filter | 6/6 | 6/6 | 146 |

Summary는 평균 55자로 가장 짧았고 목표 6개가 모두 top 3 안에 들었다. Structured JSON 전체를 그대로 dense embedding한 조건은 Recall@1이 2/6으로 가장 낮았다. 구조가 나빠서가 아니라 JSON field와 tag까지 한 text로 넣은 방식이 이 encoder와 잘 맞지 않았다는 뜻이다.

Hybrid는 keyword, recency, importance, confidence를 더해 Recall@1을 5/6으로 높였다. 남은 실패는 `Atlas가 HTTP 429로 실패했다`는 episodic memory가 `429에서 지수 backoff를 쓴다`는 procedural memory보다 앞선 경우다. 둘 다 Atlas와 rate limit 단어를 담아 topic similarity만으로 구분하기 어려웠다.

Query가 재시도 규칙을 묻는다고 분류한 뒤 procedural type만 허용하자 목표 memory가 1위가 되어 6/6이 됐다. 이 결과에는 사람이 미리 적은 정답 type이 쓰였다. 실제 router가 type을 틀리게 예측하면 올바른 memory를 후보에서 없앨 수 있다.

!!! warning "6/6은 model의 기억 능력 점수가 아니다"

    Memory 15개와 query 6개를 사람이 만들고 score weight도 같은 자료에서 정했다. 별도 validation과 test split이 없다. CLIP text encoder도 sentence retrieval 전용이 아니다. 이 실험은 score와 filter가 code에서 어떻게 이어지는지 확인한 smoke test다.

## 9. Top-k 뒤에는 원문을 다시 본다

Vector 검색 결과만 prompt에 넣기 전에 source를 확인한다. Summary가 `땅콩을 피한다`라고 되어 있어도 원문이 단순한 기호인지 심한 알레르기인지에 따라 행동이 달라진다.

Reranker가 점검할 항목은 다음과 같다.

- Query가 요구한 memory type과 맞는가?
- 동일한 `key`에 여러 active version이 있는가?
- Source가 사용자 발화인지 agent 추측인지 구분되는가?
- 시간 조건이 맞는가? “지난 여행”과 “다음 여행”을 섞지 않았는가?
- 서로 모순되는 record가 있으면 답하기 전에 확인해야 하는가?
- Top-k를 넣었을 때 context token 예산을 넘지 않는가?

중요한 action이라면 memory text와 함께 `source`, `created_at`, `confidence`를 model에 전달한다. 결제, 건강, 법률처럼 실수가 큰 분야에서는 memory만 믿고 실행하지 않고 현재 사용자에게 다시 확인한다.

## 10. 충돌을 발견하면 version을 연결한다

실습에는 다음 두 record가 있었다.[^5]

```text
m09 inactive: Jisu prefers concise replies in English.
m13 active:   Jisu currently prefers concise replies in Korean.
```

새 선호를 저장하면서 `m13.supersedes = m09`, `m09.superseded_by = m13`으로 연결했다. 이전 record를 물리적으로 바로 지우지 않았으므로 언제 어떤 source로 바뀌었는지 감사 기록에서 확인한다. Retrieval에서는 `active=true`인 m13만 사용한다.

충돌 처리 방법은 상황에 따라 다르다.

| 상황 | 처리 |
| --- | --- |
| 사용자가 명시적으로 선호를 정정 | 새 version 저장, 이전 version 비활성화 |
| 같은 사실을 같은 source가 반복 | 중복을 합치고 source 목록 추가 |
| 두 tool이 서로 다른 값을 반환 | 둘 다 보존하고 timestamp와 신뢰도를 비교 |
| Agent 추론과 사용자 발화가 충돌 | 사용자 발화를 우선하고 추론 record 비활성화 |
| 신뢰도가 비슷한 두 사용자 발화가 충돌 | 확인 질문 후 갱신 |
| 시간에 따라 변하는 상태 | 유효 기간을 두고 최신 record 선택 |

단순 `upsert(key, value)`로 덮어쓰면 이력을 잃는다. Append-only event와 현재 materialized view를 함께 두면 감사 기록과 빠른 조회를 모두 얻는다.

## 11. 삭제는 검색에서 숨기는 것보다 넓다

실습의 Kyoto 여행 계획 `m11`은 `active=false`, `deleted=true`로 표시해 retrieval 후보에서 뺐다.[^5] 이것은 soft delete 예시다. 실제 삭제 요청에는 더 많은 위치를 확인한다.

1. Primary memory record를 tombstone으로 표시한다.
2. Vector index와 keyword index에서 제거한다.
3. Summary, link, graph edge 같은 파생 자료를 찾는다.
4. Prompt cache와 application cache를 비운다.
5. Backup 보존 기간과 삭제 완료 시점을 기록한다.
6. 정책이 허용하면 원문과 파생물을 hard delete한다.

Tombstone은 삭제된 ID가 background re-index 과정에서 다시 살아나는 일을 막는다. 다만 tombstone 자체에도 민감한 원문을 넣지 않는다. 삭제 여부와 최소 ID만 남긴다.

## 12. 작은 pipeline을 구현한다

핵심 순서는 다음 code로 줄일 수 있다.

```python
def retrieve(user_id, query, expected_type=None, top_k=3):
    candidates = store.search(
        namespace=(user_id, "memories"),
        active=True,
        deleted=False,
    )

    scored = [hybrid_score(query, memory) for memory in candidates]
    ranked = sorted(scored, key=lambda item: item.score, reverse=True)

    if expected_type is not None:
        ranked = [item for item in ranked if item.type == expected_type]

    return ranked[:top_k]
```

Search log에는 query, filter, 후보 수, score component, 최종 rank, context에 들어간 memory ID를 남긴다. 저장 log에는 candidate extraction 결과, 중복 판단, 충돌 처리, source를 기록한다. 이 두 log가 있어야 Recall이 낮을 때 write와 retrieve 중 어디를 고칠지 알 수 있다.

## 확인 문제

1. 모든 Conversation History를 long-term memory로 복사하면 어떤 문제가 생기는가?
2. Agent가 추측한 사용자 선호를 곧바로 semantic memory로 저장하면 위험한 이유는 무엇인가?
3. Raw message, summary, structured JSON을 함께 보존하면 각각 어디에 쓸 수 있는가?
4. `user_id`와 `active` filter를 vector search 뒤에 적용하면 안 되는 이유는 무엇인가?
5. Keyword 검색이 dense similarity보다 잘 찾을 수 있는 정보 세 가지를 적어보자.
6. 오래된 땅콩 알레르기가 recency 때문에 밀리지 않도록 어떤 score를 함께 써야 하는가?
7. Generative Agents의 retrieval score를 이루는 세 항목은 무엇인가?
8. Structured dense 조건이 낮았다는 결과를 “JSON memory는 나쁘다”로 해석하면 안 되는 이유는 무엇인가?
9. Hybrid search가 Atlas failure와 retry rule을 혼동한 원인은 무엇인가?
10. Memory type filter가 검색을 고칠 수도 있고 망칠 수도 있는 까닭은 무엇인가?
11. 새 선호를 저장할 때 `supersedes`와 `active` field를 함께 쓰는 이유는 무엇인가?
12. Soft delete 뒤에도 vector index, cache, backup을 확인해야 하는 이유는 무엇인가?
13. Recall@1과 Recall@3이 각각 4/6과 6/6이라면 검색 결과를 어떻게 해석해야 하는가?

## 완료 체크

- [x] Conversation History에서 장기 보관할 memory 후보를 고르는 기준을 정했다.
- [x] Raw message, summary, structured JSON의 보존 목적을 비교했다.
- [x] User, source, timestamp, type, importance, confidence, active field를 설계했다.
- [x] Dense similarity, keyword, recency, importance, confidence를 계산했다.
- [x] Hybrid score와 memory type filter를 적용했다.
- [x] 정답을 아는 query 6개에서 Recall@1과 Recall@3을 측정했다.
- [x] 다른 사용자의 record, 비활성 record, 삭제 record를 검색 전에 제외했다.
- [x] 새 한국어 선호가 예전 영어 선호를 대체하도록 version을 연결했다.
- [x] Soft delete 뒤에 확인할 primary store, index, cache, backup을 정리했다.
- [x] 작은 합성 실험을 실제 memory benchmark와 구분했다.

---

[^1]: LangChain. [LangGraph: Add memory](https://docs.langchain.com/oss/python/langgraph/add-memory). Thread 범위 short-term memory, namespace 범위 long-term store, semantic search와 memory write 위치를 참고했다. 확인일: 2026-08-04.
[^2]: Xu, W. et al. (2025). [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110). Contextual description, keyword, tag가 있는 note, memory link와 기존 memory를 갱신하는 memory evolution을 참고했다.
[^3]: Chhikara, P. et al. (2025). [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413). 대화에서 중요한 정보를 추출하고 consolidation, retrieval하는 memory-centric architecture와 graph variant를 참고했다.
[^4]: Park, J. S. et al. (2023). [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442). Figure 6과 §4.1의 memory stream, recency·importance·relevance retrieval score를 참고했다.
[^5]: 직접 실행한 `llm_lecture/week24_memory_pipeline.py`의 결과다. 영어 memory 15개, active Jisu record 11개, query 6개를 사용했다. CLIP text encoder는 교육용 dense embedder로만 사용했고 score weight와 정답 type은 사람이 정했다. Script, CSV, JSON과 논문 원본은 Git에서 제외하고 최종 plot만 공개했다. 실행일: 2026-08-04.

<!-- HUMANIZE-SUMMARY
장르: 교육용 강의 노트
검토 단위: 문서를 5,000자 이하의 절 묶음으로 나누어 점검
원본/윤문본: 14,595자 / 14,588자, 글자 수 기준 변경률 0.05%
카테고리별 탐지/수정: C-11 연결어미 뒤 쉼표 9→0, A-10 가능 표현 4→0, A-18 긴 복합문 5→0, D-1 관용구 0→0
정량 점검: humanize-korean metrics v2.0 risk score 3→1, risk band low 유지
자체검증: 고유명사·수치 보존 / 변경률 30% 이하 / 장르 유지 / 평어체 유지 / S1 잔존 없음 / 인공 수사 추가 없음
등급: B — 자체검증 6/6을 통과했고 논문 개념, model revision, score 식, 실험 결과와 reference를 보존함
주요 변경 1: “무엇을 오래 남길지 고르고, 필요한 memory를” → “무엇을 오래 남길지 고른다. 필요한 memory를”
주요 변경 2: “저장 후보를 만들고, 기존 record와” → “저장 후보를 만든 뒤 기존 record와”
주요 변경 3: “rule과 LLM을 섞어 만들 수 있다” → “rule과 LLM을 섞어 만든다”
주요 변경 4: “assistant 답변까지 섞여 길고” → “assistant 답변까지 섞여 길다”
주요 변경 5: “감사할 수 있다” → “감사 기록에서 확인한다”
-->
