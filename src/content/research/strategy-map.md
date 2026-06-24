---
title: "StrategyMap"
subtitle: "A workflow map as state memory for dependency-sensitive intelligent agents."
subtitleKo: "절차와 순서가 중요한 일을 에이전트가 현재 상태를 기억하며 처리하도록 돕는 업무 지도입니다."
year: "2026"
status: "Under review"
image: "/research/img/strategy-map-overview.png"
showHeroImage: false
summary: "StrategyMap turns procedural domain knowledge into a workflow map as state memory, helping agents track completed, admissible, and blocked work during execution."
summaryKo: "복잡한 업무 지식을 긴 프롬프트에 넣어 두는 대신, 상태를 기억하는 업무 지도로 바꿨습니다. 에이전트는 무엇이 완료됐고, 무엇이 가능하며, 무엇이 막혀 있는지 보며 다음 단계를 고릅니다."
sourceFolder: "workdir/papers/StrategyMap"
demoVideos:
  - title: "StrategyMap Demo"
    titleKo: "데모 영상"
    titleEn: "StrategyMap Demo"
    url: "https://youtu.be/wQ003_-T9ZY"
    embedUrl: "https://www.youtube.com/embed/wQ003_-T9ZY"
    descriptionKo: "에이전트가 사람이 작성한 작업 지도를 따라 과제를 해결하는 데모입니다."
    descriptionEn: "How the agent uses the manual Strategy Map as navigation to complete the task."
references:
  - text: "Yao, Shunyu, et al. “ReAct: Synergizing Reasoning and Acting in Language Models.” International Conference on Learning Representations, 2023."
  - text: "Wei, Jason, et al. “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.” Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 24824–24837."
  - text: "Yao, Shunyu, et al. “Tree of Thoughts: Deliberate Problem Solving with Large Language Models.” Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 11809–11822."
  - text: "Besta, Maciej, et al. “Graph of Thoughts: Solving Elaborate Problems with Large Language Models.” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 16, 2024, pp. 17682–17690."
  - text: "Anthropic. “Model Context Protocol: A Protocol for Seamless Integration between LLM Applications and External Data Sources.” 2024."
    href: "https://github.com/modelcontextprotocol"
  - text: "Xiao, Ruixuan, et al. “FlowBench: Revisiting and Benchmarking Workflow-Guided Planning for LLM-Based Agents.” Findings of the Association for Computational Linguistics: EMNLP 2024, 2024, pp. 10883–10900."
sections:
  - key: "background"
    navKo: "배경"
    navEn: "Background"
    titleKo: "절차가 있는 업무를 위한 길잡이"
    titleEn: "Procedural Navigation for Agents"
    bodyKo:
      - >
        비즈니스 사용자가 원하는 것은 단순한 Q&A 답변이 아니라, 실제 도구와 업무 규칙 안에서 실행 가능한 의사결정인 경우가 많다. 예를 들어 S&OP 업무에서 재고 이동을 추천하려면 현재 재고, 수요 약속, 서비스 수준 조건을 먼저 확인해야 한다. 최종 추천 문장이 그럴듯해도, 이런 선행 검증을 건너뛰었다면 운영 관점에서는 잘못된 답이다.
      - >
        이 문제는 에이전트가 어떤 도구를 쓸 수 있는지 아는 것만으로 해결되지 않는다. 현재까지 어떤 단계가 완료되었는지, 어떤 단계가 지금 가능한지, 아직 어떤 단계가 막혀 있는지를 알아야 한다. 긴 실행 기록에는 이 정보가 숨어 있지만, 에이전트가 매 순간 raw trace에서 절차 상태를 다시 추론하는 것은 불안정하다.
      - >
        완전히 고정된 workflow script도 충분하지 않다. script는 순서를 강하게 보장하지만, 실제 업무에서는 중간 관찰값이나 사용자 조건에 따라 다음 행동이 달라질 수 있다. 반대로 자율적인 trace 기반 agent는 유연하지만, process state가 길고 noisy한 history 안에 묻힌다. 필요한 것은 더 긴 프롬프트나 더 많은 tool catalog가 아니라, 실행 중에 현재 절차 상태를 명시적으로 보여주는 계층이다.
      - >
        StrategyMap은 사람이 검토한 업무 절차를 milestone과 dependency로 표현하고, 실행 중에는 상태를 기억하는 업무 지도로 completed, admissible, blocked 상태를 드러낸다. 에이전트는 전체 절차를 외워서 움직이는 것이 아니라, 현재 확인된 process state에서 다음으로 유효한 지점을 고른다. 그래서 StrategyMap은 workflow를 프롬프트에 붙이는 방식이 아니라, 업무 지식을 실행 시점의 상태 메모리로 바꾸는 방법이다.
    bodyEn:
      - >
        Business users often need more than an answer. They need a decision or service outcome that can be executed through real tools under explicit business rules. In an S&OP allocation task, for example, an agent should not recommend a supply transfer before checking inventory, demand commitments, and service-level constraints. A final recommendation may sound plausible, but it is operationally invalid if prerequisite checks were skipped.
      - >
        This problem is not solved by giving the agent more tools. The agent must know which process states are completed, which next steps are admissible, and which future steps remain blocked. Long traces contain some of this information, but requiring the agent to infer process state from raw history at every turn is fragile.
      - >
        Fully scripted workflows are also insufficient. They make prerequisite order explicit, but become brittle when local context changes. Autonomous trace-based agents are flexible, but their execution state remains implicit in long and noisy histories. The missing abstraction is not a longer prompt or a larger tool catalog, but an explicit execution-time layer for process state.
      - >
        StrategyMap represents reviewed process knowledge as milestones and dependencies, then instantiates it as a workflow map as state memory during execution. The map exposes completed, admissible, and blocked states, so the agent can choose the next valid objective from checked process progress. In this sense, StrategyMap is not just workflow text in a prompt; it turns workflow knowledge into state memory.
  - key: "method"
    navKo: "방법"
    navEn: "Method"
    titleKo: "상태를 기억하는 업무 지도"
    titleEn: "Workflow Map as State Memory"
    bodyKo:
      - >
        사람이 검토한 업무 절차를 milestone과 dependency로 구조화했다. 각 milestone은 단순한 설명 문장이 아니라, 필요한 입력, 만들어야 할 출력, 완료 여부를 확인하는 contract를 가진 checkpoint다.
      - >
        실행이 시작되면 published map으로부터 세션 안에서만 쓰는 업무 지도를 만든다. 이 지도는 전체 map의 dependency를 유지하면서, 현재 세션에서 어떤 milestone이 완료되었고 어떤 milestone이 다음으로 실행 가능한지 계산한다. 즉 에이전트는 전체 절차 구조를 보되, dependency와 현재 process state 때문에 admissible frontier 안에서 다음 목표를 고르게 된다.
      - >
        업데이트는 실행 가능한 상태만 대상으로 한다. 도구 호출이나 milestone transition 뒤에 관찰값이 해당 milestone의 output contract와 맞으면, 그 milestone을 satisfied로 표시하고 contract-checked output을 node memory에 기록한다. 조건이 맞지 않으면 상태를 커밋하지 않는다. 그래서 StrategyMap은 단순한 고정 그래프가 아니라, 실행 중에 확인된 증거만 상태 메모리로 반영하는 업무 지도다.
      - >
        TRA 설정에서는 Reasoning Engine과 Librarian을 분리했다. Reasoning Engine은 상태를 기억하는 업무 지도가 제시한 다음 milestone과 process progress에 집중하고, Librarian은 MCP 서버에서 후보 도구를 찾아 제공한다[5]. 이 분리는 StrategyMap이 정답을 직접 고르는 장치가 아니라, 에이전트가 process-valid objective를 유지한 채 필요한 도구를 찾도록 돕는 실행 프로토콜임을 보여준다.
    bodyEn:
      - >
        StrategyMap structures reviewed procedural knowledge as milestones and dependencies. Each milestone is not just a textual instruction: it is a checkpoint with required inputs, expected outputs, and a completion contract.
      - >
        At the start of execution, the published map initializes a session-local workflow map as state memory. The map preserves the full dependency structure while computing which milestones are completed and which milestones are currently admissible. The agent can see the process structure, but dependency state determines the admissible frontier from which the next objective should be chosen.
      - >
        Updates are committed only for admissible process progress. After a tool call or milestone transition, the system checks whether the observation satisfies the milestone output contract. If it does, the milestone is marked as satisfied and the contract-checked output is stored in node-local memory. If it does not, the workflow map state is not committed. This makes StrategyMap a workflow map as state memory, not a static graph prompt.
      - >
        In the TRA setting, the Reasoning Engine is separated from the Librarian. The Reasoning Engine focuses on the next milestone and process progress exposed by the workflow map as state memory, while the Librarian retrieves candidate tools from MCP servers [5]. This separation frames StrategyMap as an execution protocol for preserving process-valid objectives while tool provisioning is handled separately.
    figure:
      src: "/research/img/strategy-map-overview.png"
      alt: "StrategyMap runtime architecture"
      captionKo: "StrategyMap의 상태를 기억하는 업무 지도 구조."
      captionEn: "Workflow map as state memory architecture in StrategyMap."
  - key: "results"
    navKo: "결과"
    navEn: "Results"
    titleKo: "Strategy Map 가이드 효과"
    titleEn: "Strategy Map Guidance Effects"
    bodyKo:
      - >
        실험은 두 축으로 구성했다. Runtime type은 ReAct[1]와 TRA를 비교했고, reasoning type은 CoT[2], ToT[3], GoT[4]를 비교했다. 주요 실험은 자동차 S&OP 절차에서 진행했고, 추가로 FlowBench[6] workflow specification에서도 같은 Strategy Map 방식이 동작하는지 확인했다.
      - >
        S&OP에서는 StrategyMap이 정확도를 22.9~38.6%p 높였다. 고정 체크리스트나 고정 Strategy Map context보다, 상태를 기억하는 업무 지도가 completed, admissible, blocked 상태를 유지하는 조건이 더 좋은 결과를 냈다.
      - >
        작동 방식 진단 지표도 같은 방향을 보였다. StrategyMap을 사용하면 path conformity, milestone alignment, required-tool coverage가 함께 좋아졌다. 이는 성능 향상이 단순히 마지막 답변 문장을 더 그럴듯하게 만든 결과가 아니라, 에이전트가 실제로 더 의존성에 맞는 절차 경로를 따라갔다는 근거다.
      - >
        FlowBench에서는 benchmark가 제공한 workflow specification을 milestone dependency로 바꾸어 사용했다. 이 비교는 숨겨진 정답을 더 준 것이 아니라, 같은 workflow 지식을 prompt-only context로 둘 때와 Strategy Map artifact로 유지할 때의 차이를 보는 실험이다.
    bodyEn:
      - >
        The experiments were organized along two axes. The runtime type compared ReAct [1] and TRA, while the reasoning type compared CoT [2], ToT [3], and GoT [4]. The primary study used an automotive S&OP procedure-derived benchmark, and FlowBench [6] tested whether the same Strategy Map representation transfers to benchmark-provided workflow specifications.
      - >
        In S&OP, StrategyMap improved accuracy by 22.9 to 38.6 percentage points. Ordered checklists and static Strategy Map context underperformed the full workflow map as state memory, showing that maintaining completed, admissible, and blocked states during execution matters.
      - >
        The mechanism diagnostics pointed in the same direction. With StrategyMap, path conformity, milestone alignment, and required-tool coverage improved together. This suggests that the accuracy gain did not come merely from different final-answer phrasing, but from the agent following a more dependency-consistent execution path.
      - >
        In FlowBench, benchmark-provided workflow specifications were converted into milestone dependencies. The comparison does not add hidden answers; it tests the difference between exposing workflow knowledge as prompt-only context and maintaining it as a Strategy Map artifact.
    highlightsKo:
      - "S&OP 정확도 22.9~38.6%p 향상"
      - "고정 체크리스트와 고정 지도보다 상태를 기억하는 업무 지도가 더 좋은 성능"
      - "Path conformity, milestone alignment, required-tool coverage가 함께 개선"
      - "절차 설명을 주는 것보다 현재 단계에 맞춰 길을 좁히는 방식이 더 효과적"
    highlightsEn:
      - "S&OP accuracy improved by 22.9 to 38.6 percentage points"
      - "Full workflow map as state memory outperformed static checklists and static maps"
      - "Path conformity, milestone alignment, and required-tool coverage improved together"
      - "Runtime navigation mattered more than simply providing procedural knowledge"
    table:
      captionKo: "S&OP ablation 결과. 같은 executor 안에서 상태를 기억하는 Full SM 업무 지도가 고정 체크리스트와 고정 Strategy Map context보다 높은 정확도를 보였다."
      captionEn: "S&OP ablation results. Within each executor, the full SM workflow map as state memory outperformed ordered checklists and static Strategy Map context."
      columns:
        - key: "executor"
          labelKo: "Executor"
          labelEn: "Executor"
        - key: "condition"
          labelKo: "조건"
          labelEn: "Condition"
        - key: "accuracy"
          labelKo: "정확도"
          labelEn: "Accuracy"
        - key: "delta"
          labelKo: "No map 대비"
          labelEn: "Delta vs no map"
        - key: "tokens"
          labelKo: "토큰(k)"
          labelEn: "Tokens (k)"
        - key: "time"
          labelKo: "시간(s)"
          labelEn: "Time (s)"
      rows:
        - executor: "ReAct"
          condition: "No map / trace only"
          accuracy: "46.4%"
          delta: "-"
          tokens: "104.2"
          time: "24.0"
        - executor: "ReAct"
          condition: "Ordered checklist"
          accuracy: "27.1%"
          delta: "-19.3%p"
          tokens: "111.2"
          time: "27.5"
        - executor: "ReAct"
          condition: "Static SM context"
          accuracy: "28.6%"
          delta: "-17.9%p"
          tokens: "109.8"
          time: "25.3"
        - executor: "ReAct"
          condition: "Full SM state memory"
          accuracy: "67.1%"
          delta: "+20.7%p"
          tokens: "275.0"
          time: "98.4"
        - executor: "TRA"
          condition: "No map / trace only"
          accuracy: "29.3%"
          delta: "-"
          tokens: "55.4"
          time: "38.2"
        - executor: "TRA"
          condition: "Ordered checklist"
          accuracy: "22.9%"
          delta: "-6.4%p"
          tokens: "66.7"
          time: "52.6"
        - executor: "TRA"
          condition: "Static SM context"
          accuracy: "20.7%"
          delta: "-8.6%p"
          tokens: "76.9"
          time: "47.7"
        - executor: "TRA"
          condition: "Full SM state memory"
          accuracy: "72.9%"
          delta: "+43.6%p"
          tokens: "99.7"
          time: "114.6"
    figure:
      src: "/research/img/strategy-map-result-exp1.png"
      alt: "StrategyMap experiment results"
      captionKo: "StrategyMap의 주요 실험 결과 비교."
      captionEn: "Main experimental comparison for StrategyMap."
  - key: "reflection"
    navKo: "시사점"
    navEn: "Reflection"
    titleKo: "절차적 지식이 에이전트의 업무 실행에 도움이 됨"
    titleEn: "Procedural Knowledge Helps Agents Execute Work"
    bodyKo:
      - >
        이 프로젝트에서 가장 크게 배운 점은 업무 절차 지식을 긴 프롬프트로 넣는 것만으로는 부족하다는 점이다. 복잡한 도구 사용 업무에서는 Strategy Map처럼 현재 무엇이 완료되었고, 무엇이 다음으로 가능하며, 무엇이 아직 막혀 있는지를 상태 메모리로 관리하는 방식이 더 중요했다.
      - >
        좋은 안내가 항상 더 짧은 실행을 뜻하지도 않았다. S&OP에서는 StrategyMap을 쓸 때 더 많은 대화 턴과 비용이 들 수 있었다. 중요한 것은 단계 수를 줄이는 것이 아니라, 유효한 단계를 밟는 것이었다. 즉, 절차를 빨리 끝내는 것보다 필요한 확인을 빠뜨리지 않고 진행하는 것이 더 중요했다.
      - >
        결과를 설명하려면 정확도만으로는 부족했다. 경로 일치도(path conformity), 마일스톤 정렬도(milestone alignment), 필요한 도구 사용 범위(tool coverage) 같은 절차 지표가 있어야 에이전트가 왜 좋아졌는지 볼 수 있었다. 단순히 마지막 답변이 달라진 것이 아니라, 더 의존성에 맞는 경로를 따라갔는지를 확인하는 과정이 필요했다.
      - >
        또한 단순히 절차를 보여주는 것과 Strategy Map으로 현재 업무 상태를 기억하는 것은 달랐다. 순서가 있는 체크리스트나 고정 지도는 업무 흐름을 알려주지만, 현재 세션에서 어떤 선행 조건이 실제로 충족되었는지는 관리하지 못한다. 상태를 기억하는 업무 지도가 더 좋은 결과를 낸 이유는 이 차이에 있었다.
      - >
        한계도 분명하다. StrategyMap은 비용을 줄이는 방법이 아니라, 추가 토큰, 실행 시간, 단계를 더 나은 업무 품질과 절차 품질로 바꾸는 방식에 가깝다. 또 지도가 잘못 작성되었거나 오래되면 잘못된 행동을 그럴듯하게 만들 수 있다. 실제 적용을 위해서는 작성, 검토, 배포, 실행 기록 기반 유지보수를 갖춘 관리 가능한 업무 산출물로 다뤄야 하고, 외부에서 만든 업무 흐름과 실제 운영 로그에서도 검증해야 한다.
    bodyEn:
      - >
        The main lesson is that workflow knowledge is not enough as long prompt text. In complex tool-use tasks, Strategy Map was useful because it worked as state memory for what had been completed, what was currently admissible, and what remained blocked.
      - >
        Good guidance also does not always mean shorter execution. In S&OP, StrategyMap could require more turns and higher cost. The important target was not fewer steps, but valid steps: the agent should avoid skipping prerequisite checks even when that makes execution longer.
      - >
        Accuracy alone was not enough to explain the result. Process metrics such as path conformity, milestone alignment, and tool coverage were necessary to see why the agent improved. They helped distinguish better procedural execution from merely different final-answer wording.
      - >
        It also mattered that static context was not the same as Strategy Map guidance. Ordered checklists and static maps can describe the workflow, but they do not track which prerequisites have actually been satisfied in the current session. The workflow map as state memory was stronger because it maintained this state during execution.
      - >
        The limitations are equally important. StrategyMap does not reduce cost by default; it converts additional tokens, runtime, and steps into better task and process quality. A stale or misspecified map can also make an invalid action look justified. For deployment, maps should be treated as governed process artifacts with authoring, review, publishing, and trace-based maintenance, and they still need validation on externally authored workflows and real operational logs.
---
