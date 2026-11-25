# Reasoning 학습 필독 논문

처음 reasoning 모델 학습(SFT → DPO → GRPO)을 시작하는 엔지니어를 위한 읽기 목록이다. 

---

## 한눈에 보기

| Part | 주제 | 이 Part가 답하는 질문 | 편수 | 논문 |
|---|---|---|---|---|
| 1 | [기초와 평가](#part-1--기초와-평가) | 추론이 정말 필요한가? pass@k는 무엇을 말해 주나? | 4 | 1, 2, 3, 4 |
| 2 | [Off-policy 문제](#part-2--off-policy-문제) | 남이 쓴 trace로 배우면 왜 안 되나? | 5 | 7, 8, 9, 10, 11 |
| 3 | [SFT → RL 전환](#part-3--sft--rl-전환) | SFT는 어디까지 하고, 언제 RL로 넘어가나? | 6 | 12, 13, 14, 15, 16, 17 |
| 4 | [선호 학습 (DPO)](#part-4--선호-학습-dpo) | 같은 오답을 반복하는 샘플에 DPO를 쓸 때 알아야 할 것은? | 4 | 18, 19, 20, 21 |
| 5 | [RL 방법론](#part-5--rl-방법론) | GRPO는 어떻게 동작하고, 어디서 함정에 빠지나? | 8 | 22, 23, 24, 25, 26, 33, 27, 28 |
| 6 | [On-policy distillation](#part-6--on-policy-distillation) | 좋은 teacher가 있는데 SFT가 성능을 깎는다면? | 4 | 29, 31, 34, 35 |
| 7 | [도메인 적용](#part-7--도메인-적용) | 앞의 기법들은 실제 relevance 모델에서 어떻게 조합되나? | 1 | 32 |

### 시간이 없다면 이 9편부터

1 Self-Consistency → 2 To CoT or not to CoT → 4 Does RL Incentivize → 9 I Learn Better → 13 RL's Razor → 22 GRPO → 24 DAPO → 25 Dr. GRPO → 32 TaoSR1

### 이 목록이 답하는 네 가지 질문

| 질문 | 주로 보는 Part | 빠른 참조 |
|---|---|---|
| Q1. SFT는 어디까지 하고, 언제 RL로 넘어가나? | 3 | [Q1](#q1-sft는-어디까지-언제-rl로-넘어가나) |
| Q2. 어떤 상황에서 어떤 학습 방법을 쓰나? | 4, 5, 6 | [Q2](#q2-상황별-학습-방법) |
| Q3. 학습 중 어떤 메트릭을 보나? | 1, 3, 5 | [Q3](#q3-학습-중-반드시-볼-메트릭) |
| Q4. off-policy는 왜 문제이고, 왜 on-policy로 학습하나? | 2, 6 | [Q4](#q4-왜-on-policy인가) |

---

## Part 1 — 기초와 평가

> **추론이 정말 필요한가? pass@k는 무엇을 말해 주나?**  
> 읽고 나면: CoT가 언제 필요한지, pass@k를 어떻게 읽는지 · 가이드 1~3절

### 1. Self-Consistency
Wang et al., 2022 · [원문](https://arxiv.org/abs/2203.11171) · [Q&A](./reasoning_qa/01_self_consistency.md) · **가이드** 2절 MV@k, sharpen_gap

- **질문** CoT prompting에서 greedy decoding 대신 여러 reasoning path를 sampling한 뒤 최종 답을 다수결로 모으면 추론 정확도가 오르는가?
- **요약** CoT 출력을 여러 번 sampling해 최종 답을 다수결로 모으는 것만으로, 추가 학습 없이 GSM8K +17.9 등 추론 정확도를 크게 올리며, 답의 일치율은 모델 confidence 지표로도 쓸 수 있다.

### 2. To CoT or not to CoT?
Sprague et al., 2024 · [원문](https://arxiv.org/abs/2409.12183) · [Q&A](./reasoning_qa/02_to_cot_or_not.md) · **가이드** 원칙 6, think 기여

- **질문** Prompt 기반 CoT는 어떤 종류의 과제에서 실제로 direct answer(DA)보다 도움이 되며, 그 이유는 무엇인가?
- **요약** Prompt 기반 CoT의 이득은 거의 전부 수학·기호·논리처럼 중간 계산을 추적해야 하는 과제에서 나오며, commonsense·knowledge 류 과제에서는 DA와 거의 같으므로 CoT는 선택적으로 쓰고 symbolic 과제에는 tool을 붙이는 편이 낫다.

### 3. Measuring Faithfulness in CoT
Lanham et al., 2023 · [원문](https://arxiv.org/abs/2307.13702) · [Q&A](./reasoning_qa/03_cot_faithfulness.md) · **가이드** 3.4 추론 품질

- **질문** LLM이 생성한 CoT는 모델이 실제로 답에 도달한 과정을 충실히(faithful) 반영하는가, 그리고 어떤 조건에서 더 faithful한가?
- **요약** CoT에 잘라내기·오류 삽입·filler·paraphrase로 개입해 보면 CoT 의존도(faithfulness)는 과제별로 크게 다르고 더 큰 모델일수록 낮아지는 경향이 있어, CoT를 모델 판단의 설명으로 믿으려면 모델과 과제를 신중히 골라야 한다.

### 4. Does RL Really Incentivize Reasoning?
Yue et al., 2025 · [원문](https://arxiv.org/abs/2504.13837) · [Q&A](./reasoning_qa/04_does_rl_incentivize.md) · **가이드** 2절, 원칙 2

- **질문** RLVR(Reinforcement Learning with Verifiable Rewards)은 base model에 없던 새로운 reasoning 능력을 만들어내는가, 아니면 base model이 이미 가진 reasoning path를 더 자주 뽑도록 만들 뿐인가?
- **요약** 현재 RLVR은 base model이 이미 가진 reasoning path의 sampling efficiency만 높일 뿐 boundary를 넓히지 못하고(큰 k에서 base가 역전), 새로운 reasoning pattern은 distillation으로 들어온다.

---

## Part 2 — Off-policy 문제

> **남이 쓴 trace로 배우면 왜 안 되나?**  
> 읽고 나면: 남이 쓴 trace로 SFT하면 왜 망가지는지 · 가이드 3.3, 7.5~7.7

### 7. Fine-Tuning on New Knowledge → Hallucinations?
Gekhman et al., 2024 · [원문](https://arxiv.org/abs/2405.05904) · [Q&A](./reasoning_qa/07_new_knowledge_hallucination.md) · **가이드** 3.3 따라 할 수 없는 경로

- **질문** SFT 데이터에 pre-training에서 배우지 못한 새로운 사실(new knowledge)이 섞여 있으면, fine-tuned 모델이 기존 지식에 대해 hallucination을 더 하게 되는가?
- **요약** 모델이 모르는 사실을 담은 SFT 예제는 늦게 학습되고, 결국 fit되면 기존 지식에 대한 hallucination을 선형적으로 늘리므로 early stopping·Unknown 필터링(또는 "I don't know" relabel)이 필요하다.

### 8. Small Models Struggle to Learn from Strong Reasoners
Li et al., 2025 · [원문](https://arxiv.org/abs/2502.12143) · [Q&A](./reasoning_qa/08_small_models_struggle.md) · **가이드** 7.7 teacher 선택

- **질문** 작은 student 모델(≤3B)은 강한 teacher의 long CoT나 대형 teacher의 CoT를 distillation 받으면 항상 이득을 보는가?
- **요약** 3B 이하 student는 강한 teacher의 long/large CoT를 그대로 모방하면 오히려 성능이 떨어지며(teacher-student distribution mismatch), long/large CoT를 20%만 섞는 Mix Distillation이 이 learnability gap을 줄인다.

### 9. I Learn Better If You Speak My Language
Ren et al., 2024 · [원문](https://arxiv.org/abs/2402.11192) · [Q&A](./reasoning_qa/09_i_learn_better.md) · **가이드** 3.3 rel_NLL

- **질문** 같은 질문에 대해 사람이 쓴 정답(human-annotated response) 대신 LLM이 생성한 응답으로 SFT하면 왜 reasoning 성능이 더 좋은가?
- **요약** 같은 정답이라도 target 모델이 "익숙한"(낮은 perplexity) 스타일로 쓰인 응답이 더 잘 학습되고 cross-task forgetting도 적으므로, human label을 그대로 쓰기보다 모델 자신의 출력을 최소 수정해 정답으로 만드는 편이 낫다.

### 10. Self-Distillation Fine-Tuning (SDFT)
Yang et al., 2024 · [원문](https://arxiv.org/abs/2402.13669) · [Q&A](./reasoning_qa/10_sdft.md) · **가이드** 7.7 재작성

- **질문** 이미 instruction-tuned된 seed LM을 특정 downstream task에 fine-tuning할 때, task 성능을 올리면서 일반 instruction-following·safety·helpfulness를 잃지 않으려면 어떻게 해야 하는가?
- **요약** 정답 응답을 seed 모델 자신이 다시 쓰게 해(self-distillation) 학습 target을 모델 분포 안으로 끌어오면, target task 성능은 유지하거나 약간 올리면서 vanilla SFT가 일으키는 safety·helpfulness·타 task forgetting을 크게 줄일 수 있다.

### 11. STaR
Zelikman et al., 2022 · [원문](https://arxiv.org/abs/2203.14465) · [Q&A](./reasoning_qa/11_star.md) · **가이드** 7.7 자기 trace

- **질문** 소수의 rationale 예시와 rationale 없는 (질문, 정답) 데이터만으로, 모델이 스스로 생성한 rationale을 학습해 reasoning 능력을 반복적으로 키울 수 있는가?
- **요약** 정답으로 필터링한 자기 생성 rationale로 반복 fine-tuning하고, 못 푼 문제는 정답 hint로 rationale을 역생성(rationalization)해 보충하면, rationale 데이터 없이도 reasoning이 bootstrapping되며 이는 binary reward policy gradient의 근사로 볼 수 있다.

---

## Part 3 — SFT → RL 전환

> **SFT는 어디까지 하고, 언제 RL로 넘어가나?**  
> 읽고 나면: SFT를 어디서 멈추고 RL로 넘어갈지 · 가이드 7.1, 7.4

### 12. SFT Memorizes, RL Generalizes
Chu et al., 2025 · [원문](https://arxiv.org/abs/2501.17161) · [Q&A](./reasoning_qa/12_sft_memorizes_rl_generalizes.md) · **가이드** 원칙 3·4

- **질문** SFT와 RL은 각각 학습 데이터를 "암기(memorization)"하는가, 아니면 새로운 변형에도 통하는 "일반화 가능한 규칙"을 학습하는가?
- **요약** 같은 compute에서 outcome reward 기반 multi-turn RL은 규칙과 시각 변형 모두에서 OOD 성능을 올리지만 SFT는 학습 규칙을 외워 OOD가 무너진다. 다만 RL이 작동하려면 SFT로 출력 format을 먼저 잡아 줘야 한다.

### 13. RL's Razor
Shenfeld et al., 2025 · [원문](https://arxiv.org/abs/2509.04259) · [Q&A](./reasoning_qa/13_rls_razor.md) · **가이드** 7.3, 7.4

- **질문** 새 task에서 SFT와 RL이 같은 성능에 도달했는데도 RL이 기존 능력을 훨씬 덜 잊는 이유는 무엇이며, 무엇이 forgetting의 양을 결정하는가?
- **요약** Forgetting은 알고리즘이 아니라 새 task에서 잰 base 대비 KL로 예측되고, on-policy RL은 자기 분포에서 sampling하기 때문에 KL이 가장 작은 해를 고르며(RL's Razor), 그래서 SFT보다 덜 잊는다.

### 14. Retaining by Doing
Chen et al., 2025 · [원문](https://arxiv.org/abs/2510.18874) · [Q&A](./reasoning_qa/14_retaining_by_doing.md) · **가이드** 7.5 쉬운 구간 하락

- **질문** LM post-training에서 SFT와 RL 중 어느 쪽이 기존 능력을 덜 잊는가, 그 차이는 어디서 오는가, 그리고 실용적으로 어떻게 줄일 수 있는가?
- **요약** RL이 SFT보다 덜 잊는 핵심은 KL 정규화나 advantage가 아니라 on-policy 데이터의 mode-seeking 성질이고, epoch마다 다시 생성한 approximately on-policy 데이터로 SFT해도 forgetting 대부분을 막을 수 있다.

### 15. Cognitive Behaviors that Enable Self-Improving Reasoners
Gandhi et al., 2025 · [원문](https://arxiv.org/abs/2503.01307) · [Q&A](./reasoning_qa/15_cognitive_behaviors.md) · **가이드** 7.1 SFT 목표

- **질문** 같은 RL을 적용해도 어떤 모델은 크게 좋아지고 어떤 모델은 금방 정체되는데, 초기 모델의 어떤 성질이 RL을 통한 self-improvement를 가능하게 하는가?
- **요약** RL은 base model에 이미 있는 verification과 backtracking 같은 cognitive behavior를 증폭할 뿐이므로, 이런 행동을 priming SFT나 pretraining data curation으로 먼저 심어 주면(답이 틀린 예시로도) RL이 정체되던 모델도 self-improve한다.

### 16. Demystifying Long CoT Reasoning
Yeo et al., 2025 · [원문](https://arxiv.org/abs/2502.03373) · [Q&A](./reasoning_qa/16_demystifying_long_cot.md) · **가이드** 7.4, Stage 3

- **질문** long CoT(branching/backtracking, error validation/correction)는 어떤 조건에서 생기며, SFT와 RL은 각각 어떤 역할을 하는가?
- **요약** long CoT는 강한 long CoT 모델에서 distill한 SFT로 먼저 심고, RL에서는 길이 의존 Cosine Reward + repetition penalty + 필터링된 verifiable reward로 길이 폭주와 reward hacking을 막아야 안정적으로 강화된다.

### 17. Rethinking "RL Generalizes, SFT Memorizes" 🆕
Hou et al., NeurIPS 2026 · [원문](https://arxiv.org/abs/2611.14912) · [Q&A](./reasoning_qa/17_rethinking_sft_memorizes.md) · **가이드** 원칙 3

- **질문** "RL은 일반화하고 SFT는 암기한다"는 결론은 SFT에 RL과 동등한 데이터(출처·양)를 줬을 때도 성립하는가?
- **요약** SFT의 일반화 열세는 데이터 출처와 rollout 규모 불일치 탓이 크며, RL-tuned 모델 rollout 중심 혼합 데이터를 RL과 같은 총 rollout 수(오답 포함)로 학습하면 SFT도 RL 수준의 OOD reasoning 성능에 도달한다.

---

## Part 4 — 선호 학습 (DPO)

> **같은 오답을 반복하는 샘플에 DPO를 쓸 때 알아야 할 것은?**  
> 읽고 나면: DPO를 언제 쓰고, 왜 offline DPO가 약한지 · 가이드 Stage 2

### 18. DPO
Rafailov et al., 2023 · [원문](https://arxiv.org/abs/2305.18290) · [Q&A](./reasoning_qa/18_dpo.md) · **가이드** confident_wrong

- **질문** reward model 학습과 RL 없이, 선호 데이터만으로 RLHF와 같은 KL-constrained reward maximization 목표를 정확히 최적화할 수 있는가?
- **요약** KL-constrained RLHF의 최적 policy를 closed form으로 풀어 reward를 policy log-ratio로 바꾸면, reward model과 RL 없이 선호쌍에 대한 logistic loss 하나로 PPO와 같거나 더 좋은 선호 정렬을 얻는다.

### 19. Is DPO Superior to PPO?
Xu et al., 2024 · [원문](https://arxiv.org/abs/2404.10719) · [Q&A](./reasoning_qa/19_dpo_vs_ppo.md) · **가이드** 3.3 분포 이동

- **질문** reward-free인 DPO가 정말 reward-based인 PPO보다 우수한가, 그리고 학술 benchmark에서 PPO가 약한 이유는 무엇인가?
- **요약** DPO는 이론적으로 OOD 응답을 선호하는 biased solution을 허용해 distribution shift에 취약하며, advantage normalization·large batch·reference EMA로 제대로 튜닝한 PPO가 대화·코드 생성 전반에서 DPO와 DPO-Iter를 일관되게 앞선다.

### 20. Preference FT Should Leverage On-Policy Data
Tajwar et al., 2024 · [원문](https://arxiv.org/abs/2404.14367) · [Q&A](./reasoning_qa/20_preference_ft_on_policy.md) · **가이드** Stage 2 vs 3 선택

- **질문** preference fine-tuning에서 어떤 요소(on-policy sampling, negative gradient)가 왜 중요하며, 언제 필요한가?
- **요약** reward peak가 reference policy의 저확률 영역에 있을 때는 on-policy sampling과 negative gradient(둘 다 mode-seeking)가 필수적이며, 현재 policy 샘플로 pair를 만들어 DPO를 돌리는 on-policy contrastive 학습이 둘의 장점을 함께 얻는다.

### 21. Iterative Reasoning Preference Optimization
Pang et al., 2024 · [원문](https://arxiv.org/abs/2404.19733) · [Q&A](./reasoning_qa/21_iterative_rpo.md) · **가이드** Stage 2

- **질문** iterative preference optimization을 CoT reasoning task에 적용해, 정답 라벨만으로 reasoning 능력을 반복적으로 끌어올릴 수 있는가?
- **요약** 정답 라벨로 자기 생성 CoT를 맞음/틀림 pair로 만들어 DPO+NLL로 반복 학습하면 Llama-2-70B-Chat의 GSM8K 정확도가 55.6%에서 81.6%로 오르며, NLL 항과 반복(iteration)이 핵심이다.

---

## Part 5 — RL 방법론

> **GRPO는 어떻게 동작하고, 어디서 함정에 빠지나?**  
> 읽고 나면: GRPO 계열의 구조와 구현 함정 · 가이드 Stage 3

### 22. DeepSeekMath (GRPO)
Shao et al., 2024 · [원문](https://arxiv.org/abs/2402.03300) · [Q&A](./reasoning_qa/22_deepseekmath.md) · **가이드** GRPO 원전

- **질문** 공개 웹 데이터와 PPO보다 저렴한 RL만으로 7B open model이 GPT-4급 수학 추론에 얼마나 다가갈 수 있는가?
- **요약** critic을 없애고 같은 질문에서 뽑은 G개 sample의 정규화 reward를 advantage로 쓰는 GRPO를 제안해 7B 모델의 MATH 성능을 46.8%에서 51.7%로 올렸다. 다만 이 향상은 Pass@K가 아니라 Maj@K에서만 나타나므로, 새 능력을 만들기보다 출력 분포를 날카롭게 한 것으로 보인다.

### 23. DeepSeek-R1
DeepSeek-AI, 2025 · [원문](https://arxiv.org/abs/2501.12948) · [Q&A](./reasoning_qa/23_deepseek_r1.md) · **가이드** SFT → RL

- **질문** 사람이 쓴 reasoning trajectory로 SFT를 하지 않고, 정답 여부만 보는 rule-based reward로 base model에 pure RL을 하면 고급 추론 능력이 생기는가?
- **요약** 정답 여부와 형식만 보는 rule-based reward로 671B base model에 GRPO를 적용하자, SFT 없이도 긴 CoT와 self-reflection이 나타나 AIME pass@1이 15.6%에서 77.9%로 올랐다. 그 위에 cold start, rejection-sampling SFT, 2단계 RL을 쌓은 R1은 o1급 성능을 보이며, 이렇게 얻은 추론 능력은 SFT distillation만으로 작은 모델에 효율적으로 옮겨진다.

### 24. DAPO
Yu et al., 2025 · [원문](https://arxiv.org/abs/2503.14476) · [Q&A](./reasoning_qa/24_dapo.md) · **가이드** zero_adv_frac

- **질문** naive GRPO로는 재현되지 않는 R1 수준의 large-scale long-CoT RL을, 어떤 알고리즘 수정으로 안정적으로 재현할 수 있는가?
- **요약** DAPO는 GRPO에서 KL을 빼고 Clip-Higher, Dynamic Sampling, Token-level loss, Overlong reward shaping을 더해 Qwen2.5-32B base의 AIME 2024 점수를 30점에서 50점으로 올린 오픈소스 large-scale RL 레시피다.

### 25. Dr. GRPO
Liu et al., 2025 · [원문](https://arxiv.org/abs/2503.20783) · [Q&A](./reasoning_qa/25_dr_grpo.md) · **가이드** GRPO 구현 함정

- **질문** R1-Zero식 학습(base model에 바로 RL)에서 성능 향상과 응답 길이 증가는 정말 RL이 만든 것인가, 아니면 base model의 사전 특성과 GRPO의 최적화 bias 때문인가?
- **요약** R1-Zero식 RL의 성능 향상 일부는 base model의 사전 능력과 template 때문이고, 응답 길이 증가 일부는 GRPO의 length/std 정규화 bias 때문이다. 두 항을 지운 Dr. GRPO는 성능을 유지하면서 오답의 불필요한 길이 증가를 막는다.

### 26. Online Difficulty Filtering
Bae et al., 2025 · [원문](https://arxiv.org/abs/2504.03380) · [Q&A](./reasoning_qa/26_online_difficulty_filtering.md) · **가이드** learnable_frac

- **질문** RLVR(GRPO)에서 학습 중인 policy 기준으로 어떤 난이도의 prompt를 골라야 학습이 가장 효율적이며, 그 이유를 이론적으로 설명할 수 있는가?
- **요약** 학습 중인 policy 기준 pass rate가 중간(예: 0.3~0.7)인 prompt만 online으로 골라 고정 크기 batch를 채우면, reward variance로 잡히는 learnability 하한이 최대가 되어 GRPO가 더 적은 step으로 더 높은 성능에 도달한다.

### 33. Too Correct to Learn 🆕
Liang et al., ACL 2026 · [원문](https://arxiv.org/abs/2604.18493) · [Q&A](./reasoning_qa/33_too_correct_to_learn.md) · **가이드** zero_adv_frac, sharpen_gap

- **질문** 강한 base model이 이미 대부분 맞히는 "포화된(saturated)" 데이터에서 GRPO를 돌리면 왜 학습이 멈추고, 정답 보상만으로 학습 신호를 되살릴 수 있는가?
- **요약** 쉬운 데이터에서 GRPO 그룹이 모두 정답이면 advantage가 사라져 mode collapse가 오므로, rollout 절반을 Top-K 안에서 균등 sampling(CUTS)해 그룹 내 분산을 구조적으로 되살리면 쉬운 데이터만으로도 어려운 벤치마크의 pass@1과 maj@k가 크게 오른다.

### 27. LUFFY
Yan et al., 2025 · [원문](https://arxiv.org/abs/2504.14945) · [Q&A](./reasoning_qa/27_luffy.md) · **가이드** p=0 guidance

- **질문** on-policy RLVR은 모델이 스스로 만든 출력에서만 배우는데, 더 강한 모델의 off-policy reasoning trace를 섞어 넣으면 초기 능력의 경계를 넘어서는 추론을 학습시킬 수 있는가?
- **요약** LUFFY는 GRPO 그룹에 teacher(R1)의 off-policy trace를 섞고, 낮은 확률 token의 gradient를 키우는 policy shaping으로 entropy collapse를 막는다. 이를 통해 SFT식 경직된 모방 없이 on-policy RL보다 높은 math 성능을 얻고, on-policy가 실패하는 어려운 데이터에서도 학습을 가능하게 한다.

### 28. Pass@k Training
Chen et al., 2025 · [원문](https://arxiv.org/abs/2508.10751) · [Q&A](./reasoning_qa/28_passk_training.md) · **가이드** 탐색 부족 시

- **질문** RLVR의 reward를 Pass@1 대신 Pass@k로 바꾸면 exploration이 살아나서, 결국 Pass@1(exploitation)까지 더 높일 수 있는가?
- **요약** GRPO의 advantage를 Pass@k reward의 닫힌 해로 바꾸면 오답에 관대해지고 어려운 문제에 최적화가 집중되어 entropy가 유지되며, 이후 Pass@1 학습을 이어 붙이면 Pass@1 단독 RLVR보다 높은 성능에 도달한다.

---

## Part 6 — On-policy distillation

> **좋은 teacher가 있는데 SFT가 성능을 깎는다면?**  
> 읽고 나면: teacher 지식을 on-policy로 옮기는 법 · 가이드 7.6 마지막 대안

### 29. GKD
Agarwal et al., 2023 · [원문](https://arxiv.org/abs/2306.13649) · [Q&A](./reasoning_qa/29_gkd.md) · **가이드** on-policy distillation

- **질문** auto-regressive student를 distill할 때, 고정된 output sequence 대신 student가 직접 생성한 sequence에 teacher의 token-level 피드백을 주면 train-inference mismatch를 줄이고 성능을 높일 수 있는가?
- **요약** student가 스스로 생성한 sequence 위에서 teacher의 token-level 분포를 따라가게 하는 on-policy GKD는 sampling으로 gradient를 흘리지 않는 단순한 형태로 exposure bias를 줄이며, 요약·번역·수학 추론에서 기존 KD를 일관되게 앞서고 RL fine-tuning과도 바로 결합된다.

### 31. Thinking Machines — On-Policy Distillation
블로그, 2025.10 · [원문](https://thinkingmachines.ai/blog/on-policy-distillation/) · [Q&A](./reasoning_qa/31_thinking_machines_opd.md) · [저장소 정리](../2026-07-26/opd_follow_up_research.md)

- **질문** RL의 on-policy 장점과 distillation의 dense reward를 결합하면, post-training을 RL이나 off-policy distillation(SFT)보다 훨씬 싸게 같은 수준으로 끝낼 수 있는가?
- **요약** student가 직접 만든 trajectory의 각 토큰에 teacher logprob 기반 reverse KL을 advantage로 주는 on-policy distillation은 RL 스크립트의 한 줄 변경만으로 RL 수준의 성능을 9~30배(RL 대비 최대 50~100배) 적은 compute로 얻고, mid-training으로 잃은 행동도 이전 모델을 teacher로 써서 복원할 수 있다.

### 34. Rethinking On-Policy Distillation 🆕
Li et al., 2026.04 · [원문](https://arxiv.org/abs/2604.13016) · [Q&A](./reasoning_qa/34_rethinking_opd.md) · [요약](../2026-07-26/opd%20papers/rethinking_opd.md) · **가이드** teacher 선택

- **질문** OPD(on-policy distillation)는 언제 성공하고 언제 실패하며, 성공할 때 토큰 수준에서 무슨 일이 일어나는가?
- **요약** OPD는 teacher 점수가 아니라 "student와 사고 패턴이 호환되고 student가 못 본 능력을 가진" teacher일 때만 작동하며, 성공은 확률질량 97~99%를 차지하는 공유 고확률 토큰에 대한 점진적 정렬로 나타나므로, teacher는 초기 overlap 지표로 고르고 패턴 차이는 cold start SFT와 teacher-aligned prompt로 좁힌다.

### 35. A Survey of On-Policy Distillation for LLMs 🆕
Song & Zheng, 2026.04 (v5 2026.10) · [원문](https://arxiv.org/abs/2604.00626) · [Q&A](./reasoning_qa/35_opd_survey.md) · **가이드** forward vs reverse KL

- **질문** OPD 방법이 빠르게 늘어나면서 supervision 형태와 최적화 방식이 제각각이라 서로 비교하기 어렵다. 이들을 하나의 틀로 묶으면 언제 어떤 설계가 효과적인지 설명할 수 있는가?
- **요약** OPD를 "student가 방문한 state × teacher가 쓰는 정보 × update 방식"의 조합으로 보고 f-divergence matching 하나로 묶은 서베이로, teacher 강도나 divergence 하나로는 성공을 설명할 수 없고 coverage·보존·이후 RL 학습 가능성까지 봐야 한다고 정리하므로 처음부터 읽기보다 장별 인덱스로 쓴다.

---

## Part 7 — 도메인 적용

> **앞의 기법들은 실제 relevance 모델에서 어떻게 조합되나?**  
> 읽고 나면: CoT SFT → DPO → GRPO 전체 파이프라인이 실제 서비스에서 어떻게 조합되는지

### 32. TaoSR1
Dong et al., KDD 2026 · [원문](https://arxiv.org/abs/2508.12365) · [Q&A](./reasoning_qa/32_taosr1.md) · [요약](../2026-11-04/taosr1.md)

- **질문** CoT reasoning을 하는 생성형 LLM을 e-commerce query-item relevance 분류에 직접 학습·배포하려면 어떤 학습 파이프라인(SFT → DPO → GRPO)과 서빙 설계가 필요한가?
- **요약** 4단계 relevance 분류를 생성형 LLM으로 풀 때, RAG로 합성한 CoT를 label 뒤에 두는 post-CoT SFT → pass@N 기반 DPO(hard case는 DeepSeek-R1 정답 주입) → label 균형을 맞춘 difficulty-filtered GRPO 순서로 학습하고 CumPT 단일 threshold로 tiering하면, 최강 SFT baseline 대비 Macro-F1 +4.9pt를 얻으면서 첫 토큰만으로 온라인 서빙할 수 있다.

---

## 질문별 빠른 참조

### Q1. SFT는 어디까지, 언제 RL로 넘어가나

| 판단 기준 | 볼 메트릭 | 근거 논문 |
|---|---|---|
| SFT 목표는 정확도가 아니라 "RL을 시작하기 좋은 상태" | 형식 오류율, think 기여 | 12 SFT Memorizes, 15 Cognitive Behaviors |
| 기존 능력을 잃기 시작하면 멈춘다 | base 대비 KL, 쉬운 구간 pass@1, self_NLL_drift | 13 RL's Razor, 14 Retaining by Doing |
| 다양성이 줄기 시작하면 멈춘다 | pass@8, entropy, 서로 다른 답의 수 | 4 Does RL Incentivize |
| zero-shot이 강하면 SFT를 생략한다 | 답만 내는 대조군 대비 위치 | 2 To CoT or not to CoT, 12 SFT Memorizes |

### Q2. 상황별 학습 방법

| 상황 | 방법 | 근거 논문 |
|---|---|---|
| 맞았다 틀렸다 하는 샘플(0<p<1)이 많고 sharpen_gap이 큼 | GRPO / DAPO | 22 GRPO, 24 DAPO, 25 Dr. GRPO |
| zero_adv_frac이 높음 (rollout 낭비) | dynamic sampling, 난이도 필터링 | 24 DAPO, 26 Difficulty Filtering, 33 Too Correct to Learn |
| 같은 오답을 반복하는 샘플(confident_wrong)이 많음 | DPO → iterative / on-policy DPO | 18 DPO, 20 On-Policy Preference FT, 21 Iterative RPO |
| 한 번도 못 맞히는 샘플(p=0)이 많음 | 라벨 검수 후 off-policy guidance | 27 LUFFY |
| 좋은 teacher가 있는데 SFT가 성능을 깎음 | on-policy distillation | 29 GKD, 31 Thinking Machines OPD, 34 Rethinking OPD |
| RL 중 pass@k가 떨어짐 | pass@k 보상 | 28 Pass@k Training |

### Q3. 학습 중 반드시 볼 메트릭

| 메트릭 | 왜 보나 | 근거 논문 |
|---|---|---|
| pass@1 / MV@k / pass@k | 분포가 날카로워진 것인지, 탐색이 줄어든 것인지 구분 | 1 Self-Consistency, 4 Does RL Incentivize, 33 Too Correct to Learn |
| base 대비 KL | forgetting 예측 | 13 RL's Razor |
| 응답 길이 | 길이 편향 감지 | 25 Dr. GRPO |
| think 제거·오염 실험 | 추론이 장식인지 확인 | 3 CoT Faithfulness |

### Q4. 왜 on-policy인가

| 단계 | 논지 | 근거 논문 |
|---|---|---|
| ① 증상 | 모르는 것·따라갈 수 없는 것을 SFT로 강요하면 해롭다 | 7 New Knowledge, 8 Small Models Struggle |
| ② 완화 | 학생 분포에 맞춘 데이터가 낫다 | 9 I Learn Better, 10 SDFT |
| ③ 망각 | on-policy 학습이 기존 능력을 덜 잃는다 | 13 RL's Razor, 14 Retaining by Doing |
| ④ 선호 학습 | DPO에서도 같은 원리가 적용된다 | 19 DPO vs PPO, 20 On-Policy Preference FT |
| ⑤ 해법 | teacher 지식을 on-policy로 옮긴다 | 29 GKD, 31 Thinking Machines OPD, 34 Rethinking OPD, 35 OPD Survey |
