# Self-Distillation Enables Continual Learning

> **Venue**: arXiv preprint (2026.01.27)
> **Authors**: Idan Shenfeld, Mehul Damani, Jonas Hübotter, Pulkit Agrawal (MIT / Improbable AI Lab / ETH Zurich)
> **arXiv**: [2601.19897v1](https://arxiv.org/abs/2601.19897)
> **Code / Data**: http://idanshenfeld.com/SDFT

**한 줄 정의**: demonstration을 문맥으로 받은 **자기 자신**을 teacher로 삼아, reward 함수 없이 demonstration만으로 on-policy 학습 신호를 만드는 self-distillation (SDFT).
SFT가 본질적으로 off-policy라서 생기는 파괴적 망각을, reward engineering 없이 제거한다.

---

## 1. Background

### Continual Learning의 현황

- foundation model은 배포 이후 **정적(static)** 이다. retrieval·prompting으로 추론 시 행동은 바꿀 수 있지만 **파라미터를 갱신해 새 스킬·지식을 내재화하지는 못한다.**
- 최근 연구가 반복 확인한 사실: **모델이 자기 현재 정책으로 생성한 데이터로 학습하면(on-policy) off-policy 대비 파괴적 망각이 크게 줄어든다.** (Shenfeld et al. 2025 "RL's Razor", Chen et al. 2025 "Retaining by Doing", Lai et al. 2025)
- 그런데 지금까지 성공한 on-policy 방법은 **거의 전부 RL**이고, RL은 **명시적 reward 함수**를 요구한다. 현실의 지속 학습 데이터는 대부분 **expert demonstration** 형태이지 reward가 아니다.

> 이 논문의 질문: **demonstration만 있을 때 on-policy 학습의 이점을 어떻게 얻는가?**

### 기존 방법의 한계

| 방법 | 학습 신호 | 분포 | 필요한 것 | 문제점 |
|---|---|---|---|---|
| **SFT** | demonstration에 cross-entropy | off-policy (expert 궤적) | demonstration | **파괴적 망각**, compounding error |
| **on-policy RL** | scalar reward | on-policy | **명시적 reward 함수** | 현실 데이터에는 reward가 없다 |
| **IRL** | demonstration에서 reward를 역추론 | on-policy (2단계) | reward 구조의 강한 사전가정 | max-ent / adversarial / preference 가정 없이는 ill-posed, 대규모 상태공간 확장 실패 |
| **Context Distillation** | 문맥 조건 teacher를 모방 | **off-policy** (고정 문맥 few-shot) | 고정 prompt prefix | 정적 문맥의 prompt 압축에 그침, 오류 교정 불가 |
| **LoRA / re-invoke 류 보정** | 파라미터 제약 or 사후 복구 | — | — | 잃은 능력을 **부분적으로만** 복구 |
| **SDFT (본 논문)** | **demonstration 조건 teacher와의 reverse KL** | **on-policy** | **demonstration만** | ICL이 약한 모델에는 적용 불가 |

- SFT의 근본 문제는 **fixed offline data distribution 위에서 expert action을 모방**한다는 점이다. sequential SFT는 도메인이 바뀔 때 일반화 저하와 심각한 망각을 낳는다 (Kirkpatrick et al. 2017, Li & Hoiem 2017). 이 논문은 IRL 노선을 명시적으로 포기하고, reward 구조 사전가정의 자리를 **모델의 in-context learning 능력**으로 대체한다.

---

## 2. Motivation

### 핵심 통찰 1: SFT는 본질적으로 off-policy다

- SFT는 학습 내내 **expert가 방문한 상태**에서만 라벨을 받는다. 추론 시 student가 스스로 만든 prefix는 학습에서 본 적 없는 상태다 → compounding error (Ross et al. 2011).
- continual learning 관점의 대가는 여기서 나온다 — 파라미터가 기존 분포에서 크게 이탈하면서 이전 능력이 무너진다. 실측: **SFT 모델의 base policy 대비 KL = 1.26 nats** (뒤의 demonstration 조건 teacher 0.68 nats와 대비된다).

### 핵심 통찰 2: demonstration을 문맥에 넣은 모델은 "이미 학습이 끝난 자기 자신"에 가깝다

- 실측 (Qwen2.5-7B-Instruct, ToolAlpaca): demonstration 없는 base는 **42%**, demonstration을 문맥으로 준 teacher는 **100%** test reward. teacher 추론 trace **50개를 수동 검사**한 결과 최종 tool call뿐 아니라 **중간 chain-of-thought도 타당하고 의미적으로 근거 있음** — demonstration을 베끼는 것이 아니라 **올바른 추론 과정을 재구성**한다.
- 동시에 그 teacher는 base에서 멀리 가지 않는다 — base 대비 KL **0.68 nats**로 SFT 모델(1.26)의 **약 절반**.

> **핵심**: demonstration-conditioned 모델은 "높은 성능"과 "base policy 근접"을 **동시에** 만족한다. 이것이 trust-region 정식화가 요구하는 최적 정책의 두 조건 그 자체다.

### 핵심 통찰 3: 그러므로 reward 없이 on-policy RL을 할 수 있다

- trust-region RL의 최적 정책은 tilted distribution 형태의 닫힌 해를 갖는다. 뒤집으면 **reward = "최적 정책과 현재 정책의 log 비율"** 로 쓸 수 있고, 미지의 최적 정책 자리에 **demonstration 조건 모델**을 대입하면(In-Context Assumption) reward가 곧바로 계산된다.
- 결과적으로 SDFT는 **reward 함수 없는 on-policy RL**이자 동시에 **자기 자신을 teacher로 쓰는 on-policy distillation**이다. 두 관점이 수학적으로 같다.

---

## 3. Contributions

1. **SDFT 제안** — demonstration을 문맥으로 받은 동일 모델을 teacher로 삼는 on-policy self-distillation. 외부 teacher도, reward 함수도 필요 없다.
2. **Inverse RL 해석의 정식화** — self-distillation 목적함수가 demonstration과 모델의 ICL 능력이 정의하는 **암묵적 reward 최대화와 수학적으로 동치**임을 유도 (In-Context Assumption).
3. **ICL 가정의 경험적 검증** — Optimality(teacher ToolAlpaca 100%, base 42%)와 Minimal Deviation(base 대비 KL 0.68 vs SFT 1.26 nats)을 각각 측정.
4. **skill learning · knowledge acquisition 양쪽에서 SFT 능가** — 새 태스크 정확도와 이전 능력 보존을 **동시에** 개선하는 Pareto 우위. knowledge acquisition OOD 정확도 **80 → 98**.
5. **순차 학습에서 파괴적 망각 없는 스킬 누적 입증** — 3개 스킬을 순차 학습해도 이전 스킬이 유지된다. SFT는 진동(oscillation)만 반복.
6. **reasoning trace 없는 데이터로도 reasoning 모델 학습** — Olmo-3-7B-Think에 답만 있는 의료 데이터를 SFT하면 31.2 → 23.5로 붕괴하지만 SDFT는 **43.7**로 상승.

---

## 4. Method

### 4.1 SDFT — demonstration-conditioned 자기 teacher

하나의 모델 π를 **두 역할**로 쓴다. Student는 질문 x만 조건으로 받아 `P = π_θ(y | x)`, Teacher는 같은 모델이 질문 x와 demonstration c를 함께 받아 `Q = π(y | x, c)`.

teacher context는 다음 prompt로 구성한다 (논문 §3 원문).

```
<Question>
This is an example for a response to the question:
<Demonstration>
Now answer with a response of your own, including the thinking process:
```

이 프롬프트만으로 모델이 demonstration `c`를 **그대로 복창하는 것을 막고**, 의도를 이해한 자기 응답을 내놓게 만든다.

#### Training Objective

```
[목적함수 — reverse KL]
L(θ) = D_KL( π_θ(·|x) || π(·|x, c) )
     = E_{y ~ π_θ(y|x)} [ log ( π_θ(y|x) / π(y|x, c) ) ]

[token-level gradient estimator]  (V = token vocabulary, teacher 분포는 고정으로 취급)
∇_θ L(θ) = E_{y ~ π_θ} [ Σ_t Σ_{v ∈ V}
             log( π_θ(v | y_<t, x) / π(v | y_<t, x, c) ) · ∇_θ log π_θ(v | y_<t, x) ]
```

#### Inverse RL 유도 — 왜 이것이 reward 최대화인가

```
(1) trust-region RL:  π_{k+1} = argmax_π E_{y~π}[r(y,x)] − β · D_KL(π(·|x) || π_k(·|x))
(2) 닫힌 해:          π*_{k+1}(y|x) ∝ π_k(y|x) · exp( r(y,x) / β )
(3) 뒤집으면 reward:  r(y,x) = β [ log π*_{k+1}(y|x) − log π_k(y|x) ] + C
(4) In-Context Assumption — 미지의 최적 정책을 ICL로 근사:  π*_{k+1}(y|x) ≈ π(y|x, c)
(5) 암묵적 reward (β, C는 최적 정책을 바꾸지 않으므로 소거):
      r(y, x, c) = log π(y|x, c) − log π_k(y|x)
    token 단위 분해: r_t(y_t|y_<t,x,c) = log( π(y_t|y_<t,x,c) / π_k(y_t|y_<t,x) ),  Σ_t r_t = r(y,x,c)
(6) policy gradient:  ∇_θ J(π_k) = E_{y~π_k}[ log( π(y|x,c) / π_k(y|x) ) · ∇_θ log π_k(y|x) ]
      → 위 reverse KL gradient와 기대값에서 동치
```

> **해석**: SDFT는 "현재의 나"와 "demonstration을 본 더 현명한 나"를 비교해 추론한 reward를 최대화하는 on-policy RL이다.

**ICL 가정이 성립할 두 조건** (§3.2에서 각각 실측)

| 조건 | 내용 | 실측 |
|---|---|---|
| **Optimality** | demonstration 조건 정책의 기대 reward가 미지의 최적 정책과 비슷해야 한다 | ToolAlpaca teacher **100%** (base 42%), 추론 trace 50개 수동 검증 통과 |
| **Minimal Deviation** | 최적 reward를 내는 정책들 중 **현재 정책에 KL로 가장 가까운** 것이어야 한다. teacher가 demonstration을 그대로 베끼면 base에서 크게 이탈해 on-policy의 이점이 사라진다 | teacher **0.68 nats** vs SFT 모델 **1.26 nats** (약 1/2) |

**Teacher 가중치 선택** — teacher는 항상 demonstration에 조건화되지만, 가중치를 무엇으로 둘지는 별개의 설계 선택이다 (Appendix A.3, Figure 8).

| teacher 가중치 | 결과 |
|---|---|
| **frozen base θ₀** | 학습은 안정적이나 **일관되게 성능 미달** (≈65 정체) — student가 얻은 개선을 반영하지 못한다 |
| **현재 student θ** | **심각한 불안정 → 발산.** 약 500 generation 지점에서 **≈33까지 붕괴**. token 확률의 작은 요동이 on-policy 피드백 루프로 급속 증폭 |
| **EMA(θ)** ✅ | student를 추적하면서 고분산 업데이트를 평활화 → 안정적 학습 + 최고 성능 (**≈70**) |

```
Algorithm 1 (SDFT) 요약
φ ← θ                              # teacher 가중치 초기화
for each training step:
    B = {(x_i, c_i)} ~ D           # demonstration 데이터셋에서 미니배치
    for all (x_i, c_i) in parallel:
        s_i ← Ctx_S(x_i);  y_i ~ P_sample(· | s_i)   # 질문만 조건, on-policy rollout 1개
        t_i ← Ctx_T(x_i, c_i)                        # 질문 + demonstration
        ℓ^S_{i,t} = log π_θ(y_{i,t} | y_{i,<t}, s_i) ; ℓ^T_{i,t} = log π_φ(y_{i,t} | y_{i,<t}, t_i)
    g ← (1/B) Σ_i g_analytic({(ℓ^S, ℓ^T)}_t)   # 엔진(vLLM)-학습 차이는 importance sampling으로 보정
    θ ← θ − η g
    φ ← α θ + (1 − α) φ            # teacher EMA 갱신
```

#### 학습 vs 추론

| 단계 | 과정 |
|---|---|
| **학습** | student(질문만)가 rollout 생성 → 같은 모델의 EMA 복사본을 질문+demonstration으로 조건화해 teacher logprob 계산 → per-token reverse KL gradient로 student 갱신 → teacher EMA 갱신 |
| **추론** | student 단독. demonstration도, teacher도, 별도 모델도 배포에 관여하지 않는다 |

- **prompt당 rollout 1개**로 충분하다. GRPO류가 group 샘플링으로 상대 advantage를 추정해야 하는 것과 대조적이며, 감독이 token(또는 logit) 수준이라 trajectory-level advantage보다 credit assignment가 촘촘하다. 비용은 SFT 대비 **FLOPs 약 2.5배, wall-clock 약 4배**이지만, re-invoke처럼 SFT→복구의 다단계가 필요한 방법과 비교하면 **총 학습 시간은 오히려 줄어든다.**

---

## 5. Experiments

### 5.1 Setup / Dataset

| 축 | 태스크 | 데이터 | 규모 / 구성 | 평가 |
|---|---|---|---|---|
| **Skill Learning** | Science Q&A | SciKnowEval Chemistry L-3 | train 75% / val 5% / test 20%. demonstration은 **GPT-4o**로 prompt당 최대 8회 샘플링해 정답 일치 응답 1개 채택 → 학습셋 **100% 커버** | 객관식 exact match |
| | Tool Use | ToolAlpaca | 원 논문 train-test split, demonstration 원본 포함 | ground-truth API call 정규식 매칭 |
| | Medical | HuatuoGPT-o1 (stage 1 학습 / stage 2 평가) | 영어 문항 약 **20,000개** 학습, verifiable 문제에서 **1,000개** 평가 샘플링 | **GPT-5-mini** judge |
| **Knowledge Acquisition** | 2025 자연재해 | 직접 구축 Wikipedia 코퍼스 **약 200K 토큰** (2025 미얀마 지진, 캄차카 지진, 우타라칸드 홍수, 태풍 Kalmaegi, 열대폭풍 Wipha, 사이클론 Ditwah, 허리케인 Melissa, Kentwood Carson 토네이도, 2025.07 중앙 텍사스 홍수) | **GPT-5**로 문서당 100문항 QA 생성 → SFT 데이터셋이 원 코퍼스의 **약 5배**. 중복 문항 수동 검증 | GPT-5-mini judge (CORRECT / PARTIALLY_CORRECT / INCORRECT) |

- Skill Learning 태스크는 **의도적으로 Math·Coding을 피했다.** 이미 명시적으로 파인튜닝된 영역이 아니어야 continual learning의 이득이 드러나기 때문.
- 평가 축 3종: **In-Distribution Accuracy**(새 태스크 held-out. Knowledge는 Strict=모든 세부 정확 / Lenient=정확 정보 포함 + 틀린 진술 없음), **Previous Capabilities**(HellaSwag·TruthfulQA·MMLU·IFEval·Winogrande·HumanEval 평균 — 파괴적 망각 지표), **OOD Accuracy**(Knowledge만. 주입 지식에 의존하지만 직접 참조하지 않는 간접 질문, 예: "2025년에 어느 나라가 국제 인도적 지원을 필요로 했는가?" → 좁게 암기했는지 vs 내부 지식에 통합됐는지를 구분).

### 5.2 Implementation Details

| 항목 | 값 |
|---|---|
| Base model / 환경 | **Qwen2.5-7B-Instruct**, **NVIDIA H200 1장**, Hugging Face TRL, **full fine-tuning** |
| LR sweep | {5e-6, 1e-5, 5e-5} (CPT는 {1e-6, 5e-6, 1e-5}) |
| Optimizer / Scheduler | AdamW, cosine with warmup, warmup 10 step, max grad norm 1, weight decay 0, bfloat16 |
| Batch size sweep | {16, 32, 64} |
| Epochs | SDFT는 **다중 epoch 이득** — Skill 2 epoch, Knowledge 4 epoch 부근이 최적. **SFT는 1 epoch 이후 이득 없이 빠르게 overfit** |
| EMA α sweep / Max gen. length | {0.01, 0.02, 0.05} / Skill 2048, Knowledge 1024 |
| Rollout | **prompt당 1개**, analytic per-token KL estimator |
| 평가 세팅 | accuracy는 greedy(temp=0), pass@k는 temp=1.0 / top-p 0.95. **seed 3개 평균 + 95% CI**, LM Evaluation Harness |

**베이스라인**: **SFT**(표준) / **DFT**(Wu et al. 2025b — importance sampling으로 offline 데이터를 on-policy 샘플처럼 취급하는 근사 on-policy) / **SFT + Re-invoke**(Lu & Thinking Machines Lab, 2025 — **OPD 블로그의 방법.** SFT 후 범용 prompt에서 base policy를 teacher로 추가 on-policy distillation해 이전 능력 복구) / **CPT**(코퍼스 next-token prediction, Knowledge만) / **Oracle RAG**(항상 정답 문서를 주는 검색기, Knowledge 상한선).

### 5.3 Main Results — skill learning / knowledge acquisition

#### (a) Skill Learning — Table 5 전체 수치

Base (Qwen2.5-7B)의 이전 능력: HellaSwag 62.0 / HumanEval 65.8 / IFEval 74.3 / MMLU 71.7 / TruthfulQA 47.9 / Winogrande 71.1 → **평균 65.5**. 새 태스크 base 정확도는 Science Q&A 32.1, Tool Use 42.9, Medical 30.1.

| 태스크 | 방법 | New Task | HellaSwag | HumanEval | IFEval | MMLU | TruthfulQA | Winogrande | **Prior Avg** |
|---|---|---|---|---|---|---|---|---|---|
| **Science Q&A** | SFT | 66.2 | 55.0 | 54.8 | **35.3** | 64.6 | 36.8 | 73.7 | **53.4** |
| | SFT + Re-invoke | 66.0 | 61.6 | 63.4 | 52.9 | 68.7 | 45.2 | 70.0 | **60.2** |
| | DFT | 54.8 | 57.6 | 67.0 | 60.4 | 69.4 | 38.8 | 68.2 | **60.2** |
| | **SDFT (Ours)** | **70.2** | 60.9 | 68.9 | 66.8 | 70.7 | 46.5 | 73.1 | **64.5** |
| **Tool Use** | SFT | 63.2 | 57.3 | 50.0 | 49.8 | 70.2 | 37.5 | 73.1 | **56.0** |
| | SFT + Re-invoke | 63.1 | 61.7 | 68.9 | 59.1 | 71.5 | 49.1 | 71.6 | **63.7** |
| | DFT | 64.2 | 59.7 | 61.4 | 60.2 | 71.6 | 40.2 | 71.5 | **60.8** |
| | **SDFT (Ours)** | **70.6** | 61.6 | 68.3 | 71.9 | 71.5 | 47.3 | 71.7 | **65.4** |
| **Medical** | SFT | 35.5 | 59.5 | 62.1 | 56.6 | 70.5 | 39.8 | 72.9 | **60.2** |
| | SFT + Re-invoke | 35.6 | 61.5 | 63.1 | 67.6 | 70.0 | 42.3 | 71.4 | **62.6** |
| | DFT | 36.2 | 61.9 | 64.6 | 74.6 | 71.6 | 40.1 | 71.3 | **64.0** |
| | **SDFT (Ours)** | **40.2** | 61.4 | 67.7 | 72.3 | 71.5 | 47.3 | 71.9 | **65.4** |

> **Figure 1 / Figure 4 산점도의 수치 재현 (Tool Use 기준, x = New Task Accuracy / y = Prior Tasks Performance)**
> ```
> Prior
>  65.5 ┤ ● Base(42.9, 65.5)                     ● SDFT(70.6, 65.4)   ← 우상향
>  63.7 ┤                       ● SFT+Re-invoke(63.1, 63.7)
>  60.8 ┤                       ● DFT(64.2, 60.8)
>  56.0 ┤                       ● SFT(63.2, 56.0)                     ← 우하향
>       └───┬─────────┬─────────┬─────────┬─────────┬──── New Task
>          40        50        60        70        75
> ```
> SFT는 base에서 **오른쪽 아래로** 이동한다 — 새 태스크 +20.3점을 이전 능력 −9.5점과 맞바꾼다.
> SDFT는 **오른쪽으로만** 이동한다 — 새 태스크 +27.7점, 이전 능력 −0.1점.
> Science Q&A에서는 대비가 더 극적이다: SFT는 IFEval **74.3 → 35.3**(−39.0점) 붕괴, SDFT는 66.8로 유지.

- SDFT는 **세 태스크 전부에서 새 태스크 정확도 1위이면서 동시에 이전 능력 평균 1위**다(Pareto 우위). SFT 대비 새 태스크 **+4.0 / +7.4 / +4.7**점, 이전 능력 평균 **+11.1 / +9.4 / +5.2**점.
- Re-invoke는 잃은 능력을 **부분적으로만** 복구한다 (Science: 53.4 → 60.2, base 65.5에 6.7점 부족). DFT(근사 on-policy)는 SFT보다 망각이 적지만 여전히 눈에 띄는 저하가 남는다.
- 새 태스크 정확도 자체가 SFT보다 높다는 점이 중요하다 — 망각을 줄이려고 학습을 덜 한 것이 **아니다.** off-policy는 expert 궤적에서만 배우므로 test time에 미방문 상태로 밀려 compounding error를 겪는 반면, on-policy는 학습된 정책 자신이 유도한 상태 분포 위에서 배워 이 불일치를 피한다.

#### (b) Knowledge Acquisition — Table 1

| 방법 | Accuracy (strict) | Accuracy (lenient) | **OOD Accuracy** |
|---|---|---|---|
| Base | 0 | 0 | 0 |
| Oracle RAG (상한) | 91 | 100 | 100 |
| CPT | 9 | 37 | 7 |
| SFT | 80 | 95 | 80 |
| **SDFT (Ours)** | **89** | **100** | **98** |

- 지식이 pretraining에 없으므로 base는 **전 항목 0점**. **CPT는 사실상 실패**(strict 9, OOD 7) — 코퍼스 next-token prediction만으로는 질의응답 형태로 지식이 인출되지 않는다.
- SFT는 strict 80까지 오지만 **OOD 80에서 멈춘다.** SDFT는 strict 89 / lenient 100 / **OOD 98** 로 oracle RAG(91/100/100)를 거의 따라잡는다. **격차가 OOD에서 가장 크게 벌어진다(+18점)** — SFT는 특정 답변을 재생산하도록 가르칠 뿐 사실을 모델의 넓은 지식 베이스에 통합하지 못한다.

#### (c) pass@k — entropy collapse가 아니다

on-policy RL의 흔한 비판은 "새 능력 습득이 아니라 분포 샤프닝(entropy 감소)일 뿐"이라는 것이다(Yue et al. 2025). Skill Learning에서 **k = 1 ~ 128** 범위의 pass@k를 측정한 결과 SDFT의 이득은 **모든 k에서 균일하게 유지**된다(Figure 5 우측: 전 구간 Distillation > SFT > Base 순서 유지). 큰 k에서도 우위가 사라지지 않으므로 **진짜 스킬 습득**이다.

### 5.4 Sequential Learning — 스킬 순차 누적

**세팅**: 단일 모델을 **Tool Use → Science Q&A → Medical** 순으로 연속 학습. 성능은 태스크별 선형 정규화 — **0 = base model 정확도, 1 = 두 알고리즘을 통틀어 얻은 최대 정확도.**

**Figure 3에서 읽은 정규화 성능 (그래프 판독 근사값)**

| 학습 단계 | 지표 | **SDFT** | **SFT** |
|---|---|---|---|
| **Phase 1 (~0–250 step): Tool Use 학습** | Tool Use | 0 → **≈0.90** | 0 → ≈0.85 |
| | Science Q&A / Medical | ≈0.0 / ≈0.05 (미학습) | ≈0.0 / **≈ −0.3 (학습 전부터 이미 저하)** |
| **Phase 2 (Science Q&A 학습)** | Tool Use | **≈0.85–0.95 유지** | **≈0.85 → ≈ −0.35 붕괴** |
| | Science Q&A | 0 → **≈0.85** | 0 → ≈0.85 |
| **Phase 3 (Medical 학습)** | Tool Use | **≈0.90 유지** | ≈ −0.35 ~ −0.5 (회복 없음) |
| | Science Q&A | **≈0.85 유지** | **≈0.85 → ≈ −0.4 붕괴** |
| | Medical | 0 → **≈0.90** | 0 → ≈0.55 |
| **총 gradient step** | | 약 **700** | 약 870 |

- **SDFT**: 각 phase에서 새 스킬 곡선만 올라가고 이미 올라간 곡선은 **평평하게 유지**된다. 세 곡선이 최종적으로 모두 0.85~0.95 구간에 함께 머문다 → **진짜 누적(cumulative)**. 마지막 단계에서 Medical을 **≈0.90**까지 올리는데, SFT는 같은 태스크를 **≈0.55**에서 멈춘다 — 이전 태스크의 간섭이 새 태스크 학습까지 방해한다.
- **SFT**: 곡선들이 **서로 밀어낸다.** 새 태스크를 시작하는 순간 직전 태스크가 base 이하(음수 정규화 성능)로 급락하고 돌아오지 않는다. 학습이 아니라 **진동(oscillatory behavior)** 이다. 붕괴 폭은 정규화 기준 **약 1.2~1.25**(0.85 → −0.35~−0.4) — 얻은 것보다 **더 많이** 잃어 base model보다도 나빠진다.

> 이 실험이 논문 제목의 근거다. **여러 스킬을 순차 누적하되 이전 스킬과 사전 일반 능력 양쪽에서 퇴행이 없다**는 것을 한 모델로 보였다.

**보너스 — reasoning trace 없는 데이터 (Table 2)**: reasoning 모델 post-training은 중간 추론 trace를 요구하는데 현실 데이터는 최종 답만 있는 경우가 많다. Olmo-3-7B-Think + 답만 있는 의료 데이터.

| 방법 | Accuracy | 평균 생성 토큰 수 |
|---|---|---|
| Olmo-3-7B-Think (base) | 31.2 | 4612 |
| + SFT | **23.5** | **3273** |
| + **SDFT (Ours)** | **43.7** | 4180 |

SFT는 정확도를 **31.2 → 23.5**로 떨어뜨리며 응답 길이도 4612 → 3273으로 급감시킨다 — **추론 깊이의 붕괴(collapse in reasoning depth)**. SDFT는 **43.7**(+12.5점)로 올리면서 토큰 수 4180을 유지한다. teacher가 같은 모델에서 파생되므로 감독 신호가 **모델 고유의 CoT 스타일을 보존**하기 때문이다.

### 5.5 Ablation Study

#### (a) on-policy가 정말 필요한가 — teacher 품질만으로 설명되나 (§4.6, Figure 6, Tool Use)

같은 teacher를 쓰되 학습 방식만 바꾼 3가지 비교. 약 2,000+ generation 지점의 정확도.

| 방식 | 정확도 |
|---|---|
| **SDFT (on-policy distillation)** | **≈70** |
| Offline distillation (teacher 생성 고정 데이터셋에 KL loss) | ≈63 |
| SFT from Teacher (teacher 샘플을 정답으로 SFT) | ≈61 |

teacher 출력으로 SFT만 해도 표준 SFT보다는 낫다 → teacher 품질은 실제로 좋다. 그러나 **offline 두 방식 모두 on-policy SDFT에 일관되게 미달**한다. 이득이 teacher 품질만으로 설명되지 않으며 **on-policy 학습 자체가 필수 성분**이다.

#### (b) demonstration 조건화 방식 (Appendix A.2, Knowledge Acquisition)

on-policy 절차는 고정한 채 teacher context 구성만 바꿈 — **Only Answers 37% / Only Text 75% / Text and Answers 89%** (strict accuracy). 최근 self-distillation 계열 지식 주입 연구(Eyuboglu et al. 2025, Kujanpää et al. 2025)는 **원문 코퍼스만 문맥으로 쓰고 offline distillation**을 하는데, 이 논문은 두 축에서 다르다 — (i) 원문 + **작성된 답변(worked answer)** 을 함께 조건화, (ii) **on-policy** distillation. text-only 대비 **+14점**으로, 답변 조건화가 student에게 훨씬 강한 안내를 제공한다.

#### (c) KL gradient estimator (Appendix A.1)

sequence-level KL의 gradient는 π_θ가 샘플링 분포와 log 안쪽 양쪽에 등장해 미분이 비자명하다. 세 estimator를 모두 ablation.

| Estimator | 성질 | 결과 |
|---|---|---|
| **Token-level (partial)** — 샘플된 토큰만 사용 | sequence-level KL의 **부분 도함수**, 초기 토큰이 후속 분포에 미치는 영향 무시 → **편향** | 분산이 크고 **KL 제어가 약함** |
| **Full analytic per-token** — 매 시점 vocabulary 전체로 해석적 계산 | sample-based보다 **분산이 확실히 낮음**, sequence 수준에서는 여전히 편향. forward pass 산출물 재활용 | ✅ **가장 안정적 + 최고 성능 → 전 실험 채택** |
| **Rao-Blackwellized** (Amini et al. 2025) | next-token 분포는 해석적 적분, prefix는 MC → **KL·gradient 모두 unbiased**, 증명 가능한 저분산 | 추가 복잡도 대비 **측정 가능한 이득 없음** |

prompt당 궤적 수를 늘리는 것도 이론상 분산을 줄이지만 **실측 개선은 미미하고 compute만 크게 증가** → **prompt당 단일 궤적** 채택.

#### (d) 모델 크기 (§4.4, Figure 5 좌측, Science Q&A)

SDFT의 신호는 ICL 능력에서 나오므로 규모가 커질수록 유리해야 한다. **SDFT − SFT: 3B −3.3 / 7B +4.0 / 14B +6.9.** 3B는 ICL이 너무 약해 의미 있는 teacher 안내를 만들지 못한다. **단조 증가 추세**는 SDFT의 효과가 in-context reasoning 능력에 직결됨을 보여준다.

#### (e) teacher 가중치 및 알려진 실패 모드

- teacher 가중치 ablation은 §4.1 표 참조 — frozen base는 정체(≈65), 현재 student 자신은 발산(≈33), **EMA만 ≈70에 안정 도달**. **적용 한계**로는, 비추론 모델을 명시적 CoT 생성 모델로 바꾸는 것처럼 **생성 패턴의 근본적 전환**이 필요한 적응은 SDFT로 어렵다고 저자들이 명시한다.
- **learned artifacts**: teacher가 demonstration/원문에 조건화되어 있으므로 `"Based on the text..."`, `"Following the example..."` 같은 문구를 앞에 붙인다. student는 그런 문맥을 받지 않는데도 이 marker를 **teacher 출력 분포의 일부로 학습**해 재생산한다. **처음 몇 토큰의 loss를 마스킹**하면 downstream 정확도 손실 없이 억제되지만, 저자들도 **heuristic fix**라고 명시하며 원리적 해법은 미해결로 남긴다.

---

## 6. Key Takeaways

1. **SFT의 파괴적 망각은 "off-policy이기 때문"이지 데이터 부족 때문이 아니다.** 같은 demonstration을 쓰고 학습 방식만 on-policy로 바꾼 SDFT는 Tool Use에서 **새 태스크 63.2 → 70.6, 이전 능력 평균 56.0 → 65.4**(base 65.5)로 두 축을 동시에 개선한다. trade-off가 아니라 **잘못된 학습 방식의 부산물**이었다.
2. **in-context learning이 reward 함수를 대체한다.** trust-region RL의 최적 정책 자리에 demonstration 조건 모델을 대입하면 reward가 `log π(y|x,c) − log π_k(y|x)`라는 닫힌 형태로 나온다. IRL이 요구하던 reward 구조 사전가정을 **모델 자신의 ICL 능력**으로 갈음한다.
3. **"좋은 teacher"의 조건은 성능만이 아니라 base policy 근접성이다.** demonstration 조건 teacher는 ToolAlpaca **100%**(base 42%)를 달성하면서 base 대비 KL이 **0.68 nats** — SFT 모델(**1.26 nats**)의 약 **절반**. 이 두 조건의 동시 만족이 SDFT가 작동하는 이유다.
4. **지식은 암기되는 것이 아니라 통합되어야 한다.** SFT는 strict 80 / OOD 80에서 멈추지만 SDFT는 **strict 89 / lenient 100 / OOD 98** 로 oracle RAG(91/100/100)에 근접한다. **OOD +18점** 격차가 SFT의 한계를 드러낸다.
5. **순차 학습에서 스킬이 실제로 누적된다.** Tool Use → Science → Medical 3단계에서 SDFT는 세 곡선이 모두 정규화 성능 **0.85~0.95**에 함께 도달·유지한다. SFT는 다음 태스크를 시작하는 순간 직전 스킬이 **0.85 → −0.35~−0.4**로 붕괴해 회복하지 않으며, 마지막 Medical조차 **0.55**에서 멈춘다.
6. **효과는 모델 규모에 단조 증가하며 3B에서는 오히려 손해다.** Science Q&A 기준 SDFT−SFT는 **3B −3.3 / 7B +4.0 / 14B +6.9**. 신호가 ICL에서 나오므로 **적용 가능 여부를 먼저 판단해야 하는 방법**이다.
7. **on-policy 자체가 필수 성분이다 — teacher 품질만으로는 설명되지 않는다.** 같은 teacher로 offline distillation(**≈63**)이나 teacher 샘플 SFT(**≈61**)를 해도 on-policy SDFT(**≈70**)에 일관되게 미달한다. 비용은 SFT 대비 FLOPs **2.5배** / wall-clock **4배**이지만, SFT→복구 다단계 파이프라인 대비로는 총 시간이 오히려 짧다.

---

## 7. 원문 블로그 대비 갱신점

| # | 원문(2025.10)의 주장 | 이 논문의 처리 | 근거 |
|---|---|---|---|
| ① | **OPD = dense × on-policy** | **계승 + 확장.** 같은 per-token reverse KL 골격을 쓰되 원문이 전제한 "외부 teacher"를 제거하고 **teacher를 문맥으로 합성**한다. 나아가 이것이 IRL의 암묵적 reward 최대화와 동치임을 유도 — OPD를 **reward-free 학습 도구**로 재해석 | Eq. 1~6, In-Context Assumption |
| ② | **teacher는 별도의 더 강한 모델이어야 유리하다** (Qwen3-8B ← Qwen3-32B) | **부분 반박.** 별도 모델이 전혀 필요 없다. **동일 모델 + demonstration 문맥**만으로 유효한 teacher가 만들어진다. 단 모델의 ICL이 충분히 강해야 하며 **3B에서는 SFT보다 −3.3점 손해** | Figure 5(좌), §3.2 (42% → 100%) |
| ③ | **모델 자신의 샘플로 SFT하면 이론상 KL=0인데도 열화된다 — "0보다 큰 어떤 학습률에서도"** | **정교화.** SDFT의 teacher도 "자기 자신"이지만 **문맥이 다르므로 분포가 다르다**(KL ≠ 0) — 여기서 학습 신호가 나온다. 반대로 **문맥 조건 없이 현재 student를 그대로 teacher로 쓰면 학습이 발산**한다(≈500 generation에서 33까지 붕괴). 즉 self-distillation이 성립하려면 **teacher–student 간 정보 비대칭(demonstration 조건화)** 과 **EMA 평활화**가 필수. 블로그의 경고가 구체적 실패 모드로 재현됐다 | Appendix A.3, Figure 8 |
| ④ | **teacher가 고정된 OPD는 항상 on-policy로 남아 continual learning에 유망하다 (제안 수준)** | **✅ 이 논문의 핵심 — 실증. 다만 "고정"은 틀렸다.** 3스킬 순차 학습에서 SDFT는 퇴행 없이 누적한다. 그러나 **teacher를 frozen으로 두면 student의 진전을 추적하지 못해 성능이 정체**한다 — **EMA로 천천히 따라가는 teacher**가 최적. 블로그의 "고정 teacher" 처방은 **EMA teacher로 수정**되어야 한다 | Figure 3, Appendix A.3 |
| ⑤ | **mid-training으로 무너진 행동을 OPD로 복구할 수 있다 (IF-eval 79% → 83%)** | **검증 + 한계 규명.** 블로그의 복구 절차가 **"SFT + Re-invoke" 베이스라인으로 직접 구현·비교**됐다(Lu & Thinking Machines Lab, 2025로 인용). 복구는 실제로 된다 — Science Q&A 이전 능력 평균 **53.4 → 60.2**, IFEval **35.3 → 52.9**. 그러나 **base의 65.5 / 74.3에는 못 미친다.** SDFT는 사후 복구 단계 없이 **한 번에 64.5 / 66.8** | Table 5 |
| ⑥ | **복구 단계에 도메인 문서를 전혀 쓰지 않아도 지식이 유지된다** | **구조가 다르다.** 블로그는 "① 도메인 mid-training → ② 무관한 chat prompt로 복구"의 **2단계**. SDFT는 **도메인 demonstration을 teacher 문맥에 넣어 1단계**로 처리한다. 오히려 **문맥에 무엇을 넣느냐가 결정적**임을 보였다 — 답만 37%, 원문만 75%, **원문+답 89%** | Appendix A.2 |

**블로그가 다루지 않았던 영역**

| 항목 | 이 논문의 기여 |
|---|---|
| **reward도 teacher 모델도 없는 도메인** | 블로그의 OPD는 teacher 모델의 존재를 전제한다. SDFT는 **demonstration 데이터셋만 있으면** 성립하도록 요구사항을 낮춘다 |
| **지식 주입** | 블로그는 행동(instruction following) 복구만 다뤘다. SDFT는 **새 사실 주입** 축을 추가하고 OOD 통합까지 측정 (SFT 80 → SDFT 98). 반대 방향의 한계도 명시 — **생성 패턴의 근본적 전환(비추론 모델 → 명시적 CoT)은 SDFT로 어렵다** |
| **gradient estimator** | 블로그의 `advantages = -reverse_kl`은 sampled-token 추정. 이 논문은 세 estimator를 ablation해 **full analytic per-token**을 권고 (token-level은 분산이 크고 KL 제어가 약함) — Revisiting OPD(2603.25562)의 truncated reverse-KL 처방과 같은 방향 |
| **rollout 수 / 비용** | 블로그는 prompt당 4 샘플. 이 논문은 **prompt당 1 rollout**으로 충분함을 실증. 비용은 SFT 대비 FLOPs **2.5배** / wall-clock **4배**이나 다단계 파이프라인 대비로는 총 시간 단축 |
| **실패 모드** | learned artifacts("Based on the text...") 오염 규명 + 첫 토큰 loss masking이라는 heuristic 대응 |

> **한 줄 정리**: 블로그의 ④(제안 수준)를 이 논문이 실증하면서, 그 과정에서 **② teacher는 별도 모델일 필요가 없고**, **④의 "고정 teacher"는 EMA teacher여야 하며**, **⑤의 사후 복구는 애초에 필요 없다**는 세 가지 수정을 함께 내놓았다.

---

[← 후속 연구 정리](opd_follow_up_research.md) · [원문 요약](on_policy_distillation.md)
