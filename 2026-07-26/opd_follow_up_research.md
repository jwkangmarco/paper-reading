# On-Policy Distillation — 후속 연구 정리 및 읽기 순서

> **기준일**: 2026.07.26
> **원 출처**: [Thinking Machines Lab — On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) (2025.10.27)
> **원문 요약**: [`on_policy_distillation.md`](./on_policy_distillation.md)

---

## 0. 조사 방법과 신뢰도

표기 규칙:

| 표기 | 의미 |
|---|---|
| 📄 | **PDF 본문 전체를 읽고 별도 요약 노트를 작성함** (필독 6편) |
| ✅ | arXiv abs 페이지에서 제목·저자·투고일·초록을 직접 확인함 |
| ▫️ | 큐레이션 저장소 목록에 존재하나 초록 미확인 |

> ⚠️ **주의**: 이 분야는 2026년 상반기에 논문 수가 폭증했고, 그 때문에 웹 검색 요약이
> **존재하지 않는 arXiv ID를 그럴듯하게 생성**하는 경우가 관측되었다.
> ▫️ 항목은 인용 전 반드시 `arxiv.org/abs/{id}` 로 실재를 확인할 것.

### 1차 소스

| 소스 | 링크 | 비고 |
|---|---|---|
| 큐레이션 저장소 | [chrisliu298/awesome-on-policy-distillation](https://github.com/chrisliu298/awesome-on-policy-distillation) | 300편+ 를 8개 카테고리로 분류 |
| 전용 서베이 | [arXiv:2604.00626](https://arxiv.org/abs/2604.00626) | Song & Zheng, 2026.04 (v4 06.18) |
| 수식 중심 서베이 | [arXiv:2606.22793](https://arxiv.org/abs/2606.22793) | B. Zhang, 2026.06 |

---

## 1. 필독 6편 — 이 순서대로

### 1️⃣ Learning beyond Teacher: Generalized OPD with Reward Extrapolation
**📄 [요약 노트](./exopd_learning_beyond_teacher.md)** · [arXiv:2602.12125v2](https://arxiv.org/abs/2602.12125)
Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, Yankai Lin
(Gaoling School of AI, Renmin University of China / LLM Department, Tencent) · 2026.02.26
Code: `github.com/RUCBM/G-OPD`

- **역할**: 이론 골격
- **핵심 정리**: OPD는 **"reward 함수와 KL 정규화가 항상 1:1(β=1)로 고정되고 reference model은 임의 선택 가능한 dense KL-constrained RL"** 의 특수 케이스
- **⚠️ 명칭 구분**
  - **G-OPD** = *프레임워크*. 유연한 reference model π_ref + reward scaling factor λ(≡1/β)를 노출한 일반화 정식화
  - **ExOPD** = *방법*. λ > 1로 두는 "reward extrapolation" 인스턴스 (논문 전체에서 **λ = 1.25 고정**)
- **최적해**: `log π_θ = log π* + (λ−1)(log π* − log π_ref)` — λ>1은 teacher 개선 방향을 **연장**해 teacher를 넘는다
- **결과**: same-size single-teacher 수학 평균 **48.0** vs teacher **46.0** (표준 OPD 46.5). multi-teacher 병합에서 수학 4 + 코드 3 벤치마크 **전부** domain teacher 초과 — SFT·ExPO·OPD 중 유일
- **strong-to-weak**: OPD 대비 1.7B **+2.3** / 4B **+2.7**. *reward correction*(π_ref = teacher의 pre-RL base)으로 math 28.1→28.7, code 51.3→52.3 추가 향상 (단 pre-RL 체크포인트 접근 + 큰 모델 logprob 비용)
- **λ의 양면성**: 0<λ<1은 interpolation(budget-controlled reasoning), **λ=1.5는 implicit reward hacking으로 붕괴**
- **왜 1번인가**: 원문 블로그가 각주로만 언급한 "RL 자체가 sequence-level reverse KL 계열을 최적화한다"를 정식화. 이 정리를 손에 쥐면 이후 논문들이 전부 **"그 1:1 결합을 어디서 어떻게 푸느냐"의 변주**로 읽힌다.

### 2️⃣ Rethinking OPD of LLMs: Phenomenology, Mechanism, and Recipe
**📄 [요약 노트](./rethinking_opd.md)** · [arXiv:2604.13016v2](https://arxiv.org/abs/2604.13016)
Yaxuan Li, Yuxin Zuo, Bingxiang He (공동 1저자) 외 8인
(Tsinghua / ShanghaiTech / UIUC / Renmin University of China) · 2026.04.15
Code: `github.com/thunlp/OPD`

- **역할**: 성립 조건 규명 — "언제 되는가"
- **성공 조건 2가지 (서로를 대신하지 못한다)**
  1. student와 teacher의 **thinking pattern 호환성**
  2. thinking pattern이 맞고 점수가 높더라도, teacher는 **student가 학습 중 본 것을 넘어서는 새 능력**을 제공해야 함
- **결정적 증거**: same-pipeline teacher는 gap recovery **5.3%**(DS-7B) / **15.6%**(Qwen3-4B)인 반면 RL post-trained teacher는 **16.9%** / **58.6%**. 그런데 후자는 **초기 overlap이 오히려 낮다** (71.5% vs 74.2%, 70.3% vs 75.7%)
- **reverse distillation**: 같은 계열 1.5B/7B teacher는 student 관점에서 분포적으로 구분 불가 — 둘 다 0.54 → 약 0.30으로 동일 퇴행
- **토큰 수준 메커니즘**: overlap ratio 72% → 91% 이상으로 점진 정렬, 그 공유 토큰 집합이 확률질량의 **97~99%** 를 차지. Overlap Top-k만 최적화해도 Student Top-k를 재현 → **인과성 확인**
- **복구 레시피 2종**: off-policy cold start (AMC'23 0.26→0.40, 최종 상한 자체 상승), teacher-aligned prompt selection (템플릿 한 줄로 gap recovery 80%→85%)
- **⚠️ long-horizon 경고**: teacher continuation 이득이 prefix 1K에서 **+0.3659** → 16K에서 **+0.0237** 로 소멸. 3K~7K가 sweet spot이고 10K/15K에서는 step 200~220에 붕괴
- **왜 2번인가**: teacher 선정 단계에서 반드시 필요. 블로그의 Qwen3-8B ← Qwen3-32B 성공을 아무 모델 쌍에나 적용하면 안 되는 이유를 준다.

### 3️⃣ Revisiting OPD: Empirical Failure Modes and Simple Fixes
**📄 [요약 노트](./revisiting_opd.md)** · [arXiv:2603.25562v2](https://arxiv.org/abs/2603.25562)
Yuqian Fu, Haohuan Huang (공동 1저자), Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, Dongbin Zhao
(State Key Lab of Multimodal AI Systems CASIA / School of AI UCAS / Fudan / Independent) · 2026.04.27

- **역할**: 구현 직전 필독
- **이론**: token-level OPD는 sequence-level reverse-KL에 대해 **biased**(버려지는 항 = future-reward coupling)이지만, worst-case 분산 상한이 **O(T²) vs O(T⁴)** — long-horizon에서 token-level을 쓰는 것에 정식 근거를 준다
- **synthetic study**: γ ∈ {0, 0.25, 0.5, 0.75, 1.0} 보간에서 γ↑ → variance 1~수 자릿수↑, γ=1.0은 수렴 실패
- **실패 모드 3종**
  1. 대다수 sampled token이 음의 reward라 **소수 filler token이 최적화를 지배**
  2. repetition loop에서도 teacher가 국소 동의 — teacher−student gap의 lower tail이 뒤쪽 position일수록 넓어짐
  3. **tokenizer / special-token mismatch** — 의미상 옳은 출력에 −19.16 / −58.71 벌점
- **처방**: teacher top-K(**K=32**) local support matching = 지지집합 내부 renormalized **truncated reverse-KL** + **top-p 0.9** rollout + special-token masking
- **결과**: multi-task 수학 평균 34.8 → **41.7 (+19.8%)**, ALFWorld 90.6 → 95.3 동반 상승. single-task 36.4 → 41.7, WebShop success 50.0 → 57.8
- **왜 3번인가**: 블로그의 `advantages = -reverse_kl` 한 줄을 그대로 쓰면 안 된다는 것이 이 논문의 메시지. renormalization 없으면 rapid collapse, top-p 없으면 top-K가 baseline보다 나빠진다(17.7 < 20.4).

### 4️⃣ Demystifying OPD: Length Inflation and Stabilization Strategies
**📄 [요약 노트](./demystifying_opd.md)** · [arXiv:2604.08527v1](https://arxiv.org/abs/2604.08527)
Feng Luo, Yu-Neng Chuang, Guanchu Wang, Zicheng Xu, Xiaotian Han, Tianyi Zhang, Vladimir Braverman
(Rice / UNC Charlotte / Johns Hopkins / Case Western Reserve) · 2026.04.09

- **역할**: 학습 안정성 — 돌리면 반드시 만나는 문제
- **인과 사슬**: **abrupt repetition saturation** → 반복 token이 reverse-KL advantage를 **4~9배** 크게 받음(붕괴 후 −0.02~−0.05 vs 일반 token −0.15~−0.17) → on-policy 업데이트가 자기강화 → **length inflation** → TruncRate ≈ 1 → biased gradient → MATH500 accuracy **0.72 → 0.60~0.63** 급락
- **전이 구간**: Figure 1 기준 step 약 **280~310의 30-step 창** 안에 완료되며 **회복이 없다**. 3개 student–teacher 조합 전부에서 재현
- **핵심 논증**: 이것은 GRPO/RLVR의 sequence-level length bias(Dr.GRPO/DAPO)와 **층위·원인·증폭 경로·성격이 모두 다르다.** token 수준 신호 편중 × on-policy 방문 빈도 증폭이며 재정규화로는 못 잡는다
- **처방** — **Stable-OPD** (표기 주의: `StableOPD` 아님): `L_OPD + λ_gold·L_SFT(mixture) + β_KL·KL(π_θ‖π_ref)`
- **결과**: 1.5B **28.9 → 36.1 (+7.2)**, 7B **43.8 → 47.6 (+3.8, 전 방법 1위)**
- **Ablation**: OPD 28.0 → +KL 29.7(+1.7) → +Mixture 35.7(+6.0) — 두 항이 인과 사슬의 **다른 고리**를 끊어 상보적
- **왜 4번인가**: 3번이 "신호가 틀리는 문제"라면 이건 "학습이 발산하는 문제". thinking budget이 있는 추론 모델을 OPD로 학습시키면 거의 확실히 겪는다.

> 💡 3번과 4번은 묶어서 읽어도 된다. 둘 다 "블로그 레시피가 실제로 어디서 터지나" 계열.

### 5️⃣ SOD: Step-wise On-policy Distillation for Small Language Model Agents
**📄 [요약 노트](./sod_stepwise_opd.md)** · [arXiv:2605.07725v1](https://arxiv.org/abs/2605.07725)
Qiyong Zhong, Mao Zheng, Mingyang Song (공동 1저자) 외 5인
(Zhejiang University / LLM Department Tencent / USTC / National University of Singapore) · 2026.05.08
Code: `github.com/YoungZ365/SOD`

- **역할**: SLM 에이전트 / tool-calling 도메인 착지
- **문제 진단 (이론 + 실측)**: tool observation이 만드는 **이산적 state transition** 때문에 divergence가 `Ω(m·η_tool)`로 점프하고 연속 오류에서 super-linear로 증폭(Prop. 1). 저-overlap 상태에서 OPD gradient의 **SNR이 0으로 붕괴**(Prop. 2). 실측: teacher entropy 0.85 → **2.14**, 마지막 step 토큰의 **78%** 가 H>1.0
- **step의 정의**: 두 tool observation 사이의 model response (observation 토큰 제외)
- **처방**: step마다 divergence `d_k`를 재고 `w_k = min(Π(d_u+ε)/(d_{u+1}+ε), 1+δ)` 로 재가중. **추가 forward pass 0회**, 하이퍼파라미터는 사실상 **δ=0.2 하나**
- **결과**: 차선 baseline 대비 **0.6B +20.86% / 1.7B +18.50%**. **0.6B가 AIME 2025 average@32 26.13%**. 1.7B는 4B teacher 성능의 **69.8%** 회수 (표준 OPD 58.9%)
- **비용**: 0.6B에서 오히려 **3.5% 빠름**. 최대 16 tool-call 턴, Python 코드 인터프리터 환경
- **왜 5번인가**: "SFT만 한 모델이 중간 turn에서 잘못된 tool call을 한 뒤 무너진다"는 실무 관찰을 정면으로 다룬다. 2번이 제기한 long-horizon 의문에 **구성적 답변**을 준다.

### 6️⃣ Self-Distillation Enables Continual Learning (SDFT)
**📄 [요약 노트](./sdft_continual_learning.md)** · [arXiv:2601.19897v1](https://arxiv.org/abs/2601.19897)
Idan Shenfeld, Mehul Damani, Jonas Hübotter, Pulkit Agrawal
(MIT / Improbable AI Lab / ETH Zurich) · 2026.01.27
Code: `idanshenfeld.com/SDFT`

- **역할**: 운영 단계 — 지속 학습
- **핵심 주장**: SFT의 파괴적 망각은 데이터 문제가 아니라 **off-policy이기 때문**이다. SDFT는 demonstration을 문맥으로 받은 **자기 자신**을 teacher로 삼아 **reward 함수 없이** on-policy 학습 신호를 만든다
- **이론**: 이 목적함수가 trust-region RL의 최적 정책 자리에 ICL 조건 모델을 대입한 **암묵적 reward 최대화(IRL)와 수학적으로 동치**. 가정 검증 — Optimality: ToolAlpaca 42%→**100%**, Minimal Deviation: base 대비 KL **0.68** vs SFT **1.26** nats
- **결과**: skill learning 3종에서 **새 태스크 정확도와 이전 능력 평균 양쪽 모두 1위** (Tool Use: 63.2→**70.6** / 56.0→**65.4**, base 65.5). knowledge acquisition OOD **80 → 98**. 3스킬 순차 학습에서 **퇴행 없는 누적**
- **⚠️ 블로그 대비 중요한 정정**: 블로그는 "**고정된** teacher"가 핵심이라 했지만, 실제로는 **frozen teacher는 정체(≈65), 현재 student 자신은 발산(≈33), EMA teacher만 안정(≈70)**
- **한계**: ICL 의존이라 **3B에서는 SFT보다 −3.3점** (7B +4.0, 14B +6.9). 비용 FLOPs 2.5× / wall-clock 4×
- **왜 6번인가**: 블로그가 마지막에 던지고 끝낸 "OPD는 continual learning에 유망하다"를 실제로 검증한 논문.

---

## 2. 상황별 2차 읽을거리

| 상황 | 논문 | 핵심 |
|---|---|---|
| **GPU 예산이 빡빡하다** | ✅ [Lightning OPD](https://arxiv.org/abs/2604.13010) (Wu, Han, Cai · MIT · 2026.04) | SFT rollout에 대해 teacher 확률을 사전계산해 오프라인화. **teacher consistency** 조건을 지켜야 하며 위반 시 gradient bias 발생. live teacher 서버 제거, **학습 효율 4배**. Qwen3-8B로 **30 GPU h에 AIME'24 69.9%**, Qwen3-30B-A3B로 71.0% |
| **teacher가 student보다 훨씬 크다** | ✅ [TGPO](https://arxiv.org/abs/2605.13230) (Liu et al. · 동북대·메이퇀 · 2026.05) | 정책 격차가 크면 RL 탐색이 teacher 분포 밖 궤적을 만들어 **피드백이 무의미**해진다. teacher가 student 문맥을 조건으로 직접 token 생성을 안내 + RLVR 궤적 보상 결합 |
| **Pass@k / 생성 다양성이 중요** | ✅ [Entropy-Aware OPD](https://arxiv.org/abs/2603.07079) (Jin, Min, Yang et al. · KAIST·IBM · 2026.03) | reverse KL의 mode-seeking이 **생성 다양성을 줄이고** teacher 엔트로피가 높을 때 신호가 불안정. 고엔트로피 토큰에서만 forward KL 혼합. Pass@8 기준 Qwen3-0.6B **+1.37** / 1.7B **+2.39** / 4B **+5.05** |
| **teacher가 폐쇄형 API뿐** | ✅ [Black-Box OPD (GAD)](https://arxiv.org/abs/2511.10643) (Ye, Dong, Chi, Wu, Huang, Wei · MSRA · 2025.11) | teacher 내부 접근 없이 student를 generator, discriminator를 적응형 reward model로 두는 GAN 구조. Qwen2.5-14B-Instruct가 LMSYS-Chat 자동평가에서 teacher인 **GPT-5-Chat과 대등** 수준 |
| **system prompt / 긴 문맥을 파라미터에 내재화** | ✅ [OPCD](https://arxiv.org/abs/2602.12275) (Ye, Dong, Wu, Huang, Wei · MSRA · 2026.02) | 문맥을 조건으로 받은 teacher 대비 divergence를 줄이며 student 자기 출력으로 학습. 해결 이력에서 전이 가능한 인사이트 추출, **system prompt 행동 내재화** |
| **LoRA를 쓸 건데 궁합이 궁금** | ✅ [On the Geometry of OPD](https://arxiv.org/abs/2606.07082) (Shen et al. · 2026.06) | OPD 업데이트는 SFT보다 **더 적은 weight를 건드리고 principal direction을 더 강하게 회피**. 초기에 누적 업데이트가 좁은 저차원 채널로 들어가는 **subspace locking**. 원문의 "LoRA 격차 13%→6%"를 기하학적으로 설명 |
| **자기 자신을 teacher로 (privileged info)** | ✅ [Self-Distilled Reasoner / OPSD](https://arxiv.org/abs/2601.18734) (Zhao, Xie, Liu et al. · UCLA·Meta · 2026.01) | 하나의 LLM이 문맥만 달리해 teacher·student를 겸함. teacher는 검증된 추론 trace 같은 privileged info를 조건으로 받고 student는 질문만 봄 |
| **인덱스가 필요할 때** | ✅ [서베이 2604.00626](https://arxiv.org/abs/2604.00626) | OPD를 **student-sampled trajectory 위의 f-divergence 최소화**로 형식화. exposure bias 오차를 **quadratic → linear**로 줄이는 것이 본질 |

### 읽지 않아도 되는 것

- **MiniLLM** ([2306.08543](https://arxiv.org/abs/2306.08543)), **GKD** ([2306.13649](https://arxiv.org/abs/2306.13649))
  → 원문 요약 §1의 3열 비교표에 핵심이 정리되어 있다. 필요할 때 해당 절만 참조.
- **서베이를 1번으로 읽는 것** → 이미 계보와 두 함정을 정리한 상태라면 landscape는 중복이다. **인덱스로만** 쓸 것.

---

## 3. 연구 지형 — 갈래별 개관

### 3.1 메커니즘 규명 — "왜 되는가"

| 논문 | 기여 |
|---|---|
| 📄 [Rethinking OPD](./rethinking_opd.md) | 성공 조건 2가지, overlap token 인과성, long-horizon 한계 |
| 📄 [ExOPD / G-OPD](./exopd_learning_beyond_teacher.md) | OPD = β=1 고정 dense KL-constrained RL, reward extrapolation |
| ✅ [Geometry of OPD](https://arxiv.org/abs/2606.07082) | subspace locking, 파라미터 공간 업데이트 기하 |

### 3.2 실패 모드 진단과 안정화 — 최다 논문 갈래

| 논문 | 기여 |
|---|---|
| 📄 [Revisiting OPD](./revisiting_opd.md) | 실패 모드 3종, O(T²) vs O(T⁴), truncated reverse-KL, +19.8% |
| 📄 [Demystifying OPD](./demystifying_opd.md) | abrupt repetition saturation → truncation collapse, Stable-OPD, +7.2 |
| ✅ [Entropy-Aware OPD](https://arxiv.org/abs/2603.07079) | mode-seeking의 다양성 손실, forward KL 혼합 |
| ✅ [TGPO](https://arxiv.org/abs/2605.13230) | 큰 정책 격차에서의 reverse KL 붕괴 |
| ▫️ [PowerOPD](https://arxiv.org/abs/2606.17199) · [The Extrapolation Cliff](https://arxiv.org/abs/2605.08737) · [Escaping the KL Agreement Trap](https://arxiv.org/abs/2606.09471) · [OPD Reduces Output Diversity](https://arxiv.org/abs/2606.26091) · [Denser ≠ Better](https://arxiv.org/abs/2607.01763) · [On the Position Bias of OPD](https://arxiv.org/abs/2606.22600) | 미확인 |

### 3.3 Self-Distillation (OPSD) — 저장소 기준 단일 최대 카테고리 (65편)

| 논문 | 기여 |
|---|---|
| 📄 [SDFT](./sdft_continual_learning.md) | demonstration-conditioned 자기 teacher, EMA, 스킬 순차 누적 |
| ✅ [Self-Distilled Reasoner (OPSD)](https://arxiv.org/abs/2601.18734) | privileged information 조건부 teacher, 단일 모델 |
| ▫️ [Why Does Self-Distillation (Sometimes) Degrade Reasoning](https://arxiv.org/abs/2603.24472) · [When Are Teacher Tokens Reliable?](https://arxiv.org/abs/2605.21606) · [UniSD](https://arxiv.org/abs/2605.06597) · [Preference-Based Self-Distillation](https://arxiv.org/abs/2605.05040) | 미확인 |

### 3.4 효율 · 시스템

| 논문 | 기여 |
|---|---|
| ✅ [Lightning OPD](https://arxiv.org/abs/2604.13010) | 오프라인화, teacher consistency, 4배 효율 |
| ▫️ [Are Full Rollouts Necessary?](https://arxiv.org/abs/2605.31490) · [Less is More](https://arxiv.org/abs/2605.27028) · [AsyncOPD](https://arxiv.org/abs/2606.24143) · [f-OPD](https://arxiv.org/abs/2605.17862) · [DP-OPD](https://arxiv.org/abs/2604.04461) | 미확인 |

### 3.5 Teacher 접근 제약 완화

| 논문 | 기여 |
|---|---|
| ✅ [Black-Box OPD (GAD)](https://arxiv.org/abs/2511.10643) | 내부 접근 없이 discriminator를 적응형 reward model로 |
| ▫️ [Breaking the Tokenizer Barrier](https://arxiv.org/abs/2606.09456) · [Cross-Tokenizer via Byte-Level](https://arxiv.org/abs/2604.07466) · [X-Token](https://arxiv.org/abs/2605.21699) · [DSKD](https://arxiv.org/abs/2504.11426) | 미확인 |

### 3.6 에이전트 · 멀티턴 · 문맥 내재화

| 논문 | 기여 |
|---|---|
| 📄 [SOD](./sod_stepwise_opd.md) | step-wise 재가중, tool call 오류 연쇄, SNR 붕괴 형식화 |
| ✅ [OPCD](https://arxiv.org/abs/2602.12275) | 문맥 지식의 파라미터 내재화 |
| ▫️ [OEL](https://arxiv.org/abs/2603.16856) · [PMD](https://arxiv.org/abs/2607.01480) · [Sample-Efficient Learning from Agent Experience](https://arxiv.org/abs/2607.21051) | 미확인 |

---

## 4. 원문 대비 무엇이 바뀌었나

6편 본문을 모두 읽고 정리한 결과. **동일 주장을 여러 논문이 독립적으로 반박한 항목이 가장 신뢰도가 높다.**

### 4.1 가장 강하게 뒤집힌 주장 두 가지

#### ❌ "reverse KL은 unhackable" — **4편이 독립적으로 반박**

원문은 *"reward가 teacher 분포와의 거리 그 자체라 프록시와 목표가 원리적으로 일치한다"* 고 했다. 그러나:

| 논문 | 반박 근거 |
|---|---|
| ExOPD | `log(π*/π_ref)`는 결국 proxy. λ=1.5에서 log-ratio peak를 공격적으로 fitting하는 hacking + length bias |
| Rethinking OPD | unhackable ≠ usable. 실패 teacher의 reward AUROC **0.75** > 성공 teacher **0.73** 인데도 gradient norm은 더 작다 |
| Revisiting OPD | Appendix H 전체가 reward-hacking 사례집 — over-continuation / `wait` 반복 / 깨진 비영어 출력에 teacher가 고확률 부여 |
| Demystifying OPD | 논문 §1이 직접 *"student repetition may exploit or hack the teacher's likelihood-based signal"* |

> **다만 성격이 다르다**: RLHF의 hacking은 프록시가 **틀려서** 생기고, OPD의 것은 프록시가 **맞는데도 국소 최적이 degenerate**해서 생긴다.

#### ❌ "discount 0 덕분에 안정화 장치가 불필요" — **3편이 반박**

| 논문 | 반박 근거 |
|---|---|
| Demystifying OPD | GRPO-style **clipping을 되살렸고**, mixture의 `λ_gold·L_SFT`는 MiniLLM LM loss와 기능적으로 동일한 앵커, `β_KL·KL(π_θ‖π_ref)`는 MiniLLM에도 없던 **새 제약** |
| Revisiting OPD | 절반은 강화(분산 근거를 O(T²) vs O(T⁴)로 승격), 절반은 반박(장치 불필요는 거짓) |
| Rethinking OPD | 10K/15K 길이에서 step 200~220 붕괴. Top-1 OPD도 붕괴 |

> discount 0이 없앤 것은 **return의 분산**이지 **길이 폭발**이 아니다.

### 4.2 주장별 판정표

| 원문(2025.10)의 주장 | 판정 | 후속 연구의 갱신 |
|---|---|---|
| OPD = dense × on-policy, 두 함정 동시 회피 | 🟡 **조건부** | 제3의 함정이 있다 — dense 신호 자체의 오염(SOD), on-policy성이 반복을 자기강화(Demystifying). Rethinking: 3K~7K가 sweet spot |
| `advantages = -reverse_kl` 한 줄이면 구현 끝 | 🔴 **반박** | Revisiting: 4가지 장치 필요(top-K 32, renormalization, top-p 0.9, masking). Demystifying: 3항 objective. ExOPD: λ≠1이면 매 step 세 모델 logprob 필요 |
| 샘플 1개짜리 per-token 추정으로 충분 | 🔴 **반박** | Revisiting의 본론. 한 점 → 32점 지지집합 비교만으로 36.4 → 41.7 |
| discount 0으로 안정화 장치 불필요 | 🔴 **반박** | §4.1 참조 (3편) |
| reverse KL은 unhackable | 🔴 **반박** | §4.1 참조 (4편) |
| reverse KL의 mode-seeking이 장점 | 🟡 **조건부** | Entropy-Aware: 생성 다양성 손실이라는 대가. Demystifying: mode의 **품질**을 reverse KL이 보장하지 않는다 |
| teacher가 사실상 성능 상한 | 🔴 **반박** | ExOPD: λ>1로 돌파. teacher에 RL 100 step 추가 = **+0.9** vs ExOPD 50 step = **+2.0**. Rethinking: teacher는 천장이 아니라 **자석** — 아래로도 끌어당긴다 |
| 어떤 open-weight teacher든 쓸 수 있다 | 🔴 **반박** | Rethinking: 같은 계열 1.5B/7B는 student 관점에서 구분 불가. Revisiting: 같은 Qwen2.5-7B-Instruct 계열에서도 special-token mismatch |
| per-token 균일 처리로 충분 | 🔴 **반박** | SOD: uniform 34.70 vs SOD 42.98 (**−8.28pt**). GRPO를 뺀 순수 비교에서도 +20.69% |
| 벌점이 forking token에 잘 몰린다 | 🟡 **도메인 의존** | SOD: TIR에서는 성립하지 않는다. 오염 구간에 몰리고, 그 구간은 **teacher 자신이 불확실**(entropy 0.85→2.14) |
| partial rollout 가능 | 🟢 **유지** | SOD의 causal 가중치와 호환. 단 Demystifying: TruncRate 모니터링 필수 |
| 작은 batch로도 된다 | ⚪ **미검증** | Demystifying은 batch 64×4 고정. minibatch **구성**이 크기보다 중요하다는 간접 증거만 |
| teacher는 별도의 더 강한 모델이어야 | 🟡 **부분 반박** | SDFT: 별도 모델 불필요. 단 ICL 의존이라 **3B에서는 SFT보다 −3.3점** (7B +4.0, 14B +6.9) |
| **고정** teacher면 항상 on-policy로 남는다 | 🔴 **정정** | SDFT: frozen은 **정체**(≈65), student 자신은 **발산**(≈33), **EMA만 안정**(≈70) |
| continual learning에 유망 (제안) | 🟢 **실증** | SDFT: 3스킬 순차 누적, 퇴행 없음. 단 위 EMA 정정 조건부 |
| 멀티턴 / 에이전트 (미다룸) | ➕ **확장** | SOD: 최대 16 tool-call 턴, Prop.1·2로 실패를 형식화 후 해결 |
| AIME'24 74.4% / 1,800 GPU h | ⚪ **미재검증** | 6편 중 GPU hour 회계를 다시 보고한 논문 없음 |

### 4.3 후속 연구가 새로 연 축

| 축 | 논문 | 내용 |
|---|---|---|
| **reference model이 설계 변수** | ExOPD | π_ref를 teacher의 pre-RL base로 두면 reward 정확도 상승(reward correction) |
| **λ = 추론 예산 손잡이** | ExOPD | 0<λ<1은 interpolation → budget-controlled reasoning |
| **long-horizon 한계의 정량화** | Rethinking | teacher continuation 이득 1K +0.3659 → 16K +0.0237 |
| **bias–variance 정식화** | Revisiting | token-level은 biased지만 O(T²) vs O(T⁴) |
| **SNR 붕괴 형식화** | SOD | 저-overlap 상태에서 OPD gradient SNR → 0 (Prop. 2) |
| **IRL 동치성** | SDFT | trust-region RL의 최적 정책 자리에 ICL 조건 모델 대입 → 암묵적 reward 최대화와 동치 |

---

## 5. 추천 학습 계획

| 주차 | 내용 |
|---|---|
| **1주차** | 1️⃣ [ExOPD](./exopd_learning_beyond_teacher.md) → 2️⃣ [Rethinking OPD](./rethinking_opd.md) (이론 골격 + 성립 조건) |
| **2주차** | 3️⃣ [Revisiting OPD](./revisiting_opd.md) + 4️⃣ [Demystifying OPD](./demystifying_opd.md) (구현 함정, 묶어 읽기) → 5️⃣ [SOD](./sod_stepwise_opd.md) (도메인 착지) |
| **이후** | 6️⃣ [SDFT](./sdft_continual_learning.md) (운영), 그리고 §2 상황별 2차 읽을거리 |

시간이 없으면 **§4 판정표만 읽어도** 원문 블로그를 그대로 구현했을 때 어디서 터지는지는 파악된다.
