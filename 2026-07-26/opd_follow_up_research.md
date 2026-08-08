      # On-Policy Distillation — 후속 연구 정리 및 읽기 순서

> **기준일**: 2026.07.26
> **원 출처**: [Thinking Machines Lab — On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) (2025.10.27)
> **원문 요약**: [`on_policy_distillation.md`](on_policy_distillation.md)

---

## 0. 조사 방법과 신뢰도

표기 규칙:

| 표기 | 의미 |
|---|---|
| 📄 | **PDF 본문 전체를 읽고 별도 요약 노트를 작성함** (원류 2편 + 필독 7편 = 9편) |
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

## 1. 필독 9편 — 이 순서대로 (원류 2편 → 후속 7편)

### 0️⃣ 원류 2편 (2023.06) — 계보의 출발점

> ⚠️ 2026.07.28 추가분. 아래 두 편은 **원문 블로그(2025.10)보다 2년 앞선** on-policy distillation의 원류다.
> 이전 버전의 이 문서는 두 논문을 "읽지 않아도 되는 것"으로 분류했으나, 본문을 읽고 **판단을 바꿨다** —
> 후속 논문들이 반복해서 재발견하는 쟁점(안정화 장치의 필요성, divergence 선택, teacher 규모)이
> **이미 여기서 서로 다른 답을 내며 갈라져 있고**, 그 분기를 모르면 이후 논쟁이 평평하게 읽힌다.

#### 0️⃣-A. MiniLLM: On-Policy Distillation of Large Language Models
**📄 [요약 노트](minillm.md)** · [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) (v6 2026.01.31, ICLR 2024)
Yuxian Gu, Li Dong, Furu Wei, Minlie Huang
(Tsinghua CoAI / Microsoft Research) · 2023.06
Code: `github.com/microsoft/LMOps/tree/main/minillm`

- **역할**: **OPD 계보의 시작점.** 후속 논문들이 인용하는 `advantage = −reverse_kl` 형태의 원형
- **핵심 논리 사슬**: capacity mismatch에서 forward KLD(mode-covering)는 teacher의 void region에 확률을 낭비한다 → **reverse KLD**(mode-seeking)로 교체 → 기대값이 q_θ에 대해 잡히므로 **policy gradient가 필연** → 학습이 **자동으로 on-policy**가 된다
- **⚠️ 안정화 장치 3종이 필수**: Length Norm. 제거 시 Rouge-L **24.6 → 14.7 (−40%)**, Teacher-Mixed(α=0.2) 제거 시 **−4.2**, Single-Step Decomposition 제거 시 −0.9. + PPO clipping + 사전학습 코퍼스 LM loss 앵커
- **가장 중요한 관찰**: 어느 변형이든 **reverse KLD 자체는 잘 감소하는데** 생성은 반복·짧은 문자열로 붕괴한다 → **손실값이 학습 성공의 지표가 아니다**(reward hacking)
- **결과**: GPT-2/OPT/LLaMA **120M~13B** 전 구간에서 SFT·KD·SeqKD 상회. Vicuna·S-NI·UnNI에서 **student가 teacher 초과**하는 항목 다수
- **부수 성질**: ExAccErr가 낮고 **150 토큰 이후 오차 누적 정지**(노출 편향 실측) · calibration 개선(SST2 ECE **0.099** vs KD 0.191/SeqKD 0.243) · **다양성 손실은 무시할 수준**(Dist-4 99.0 vs SFT 99.5)
- **IRL 등가성**: teacher logit을 Q-function으로 두면 max-entropy IRL과 근사 동치 — 표준 KD가 behavior cloning이라면 MiniLLM은 IRL
- **⚠️ 후속 연구와 충돌**: "teacher가 클수록 student가 좋아진다"(Figure 5)는 결론은 [`rethinking_opd`](rethinking_opd.md)·[`mopd`](mopd.md)가 반박한다. MiniLLM 실험은 **같은 계열(GPT-2) 340M~1.5B 범위**에 한정된 관찰로 읽어야 한다
- **왜 먼저 읽는가**: [`mopd`](mopd.md)의 policy-gradient 구현이 *"MiniLLM을 따라"* 라고 명시하고, MOPD의 per-token advantage가 MiniLLM Eq. 2의 `r_t`와 **정확히 같은 양**이다. 이걸 알고 보면 산업 규모 논문이 훨씬 빨리 읽힌다

#### 0️⃣-B. GKD: On-policy Distillation of LMs — Learning from Self-Generated Mistakes
**📄 [요약 노트](gkd_on_policy_distillation.md)** · [arXiv:2306.13649](https://arxiv.org/abs/2306.13649) (v3 2024.01.17, **ICLR 2024**)
Rishabh Agarwal, Nino Vieillard (공동 1저자), Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem
(**Google DeepMind** / Mila / University of Toronto) · 2023.06

- **⚠️ 동명이인 주의**: [arXiv:2306.06629](https://arxiv.org/abs/2306.06629) *"GKD: A General Knowledge Distillation Framework…"*(Tan 외 · Zhipu.AI)는 **완전히 다른 논문**이다 — 100B급 PLM 증류용 메모리 최적화 툴킷이고 OPD와 무관. 이름과 arXiv 연월(둘 다 2306)이 같아 혼동하기 쉽다
- **역할**: **프레임 전환** — KD를 "interactive expert가 있는 imitation learning"으로 재정의(behavior cloning vs DAgger)
- **두 개의 손잡이**: `L_GKD = (1−λ)·E_{(x,y)}[D(p_T‖p_S)] + λ·E_{y~p_S}[D(p_T‖p_S)]`
  - **λ** = student data fraction (얼마나 on-policy로 갈 것인가)
  - **D** = divergence (forward KL / reverse KL / JSD(β))
  - Supervised KD = (λ=0, forward KL) · On-policy KD = (λ=1, forward KL) · **ImitKD·f-distill도 이 프레임의 특정 점**
- **핵심 발견**: **λ=100%가 거의 항상 최선.** WMT에서 JSD(0.1) 기준 λ=100% **0.85** vs λ=0% **0.28**(3배). GSM8K에서도 λ=100% 행이 전 divergence에서 우위
- **⚠️ 최적 divergence는 태스크 의존적**: XSum(temperature sampling) **JSD(0.9)** · WMT **JSD(0.1)** · GSM8K **forward KL** · FLAN instruction tuning **reverse KL**(forward KL은 MMLU **−0.5로 음수**, reverse KL은 **+2.0**). greedy sampling에서는 divergence 선택의 영향이 거의 없다
- **데이터 효율**: on-policy GKD를 XSum **5% 서브샘플**에 적용한 것이 supervised KD·ImitKD를 **전체 데이터셋 + ground truth**에 적용한 것보다 낫다
- **결과**: 초기 student 대비 상대 이득이 베이스라인 KD 대비 요약 **2.1×** / 번역 **1.7×** / 추론 **1.9×**. 증류된 T5가 **7000× 큰 PaLM(540B)** 의 few-shot 성능 초과
- **RL 결합**: on-policy GKD + RLAIF 동시 최적화 — RLEF\* 대비 높은 ROUGE-2와 teacher 이상의 사실 일관성. **alignment tax를 증류로 상쇄**
- **설계 결정**: student 샘플링 분포로 **역전파하지 않는다** → supervised 학습에 가깝고 안정화 장치가 불필요
- **왜 함께 읽는가**: GKD §5가 **MiniLLM을 직접 비판**한다 — *"MiniLLM은 높은 분산·reward hacking·길이 편향을 다루려 여러 안정화 트릭에 의존한다."* MiniLLM의 ablation이 이 비판을 스스로 입증하지만, 거꾸로 GKD의 λ 실험은 **MiniLLM이 처음부터 고정해둔 λ=1이 최선**임을 보여준다. **양쪽 다 맞다** — 이 긴장이 이후 3년의 논쟁 구도를 그대로 예고한다

---

### 1️⃣ Learning beyond Teacher: Generalized OPD with Reward Extrapolation
**📄 [요약 노트](exopd_learning_beyond_teacher.md)** · [arXiv:2602.12125v2](https://arxiv.org/abs/2602.12125)
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
**📄 [요약 노트](rethinking_opd.md)** · [arXiv:2604.13016v2](https://arxiv.org/abs/2604.13016)
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
**📄 [요약 노트](revisiting_opd.md)** · [arXiv:2603.25562v2](https://arxiv.org/abs/2603.25562)
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
**📄 [요약 노트](demystifying_opd.md)** · [arXiv:2604.08527v1](https://arxiv.org/abs/2604.08527)
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
**📄 [요약 노트](sod_stepwise_opd.md)** · [arXiv:2605.07725v1](https://arxiv.org/abs/2605.07725)
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
**📄 [요약 노트](sdft_continual_learning.md)** · [arXiv:2601.19897v1](https://arxiv.org/abs/2601.19897)
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

### 7️⃣ MOPD: Multi-Teacher OPD for Capability Integration in LLM Post-Training
**📄 [요약 노트](mopd.md)** · [arXiv:2606.30406v1](https://arxiv.org/abs/2606.30406)
Wenhan Ma, Jianyu Wei, Liang Zhao, Hailin Zhang 외 9인
(Peking University / **LLM Core, Xiaomi** / HKU / Renmin University of China) · 2026.06.29
> ⚠️ 2026.07.28 추가분 — 기준일 2026.07.26 시점의 6편 이후에 읽은 논문

- **역할**: 산업 규모 실전 통합 레시피 — 이 목록에서 비어 있던 **multi-teacher capability integration** 축. 프런티어 모델(**MiMo-V2-Flash 309B**) 실배포를 보고하는 유일한 논문
- **핵심 주장**: 다도메인 능력 통합은 weight space(파라미터 병합)도 dataset space(prompt 혼합)도 아닌 **policy space**에서 해야 한다. 도메인별 RL teacher를 **병렬로** 만들고, student 자기 rollout을 **도메인별 teacher로 라우팅**해 토큰 단위 reverse KL로 흡수
- **3축 프레임**: dense optimization × on-policy × parallelisable — Mix-RL·Cascade RL은 병렬성을, Off-Policy Finetune은 on-policy를, Param-Merge는 dense를 잃는다. **셋을 동시에 만족하는 것은 MOPD뿐**
- **결과**: Qwen3-30B-A3B 정규화 점수 **0.937** vs 차선 Mix-RL 0.882. 도메인별 편차 폭 **0.044** (Cascade 0.41, Off-Policy 0.36). Param-Merge는 레시피에 따라 0.328(linear) ~ 0.857(task arith.) 로 요동
- **⚠️ same-origin teacher가 전제조건**: Math teacher만 더 강한 Qwen3-235B-A22B로 교체 → **0.937 → 0.600(PG)**, top-k는 **−1.190** 으로 SFT student 아래로 붕괴. 초기 per-token KL이 0.04 → **0.19(5배)**, top-k는 step 18에 발산
- **loss 형태는 부차적**: same-origin에서 PG 0.937 vs Top-64 0.909로 등가, 학습 궤적도 거의 겹침. teacher를 바꾸면 같은 두 loss가 0.600 vs −1.190으로 갈라진다
- **인프라**: teacher prefill을 RL 트레이너 바깥의 **독립 서비스**로 빼고 비동기 호출 → 다른 시퀀스 샘플링과 겹쳐 **wall-clock 오버헤드 거의 0**. 샘플 효율도 높다(IF ~25K / SWE ~30K vs Mix-RL 150–180K)
- **multi-round**: Iter-1 student에서 teacher 재학습 → Iter-2 Teacher **1.030**(+0.093), Iter-2 MOPD **0.986**(+0.049)
- **왜 7번인가**: 1~6번이 전부 **단일 teacher · 단일 도메인** 문제를 다룬다. 실제 post-training은 다도메인 통합이고, 이 논문이 그 자리와 **인프라·조직 운영** 관점을 채운다. 2번(`rethinking_opd`)의 teacher 선정 조건을 산업 규모에서 독립 재확인하되, **"고르지 말고 same-SFT-checkpoint에서 만들어라"** 는 다른 처방을 낸다. 1번(`exopd`)의 λ 외삽을 MOPD Stage 3에 얹는 것이 자연스러운 미탐구 조합.

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

- **서베이를 1번으로 읽는 것** → 이미 계보와 두 함정을 정리한 상태라면 landscape는 중복이다. **인덱스로만** 쓸 것.

> ~~**MiniLLM**, **GKD** → 원문 요약 §1의 3열 비교표로 충분~~ — **2026.07.28 철회.**
> 본문을 읽고 판단을 바꿨다. 두 논문은 §1의 **0️⃣ 원류 2편**으로 승격했다.
> 3열 비교표는 "무엇이 다른가"만 알려주고 **"왜 갈라졌는가"** 를 알려주지 않는데,
> 후속 논문들이 반복 재발견하는 쟁점 3가지(안정화 장치의 필요성 · divergence 선택 · teacher 규모)가
> 전부 이 두 논문의 분기점에서 나온다.

---

## 3. 연구 지형 — 갈래별 개관

### 3.1 메커니즘 규명 — "왜 되는가"

| 논문 | 기여 |
|---|---|
| 📄 [Rethinking OPD](rethinking_opd.md) | 성공 조건 2가지, overlap token 인과성, long-horizon 한계 |
| 📄 [ExOPD / G-OPD](exopd_learning_beyond_teacher.md) | OPD = β=1 고정 dense KL-constrained RL, reward extrapolation |
| ✅ [Geometry of OPD](https://arxiv.org/abs/2606.07082) | subspace locking, 파라미터 공간 업데이트 기하 |

### 3.2 실패 모드 진단과 안정화 — 최다 논문 갈래

| 논문 | 기여 |
|---|---|
| 📄 [Revisiting OPD](revisiting_opd.md) | 실패 모드 3종, O(T²) vs O(T⁴), truncated reverse-KL, +19.8% |
| 📄 [Demystifying OPD](demystifying_opd.md) | abrupt repetition saturation → truncation collapse, Stable-OPD, +7.2 |
| ✅ [Entropy-Aware OPD](https://arxiv.org/abs/2603.07079) | mode-seeking의 다양성 손실, forward KL 혼합 |
| ✅ [TGPO](https://arxiv.org/abs/2605.13230) | 큰 정책 격차에서의 reverse KL 붕괴 |
| ▫️ [PowerOPD](https://arxiv.org/abs/2606.17199) · [The Extrapolation Cliff](https://arxiv.org/abs/2605.08737) · [Escaping the KL Agreement Trap](https://arxiv.org/abs/2606.09471) · [OPD Reduces Output Diversity](https://arxiv.org/abs/2606.26091) · [Denser ≠ Better](https://arxiv.org/abs/2607.01763) · [On the Position Bias of OPD](https://arxiv.org/abs/2606.22600) | 미확인 |

### 3.3 Self-Distillation (OPSD) — 저장소 기준 단일 최대 카테고리 (65편)

| 논문 | 기여 |
|---|---|
| 📄 [SDFT](sdft_continual_learning.md) | demonstration-conditioned 자기 teacher, EMA, 스킬 순차 누적 |
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
| 📄 [SOD](sod_stepwise_opd.md) | step-wise 재가중, tool call 오류 연쇄, SNR 붕괴 형식화 |
| ✅ [OPCD](https://arxiv.org/abs/2602.12275) | 문맥 지식의 파라미터 내재화 |
| ▫️ [OEL](https://arxiv.org/abs/2603.16856) · [PMD](https://arxiv.org/abs/2607.01480) · [Sample-Efficient Learning from Agent Experience](https://arxiv.org/abs/2607.21051) | 미확인 |

---

## 4. 원문 대비 무엇이 바뀌었나

9편 본문을 모두 읽고 정리한 결과. **동일 주장을 여러 논문이 독립적으로 반박한 항목이 가장 신뢰도가 높다.**

> 📌 **원류 2편(MiniLLM·GKD)은 블로그보다 2년 앞선다.** 따라서 엄밀히는 "후속 반박"이 아니라
> **블로그가 원류에 이미 있던 경고를 누락한 것**이다. 아래 표에서 이들은 `(원류)` 로 표시했다.

### 4.1 가장 강하게 뒤집힌 주장 두 가지

#### ❌ "reverse KL은 unhackable" — **5편이 독립적으로 반박 (원류 포함)**

원문은 *"reward가 teacher 분포와의 거리 그 자체라 프록시와 목표가 원리적으로 일치한다"* 고 했다. 그러나:

| 논문 | 반박 근거 |
|---|---|
| ExOPD | `log(π*/π_ref)`는 결국 proxy. λ=1.5에서 log-ratio peak를 공격적으로 fitting하는 hacking + length bias |
| Rethinking OPD | unhackable ≠ usable. 실패 teacher의 reward AUROC **0.75** > 성공 teacher **0.73** 인데도 gradient norm은 더 작다 |
| Revisiting OPD | Appendix H 전체가 reward-hacking 사례집 — over-continuation / `wait` 반복 / 깨진 비영어 출력에 teacher가 고확률 부여 |
| Demystifying OPD | 논문 §1이 직접 *"student repetition may exploit or hack the teacher's likelihood-based signal"* |
| **MiniLLM (원류, 2023.06)** | **가장 이른 증거.** Figure 8에 *"Reward Hacking"* 이 곡선 위에 직접 주석되어 있다. Teacher-Mixed Sampling을 빼면 **reverse KLD는 오히려 가장 빠르게 감소하는데** 생성은 반복·짧은·무의미한 문자열로 붕괴한다(R-L 24.6→20.4) — teacher 분포에서 확률이 높은 문자열들이다 |

> **다만 성격이 다르다**: RLHF의 hacking은 프록시가 **틀려서** 생기고, OPD의 것은 프록시가 **맞는데도 국소 최적이 degenerate**해서 생긴다.

#### ❌ "discount 0 덕분에 안정화 장치가 불필요" — **4편이 반박 (원류 포함)**

| 논문 | 반박 근거 |
|---|---|
| Demystifying OPD | GRPO-style **clipping을 되살렸고**, mixture의 `λ_gold·L_SFT`는 MiniLLM LM loss와 기능적으로 동일한 앵커, `β_KL·KL(π_θ‖π_ref)`는 MiniLLM에도 없던 **새 제약** |
| Revisiting OPD | 절반은 강화(분산 근거를 O(T²) vs O(T⁴)로 승격), 절반은 반박(장치 불필요는 거짓) |
| Rethinking OPD | 10K/15K 길이에서 step 200~220 붕괴. Top-1 OPD도 붕괴 |
| **MiniLLM (원류, 2023.06)** | **애초에 장치 없이는 학습이 안 됐다.** Length Norm. 제거 시 R-L **24.6 → 14.7 (−40%)**, Teacher-Mixed 제거 시 −4.2. 여기에 PPO clipping과 사전학습 코퍼스 LM loss 앵커까지 붙어 있다. Demystifying이 "되살렸다"는 clipping·LM loss는 **원래 여기 있던 것** |

> discount 0이 없앤 것은 **return의 분산**이지 **길이 폭발**이 아니다.
> 그리고 MiniLLM의 존재는 이 주장이 **후속 연구가 뒤집은 것이 아니라 처음부터 성립한 적이 없었음**을 뜻한다.

### 4.2 주장별 판정표

| 원문(2025.10)의 주장 | 판정 | 후속 연구의 갱신 |
|---|---|---|
| OPD = dense × on-policy, 두 함정 동시 회피 | 🟡 **조건부** | 제3의 함정이 있다 — dense 신호 자체의 오염(SOD), on-policy성이 반복을 자기강화(Demystifying). Rethinking: 3K~7K가 sweet spot |
| `advantages = -reverse_kl` 한 줄이면 구현 끝 | 🔴 **반박** | Revisiting: 4가지 장치 필요(top-K 32, renormalization, top-p 0.9, masking). Demystifying: 3항 objective. ExOPD: λ≠1이면 매 step 세 모델 logprob 필요 |
| 샘플 1개짜리 per-token 추정으로 충분 | 🔴 **반박** | Revisiting의 본론. 한 점 → 32점 지지집합 비교만으로 36.4 → 41.7. **MiniLLM(원류)의 Single-Step Decomposition이 같은 진단** — `E_{y_t~q_θ}[r_t]`를 Monte-Carlo 대신 **vocabulary 전체 합**으로 직접 계산해 분산을 줄인다(다만 성능 기여는 −0.9로 3종 중 최소) |
| discount 0으로 안정화 장치 불필요 | 🔴 **반박** | §4.1 참조 (3편) |
| reverse KL은 unhackable | 🔴 **반박** | §4.1 참조 (4편) |
| reverse KL의 mode-seeking이 장점 | 🟡 **조건부** | Entropy-Aware: 생성 다양성 손실이라는 대가. Demystifying: mode의 **품질**을 reverse KL이 보장하지 않는다. **GKD(원류)가 이미 정면으로 반박** — 최적 divergence는 **태스크 의존적**이다: XSum(temp sampling) **JSD(0.9)** · WMT **JSD(0.1)** · GSM8K **forward KL** · FLAN **reverse KL**. greedy sampling에서는 divergence 선택의 영향이 거의 없고, student가 커질수록 forward/reverse 격차도 줄어든다 |
| divergence는 reverse KL로 고정 | 🔴 **반박 (원류)** | **GKD**: λ(student data fraction)와 D(divergence)는 **독립된 두 손잡이**다. 블로그는 D를 고정한 채 λ=1만 이야기하지만, GKD는 D를 자유변수로 두는 것이 실제 이득을 준다고 보인다 — FLAN instruction tuning에서 forward KL은 MMLU **−0.5(음수)**, reverse KL은 **+2.0**. **MiniLLM(원류)은 반대로 reverse KL 고정** — 두 원류가 갈라지는 지점 |
| teacher가 사실상 성능 상한 | 🔴 **반박** | ExOPD: λ>1로 돌파. teacher에 RL 100 step 추가 = **+0.9** vs ExOPD 50 step = **+2.0**. Rethinking: teacher는 천장이 아니라 **자석** — 아래로도 끌어당긴다. **MOPD는 반대로 상한을 인정**(Δ −0.063)하되 **multi-round로 상한 자체를 올린다** — Iter-2 teacher **1.030** |
| 어떤 open-weight teacher든 쓸 수 있다 | 🔴 **반박 (3편)** | Rethinking: 같은 계열 1.5B/7B는 student 관점에서 구분 불가. Revisiting: 같은 Qwen2.5-7B-Instruct 계열에서도 special-token mismatch. **MOPD**: 더 강한 외부 teacher(Qwen3-235B-A22B) 투입 시 정규화 점수 **0.937 → 0.600(PG) / −1.190(top-k)**, 초기 KL 0.04 → **0.19** |
| per-token 균일 처리로 충분 | 🔴 **반박** | SOD: uniform 34.70 vs SOD 42.98 (**−8.28pt**). GRPO를 뺀 순수 비교에서도 +20.69% |
| 벌점이 forking token에 잘 몰린다 | 🟡 **도메인 의존** | SOD: TIR에서는 성립하지 않는다. 오염 구간에 몰리고, 그 구간은 **teacher 자신이 불확실**(entropy 0.85→2.14) |
| partial rollout 가능 | 🟢 **유지** | SOD의 causal 가중치와 호환. 단 Demystifying: TruncRate 모니터링 필수 |
| 작은 batch로도 된다 | ⚪ **미검증** | Demystifying은 batch 64×4 고정. minibatch **구성**이 크기보다 중요하다는 간접 증거만 |
| teacher는 별도의 더 강한 모델이어야 | 🟡 **부분 반박** | SDFT: 별도 모델 불필요. 단 ICL 의존이라 **3B에서는 SFT보다 −3.3점** (7B +4.0, 14B +6.9) |
| **고정** teacher면 항상 on-policy로 남는다 | 🔴 **정정** | SDFT: frozen은 **정체**(≈65), student 자신은 **발산**(≈33), **EMA만 안정**(≈70) |
| continual learning에 유망 (제안) | 🟢 **실증** | SDFT: 3스킬 순차 누적, 퇴행 없음. 단 위 EMA 정정 조건부 |
| 멀티턴 / 에이전트 (미다룸) | ➕ **확장** | SOD: 최대 16 tool-call 턴, Prop.1·2로 실패를 형식화 후 해결 |
| AIME'24 74.4% / 1,800 GPU h | ⚪ **미재검증** | 7편 중 GPU hour 회계를 다시 보고한 논문 없음. MOPD가 유일하게 인접한 보고 — teacher prefill을 사이드카 서비스로 빼면 **wall-clock 오버헤드 거의 0** |
| 다도메인 통합 (미다룸) | ➕ **확장** | MOPD: teacher를 도메인별로 두고 prompt 라우팅. Qwen3-30B-A3B 정규화 **0.937** vs Mix-RL 0.882 · Param-Merge 0.328~0.857 |

### 4.3 후속 연구가 새로 연 축

| 축 | 논문 | 내용 |
|---|---|---|
| **reference model이 설계 변수** | ExOPD | π_ref를 teacher의 pre-RL base로 두면 reward 정확도 상승(reward correction) |
| **λ = 추론 예산 손잡이** | ExOPD | 0<λ<1은 interpolation → budget-controlled reasoning |
| **long-horizon 한계의 정량화** | Rethinking | teacher continuation 이득 1K +0.3659 → 16K +0.0237 |
| **bias–variance 정식화** | Revisiting | token-level은 biased지만 O(T²) vs O(T⁴) |
| **SNR 붕괴 형식화** | SOD | 저-overlap 상태에서 OPD gradient SNR → 0 (Prop. 2) |
| **IRL 동치성** | SDFT | trust-region RL의 최적 정책 자리에 ICL 조건 모델 대입 → 암묵적 reward 최대화와 동치 |

> ⚠️ **IRL 동치성은 SDFT가 처음이 아니다.** [`MiniLLM`](minillm.md) Appendix A.1(2023.06)이 이미
> teacher logit을 Q-function으로 두면 max-entropy IRL 목적함수가 `−KL[q_θ‖p]`로 환원됨을 보였고,
> **표준 KD = behavior cloning / MiniLLM = IRL** 이라는 대비를 명시했다.
> SDFT의 기여는 IRL 동치성 자체가 아니라 그것을 **self-distillation·continual learning 설정으로 옮긴 것**이다.

### 4.4 원류 2편이 이미 열어두었던 축

블로그(2025.10)가 다루지 않았지만 **2023년에 이미 제기되어 있던** 것들. 후속 연구가 "새로 열었다"고 읽히는 축 중 일부는 사실 **재발견**이다.

| 축 | 원류 | 내용 |
|---|---|---|
| **divergence는 자유변수** | [GKD](gkd_on_policy_distillation.md) | forward KL / reverse KL / JSD(β) 중 최적은 **태스크 의존적**. Entropy-Aware OPD(2026.03)의 forward KL 혼합 처방이 이 축의 재발견 |
| **λ = on-policy 정도의 연속 손잡이** | [GKD](gkd_on_policy_distillation.md) | λ∈[0,1]로 supervised KD·on-policy KD·ImitKD·f-distill을 **한 프레임에 통합**. 결론은 λ=1이 거의 항상 최선 |
| **안정화 장치의 불가피성** | [MiniLLM](minillm.md) | 3종 장치 + PPO clipping + LM loss 앵커. Demystifying(2026)의 3항 objective와 사실상 같은 구성 |
| **reward hacking의 조기 관측** | [MiniLLM](minillm.md) | reverse KLD는 잘 내려가는데 생성은 붕괴 — **손실값이 성공 지표가 아니다** |
| **IRL 동치성** | [MiniLLM](minillm.md) | 위 주석 참조 |
| **증류 + RL 동시 최적화** | [GKD](gkd_on_policy_distillation.md) | `(1−α)·E[r(y)] − α·E[D(p_T‖p_S)]`. **alignment tax를 증류로 상쇄**. 이후 시리즈 어느 논문도 이 축을 다시 다루지 않았다 — **여전히 열린 미탐구 방향** |

---

## 5. 추천 학습 계획

| 주차 | 내용 |
|---|---|
| **1주차** | 1️⃣ [ExOPD](exopd_learning_beyond_teacher.md) → 2️⃣ [Rethinking OPD](rethinking_opd.md) (이론 골격 + 성립 조건) |
| **2주차** | 3️⃣ [Revisiting OPD](revisiting_opd.md) + 4️⃣ [Demystifying OPD](demystifying_opd.md) (구현 함정, 묶어 읽기) → 5️⃣ [SOD](sod_stepwise_opd.md) (도메인 착지) |
| **이후** | 6️⃣ [SDFT](sdft_continual_learning.md) (운영), 그리고 §2 상황별 2차 읽을거리 |

시간이 없으면 **§4 판정표만 읽어도** 원문 블로그를 그대로 구현했을 때 어디서 터지는지는 파악된다.
