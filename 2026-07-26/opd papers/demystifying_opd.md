# Demystifying OPD: Length Inflation and Stabilization Strategies for Large Language Models

> **Venue**: Preprint (2026.04.10)
> **Authors**: Feng Luo, Yu-Neng Chuang, Guanchu Wang, Zicheng Xu, Xiaotian Han, Tianyi Zhang, Vladimir Braverman
> (Rice University / UNC Charlotte / Johns Hopkins University / Case Western Reserve University)
> **arXiv**: [2604.08527v1](https://arxiv.org/abs/2604.08527) (2026.04.09)

**한 줄 정의**: OPD를 오래 돌리면 rollout이 **갑자기 길어지고 반복으로 채워지다가 잘려나가는(truncation collapse)** 고유 실패 모드가 나타난다.
이 논문은 그 인과 사슬을 rollout·token·메커니즘 세 층위에서 규명하고, **reference 기반 divergence 제약 + rollout mixture distillation**(Stable-OPD)으로 끊는다.

---

## 1. Background

### LLM 추론 학습의 현황

- **GRPO 계열 RLVR**이 수학·코드의 표준. prompt x에 대해 G개 응답 {o_1..o_G}를 π_θ_old에서 뽑고, **sequence-level reward** r_i = R(x, o_i)(주로 이진 정오답)를 준 뒤 group-normalized advantage A_i를 응답 o_i 안의 **모든 token k에 똑같이 broadcast**한다.
- 한계 두 가지: ① reward가 **sequence 수준에서만** 주어져 "어디서 틀렸는지"에 대한 token 수준 안내가 없다. ② group 내 응답이 **전부 정답이거나 전부 오답이면 advantage가 0** — group sampling 연산을 쓰고도 업데이트가 나오지 않는다.
- **OPD**는 여기에 teacher π_T를 붙여 sequence reward를 token 수준 신호로 바꾼다. student가 자기 분포에서 ŷ ~ π_S를 뽑으므로 off-policy distillation의 train-inference mismatch도 없다. 이 논문은 **GRPO의 clipped objective를 그대로 쓰되 A_i를 token-level A_{i,t}로 교체한 GRPO-style OPD**를 기본 셋업으로 삼는다.

```
표준 distillation (off-policy):
  L_SD(θ)  = E_{(x,y) ~ P(X,Y)}       [ D( π_T || π_S^θ )(y | x) ]   ← 고정 시퀀스 위
OPD (on-policy):
  L_OPD(θ) = E_{x ~ P(X), ŷ ~ π_S^θ}  [ D( π_S^θ || π_T )(ŷ | x) ]   ← student가 뽑은 ŷ 위
```

### 기존 방법의 한계

| 방법 | 신호 밀도 | 샘플 분포 | 이 논문이 지적하는 문제 |
|---|---|---|---|
| SFT / off-policy distillation | dense (token별) | 고정 teacher 시퀀스 | student가 학습 궤적에서 벗어나는 순간 train-inference mismatch |
| GRPO / RLVR | sparse (sequence 1개) | 자기 rollout | token 수준 안내 없음, all-correct/all-wrong이면 advantage 0 |
| Dr.GRPO / DAPO의 length bias 처방 | sparse | 자기 rollout | **sequence-level의 length-dependent gradient scaling**만 제거 — OPD 고유 실패 모드는 못 잡는다 |
| **표준 OPD** | **dense (token별)** | **자기 rollout** | **abrupt repetition saturation → truncation collapse** (이 논문의 발견) |

- 실제 수치가 이를 드러낸다. Qwen2.5-Math-7B backbone에서 OPD는 평균 **43.8%** 로 **SFT 44.1%, GRPO 45.5%에 오히려 못 미친다.** 1.5B backbone도 같다(OPD **28.9%** vs SFT 31.9% vs GRPO 30.1%). dense token 신호와 on-policy 샘플을 둘 다 갖고도 진다.

> 저자들의 진단: OPD가 약한 게 아니라 **학습 불안정이 OPD의 효과를 갉아먹고 있다(training instability limits the effectiveness of OPD).**

---

## 2. Motivation

### 핵심 통찰 1: reverse-KL advantage는 궤적 위에 균등하게 깔리지 않는다

token 수준 advantage는 다음 한 줄이다.

```
r^KL_{i,t} = log π_T(ŷ_{i,t} | s_{i,t}, ŷ_{i,<t}) − log π_θ(ŷ_{i,t} | s_{i,t}, ŷ_{i,<t})
A_{i,t}    = r^KL_{i,t}
```

즉 **teacher가 student보다 이 token을 더 좋아할수록 advantage가 크다.** 문제는 **반복 token**이다.

- 반복은 국소적으로 강한 예측성을 갖는다 → teacher도 그 continuation에 높은 확률을 준다 → **반복 token이 일반 token보다 체계적으로 큰 advantage를 받는다.**
- Figure 4(Qwen2.5-Math-1.5B)의 관측: 붕괴 이후 반복 token의 평균 advantage는 약 **−0.02 ~ −0.05**, 일반 token은 약 **−0.15 ~ −0.17** — 반복 token 쪽이 **4~9배 유리하다**(덜 음수다). 다만 붕괴 전에는 반복 token이 워낙 희소해서(비율 ≈ 0) advantage가 커도 총 기여가 작다. **희소성이 유일한 방파제**다.

### 핵심 통찰 2: on-policy 샘플링이 그 편향을 자기강화 루프로 증폭한다

OPD 업데이트를 state-action 형태로 쓰고(clipping 무시), 반복 tail token 집합 R의 안팎으로 쪼개면:

```
g(θ) ∝ E_{s ~ d_πθ, y ~ π_θ(·|s)} [ A(s,y) ∇_θ log π_θ(y|s) ]           ... (3)
       A(s,y) = log π_T(y|s) − log π_θ(y|s)
       d_πθ   = 현재 student 정책이 유도하는 state-visitation 분포

g(θ) = E_{s ~ d_πθ}[ 1{s ∉ R} Δ(s) ] + E_{s ~ d_πθ}[ 1{s ∈ R} Δ(s) ]    ... (4)
       Δ(s) = E_{y ~ π_θ(·|s)} [ A(s,y) ∇_θ log π_θ(y|s) ]
```

- OPD 업데이트를 지배하는 것은 **(a) 그 state를 얼마나 자주 방문하는가(d_πθ)** 와 **(b) 그 state의 action이 reverse-KL 신호에서 얼마나 선호되는가** 의 **곱**이다.
- R 방문이 늘면 두 번째 항의 지분이 커지고, 그 업데이트가 다시 R로 가는 continuation을 강화한다 → **self-reinforcing loop.** 결정적으로 **d_πθ는 student가 스스로 만든다.** off-policy distillation이라면 고정 데이터가 R 방문을 막아주지만, OPD는 자기 분포를 자기가 갱신하므로 브레이크가 없다.

> 저자 표현: "In this sense, student repetition may **exploit or hack the teacher's likelihood-based signal**."
> reverse KL이 unhackable하다는 통념에 정면으로 걸리는 문장이다(§7 참조).

---

## 3. Contributions

1. **Length Inflation**: rollout length inflation을 OPD의 **관측 가능한 학습 병리**로 정식화하고, 그 1차 실패 모드가 **abrupt repetition saturation**임을 규명. 이를 재기 위한 두 지표(TruncRate, RepRate)를 정의.
2. **Rollout Pathology**: repetition-saturated·truncation-dominated rollout이 **biased gradient**를 만들어 최적화를 불안정하게 만든다는 것을 rollout 수준·token 수준 증거로 입증.
3. **Stabilization Protocol**: 이 병리를 멈추는 데 필요한 제약을 특정. **divergence 제약 + rollout mixture**의 **dual strategy**가 student가 distillation objective를 hack하는 것을 막는다.
4. **Evaluation**: **6개 데이터셋 × 3개 LLM**에서 Stable-OPD가 일관되게 정확도를 올리고 repetition saturation을 줄임. 1.5B backbone 평균 **28.9 → 36.1 (+7.2)**, 7B backbone **43.8 → 47.6 (+3.8, 전 방법 중 1위)**.
5. **RLVR과의 구별**: Dr.GRPO / DAPO가 다루는 sequence-level length bias와 **메커니즘이 근본적으로 다른** OPD 고유 현상임을 논증.

---

## 4. Method

### 4.1 병리 분석 — abrupt repetition saturation → length inflation → truncation collapse

#### (0) 측정 도구 — 두 지표

```
Truncation rate:
  trunc(o_i) = 1  (EOS 없이 최대 생성 길이 소진으로 종료된 경우)
  TruncRate(R) = (1/N) Σ_i trunc(o_i)

Repetition rate (zlib 압축률 기반):
  CompRatio(o_i) = |bytes(o_i^tail)| / |c(bytes(o_i^tail))|      c(·) = zlib 압축
  rep(o_i)       = 1[ |o_i^tail| > L  AND  CompRatio(o_i) > τ ]   L = 10,000자, τ = 10
  RepRate(R)     = (1/N) Σ_i rep(o_i)
```

- o_i^tail은 응답의 **마지막 L자**. 즉 RepRate는 "긴 rollout 중 꼬리가 극단적으로 압축 가능한 것의 비율"이며, 눈에 보이는 저정보 반복 continuation과 잘 상관한다. 두 지표 모두 **on-policy 학습 rollout**과 **held-out validation prompt** 양쪽에서 잰다.

#### (1) 인과 사슬 (논문의 핵심 주장)

```
① abrupt repetition saturation — 반복 패턴이 짧은 구간 안에 생성 시퀀스를 지배
▼ ② 반복 token이 reverse-KL advantage를 체계적으로 크게 받음
     A = log π_T − log π_θ, 반복은 국소 예측성이 높아 teacher가 선호
     붕괴 후 반복 token 평균 ≈ −0.02~−0.05 vs 일반 token −0.15~−0.17 (4~9배)
▼ ③ on-policy 업데이트가 이 행동을 강화 — 식 (4)의 두 번째 항 지분 증가
     빈도 상승 × 큰 advantage → self-reinforcing loop
▼ ④ rollout length inflation — 응답 길이가 generation budget 쪽으로 점프
▼ ⑤ 고정 context / token limit에 의한 truncation이 학습 데이터를 지배
     TruncRate → 1 부근. 대부분의 생성이 EOS 없이 잘린다
▼ ⑥ biased gradient — 완결·비반복 궤적이 배치에서 사라지고 잘린 반복 궤적만 남음
▼ ⑦ validation 성능 급락 = truncation collapse
```

#### (2) Figure 1 재현 — 한 장에 담긴 전이

Figure 1(steps 0~800)을 표로 옮기면 다음과 같다. 세 곡선의 전이 시점이 **같은 지점에서 동시에** 일어난다는 점이 핵심이다.

| 구간 | Rollout truncation (파랑) | Rollout repetition (초록) | MATH500 accuracy (주황) |
|---|---|---|---|
| step 0 ~ 약 280 | 0.2~0.7 사이 큰 진폭으로 진동, 평균 ≈ 0.45~0.5 | **≈ 0.0 (거의 0에 붙어 있음)** | ≈ 0.70~0.725 (최고 구간) |
| step 약 280 ~ 310 | **급상승 (~30 step 안에 0.95 이상으로 점프)** | **0 → 0.3~0.6으로 spike** | **급락** |
| step 310 ~ 800 | 0.9~1.0 고착 (거의 전량 truncation) | 0.3~0.6에서 진동 지속 | ≈ 0.575~0.65, 회복 없음 |

- accuracy 하락 폭은 대략 **0.72 → 0.60~0.63**, 즉 **8~10 %p 급락 후 미회복**. 그래프 우측 축 범위 자체가 0.550~0.725이며, 전이 이후 학습은 **원래 수준으로 돌아오지 않는다.**

#### (3) Figure 2 재현 — 세 student-teacher 조합에서 모두 재현됨

OpenR1-Math-220k의 13k subset으로, 학생 규모와 teacher 선택을 모두 바꿔 3개 조합을 돌렸다.

| 조합 | Student | Teacher | 안정 구간 rollout TruncRate | 안정 구간 validation TruncRate | 안정 구간 RepRate |
|---|---|---|---|---|---|
| (a) | Qwen2.5-Math-1.5B | DeepSeek-R1-Distill-7B | **≈ 0.5** | **≈ 0.2** | ≈ 0 |
| (b) | Qwen2.5-Math-1.5B | OpenThinker3-7B | ≈ 0.5 | ≈ 0.2 | ≈ 0 |
| (c) | Qwen2.5-Math-7B | DeepSeek-R1-Distill-7B | **≈ 0.23** | **≈ 0.1** | ≈ 0 |

- 초기에는 **모두 정상**이다 — validation accuracy 점진 상승, 대부분 generation budget 안에서 종료, 반복 tail 희소. 그 뒤 **세 조합 전부** sharp phase transition. 전이는 **약 30 OPD step** 안에 완료되며 TruncRate가 1을 향해 급등, RepRate가 0에서 **0.3~0.6**으로 튄다.
- **validation set(MATH500)에서도 거의 같은 step에** 두 지표가 점프하며 accuracy가 급락한다 → 학습 rollout만의 아티팩트가 아니다. 저자들은 이를 **abrupt truncation-repetition inflation**이라 부르며, 세 조합 재현이므로 **특정 모델 쌍·split의 우연이 아니다.**

#### (4) Figure 3 재현 — rollout 수준 증거

student log-prob / teacher log-prob / reverse-KL advantage / response length를 함께 추적하면, inflation 이전에는 넷 다 완만하지만 **inflation 시점에 동시에 이동**한다. response length가 generation budget 쪽으로 점프하고, 두 log-prob 모두 덜 음수가 되는데 **teacher 쪽 상승폭이 student보다 크다.** A = log π_T − log π_θ 이므로 **평균 advantage가 위로 튄다.**

> 이 모든 일이 **teacher 모델도 loss 수식도 고정된 채** 일어난다 → 불안정성은 외부 요인이 아니라 **OPD on-policy 동역학에 내재한다.**

#### (5) Figure 4 재현 — token 수준 증거

| step 구간 | 반복 token 평균 advantage | 일반 token 평균 advantage | 반복 token 비율 |
|---|---|---|---|
| 0 ~ 100 | ≈ −0.25 ~ −0.27 (변동 큼) | ≈ −0.20 ~ −0.22 | ≈ 0.0 |
| 100 ~ 300 | −0.15 → −0.05로 상승 | ≈ −0.17 근처 수렴 | 0.0 → 0.1 |
| 300 ~ 800 | **≈ −0.02 ~ −0.05 (거의 0)** | ≈ −0.15 ~ −0.17 | **≈ 0.25~0.35 (약 30%)** |

- 반복 token은 **학습 내내** 일반 token보다 큰 advantage를 받는다. 다만 붕괴 이전에는 **비율이 0에 가까워** 총 기여가 제한된다. 붕괴 이후 **토큰의 약 30%가 반복 token**이 되고 평균 advantage는 일반 token의 **4~9배** — **빈도 × 크기**가 동시에 커지며 gradient 방향을 반복 continuation 쪽으로 돌린다.

#### (6) 왜 GRPO의 length bias와 다른가 (저자들의 논거)

| 항목 | Dr.GRPO / DAPO가 다루는 RLVR length bias | **이 논문의 OPD 실패 모드** |
|---|---|---|
| 발생 층위 | **sequence level** — length-dependent gradient scaling / normalization | **token level** — 특정 token 부류가 큰 advantage를 받음 |
| 원인 | 목적함수의 정규화 항이 긴 응답을 암묵적으로 선호 | **반복 continuation의 teacher likelihood가 높다**는 distillation 신호 자체의 성질 |
| 처방 | objective reweighting / normalization scheme | 재정규화로는 안 됨 — **분포 제약 + 데이터 anchor**가 필요 |
| 증폭 경로 | sparse reward 하의 정적 편향 | **on-policy 샘플링이 빈도를 키워 자기강화** (식 (4)) |
| 성격 | 점진적 length drift | **abrupt phase transition (약 30 step)** |

> 요컨대 RLVR의 length bias는 **"긴 응답에 gradient가 더 실린다"** 이고, OPD의 것은 **"teacher가 좋아하는 반복을 student가 스스로 더 자주 방문하게 만든다"** 이다. 후자는 student-induced data collection과 likelihood 기반 목적함수의 **상호작용**에서만 생기므로 OPD 고유다.

### 4.2 Stable-OPD

두 구성요소가 **인과 사슬의 서로 다른 고리**를 끊는다.

| 구성요소 | 끊는 고리 | 작동 방식 |
|---|---|---|
| **(ii) Rollout mixture distillation** | ⑤⑥ (truncated rollout의 데이터 지배 → biased gradient) | 완결·비반복 golden 궤적을 매 minibatch에 **앵커**로 섞어 degenerate rollout의 지분을 희석 |
| **(i) Reference 기반 divergence 제약** | ③④ (on-policy 업데이트의 자기강화 → length inflation) | π_ref 대비 per-prefix KL을 페널티로 걸어 **정책 drift 자체의 크기**를 제한 |

#### 구성요소 (ii) — Mixture Distillation: on-policy와 off-policy 감독의 결합

고정 데이터셋 D_gold를 둔다 — 각 (x, y)는 **완결되고 반복 없는 고품질 chain-of-thought**. 매 step마다 prompt x에 대해 **student의 on-policy rollout**과 **D_gold의 golden solution**을 **같은 minibatch에** 넣는다. 같은 문제에 대해 자기 궤적과 고품질 target을 동시에 보는 셈이다.

```
L_mix(θ) = L_OPD(θ) + λ_gold · E_{(x,y) ~ D_gold} [ L_SFT(θ; x, y) ]      ... (5)
           L_SFT = 표준 supervised loss,  λ_gold = golden 데이터 가중치
```

- **분포 관점**: 두 state 분포의 mixture 위에서 학습하는 것 — 현재 student가 유도하는 on-policy 분포 d_πθ 와 D_gold가 유도하는 **고정 off-policy 분포**. truncation-dominated rollout이 gradient를 독점하지 못한다.
- **최근 self-distillation과의 차이**: Hübotter et al.(2026), Zhao et al.(2026)은 golden 응답으로 **teacher 신호 자체를 다듬는다**. 그러면 Kim et al.(2026)이 지적하듯 추론 중 **teacher의 불확실성이 억제되어** 복잡한 문제에서 student 성능이 떨어질 수 있다. mixture distillation은 **on-policy rollout 위의 teacher 유래 OPD 신호를 손대지 않고**, golden 데이터를 **보조 off-policy SFT 항으로만** 쓴다.

#### 구성요소 (i) — KL-Regularized: reference 기반 divergence 제약

mixture는 학습 분포를 바꾸지만 **업데이트 크기 자체는 제어하지 못한다.** 일단 student가 길고 반복적인 궤적으로 drift하면 reverse-KL advantage는 **바로 그 state들에** 큰 양의 신호를 붙이기 시작한다. 그래서 policy 자체에 KL 정규화를 건다 — reference policy π_ref(예: **초기 student checkpoint**)를 두고, prefix state s_t마다 `KL(s_t) = D_KL(π_θ(·|s_t) || π_ref(·|s_t))`를 페널티로 문다.

#### Training Objective

```
L_Stable-OPD(θ) = L_mix(θ) + β_KL · E_{s_t} [ KL(s_t) ]                  ... (6)

               = L_OPD(θ)
                 + λ_gold · E_{(x,y) ~ D_gold}[ L_SFT(θ; x, y) ]     ← rollout mixture
                 + β_KL   · E_{s_t}[ D_KL(π_θ(·|s_t) || π_ref(·|s_t)) ]  ← divergence 제약
                   β_KL > 0 : 정규화 강도
```

- L_OPD는 **GRPO-style clipped objective의 sequence advantage A_i를 token advantage A_{i,t} = r^KL_{i,t}로 바꾼 것**을 그대로 쓴다. 두 항이 서로 다른 고리를 잡으므로 **dual strategy**이며, ablation(§5.4)이 상보성을 뒷받침한다.

---

## 5. Experiments

### 5.1 Setup / Dataset

| 항목 | 내용 |
|---|---|
| 학습 데이터 | **OpenR1-Math-220k** (prompt: NuminaMath 1.5, reasoning trace: DeepSeek-R1) |
| 필터링 | 기본 94k-prompt split → **8192 token 초과** 또는 **Math-Verify 오답** 생성 제거 → **46k prompt** |
| 학습 분할 | **SFT 33k → OPD 13k** (표준 OPD 관행, 모든 baseline 동일 적용) |
| 병리 분석용 | OpenR1-Math-220k의 **13k subset** (3개 student-teacher 조합) |
| Student / Teacher | Qwen2.5-Math-1.5B, Qwen2.5-Math-7B / DeepSeek-R1-Distill-7B, OpenThinker3-7B(= OpenThinkerV3) |
| 평가 (6종) | AIME 2024, AIME 2025, AMC, Minerva, OlympiadBench, MATH500 |
| 평가 지표 / 온도 | AIME24 / AIME25 / AMC는 test set이 작아 **avg@32**, Minerva / Olympiad / MATH500은 **pass@1**. 온도 **0.6** |

### 5.2 Implementation Details

| 항목 | 값 |
|---|---|
| rollout batch size / prompt당 trajectory | **64** / **4** |
| mixture distillation | 배치 내 각 prompt에 **off-policy golden solution 1개**를 추가로 pairing |
| rollout temperature / optimizer / lr | 1.0 / Adam / **1e-6** |
| 하드웨어 | **4 × H200 GPU** |
| baseline | SFT(46k 전량), GRPO, SimpleRL-Zero, Oat-Zero, PRIME-Zero, OpenReasonerZero, 표준 OPD |

### 5.3 Main Results

#### (a) Qwen2.5-Math-7B backbone (teacher: OpenThinkerV3) — Table 1

| Model | Avg | MATH-500 | Minerva | Olympiad | AMC | AIME24 | AIME25 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B | 19.1 | 43.6 | 7.4 | 15.6 | 31.3 | 11.5 | 4.9 |
| Qwen2.5-Math-7B-Instruct | 37.6 | 80.4 | 32.7 | 41.0 | 48.5 | 12.5 | 10.2 |
| SimpleRL-Zero | 37.4 | 76.0 | 25.0 | 34.7 | 54.9 | **27.0** | 6.8 |
| OpenReasoner-Zero | 41.0 | 82.4 | 33.1 | 47.1 | 52.1 | 16.5 | 15.0 |
| PRIME-Zero | 40.8 | 81.4 | 39.0 | 40.3 | 54.0 | 17.0 | 12.8 |
| Oat-Zero | 43.8 | 78.0 | 34.6 | 43.4 | 61.2 | 33.4 | 11.9 |
| SFT | 44.1 | 82.6 | 40.8 | 43.7 | 52.8 | 22.2 | 22.3 |
| GRPO | 45.5 | 84.4 | 39.3 | 46.8 | **62.0** | 25.1 | 15.3 |
| OPD | 43.8 | 80.0 | 37.9 | 47.5 | 53.4 | 21.7 | 22.2 |
| **Stable-OPD** | **47.6** | **84.6** | **43.4** | **49.3** | 58.1 | 24.7 | **25.2** |

- **표준 OPD(43.8)는 SFT(44.1)와 GRPO(45.5)에 모두 진다.** dense token 감독과 on-policy 샘플을 다 갖고도 그렇다. 반대로 **잘 안정화된 on-policy distillation은 공들여 튜닝한 RLVR 파이프라인을 능가할 수 있다.**
- Stable-OPD는 **43.8 → 47.6 (+3.8)** 로 **전 방법 중 1위**. RLVR 계열 중 최강인 Oat-Zero(43.8)도 **+3.8** 차이로 앞선다.
- 6개 중 4개(MATH-500, Minerva, Olympiad, AIME25)에서 best. 특히 **Minerva 37.9 → 43.4 (+5.5)**, **MATH-500 80.0 → 84.6 (+4.6)**. AMC(53.4 → 58.1)·AIME24(21.7 → 24.7)는 개선되지만 Oat-Zero·GRPO에는 못 미친다 — 소수 문제 avg@32 평가의 분산을 감안해야 한다.

#### (b) Qwen2.5-Math-1.5B backbone (teacher: OpenThinkerV3) — Table 3

| Model | Avg | MATH-500 | Minerva | Olympiad | AMC | AIME24 | AIME25 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-1.5B | 16.0 | 28.0 | 9.6 | 21.2 | 26.4 | 7.2 | 3.6 |
| Qwen2.5-Math-1.5B-Instruct | 35.7 | 77.4 | 28.7 | 39.1 | 48.1 | 12.1 | 8.9 |
| SFT | 31.9 | 70.6 | 26.8 | 31.3 | 37.8 | 11.7 | 13.2 |
| GRPO | 30.1 | 61.8 | 26.8 | 32.0 | 40.2 | 11.8 | 7.7 |
| OPD | 28.9 | 56.7 | 23.4 | 31.0 | 35.9 | 11.1 | 15.0 |
| **Stable-OPD** | **36.1** | **73.9** | **32.6** | **37.4** | **43.0** | **13.8** | **16.0** |

- **평균 28.9 → 36.1 (+7.2)** — abstract의 "improves performance by 7.2% on average"가 이 수치다. **6개 벤치마크 전부에서 best.**
- MATH-500이 **56.7 → 73.9 (+17.2)** 로 가장 크게 오른다 — truncation collapse가 pass@1 지표에 얼마나 파괴적인지를 보여준다.
- 1.5B에서도 **표준 OPD는 base보다는 낫지만(16.0 → 28.9) SFT·GRPO에는 모두 진다.** Stable-OPD만이 Instruct 모델(35.7)을 넘어선다.

#### (c) Teacher를 바꿔도 유지되는가 — Table 4 (1.5B, † = R1-Distill-7B / ‡ = OpenThinkerV3)

| Model | Avg | MATH-500 | Minerva | Olympiad | AMC | AIME24 | AIME25 |
|---|---|---|---|---|---|---|---|
| OPD † | 28.0 | 58.4 | 22.4 | 26.9 | 36.2 | 10.9 | 13.1 |
| **Stable-OPD †** | 35.7 | 72.0 | **32.7** | 34.9 | **43.0** | **14.6** | **17.2** |
| **Stable-OPD ‡** (재게) | **36.1** | **73.9** | 32.6 | **37.4** | **43.0** | 13.8 | 16.0 |

- teacher가 R1-Distill-7B든 OpenThinkerV3든 개선폭이 **+7.7(28.0→35.7) / +7.2(28.9→36.1)** 로 거의 같다 → **감독 소스에 robust**.
- 같은 student 기준 **더 강한 teacher(OpenThinkerV3)가 평균을 더 올린다**(36.1 > 35.7) — "좋은 teacher가 더 정보량 있는 token 수준 감독을 준다"는 직관과 일치. 단 세부는 갈려서 OpenThinkerV3는 MATH-500·Olympiad에서, R1-Distill-7B는 Minerva·AIME24·AIME25에서 앞선다.

#### (d) 학습 동역학 — Figure 5 / 6 (RQ2)

| 설정 | 표준 OPD | **Stable-OPD** |
|---|---|---|
| 1.5B + OpenThinker3-7B (Fig 5) | 초기 안정 이후 rollout·eval의 truncation/repetition이 **급등해 고착** | **네 곡선 전부 flat.** truncation은 중간 수준 유지, repetition은 rollout·eval 모두 **0 근처** |
| 1.5B + DeepSeek-R1-Distill-7B (Fig 6) | eval truncation이 step 400 부근에서 **1.0 부근으로 수직 상승**, eval repetition도 동반 급등 | **완만한 우상향 drift만** 관측. 상승폭이 훨씬 작고, 발생 시점도 **학습 후반으로 밀린다** |

> 두 설정 모두 **abrupt truncation-repetition inflation regime에 진입하지 않는다.** 완전 제거는 아니지만(R1-Distill teacher에서는 후반 mild drift 잔존) **phase transition이 사라진다**는 것이 요점이다.

### 5.4 Ablation Study

Qwen2.5-Math-1.5B, teacher DeepSeek-R1-Distill-7B 기준 (Table 2).

| Method | Avg. Acc (%) | 증분 |
|---|---|---|
| Qwen2.5-MATH-1.5B (base) | 16.0 | — |
| OPD | 28.0 | +12.0 (base 대비) |
| OPD + KL | 29.7 | **+1.7** |
| **OPD + KL + Mixture Distillation** | **35.7** | **+6.0** |

- **KL 단독은 modest(+1.7).** policy drift 억제만으로 일관된 이득은 있지만, degenerate rollout이 배치를 지배하는 문제는 남는다. **Mixture를 얹으면 +6.0** (29.7 → 35.7)으로 훨씬 크게 뛰어 1.5B 설정 최강 변형이 된다.
- 저자 해석: **두 항은 상보적이다.** KL 정규화는 **token 수준의 급격한 policy shift**를 제한하고, mixture distillation은 **on-policy rollout이 열화되기 시작할 때 학습을 붙잡을 고품질 궤적의 안정적 지분**을 제공한다. 어느 한쪽만으로는 §4.1의 인과 사슬을 끊지 못한다. (보고된 것은 **누적 추가 형태**이며 KL만 뺀 조건 OPD+Mixture는 보고되지 않았다.)

---

## 6. Key Takeaways

1. **OPD는 "천천히 나빠지는" 게 아니라 어느 순간 무너진다.** 세 student-teacher 조합 전부에서 **약 30 OPD step**이라는 짧은 창 안에 rollout TruncRate가 0.5(1.5B) 또는 0.23(7B) 수준에서 **1 부근으로 점프**하고, RepRate가 0에서 **0.3~0.6**으로 튄다. Figure 1 기준 MATH500 accuracy는 step 280~310 부근에서 **0.72 → 0.60~0.63으로 8~10 %p 급락하고 회복하지 않는다.**

2. **범인은 reverse-KL advantage의 token 편중이다.** 반복 token은 국소 예측성이 높아 teacher가 선호하고, 붕괴 이후 평균 advantage가 **−0.02~−0.05 vs 일반 token −0.15~−0.17 (4~9배 유리)** 가 된다. 결정적 조건은 **빈도**다 — 붕괴 전에는 반복 token 비율이 ≈ 0이지만 붕괴 후 **약 30%** 를 차지하면서 gradient 방향을 뒤집는다.

3. **on-policy 샘플링이 편향을 자기강화 루프로 만든다.** 식 (4)가 보여주듯 OPD 업데이트는 **state 방문 빈도 × advantage 크기**의 곱이고, student가 자기 방문 분포를 스스로 갱신하므로 브레이크가 없다. 이것이 **off-policy distillation에서는 일어나지 않는 이유**이기도 하다.

4. **이것은 GRPO의 length bias가 아니다.** Dr.GRPO / DAPO가 다루는 것은 **sequence-level의 length-dependent gradient scaling**이며 objective reweighting으로 처리된다. 이 논문의 실패 모드는 **token-level 신호 편중 × on-policy 증폭**이고, 점진적 drift가 아니라 **abrupt phase transition**이다. 재정규화로는 못 잡는다.

5. **표준 OPD는 실제로 SFT·GRPO에 진다.** 7B에서 OPD 43.8 vs SFT 44.1 vs GRPO 45.5, 1.5B에서 OPD 28.9 vs SFT 31.9 vs GRPO 30.1. dense 신호와 on-policy 샘플을 둘 다 가졌음에도 그렇다면, 부족한 것은 **알고리즘이 아니라 안정성**이다.

6. **Stable-OPD의 두 항은 인과 사슬의 다른 고리를 끊고, 합쳐야 효과가 난다.** KL 정규화 단독은 **28.0 → 29.7 (+1.7)** 에 그치지만 mixture distillation을 더하면 **35.7 (+6.0)**. 결과는 1.5B 평균 **28.9 → 36.1 (+7.2)**, 7B **43.8 → 47.6 (+3.8)** 로 **RLVR 계열 전부를 앞선다.** teacher를 R1-Distill-7B로 바꿔도 **+7.7**로 개선폭이 유지된다.

7. **teacher likelihood는 hack될 수 있다.** 논문이 명시하듯 student의 반복은 **"teacher의 likelihood 기반 신호를 exploit 혹은 hack"** 하는 것으로 볼 수 있다. teacher 모델도 loss 수식도 고정된 채 붕괴가 일어난다는 사실이, 불안정성이 외부 요인이 아니라 **OPD 목적함수와 student-induced 데이터 수집의 상호작용에 내재**함을 보여준다.

---

## 7. 원문 블로그 대비 갱신점

| # | 원문 블로그(2025.10)의 주장 | 이 논문의 갱신 |
|---|---|---|
| ① | **OPD = dense × on-policy, 두 함정 동시 회피** | **부분 유지, 단 조건부.** dense × on-policy가 좋은 조합인 건 맞지만, **바로 그 on-policy성이 세 번째 함정(자기강화 반복 루프)을 새로 만든다.** 식 (4)의 self-reinforcing loop는 off-policy distillation에는 없고 OPD에만 있다. 즉 두 함정을 피하면서 **새 함정을 하나 만든다.** |
| ② | `advantage = −reverse_kl` **한 줄이면 구현 끝** | **반박.** 그 한 줄만으로 돌리면 세 조합 전부에서 truncation collapse가 났다. 실제로 필요한 것은 식 (6)의 **3항 objective** — OPD + λ_gold·SFT + β_KL·KL. |
| ③ | **discount 0으로 고분산의 원천이 사라지니 MiniLLM의 안정화 장치(length norm, clipping, LM loss)가 불필요하다** | **핵심 반박.** ㄱ) 이 논문은 GRPO-style **clipped** objective를 쓴다 — clipping을 되살렸다. ㄴ) mixture distillation의 λ_gold·L_SFT 항은 **MiniLLM의 LM loss와 기능적으로 같은 앵커 역할**이다. ㄷ) β_KL·KL(π_θ‖π_ref)는 MiniLLM에는 없던 새 제약이다. **discount 0이 없앤 것은 return의 분산이지 길이 폭발이 아니다.** 길이 문제는 discount와 무관한 경로(token 편중 × 방문 빈도)로 들어온다. → **"안정화 장치 불필요"는 성립하지 않는다.** |
| ④ | **reverse KL의 mode-seeking이 장점** | **부분 반박.** mode-seeking은 "teacher가 좋아하는 한 가지 전략에 집중"을 뜻하는데, 그 mode가 **degenerate한 반복 continuation**일 수 있다. Figure 4에서 붕괴 후 토큰의 30%가 반복 token이 되는 것이 그 결과다. **집중할 mode의 품질은 reverse KL이 보장해주지 않는다.** |
| ⑤ | **reverse KL은 unhackable하다** | **핵심 반박.** 논문 §1: *"student repetition may **exploit or hack** the teacher's likelihood-based signal."* reward가 "teacher와의 거리 그 자체"인 건 맞지만, **teacher likelihood가 높은 영역과 실제로 좋은 추론이 일치한다는 보장이 없다.** 반복은 teacher에게도 국소적으로 예측 가능하므로 높은 log π_T를 받는다 — student는 **정답을 잘 맞히지 않고도 목적함수를 낮출 수 있다.** 이것은 학습된 reward model의 허점을 파는 고전적 hacking과 형태만 다를 뿐 **목적함수 악용**이 맞다. 다만 차이는 있다: RLHF의 hacking은 프록시가 **틀려서** 생기고, 여기서는 프록시가 **맞는데도 국소 최적이 degenerate**해서 생긴다. |
| ⑥ | **partial rollout이 가능하다** | **경고 추가.** reward가 시퀀스 끝에 걸려 있지 않으니 원리상 가능은 하다. 그러나 붕괴 국면에서는 **truncation이 대부분의 rollout에 걸리고**(TruncRate ≈ 1), 그 잘린 궤적이 학습 데이터를 지배해 **biased gradient**를 만든다. 즉 "끝까지 안 뽑아도 된다"와 "끝까지 못 뽑는 상태가 정상이다"는 다르다. **TruncRate를 반드시 모니터링해야 한다.** |
| ⑦ | **작은 batch로도 된다** | **직접 검증 없음.** 이 논문은 rollout batch 64 × prompt당 4 trajectory로 고정했다. 다만 mixture distillation이 **매 minibatch에 golden 궤적을 넣어야** 효과가 나온다는 점은, 배치 구성이 blog가 시사한 것보다 **더 중요하다**는 간접 증거다. |

### 실무 체크리스트

**TruncRate·RepRate를 학습·validation 양쪽 대시보드에 올려라** (RepRate는 마지막 10,000자의 zlib 압축률 > 10으로 싸게 잰다. 두 지표가 동시에 튀는 순간이 collapse의 시작이며, validation에서도 거의 같은 step에 나타난다). 그리고 **golden SFT 항과 reference KL을 처음부터 함께 켜라** — ablation상 KL 단독은 +1.7에 그친다. π_ref는 **초기 student checkpoint**로 두면 된다.

---

[← 후속 연구 정리](opd_follow_up_research.md) · [원문 요약](on_policy_distillation.md)
