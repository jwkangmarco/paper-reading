# Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation

> **Venue**: arXiv preprint (2026.02.27, Tencent HY 테크리포트 형식)
> **Authors**: Wenkai Yang¹, Weijie Liu², Ruobing Xie², Kai Yang², Saiyong Yang², Yankai Lin¹
> (¹Gaoling School of AI, Renmin University of China / ²LLM Department, Tencent)
> **arXiv**: 2602.12125v2 [cs.LG] (26 Feb 2026)
> **Code**: https://github.com/RUCBM/G-OPD

**한 줄 정의**: OPD의 목적함수를 "reference model π_ref + reward scaling factor λ"로 일반화(**G-OPD**)하고, λ > 1로 두어 **teacher의 성능 상한을 넘어서는** student를 만드는 방법(**ExOPD**).

---

## 1. Background

### On-Policy Distillation의 현황

- OPD는 student가 생성한 trajectory 위에서 teacher의 logit 분포에 정렬시키는 post-training 패러다임이다 (Agarwal et al. 2024; Lu & Thinking Machines Lab 2025). 두 용도가 이미 실증되었다.
  - **Multi-task post-training**: 서로 다른 도메인 RL로 얻은 능력들을 원래 base 모델로 (거의) 손실 없이 병합 (Xiao et al. 2026, MiMo-v2-Flash).
  - **Strong-to-weak distillation**: 큰 teacher의 능력을 작은 student로 효율적으로 이전 (Gu et al. 2024; Yang et al. 2025a).
- 저자들의 문제의식: **"왜 되는지"에 대한 mechanistic 이해가 없다.** 경험적 성공만 있고 이론적 위치가 비어 있어 잠재력이 덜 탐색되었다.

### 기존 방법의 한계

| 방법 | 신호 | 분포 | 한계 |
|---|---|---|---|
| Off-policy distillation (KD / SFT) | dense (token CE 또는 logit KL) | teacher가 만든 것 | **off-policy 본질** — teacher 행동을 모방할 뿐, 자기 행동이 유도한 reward로부터 배우지 못해 테스트 시 적응·일반화에 실패 |
| On-policy RL (GRPO 등) | **sparse** — 응답 완료 후 최종 token에만 reward | 자기 것 | 최적화가 비효율적·비효과적 (Cui et al. 2025) |
| **Standard OPD** | dense (per-token reverse KL) | 자기 것 | ① reward 항과 KL 정규화 가중치가 **항상 1:1로 고정** ② reference model 선택의 자유를 활용하지 못함 ③ **teacher가 성능 상한** |
| **G-OPD / ExOPD (본 논문)** | dense, **λ로 가중치 조절** | 자기 것 | teacher 상한을 돌파. 단 λ≠1이면 π_ref forward 비용 추가 |

- 대안 계열로 **weight extrapolation (ExPO, Zheng et al. 2025)** 이 있다. 파라미터 공간에서 base→aligned 방향으로 외삽하는데, **통제 가능성이 없다** (§5.3 정량 비교).

---

## 2. Motivation

### 핵심 통찰 1: OPD는 "β=1로 못 박힌" dense KL-constrained RL이다

표준 OPD 목적함수에 제3의 reference model π_ref를 **더하고 빼는 항등 변형**만 해도, 그것이 dense reward를 갖는 KL-제약 RL과 정확히 같아진다는 것이 이 논문의 출발점이다 (Eq. 7).

```
J_OPD(θ) = min_θ  E_{x~D, y~π_θ(·|x)} [ D_KL( π_θ(y|x) || π*(y|x) ) ]
         = min_θ  E [ log π_θ(y|x) − log π*(y|x) ]
         = max_θ  E [ log π*(y|x) − log π_θ(y|x) ]
         = max_θ  E [ ( log π* − log π_ref ) − ( log π_θ − log π_ref ) ]
         = max_θ  E [ log( π*(y|x) / π_ref(y|x) )  −  D_KL( π_θ(y|x) || π_ref(y|x) ) ]
```

> **Remark (논문 원문)**: π_ref를 도입하면 OPD는 reward가 `r(x,y) = log(π*/π_ref)`이고,
> KL 정규화가 π_θ와 π_ref 사이에 걸리며, **reward와 KL 항의 가중치가 항상 같은(β = 1)**
> KL-constrained RL과 동치가 된다.

여기서 두 가지가 즉시 따라온다. ① **π_ref는 무엇이든 될 수 있다** — 마지막 줄에서 π_ref를 되돌리면 원래 Eq. (4)로 정확히 환원되므로 λ=1일 때 π_ref 선택은 목적함수 값에 영향을 주지 않는 자유 변수다. ② **β = 1은 임의의 제약이다** — 표준 RL은 β를 하이퍼파라미터로 두는데 OPD만 이 손잡이를 잃어버렸다. 이걸 되돌려 주면 무엇이 되는가? 이것이 G-OPD.

### 핵심 통찰 2: reward 가중치를 1보다 크게 하면 teacher 너머로 외삽된다

G-OPD 목적함수의 최적해는 닫힌 형태로 구할 수 있고, 그 형태가 곧 "왜 λ>1이 teacher를 넘는가"를 설명한다 (Eq. 12).

```
log π_θ(y|x) = λ · log π*(y|x) + (1 − λ) · log π_ref(y|x)
             = log π*(y|x) + (λ − 1) · ( log π*(y|x) − log π_ref(y|x) )
```

| λ 범위 | 이름 | 의미 |
|---|---|---|
| λ = 0 | — | reward 항 소멸 → student는 π_ref(= 초기 상태)에 머문다 |
| 0 < λ < 1 | **Reward interpolation** | log-확률 분포가 π_ref와 π*의 **선형 보간**. 성능·응답 길이가 그 사이에 위치 |
| **λ = 1** | **Standard OPD** | 정확히 teacher 분포를 목표로 함 |
| **λ > 1** | **Reward extrapolation (ExOPD)** | teacher 분포에 **추가 shift 항 (λ−1)(log π* − log π_ref)** 을 얹음 → teacher를 지나쳐 같은 방향으로 더 간다 |

- 직관: `log π* − log π_ref`는 "teacher가 reference 대비 **어느 방향으로** 개선되었는가"를 가리킨다. λ>1은 그 개선 방향을 **연장**한다.
- 이는 DPO의 implicit reward (Rafailov et al. 2023, Eq. 10)와 같은 형태다: `r(x,y) = β log(π_θ/π_ref) + β log Z(x)`. log Z(x)는 x에만 의존하는 상수라 `log(π*/π_ref)`는 실제 reasoning reward의 well-defined proxy로 볼 수 있다.
- 중요한 단서: OPD의 implicit reward는 **π*가 π_ref로부터 RL로 얻어졌을 필요가 없다.** π*와 π_ref는 크기가 다른 모델이어도 "reference 분포에서 expert 분포로의 log-probability shift"를 포착하므로 의미 있는 학습 신호가 된다.

---

## 3. Contributions

1. **OPD ↔ dense RL 이론적 연결**: OPD가 "token-level reward 함수와 KL 정규화가 항상 동일 가중치이고 reference model은 임의로 고를 수 있는 dense KL-constrained RL"의 특수 케이스임을 유도로 증명 (Eq. 7, Appendix A).
2. **G-OPD 프레임워크 제안**: flexible reference model π_ref + reward scaling factor λ를 도입한 일반화 정식화 (Eq. 11). λ는 본질적으로 Eq. (2)의 1/β.
3. **ExOPD (λ > 1) 제안 및 검증**: λ=1.25에서 standard OPD와 domain teacher를 **일관되게** 상회. multi-teacher 병합에서 **모든 domain teacher를 모든 벤치마크에서 넘는 유일한 방법** (SFT·ExPO·OPD 대비).
4. **strong-to-weak에서의 reward correction 제안**: π_ref를 student base가 아닌 **teacher의 pre-RL base 모델**로 바꾸면 reward가 더 정확해진다는 분석과 실험적 검증 (수학 28.1→28.7, 코드 51.3→52.3).
5. **budget-controlled reasoning 부산물**: 0<λ<1 구간에서 성능과 응답 길이가 λ에 대해 **단조 증가** → λ 하나로 추론 예산을 조절할 수 있음.

---

## 4. Method

### 4.1 표준 OPD의 gradient (출발점)

```
J_OPD(θ) = min_θ  E_{x~D, y~π_θ(·|x)} [ D_KL( π_θ(y|x) || π*(y|x) ) ]        ... (4)

정확한 gradient (Appendix A, Eq. 20):
∇_θ J_OPD = E [ Σ_{t=1..T} ( Σ_{t'=t..T} ( log π_θ(y_t'|x,y_<t') − log π*(y_t'|x,y_<t') ) ) ∇_θ log π_θ(y_t|x,y_<t) ]

discount factor 0을 적용한 실사용 근사 (Eq. 21):
∇_θ J_OPD ≈ E [ Σ_{t=1..T} ( log π_θ(y_t|x,y_<t) − log π*(y_t|x,y_<t) ) ∇_θ log π_θ(y_t|x,y_<t) ]
```

- Appendix A의 핵심 보조정리: t' < t인 교차항의 기대값이 **정확히 0**이다 (Eq. 19). `E_{y_t}[∇_θ log π_θ(y_t|·)] = ∇_θ Σ_{y_t} π_θ = ∇_θ 1 = 0`이기 때문. 그래서 이중합이 t' ≥ t로 축약된다. 여기서 `−( log π_θ(y_t|·) − log π*(y_t|·) )`가 token-level advantage로 작동한다 → **dense credit assignment**.

### 4.2 dense reward vs sparse reward

```
RL (sparse):   r_t^RL  = 0                                  (t = 1, ..., T−1)
                       = Outcome Reward                      (t = T)          ... (8)

OPD (dense):   r_t^OPD = log( π*(y_t|x,y_<t) / π_ref(y_t|x,y_<t) )            ... (9)
```

### 4.3 G-OPD — 프레임워크

**G-OPD는 방법이 아니라 프레임워크다.** π_ref와 λ 두 손잡이를 노출시킨 일반화 정식화이며, λ=1 & π_ref=student 초기값이면 표준 OPD로 환원된다.

```
J_G-OPD(θ) = max_θ  E_{x~D, y~π_θ(·|x)} [ λ · log( π*(y|x) / π_ref(y|x) )
                                          − D_KL( π_θ(y|x) || π_ref(y|x) ) ]   ... (11)
```

- λ는 본질적으로 표준 RL 목적함수 Eq. (2)의 **1/β**에 해당한다. RL 대비 이점은 dense credit assignment + reference model 선택의 자유, OPD 대비 이점은 reward 항 가중치 조절이다.

동일한 KL 정규화 강도 아래에서 비교하기 위해 다음 등가 형태로 다시 쓸 수 있다 (Eq. 13).

```
J_G-OPD(θ) = max_θ E [ (λ − 1) · log( π*(y|x) / π_ref(y|x) )
                       − D_KL( π_θ(y|x) || π*(y|x) ) ]
```

이 형태가 중요한 이유: **λ=1이면 첫 항이 사라져 π_ref가 완전히 무관해진다.** 즉 π_ref의 선택이 의미를 갖는 것은 오직 λ≠1일 때다.

### 4.4 ExOPD — G-OPD의 인스턴스

| 구분 | 정의 |
|---|---|
| **G-OPD** | 프레임워크. flexible π_ref + reward scaling factor λ를 갖는 일반화 목적함수 (Eq. 11) |
| **Reward interpolation** | G-OPD의 0 < λ < 1 영역 |
| **ExOPD** | G-OPD의 **λ > 1** 인스턴스. 논문 전 실험에서 **λ = 1.25로 고정** (추가 튜닝 없음) |
| **ExOPD w/ reward correction** | ExOPD + π_ref를 teacher의 pre-RL base 모델로 교체 (strong-to-weak 전용) |

### 4.5 Reward correction (strong-to-weak 전용)

strong-to-weak 설정에서 π_ref 후보는 둘이다.

| 선택 | reward 형태 | 평가 |
|---|---|---|
| (i) **π_base^student** (기본값) | `log( π* / π_base^student )` | student와 teacher base 사이에 **내재적 지식·용량 격차**가 있어 noisy |
| (ii) **π_base^teacher** (reward correction) | `log( π* / π_base^teacher )` | teacher 자신의 RL post-training이 유도한 implicit reward → **Eq. (10) 기준 well-defined** |

correction 연산: `log(π*/π_base^student) + log(π_base^student/π_base^teacher) = log(π*/π_base^teacher)`

**트레이드오프 (논문이 반복 강조)**: ① teacher의 **pre-RL variant에 접근**할 수 있어야 한다 — 대부분의 공개 모델은 제공하지 않는다. ② **연산 비용 증가** — `log π_base^teacher`는 큰 모델의 forward라 `log π_base^student`보다 비싸다. 반면 multi-teacher 설정은 π_ref가 자연히 원래 base 모델(= teacher의 pre-RL 상태)이라 reward가 처음부터 Eq. (10) 형태로 정확하며 이 문제가 없다.

#### Training Objective

```
∇_θ J_G-OPD(θ) = E_{x~D, y~π_θ(·|x)} [ Σ_{t=1..T} A_t^{G-OPD} · ∇_θ log π_θ(y_t|x,y_<t) ]   ... (14)

A_t^{G-OPD} = ( log π_θ(y_t|x,y_<t) − log π*(y_t|x,y_<t) )
              + (λ − 1) · ( log π_ref(y_t|x,y_<t) − log π*(y_t|x,y_<t) )
```

- 부호 규약 주의: Eq. (14)는 **최소화(gradient descent) 기준**이라 원문 블로그의 `advantage = −reverse_kl`과 부호가 반대일 뿐 같은 것이다. 첫 항 = 표준 OPD의 per-token reverse KL, 둘째 항 = **λ−1로 가중된 extrapolation 보정항**.
- **λ = 1이면 둘째 항이 0** → π_ref forward가 아예 필요 없다. 즉 "advantage 한 줄"로 끝나는 것은 표준 OPD뿐이고, ExOPD는 **매 step 세 모델(π_θ, π*, π_ref)의 logprob**이 필요하다.

#### 학습 vs 추론

| 단계 | 과정 |
|---|---|
| **학습 (ExOPD)** | student가 rollout 생성 → teacher π*와 reference π_ref가 **같은 token 시퀀스의 logprob 계산** → A_t를 Eq. (14)로 조립 → policy gradient 업데이트 |
| **학습 (표준 OPD, λ=1)** | π_ref forward 불필요. teacher logprob 1회면 충분 |
| **추론** | student 단독. teacher·reference 모두 배포에 관여하지 않음 |

---

## 5. Experiments

### 5.1 Setup / Dataset

| 항목 | Same-size (multi-teacher, §4.1) | Strong-to-weak (§4.2) |
|---|---|---|
| Student | Qwen3-4B-Non-Thinking | Qwen3-1.7B / Qwen3-4B-Non-Thinking |
| Teacher | Qwen3-4B-Non-Thinking-RL-Math, -RL-Code (student에 도메인별 GRPO 적용) | Qwen3-30B-A3B-Instruct-2507 |
| π_ref (기본) | Qwen3-4B-Non-Thinking (= base, 자연스럽게 고정) | student base model |
| Math 데이터 | DeepMath 중 난이도 ≥ 6인 **57K** samples | 동일 |
| Code 데이터 | Eurus-RL-Code **25K** samples | (math 도메인 위주) |
| Math 평가 | AIME24, AIME25, HMMT25 (Feb), HMMT25 (Nov) | 동일 |
| Code 평가 | HumanEval+, MBPP+, LiveCodeBench (v6, 2025.02~05) | — |

- distillation 데이터 = RL 데이터와 동일. RL reward는 수학 정답 시 1.0, 코드 전 unit test 통과 시 1.0, 아니면 0.0.
- 평가: temperature 1.0, top-p 1.0, max generation 16,384. **수학은 문제당 32 sample, 코드는 4 sample** 평균 정확도. 수학 검증은 Math-Verify.

### 5.2 Implementation Details

프레임워크는 **verl**. GRPO와 G-OPD 모두 **token-level rollout correction** (Liu et al. 2025b)을 적용해 training-inference mismatch를 완화.

| GRPO (math / code) | 값 | G-OPD (공통) | 값 |
|---|---|---|---|
| Train / Micro batch size | 128 / 128 | Batch size | **1024** |
| Rollout n | 8 | Rollout n | **1** |
| Max prompt / response | 2048 / 16384 (code는 8192) | Max prompt / response | 2048 / 16384 |
| Temperature / Top-p | 1.0 / 1.0 | Temperature / Top-p | 1.0 / 1.0 |
| LR | 1e-6 | LR | **1e-5** |
| Optimization steps | 500 (math) / 300 (code) | Optimization steps | **50** (same-size) / **100** (strong-to-weak) |
| KL coefficient | **0.0** | — | — |

- Appendix B의 관측: 동일한 `prompt size × rollout n` 조건에서 **prompt size를 키우는 쪽이 수렴이 더 매끄럽다** (그래서 batch 1024 × rollout 1). 또 **distillation step을 더 늘리면 overfitting으로 일반화 성능이 떨어진다** — 50/100 step에서 멈춘 이유.
- SFT baseline: batch 1024, max seq len 32,768, warm-up ratio 0.05, LR 1e-5. teacher가 문제당 생성한 trajectory 수를 student rollout 수와 맞추고 optimization step도 동일하게 유지.
- λ 스윕: **{0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}**. λ=0은 초기 상태, λ=1은 표준 OPD.

### 5.3 Main Results

#### (a) λ 스윕 — single-teacher, same-size (Figure 2/3/4)

| 관측 | 내용 |
|---|---|
| **λ = 1 (표준 OPD)** | teacher의 post-training 행동을 **완전히 복구**. 평가 정확도와 응답 길이 모두 teacher에 근접 |
| **0 < λ < 1 (interpolation)** | 성능·응답 길이가 λ에 대해 **단조 증가**, base와 teacher 사이에 위치 → **budget-controlled reasoning**에 활용 가능 |
| **λ = 1.25 (ExOPD)** | **모든 설정에서 OPD와 domain teacher를 일관되게 상회** |
| **λ = 1.5 (과도한 외삽)** | **불안정, 성능 저하**. Eq. (9)의 implicit reward를 hacking — bias로 log ratio가 과도하게 큰 token의 peak를 공격적으로 fitting |
| 응답 길이 | ExOPD student의 길이는 계속 증가 — implicit reward의 **length bias** (Yang et al. 2025d) 가능성 |

#### (b) Single-teacher / Multi-teacher, same-size (Table 2)

teacher = 해당 도메인 RL 모델 (math: 46.0 avg / code: 61.2 avg), student 초기값 = Qwen3-4B-Non-Thinking (math 15.4 / code 52.4).

| Method | AIME24 | AIME25 | HMMT25(Feb) | HMMT25(Nov) | **Math Avg** | HumanEval+ | MBPP+ | LCB | **Code Avg** |
|---|---|---|---|---|---|---|---|---|---|
| Teacher | 58.0 | 54.6 | 32.5 | 38.9 | **46.0** | 86.0 | 70.2 | 27.3 | **61.2** |
| Student | 21.5 | 21.9 | 10.0 | 8.0 | 15.4 | 74.7 | 64.7 | 17.9 | 52.4 |
| *Single-Teacher* | | | | | | | | | |
| ExPO | 58.7 | 55.2 | 32.4 | 37.0 | 45.8 | 84.8 | 70.2 | 28.0 | 61.0 |
| OPD | 60.7 | 55.0 | 32.4 | 37.9 | 46.5 | 85.2 | 69.9 | 27.3 | 60.8 |
| **ExOPD** | **62.7** | **56.1** | **33.9** | **39.3** | **48.0** | **86.9** | **70.7** | **28.6** | **62.1** |
| *Multi-Teacher* | | | | | | | | | |
| SFT | 58.5 | 53.3 | 30.7 | 34.8 | 44.3 | 86.4 | 69.6 | 26.4 | 60.8 |
| ExPO | 57.5 | 54.5 | 31.7 | 36.3 | 45.0 | 86.7 | **72.0** | **29.0** | **62.6** |
| OPD | 60.6 | 54.1 | 32.5 | 38.3 | 46.4 | 84.6 | 69.5 | 27.6 | 60.6 |
| **ExOPD** | **61.0** | **56.0** | **34.4** | **39.2** | **47.7** | 86.3 | 70.6 | **29.0** | 62.0 |

**핵심 수치 분석**
- **teacher 상한 돌파**: single-teacher ExOPD는 math avg **48.0 (teacher 46.0 대비 +2.0)**, code avg **62.1 (+0.9)**. 반면 OPD는 math +0.5, code −0.4로 **사실상 teacher에 수렴할 뿐**이다.
- **multi-teacher에서 유일하게 모든 벤치마크 초과**: ExOPD는 4개 수학 + 3개 코드 벤치마크 **전부**에서 해당 domain teacher를 넘는다 (HMMT25 Nov 39.2 vs 38.9, HumanEval+ 86.3 vs 86.0 처럼 마진이 작은 항목 포함). SFT는 math avg 44.3으로 teacher보다 **1.7 낮아** off-policy 모방의 한계를 드러낸다.
- **ExPO(weight extrapolation)는 통제 불가**: multi-teacher에서 code는 62.6으로 ExOPD(62.0)보다 높지만 math는 **45.0으로 teacher(46.0)보다 1.0 낮다.** "모든 domain teacher를 일관되게 넘는" 성질을 보장하지 못한다 — 이것이 weight-space 외삽과 reward-space 외삽의 결정적 차이다.

#### (c) "teacher를 덜 학습시켜서 이긴 것 아닌가?" (Table 1)

math domain teacher에 RL을 **100 step 더** 돌린 것과, ExOPD **50 step**을 비교.

| Method | AIME24 | AIME25 | HMMT25(Feb) | HMMT25(Nov) | Avg |
|---|---|---|---|---|---|
| Teacher | 58.0 | 54.6 | 32.5 | 38.9 | 46.0 |
| + continued RL (100 steps) | 60.9 | 55.6 | 32.8 | 38.4 | 46.9 (+0.9) |
| **ExOPD (50 steps)** | **62.7** | **56.1** | **33.9** | **39.3** | **48.0 (+2.0)** |

> RL을 100 step 더 태워도 +0.9인데, **절반인 50 step의 ExOPD가 +2.0**을 얻는다. HMMT25(Nov)에서는 continued RL이 오히려 −0.5로 후퇴한다.

#### (d) Strong-to-weak distillation (Table 3)

teacher = Qwen3-30B-A3B-Instruct-2507 (math avg 59.7).

| Method | AIME24 | AIME25 | HMMT25(Feb) | HMMT25(Nov) | Avg |
|---|---|---|---|---|---|
| Teacher | 74.7 | 62.8 | 44.2 | 57.2 | 59.7 |
| **Student: Qwen3-1.7B-Non-Thinking** | | | | | |
| Base | 12.3 | 11.4 | 6.8 | 4.5 | 8.8 |
| SFT | 18.1 | 20.5 | 9.2 | 6.3 | 13.5 |
| OPD | 33.0 | 28.7 | 15.7 | 14.9 | 23.1 |
| **ExOPD** | **37.3** | **31.5** | **16.2** | **16.5** | **25.4 (+2.3)** |
| **Student: Qwen3-4B-Non-Thinking** | | | | | |
| Base | 21.5 | 21.9 | 10.0 | 8.0 | 15.4 |
| SFT | 45.4 | 40.9 | 22.4 | 31.6 | 35.1 |
| OPD | 55.0 | 48.0 | 29.8 | 37.7 | 42.6 |
| **ExOPD** | **58.7** | **50.8** | **33.0** | **38.8** | **45.3 (+2.7)** |

- 여기서는 teacher(59.7)를 넘지 못한다 — 1.7B/4B가 30B-A3B를 능가할 수는 없다. **이 설정에서 ExOPD의 역할은 "teacher 초과"가 아니라 "OPD의 한계를 밀어올리는 것"**이다: 1.7B 23.1→25.4 (**+2.3**), 4B 42.6→45.3 (**+2.7**). SFT 대비로는 4B 기준 35.1 vs 45.3 (**+10.2**).
- 저자 해석: `log(π*/π_base^student)`가 대소 모델 간 지식 격차·분포 편향 때문에 **noisy함에도 불구하고**, reward 외삽은 여전히 OPD의 한계를 밀어낸다.

### 5.4 Ablation Study

#### (a) Reward correction (Figure 6)

student = Qwen3-1.7B-Non-Thinking, teacher = Qwen3-4B-Non-Thinking-RL-Math / -RL-Code, pre-RL variant = Qwen3-4B-Non-Thinking.
(Qwen3-30B-A3B-Instruct-2507의 pre-RL variant를 구할 수 없어 4B 계열로 대체한 실험이다.)

| Method | Math Avg (4 benchmarks) | Code Avg (3 benchmarks) |
|---|---|---|
| SFT | 22.7 | 47.0 |
| OPD | 27.5 | 50.5 |
| ExOPD (π_ref = student base) | 28.1 | 51.3 |
| **ExOPD + reward correction (π_ref = teacher pre-RL base)** | **28.7** | **52.3** |

- 개선폭 math **+0.6**, code **+1.0**은 ExOPD가 OPD에 얹는 폭(+0.6, +0.8)과 비슷한 규모다. 즉 **π_ref 선택은 λ 조정과 대등한 수준의 설계 변수**다. 다만 더 큰 reference 모델의 logprob을 매 step 계산해야 하므로 default ExOPD보다 비싸고 pre-RL 체크포인트 접근 가정이 필요하다(논문이 "limitation"으로 반복 명시).

#### (b) 충분히 학습된 teacher에서도 성립하는가 (Table 8, Appendix C)

domain teacher를 **1200 RL step**까지 학습시킨 강한 teacher (math avg 51.9 / code avg 63.1) 기준.

| Method | Math Avg | vs Teacher | Code Avg | vs Teacher |
|---|---|---|---|---|
| Teacher | 51.9 | — | 63.1 | — |
| Student | 15.4 | −36.5 | 52.4 | −10.7 |
| *Single-Teacher* OPD | 51.7 | −0.2 | 62.9 | −0.2 |
| *Single-Teacher* **ExOPD** | **52.2** | **+0.3** | **64.3** | **+1.2** |
| *Multi-Teacher* OPD | 51.9 | +0.0 | 62.0 | −1.1 |
| *Multi-Teacher* **ExOPD** | **52.5** | **+0.6** | **64.4** | **+1.3** |

> **teacher가 강해질수록 마진은 줄지만 부호는 유지된다.** OPD는 teacher 근처에서 딱 멈추고(−0.2 ~ 0.0), ExOPD만 (+0.3 ~ +1.3) 위로 나간다. multi-teacher OPD의 code −1.1은 병합 시 도메인 간 간섭으로 손실이 나는 것을 보여주는데, ExOPD는 같은 조건에서 +1.3이다.

#### (c) 학습 동역학 (Figure 5, multi-teacher)

ExOPD는 OPD 대비 ① **training reward가 더 높고** (목적함수가 reward 항에 더 큰 가중치를 주므로 당연), ② **응답 길이가 더 길며** (Figure 4의 정확도-길이 추세와 일치), ③ **entropy가 더 높다** (긴 응답 생성 경향이 응답 다양성을 키운 결과로 해석).

---

## 6. Key Takeaways

1. **OPD는 β=1로 못 박힌 dense KL-constrained RL이다.** π_ref를 더하고 빼는 항등 변형만으로 `J_OPD = max E[ log(π*/π_ref) − D_KL(π_θ||π_ref) ]`가 유도되며, 여기서 **reward 함수와 KL 정규화의 가중치가 항상 1:1**임이 드러난다. 이 관찰이 논문 전체의 뿌리다.

2. **G-OPD(프레임워크)와 ExOPD(방법)는 다르다.** G-OPD는 flexible π_ref + reward scaling factor λ를 노출한 일반화 정식화(Eq. 11)이고, ExOPD는 그중 **λ > 1** 인스턴스다. λ=1이면 π_ref가 목적함수에서 소거되므로, **π_ref 선택이 의미를 갖는 것은 오직 λ≠1일 때**다.

3. **teacher 상한은 깨진다 — λ=1.25로.** 최적해가 `log π_θ = log π* + (λ−1)(log π* − log π_ref)`이므로 λ>1은 teacher의 개선 방향을 연장한다. same-size single-teacher에서 math avg **48.0 vs teacher 46.0 (+2.0)**, code **62.1 vs 61.2 (+0.9)**. 표준 OPD는 각각 +0.5 / −0.4로 사실상 teacher에 수렴할 뿐이다.

4. **multi-teacher 병합에서 모든 domain teacher를 넘는 유일한 방법.** ExOPD는 수학 4개 + 코드 3개 벤치마크 **전부**에서 해당 domain teacher를 상회한다. 비교 대상인 weight extrapolation **ExPO는 통제 불가** — code avg는 62.6으로 좋지만 math avg는 **45.0으로 teacher(46.0)보다 낮다.** reward-space 외삽이 weight-space 외삽보다 **예측 가능**하다는 것이 핵심 차이.

5. **λ에는 명확한 상한이 있다 — reverse KL은 무조건 unhackable하지 않다.** λ=1.5는 불안정과 성능 저하를 낳는다. 원인은 Eq. (9)의 implicit reward hacking: bias 때문에 log ratio가 과도하게 큰 token의 peak를 공격적으로 fitting한다. 응답 길이가 계속 늘어나는 **length bias**도 함께 관찰된다. λ=1.25가 논문 전반의 안전한 기본값.

6. **strong-to-weak에서 ExOPD의 역할은 "상한 돌파"가 아니라 "OPD 한계 밀어내기"다.** Qwen3-30B-A3B teacher(59.7)를 1.7B/4B가 이길 수는 없다. 그러나 OPD 대비 **1.7B 23.1→25.4 (+2.3)**, **4B 42.6→45.3 (+2.7)**. `log(π*/π_base^student)`가 대소 모델 격차로 noisy함에도 외삽은 작동한다.

7. **reward correction은 λ 조정과 대등한 효과지만 비용이 붙는다.** π_ref를 teacher의 pre-RL base로 바꾸면 reward가 Eq. (10) 기준 well-defined해져 math **28.1→28.7**, code **51.3→52.3**. 그러나 ① teacher의 pre-RL 체크포인트 접근 가정 ② 큰 reference 모델의 logprob 계산 비용이라는 두 제약이 붙는다. multi-teacher 설정에서는 π_ref가 자연히 원래 base 모델이라 이 문제가 애초에 없다.

---

## 7. 원문 블로그 대비 갱신점

| # | 블로그(2025.10.27)의 주장 | 이 논문의 판정 | 근거 |
|---|---|---|---|
| ① | OPD = dense × on-policy, 두 함정 동시 회피 | **정밀화** | 맞지만 불완전한 서술. 정확히는 "reward 함수 = log(π*/π_ref), β=1로 고정된 **dense KL-constrained RL**"의 특수 케이스 (Eq. 7). 고정된 β가 곧 미사용 자유도임을 드러냄 |
| ② | `advantage = −reverse_kl` 한 줄이면 구현 끝 | **부분 반박** | λ=1일 때만 참이다. λ≠1이면 A_t에 `(λ−1)(log π_ref − log π*)` 항이 추가되어 **매 step 세 모델의 logprob**이 필요하다 (Eq. 14). 논문이 λ≠1의 추가 연산 비용을 Remark에서 명시 |
| ③ | discount 0 덕분에 안정화 장치 불필요 | **유지 + 보강** | Appendix A에서 t'<t 교차항의 기대값이 정확히 0임을 증명(Eq. 19)하여 근사의 근거를 제공. 다만 실제 학습에는 verl의 **token-level rollout correction**이 들어갔고, distillation step을 늘리면 overfitting이 나므로 50/100 step에서 멈춘다 |
| ④ | reverse KL의 mode-seeking이 장점 | **유지, 대상 변경** | G-OPD에서 정규화 항은 `D_KL(π_θ||π_ref)`(Eq. 11) 또는 `D_KL(π_θ||π*)`(Eq. 13)로 재배치된다. mode-seeking 성질 자체는 유지되나, **"어느 분포에 대해 mode-seeking인가"가 설계 변수**가 됨 |
| ⑤ | reverse KL은 **unhackable**하다 | **반박** | `log(π*/π_ref)`는 어디까지나 true reward의 **proxy**다. λ=1.5에서 student가 implicit reward를 hacking — bias로 log ratio가 큰 token의 peak를 공격적으로 fitting해 불안정·성능 저하. 응답 길이의 무한 증가(length bias)도 hacking의 징후 |
| ⑥ | **teacher가 사실상 성능 상한** | **정면 반박 — 이 논문의 핵심** | λ>1의 최적해가 `log π* + (λ−1)(log π* − log π_ref)`라 teacher 분포를 지나쳐 간다. same-size single-teacher math **48.0 vs 46.0**, multi-teacher에서 **7개 벤치마크 전부 domain teacher 초과**. teacher에 RL 100 step을 더 태워도 +0.9인데 ExOPD 50 step은 +2.0 |
| ⑦ | AIME'24 74.4% / 1,800 GPU h로 RL 압도 | **미검증(범위 밖)** | 이 논문은 GPU hour 비용 회계를 보고하지 않는다. 다만 continued RL 100 step vs ExOPD 50 step 비교(Table 1)는 **더 적은 step으로 더 큰 이득**이라는 방향성과 일치 |
| ⑧ | continual learning에 유망 | **미검증(범위 밖)** | 다루지 않음. multi-teacher 능력 병합은 다루지만 시간축 지속 학습은 future work |

**추가로 이 논문이 새로 여는 축**: ① **π_ref = 설계 변수** — 블로그는 teacher만 손잡이로 봤으나 reference model 선택이 reward 정의 자체를 바꾼다(reward correction: math +0.6, code +1.0). ② **λ = budget 손잡이** — 0<λ<1에서 성능·응답 길이가 단조 증가하므로 λ 하나로 추론 예산을 연속 조절 가능. ③ **weight extrapolation과의 경계 확정** — ExPO는 파라미터 공간, ExOPD는 reward 공간이며 후자만이 "모든 domain teacher를 일관되게 넘는" 통제 가능성을 준다.

---

[← 후속 연구 정리](opd_follow_up_research.md) · [원문 요약](on_policy_distillation.md)
