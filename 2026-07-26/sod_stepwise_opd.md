# SOD: Step-wise On-policy Distillation for Small Language Model Agents

> **Venue**: Preprint (arXiv, 2026.05.08)
> **Authors**: Qiyong Zhong, Mao Zheng, Mingyang Song (공동 1저자), Xin Lin, Jie Sun, Houcheng Jiang, Xiang Wang, Junfeng Fang
> (Zhejiang University / LLM Department, Tencent / USTC / National University of Singapore)
> **arXiv**: 2605.07725v1
> **Code**: https://github.com/YoungZ365/SOD

**한 줄 정의**: multi-turn tool-integrated reasoning(TIR)에서 OPD의 per-token 균일 감독을 버리고,
**reasoning step 단위로 student-teacher divergence를 재어 그 비율로 distillation 강도를 조절**한다.
잘못된 tool call로 오염된 구간에서는 teacher 신호를 끄고, 정렬된 구간에서는 full-strength로 남긴다.

---

## 1. Background

### Tool-Integrated Reasoning(TIR)과 SLM의 현황

- **TIR** = 모델이 외부 환경(코드 인터프리터 등)과 여러 턴 상호작용하며 푸는 방식. trajectory는 `τ = (x, y_1, o_1, ..., y_K, o_K, y_{K+1})` — `y_k`는 모델 생성분(자연어 추론 / tool 호출 / 최종 답), `o_k`는 **환경이 만든 observation**. 정책 π_θ는 `y` 토큰만 생성하고 `o`는 외부에서 주입된다.
- 온디바이스 배포·프라이버시·추론 비용 때문에 **SLM에 에이전트 능력을 이식**하는 것이 실무적으로 중요하지만, SLM은 long-horizon trajectory에서 tool 사용이 불안정하고 capacity가 작아 exploration도 약하다.
- 이 논문이 다루는 규모는 극단적이다 — **Qwen3-0.6B / Qwen3-1.7B student**, teacher는 **Qwen3-4B**(및 14B).

### 기존 방법의 한계

| 방법 | 신호 밀도 | 분포 | TIR에서의 문제 |
|---|---|---|---|
| **SFT / off-policy distillation** | dense — O(N) | 남의 것 | exposure bias. 0.6B에서 **Vanilla보다도 나쁨**(avg 8.97 vs 12.16) |
| **RL / GRPO** | sparse — O(1) | 자기 것 | credit assignment 부재 + cold-start. 0.6B avg **11.32 < Vanilla 12.16**, entropy collapse로 tool 호출 자체를 포기 |
| **표준 OPD** | dense — O(N) | 자기 것 | **tool 오류 이후 구간의 teacher 감독이 신뢰 불가**. 1.7B에서 학습 중반 peak 후 성능 붕괴 |
| **OPSD_gt / OPSD_hint** (self-distill) | dense | 자기 것 | teacher를 ground-truth·hint로 조건화한 자기 자신. 1.7B avg 29.07 / 28.11로 OPD(36.27) 미달 |
| **SOD (제안)** | **dense + step별 신뢰도 가중** | 자기 것 | — |

> 원문 블로그가 다룬 것은 **단일 턴 수학 추론**이다. 거기서는 "dense × on-policy면 두 함정을 동시에
> 피한다"가 성립했지만, TIR에서는 **dense 신호 자체가 오염되는 제3의 함정**이 새로 생긴다.

---

## 2. Motivation

### 핵심 통찰 1: tool call 오류의 연쇄 증폭 (cascading divergence)

text-only 추론에서 step k → k+1은 student가 뽑은 토큰 하나가 붙는 것이라 drift가 점진적이다. TIR에서는
`y_{<t_{k+1}^start} = y_{<t_k^end} ⊕ o_k` — **길이 m의 외부 observation 블록이 통째로** 삽입된다.

```
step-level mismatch (Eq. 5):
  Δ_k = (1/|I_k|) · Σ_{t ∈ I_k} D_KL( π_θ(· | y_<t) || π_teacher(· | y_<t) )
  I_k = k번째 model-generated step의 토큰 위치 (tool observation 토큰은 제외)

Proposition 1 (Discontinuous divergence amplification):
  text-only :  Δ_{k+1} − Δ_k = O(η)              η = per-token TV shift
  TIR       :  Δ_{k+1} − Δ_k = Ω(m · η_tool)     η_tool >> η
  j회 연속 tool 오류:
    Δ_{k+j} − Δ_k = Ω( Σ_{i=0}^{j-1} m_i · η_tool^(i) ),  η_tool^(i+1) ≥ η_tool^(i)
```

핵심은 마지막 줄의 **부등식 η_tool^(i+1) ≥ η_tool^(i)** 다. 단발 오류 하나는 teacher가 학습 중 비슷한
에러 메시지를 본 적이 있어 감당하지만, **j개 오류가 누적된 context**는 teacher 학습 분포에서 확률이
대략 `p_err^j`로 지수적으로 희박해진다. 그래서 divergence가 step 수에 대해 **super-linear**로 자란다 —
Figure 1(a), 800 trajectory 기준 text-only는 완만한 직선이지만 TIR은 잘못된 tool call이 끼면 **가속**한다.

### 핵심 통찰 2: teacher 감독의 SNR 붕괴

```
Proposition 2 (Gradient SNR degradation):
  teacher-supported region:  S_t^ε = { v ∈ V : π_teacher(v | y_<t) ≥ ε }
  overlap:                   ρ_t   = Σ_{v ∈ S_t^ε} π_θ(v | y_<t)

  ρ_t ≤ ρ 일 때   E[ ℓ_t² ] ≥ (1 − ρ) · log²(1/ε),   SNR(g_t) → 0  as  ρ_t → 0
  (ℓ_t = log π_θ(y_t) − log π_teacher(y_t))
```

student가 teacher-supported 영역 밖으로 밀려나면 토큰마다 `−log π_teacher(y_t) > log(1/ε)` 라서 loss는
커지지만, 그 크기는 **teacher가 그 OOD 상태를 모델링하지 못한다는 사실**만 반영한다. 기댓값(신호)은 유한한데
분산은 `log²(1/ε)`로 커져 SNR이 0으로 간다. 즉 **"큰 KL = 큰 학습 신호"가 아니라 "큰 KL = 신뢰할 수 없는
신호"** 라는 것이 이 논문의 반전이다. 실증 — Figure 1(b), 800 trajectory의 teacher conditional entropy는
erroneous trajectory에서 step이 갈수록 mean과 std가 **함께** 커진다. Appendix F Case B의 구체 수치는
**teacher mean entropy가 step 1→3에서 0.85 → 1.12 → 2.14**, 마지막 step 토큰의 **78%가 H > 1.0**, 55%가 H > 2.0.

### 핵심 통찰 3: 균일 집계가 오염을 증폭한다

OPD loss는 모든 step의 token loss를 **같은 가중치로 합산**한다. 후반 오염 step이 trajectory의 상당 부분을
차지하면 전체 gradient가 고분산·저정보 항에 **체계적으로 지배**된다(Appendix D.3). 정렬된 초반 step(높은 ρ_t)과
오염된 후반 step(낮은 ρ_t)을 동일 취급하는 것이 근본 문제다.

---

## 3. Contributions

1. **TIR 고유의 실패 캐스케이드를 이론화** — tool observation의 이산적 state transition이 divergence를 super-linear로 증폭시키고(Prop. 1), 저-overlap 상태에서 OPD gradient의 SNR이 0으로 붕괴한다(Prop. 2).
2. **step-level divergence score `d_k` 제안** — full KL(Δ_k)의 monotone proxy이면서 **OPD forward pass에서 이미 나온 log-prob만으로 zero marginal cost로 계산**된다(Appendix D.5).
3. **적응적 step-wise 재가중** — 연속 step divergence **비율의 누적곱**으로 발산 구간은 억제하고 회복 구간은 복원한다. 분산 억제 배수 `O((d_1/d_k)²)`를 증명(Prop. 3).
4. **SLM 에이전트에서의 대폭 개선** — 차선 baseline(OPD) 대비 **0.6B +20.86%, 1.7B +18.50%**(상대 평균), **0.6B student가 AIME 2025에서 average@32 26.13%**(저자 주장 sub-billion 최초).
5. **비용 중립성 실증** — 0.6B에서 OPD보다 **오히려 3.5% 빠르고**(1052.3s vs 1090.5s/step), 1.7B는 +4.9%뿐.

---

## 4. Method

### 4.1 step-level divergence 정의

**"step"의 정의가 이 논문의 핵심 설계 결정이다.**

```
trajectory를 K+1개 reasoning step으로 분할:
  step k    = 두 tool observation 사이의 model response 한 덩어리 (자연어 추론 + tool 호출)
  step K+1  = 마지막 tool 응답 이후의 final answer step
  I_k       = step k의 model-generated 토큰 집합
              ★ tool observation o_k 토큰은 정책이 만든 것이 아니므로 제외

step-level divergence score (Eq. 6):
  d_k = (1/|I_k|) · Σ_{t ∈ I_k} | log π_θ(y_t | y_<t) − log π_teacher(y_t | y_<t) |
```

즉 한 step은 "추론 블록 + tool call"이고, **tool 응답은 step의 경계선**이지 내용이 아니다.
observation 토큰을 loss에서 빼는 것은 OPD 구현의 표준이지만, SOD는 한 발 더 나가 **observation을 step
분할의 기준점으로** 삼는다.

- `d_k`는 Δ_k(full KL)와 달리 **student가 실제로 뽑은 토큰의 log-prob 차이 절댓값 평균**으로, Jensen 부등식에 의해 Δ_k의 단일 샘플 Monte Carlo 추정의 절댓값 하한이 된다.
- **어휘 전체에 대한 teacher 분포가 필요 없다** → 추가 forward pass 0회. OPD를 이미 돌리고 있다면 공짜. 가중치 식이 d의 **비율**에만 의존하므로 Δ_k의 monotone proxy이기만 하면 충분하다(D.5).

### 4.2 적응적 재가중

```
w_1 = 1                                                       (첫 step은 항상 full strength)
w_k = min( Π_{u=1}^{k-1} (d_u + ε) / (d_{u+1} + ε),  1 + δ ),   k ≥ 2

  ε = 수치 안정 상수 (1e-6),  δ = 증폭 상한 offset (0.2) → w_k ≤ 1.2
```

| 상황 | 비율 `(d_u+ε)/(d_{u+1}+ε)` | w_k | 효과 |
|---|---|---|---|
| divergence 증가 (tool 오류로 drift 중) | < 1 | 급감 | 오염 구간의 distillation을 **감쇠** |
| divergence 유지 | ≈ 1 | 유지 | full-strength dense 감독 보존 |
| divergence 감소 (**recovery from earlier errors**) | > 1 | 회복 (상한 1.2) | 자가 교정한 step의 감독을 **복원** |

- **누적곱(telescoping)이 핵심**: 단조 증가 구간에서 `w_k ≤ (d_1+ε)/(d_k+ε)`로 접혀서 가중치가 "**첫 step 대비 지금 얼마나 벌어졌나**"에 반비례한다. 국소 비율이 아니라 궤적 전체의 누적 drift를 본다.
- **Prop. 3 (Bounded variance)**: `E[w_k² ℓ_t²] ≤ ((d_1+ε)/(d_k+ε))² · E[ℓ_t²]`. ρ_t → 0으로 `E[ℓ_t²] → ∞`가 되어도 d_k가 함께 커지므로 **가중 기여는 유한하게 유지**된다. 분산 억제 배수 `O((d_1/d_k)²)` — Prop. 2의 SNR 붕괴가 복구된다.
- **상한 1+δ가 필요한 이유**: recovery 구간에서 w_k가 무한정 커지면 gradient가 폭발한다 (ablation에서 clipping 제거 시 avg 42.98 → 38.10).

#### Training Objective

```
ℓ_OPD(y_t) = log π_θ(y_t | y_<t) − log π_teacher(y_t | y_<t)                    (Eq. 8)
L_OPD^step = E_{y ~ π_θ} [ Σ_{k=1}^{K+1} w_k · Σ_{t ∈ I_k} ℓ_OPD(y_t) ]         (Eq. 9)
L          = L_GRPO + L_OPD^step                                                (Eq. 10)
```

- `L_GRPO`는 group-relative advantage `Â_i = (r_i − mean{r_j}) / (std{r_j} + ε_A)` 기반 clipped surrogate. **sparse outcome reward로 trajectory 탐색**을 담당한다.
- `L_OPD^step`은 **dense token guidance**를 담당하되 강도가 divergence로 변조된다.
- 두 항의 역할은 직교적이되 **비대칭**이다 — ablation에서 GRPO 제거 −2.20, step-wise OPD 제거 −17.59.
- Algorithm 1은 3 stage: (I) 환경과 상호작용하며 G개 trajectory 샘플링 + Â_i 산출 → (II) {I_k} 분할 후 d_k, w_k 계산 → (III) `L_GRPO + L_OPD^step`으로 업데이트, θ_old ← θ.

#### 학습 vs 추론

| 단계 | 과정 |
|---|---|
| **학습** | student가 코드 인터프리터와 상호작용하며 rollout(최대 16턴) → teacher가 같은 토큰 시퀀스의 log-prob 계산 → step 분할 후 d_k, w_k 산출 → GRPO advantage와 합쳐 1회 업데이트 |
| **추론** | student 단독. teacher도 divergence 계산도 배포에 관여하지 않는다. 필요한 것은 tool 환경뿐 |
| **실무 포인트** | w_k는 prefix(d_1..d_k)만으로 계산되는 **causal** 양이라 원문 블로그가 강조한 partial rollout 학습과 원리적으로 충돌하지 않는다 |

---

## 5. Experiments

### 5.1 Setup / Dataset / Benchmarks

| 구분 | 구성 |
|---|---|
| **SFT (3k)** | s1-1k(수학) 1k · LeetCodeDataset(코딩) 1k · ReTool(tool-use) 1k |
| **RL (약 30k)** | DAPO-Math 17k · Skywork-or1 Math 4,902 / Code 3,586 · MegaScience(과학) 3k |
| **벤치마크** | AIME 2024 / 2025(각 30문제, 정수 exact match) · GPQA-Diamond(198문항, 전문가 65% / 랜덤 25%) · LiveCodeBench v6(1,055문제, 2023.05~2025.04, contamination-aware) |

- SFT trajectory는 **Qwen3-Coder-30B-A3B**가 agent framework 안에서 **SandBoxFusion** 코드 인터프리터로 end-to-end 생성. LeetCode·ReTool 부분은 **ReasonFlux-PRM**으로 채점해 상위 1k씩만 남기고 s1-1k는 전량 사용.

**평가 프로토콜 — average@32 (pass@1과 혼동 금지)**

```
temperature = 1.0, top_p = 0.6, 문제당 독립 샘플 32개 생성
average@32 = 32개 샘플의 정답률 평균 (%)
  ← pass@1(1개 뽑아 맞는가)도, pass@32(32개 중 하나라도 맞는가)도 아니다
전체 실험을 서로 다른 random seed로 5회 반복, 평균 ± 표준편차 보고
```

pass@32라면 훨씬 높았을 것이므로 **26.13%는 상당히 보수적인 지표**임을 유념할 것.

### 5.2 Implementation Details

| 항목 | 값 |
|---|---|
| **Student / Teacher** | Qwen3-0.6B, Qwen3-1.7B / **Qwen3-4B를 RL 데이터셋에 GRPO로 추가 최적화한 버전** (확장 실험 Qwen3-14B) |
| **Tool 환경** | Python **code interpreter** (SFT 데이터 생성은 SandBoxFusion) |
| **프레임워크 / 하드웨어** | **VeRL + Open-AgentRL** / 단일 노드 **8× NVIDIA H20 (96GB)** |
| **Rollout** | vLLM 비동기, tensor parallel = 4, **최대 16 tool-call 턴** |
| **Optimizer** | AdamW, lr **1e-6**, train batch **64**, mini-batch **16**, 최대 1 epoch (teacher는 2) |
| **길이 / 샘플 수** | max prompt 2,560 / max response **20,480** tokens · 학습 시 prompt당 **16 responses**, validation 32 |
| **SFT 초기화** | 3K 코퍼스, 5 epoch, global batch 128, lr **5e-5**, max seq 32,768 (right truncation) |
| **SOD 전용 하이퍼파라미터** | **w_1 = 1, ε = 1e-6, δ = 0.2** — 이 셋이 전부 |
| **학습 스텝 / 시간** | 0.6B 120 step / 1.7B 150 step · 0.6B·1.7B 2~3일, 4B·14B teacher 5~6일, SFT는 수 시간 |

> **재현 관점 핵심**: 모든 RL/distillation 방법이 **동일한 SFT 체크포인트에서 출발**하고 옵티마이저·배치·
> 인프라를 공유한다. SOD가 추가하는 하이퍼파라미터는 사실상 **δ 하나**(w_1, ε는 상수).
> VeRL 기반 OPD 파이프라인이 이미 있다면 step 분할 로직 + 스칼라 3~7개의 누적곱만 얹으면 된다.

**학습 비용 (Table 4)**

| Params | GRPO | OPD | **SOD** |
|---|---|---|---|
| **0.6B** — s/step · GB · 총 h | 680.6 · 78.5 · 22.7 | 1090.5 · 87.7 · 36.4 | **1052.3 · 87.6 · 35.1** |
| **1.7B** — s/step · GB · 총 h | 784.6 · 81.7 · 32.7 | 1053.6 · 88.2 · 43.9 | **1105.4 · 88.7 · 46.1** |

- **알고리즘 오버헤드는 사실상 0.** d_k·w_k는 이미 계산된 log-prob의 per-step 평균 + K개(약 3~7) 스칼라 곱. 추가 forward pass 없음, peak memory 차이 0.5GB 미만.
- 0.6B에서 SOD가 OPD보다 **3.5% 빠른** 이유: 재가중이 잘못된 tool-call 패턴의 학습을 억제해 **실패한 재시도가 줄고 응답이 짧아지기 때문**. 성능 개선이 속도 개선으로도 이어진다.
- GRPO가 1.4~1.5배 빠른 것은 장점이 아니다. GRPO student가 **tool 호출 능력 자체를 잃어** 응답이 짧아진 결과(Figure 4(c)(f))로, per-step 시간이 인위적으로 줄어든 것이다.

### 5.3 Main Results

**Table 1 — average@32, 5회 실행 평균 (%)** · *기울임 = 차선, 굵게 = 최고*

| Params | Method | AIME 2024 | AIME 2025 | GPQA-D | LiveCodeBench | **Average** |
|---|---|---|---|---|---|---|
| **4B (Teacher)** | GRPO | 67.60 ±1.34 | 60.42 ±1.47 | 55.19 ±0.81 | 63.13 ±0.93 | **61.59** |
| **0.6B** | Vanilla | 7.71 ±0.33 | 12.81 ±0.38 | 13.24 ±0.24 | 14.89 ±0.29 | 12.16 |
| | SFT | 5.67 ±0.26 | 5.42 ±0.22 | 15.20 ±0.30 | 9.61 ±0.27 | 8.97 |
| | GRPO | 4.06 ±0.37 | 4.90 ±0.31 | _20.38_ ±0.47 | 15.95 ±0.51 | 11.32 |
| | OPD | _16.82_ ±0.81 | _22.95_ ±0.97 | 17.76 ±0.41 | _22.65_ ±0.75 | _20.04_ |
| | OPSD_gt | 12.63 ±0.58 | 17.04 ±0.64 | 17.32 ±0.35 | 16.73 ±0.44 | 15.93 |
| | OPSD_hint | 9.77 ±0.43 | 14.12 ±0.48 | 15.98 ±0.30 | 12.65 ±0.37 | 13.13 |
| | **SOD** | **20.84** ±0.90 | **26.13** ±1.07 | **22.19** ±0.59 | **27.72** ±0.83 | **24.22** |
| **1.7B** | Vanilla | 9.90 ±0.38 | 8.96 ±0.33 | 26.80 ±0.36 | 22.73 ±0.41 | 17.10 |
| | SFT | 26.77 ±0.66 | 22.40 ±0.72 | 29.85 ±0.51 | 24.63 ±0.57 | 25.91 |
| | GRPO | 25.63 ±1.03 | 21.67 ±1.12 | 33.55 ±0.76 | 20.70 ±0.87 | 25.39 |
| | OPD | _43.86_ ±1.23 | _37.04_ ±1.31 | 31.73 ±0.55 | _32.45_ ±0.93 | _36.27_ |
| | OPSD_gt | 33.85 ±0.88 | 24.69 ±0.94 | _35.02_ ±0.67 | 22.73 ±0.61 | 29.07 |
| | OPSD_hint | 34.42 ±0.76 | 21.43 ±0.83 | 33.46 ±0.59 | 23.12 ±0.52 | 28.11 |
| | **SOD** | **50.83** ±1.15 | **41.72** ±1.24 | **38.72** ±0.73 | **40.63** ±0.91 | **42.98** |

- **차선 대비 상대 개선**: 0.6B 24.22/20.04 = **+20.86%** · 1.7B 42.98/36.27 = **+18.50%**. 두 스케일 모두 4개 태스크 전부 1위.
- **0.6B가 AIME 2025에서 26.13%** — sub-billion 모델로는 저자 주장 최초, teacher(60.42)의 43.2% 수준. **teacher 성능 회수율**은 1.7B SOD **69.8%**(42.98/61.59) vs OPD **58.9%**(36.27/61.59).
- **0.6B SOD(24.22)가 1.7B Vanilla(17.10)를 추월** — 재가중이 **모델 크기 격차를 부분 보상**한다.
- **0.6B에서 SFT(8.97)·GRPO(11.32)는 Vanilla(12.16)보다 나쁘다** — sparse reward와 static demo는 0.6B TIR에 부적합. 단 GPQA는 예외로 0.6B GRPO(20.38)가 OPD(17.76)를 앞선다(tool 의존도가 낮은 객관식).

**확장성 — teacher 크기 (Figure 3, Obs 5)**: **OPD는 강한 teacher에서 오히려 손해를 본다.**
0.6B student에 teacher를 4B → 14B로 바꾸면 OPD 평균 정확도가 **눈에 띄게 하락**한다 — capacity gap이
커질수록 distribution mismatch가 커지고 균일 distillation이 신뢰 못 할 감독을 그대로 전파하기 때문.
**SOD는 반대로 14B teacher에서 0.6B·1.7B 양쪽 모두 4B 대비 일관되게 개선**된다.
→ **"teacher는 클수록 좋다"는 통념은 SLM+TIR에서 성립하지 않는다.**

**학습 동역학 (Figure 4)**: GRPO는 0.6B에서 policy entropy가 0.8대 → 0.1대로 붕괴하고 mean tool turns가
급감해 **multi-step tool 상호작용을 완전히 포기**한다(TIR의 치명적 실패 모드). OPD는 1.7B에서 AIME2025
정확도가 초반 peak 후 **크게 열화**한다. SOD는 entropy를 OPD 수준으로 유지하면서 열화 없이 단조 개선하고,
mean tool turns는 OPD보다 **적다**(불필요한 재시도 감소).

### 5.4 Ablation Study

**Table 2 — 1.7B student, average@32 (%)**

| 구분 | Variant | AIME24 | AIME25 | GPQA-D | LCB | **Avg** | Δ |
|---|---|---|---|---|---|---|---|
| **재가중** | (1.1) Uniform Weighting (w_k = 1) | 41.68 | 35.58 | 30.12 | 31.43 | 34.70 | −8.28 |
| | (1.2) Heuristic Weighting (w_k = 0.9^(k−k_err)) | 44.96 | 38.75 | 31.14 | 33.71 | 37.14 | −5.84 |
| | (1.3) Mask After Wrong (첫 오류 이후 전부 0) | 39.11 | 31.59 | 26.56 | 30.12 | 31.85 | **−11.13** |
| | (1.4) w/o Weight Clipping (δ 상한 제거) | 45.78 | 37.93 | 33.57 | 35.12 | 38.10 | −4.88 |
| **목적함수** | (2.1) w/o GRPO | 48.87 | 39.73 | 35.89 | 38.62 | 40.78 | −2.20 |
| | (2.2) w/o Step-wise OPD (= GRPO only) | 25.63 | 21.67 | 33.55 | 20.70 | 25.39 | **−17.59** |
| | **SOD** | **50.83** | **41.72** | **38.72** | **40.63** | **42.98** | — |

- **uniform (= 표준 OPD의 step 취급)** 34.70 — **원문 블로그의 "per-token 균일 처리로 충분하다"에 대한 직접적 반증**. 이 한 항목만으로 8.28pt(상대 19.3%)가 사라진다.
- **heuristic 고정 감쇠** 37.14 — 오류 이후 지수 감쇠는 부분적으로 돕지만 **student가 회복하는 non-monotonic 패턴을 고정 스케줄로는 포착할 수 없다.**
- **hard masking** 31.85로 **최악** — 오류 이후를 전부 버리면 부분 교정된 trajectory의 유용한 신호까지 폐기하고 복구 학습 기회를 없앤다. **"오류 이후는 잘라내면 된다"는 손쉬운 해법이 명시적으로 기각**된다.
- **clipping 제거** 38.10 — 상한 없는 증폭은 학습을 불안정하게 만든다. **비대칭성**: step-wise OPD 제거(−17.59)가 GRPO 제거(−2.20)보다 8배 큰 타격 — dense 감독이 주 동력, sparse reward는 탐색 보조.

**Table 5 — naive OPD 분리 실험 (1.7B, Average)**: naive OPD(KD loss만) **33.79** ·
OPD(+GRPO, 본문 baseline) **36.27** · SOD w/o GRPO(재가중만) **40.78** · SOD **42.98**.
→ 순수 distillation끼리 비교하면 naive OPD 33.79 → SOD w/o GRPO 40.78 = **+6.99pt (+20.69%)**.
**RL 보상 없이 재가중만으로 20% 개선** — 성능 향상의 주 원인이 재가중임을 분리 입증한다(GRPO 기여는 +2.20pt).

**세 가지 distillation 패턴 (Figure 5·6, Appendix G, δ = 0.2 실측)**

| 패턴 | Appendix G 실측 |
|---|---|
| **Stable** | Case A: d_k 0.22~0.50, teacher entropy 0.24~0.41, w_k **1.00 → 1.20 유지** |
| **Erroneous** | Case B: numpy 샌드박스 3연속 실패 → d_k **0.218 → 1.284(약 6배)**, entropy **0.32 → 2.38**, w_k **1.00 → 0.17** |
| **Recovery** | Case C: IndentationError 3연속 후 코드 재작성 성공 → w_k **1.00 → 0.73 → 0.56 → 0.41 → 0.76 → 1.12 → 1.20** |

Figure 6(학습 스텝별 분포): **stable 비율은 꾸준히 증가**, **erroneous는 크게 감소**, **recovery는 증가 후
유지**(재가중이 학습 끝까지 계속 작동). 1.7B가 더 빨리 stable로 수렴하고 0.6B는 초반 erroneous 비율이 훨씬
높다 → **작은 모델일수록 초기 오류 억제가 결정적**. Case C가 이 방법의 정체성을 가장 잘 보여준다 —
trajectory-level 필터링이라면 궤적 전체를 버리거나 전부 살렸겠지만, SOD는 **오류 구간(s2~s4)만 국소화**하고
회복 구간(s5~s7)에서는 감독을 온전히 쓴다.

**Limitations(저자 명시)**: ① tool 환경이 **Python 코드 인터프리터 단일** — 웹 브라우징·API 호출은 drift
패턴이 다를 수 있다. ② 전 실험이 **Qwen3 계열**, 다른 모델 패밀리 검증은 미수행.

---

## 6. Key Takeaways

1. **TIR에서 dense 감독은 공짜가 아니다 — "큰 KL = 신뢰 못 할 신호"다.** tool observation은 길이 m의 외부 텍스트를 통째로 삽입해 `Ω(m·η_tool)` 크기의 divergence 점프를 만들고, 연속 오류에서 `η_tool^(i+1) ≥ η_tool^(i)`로 **super-linear 증폭**된다(Prop. 1). 저-overlap 상태에서 gradient **SNR은 0으로 붕괴**(Prop. 2). 실측으로 teacher entropy가 step 1→3에서 **0.85 → 2.14**, 마지막 step 토큰 **78%가 H > 1.0**.

2. **"step"은 tool 응답을 경계로 한 추론 블록이고, 신뢰도 지표는 공짜다.** step k = 두 observation 사이의 model response(추론 + tool 호출), observation 토큰은 I_k에서 제외. `d_k`는 student가 뽑은 토큰의 log-prob 차이 절댓값 평균이라 **OPD forward pass 결과만으로** 나온다 — 추가 forward 0회, 메모리 차이 0.5GB 미만.

3. **누적 비율곱 가중치가 억제와 회복을 동시에 처리한다.** `w_k = min(Π (d_u+ε)/(d_{u+1}+ε), 1+δ)`, 단조 증가 구간에서 `(d_1+ε)/(d_k+ε)`로 접혀 분산을 `O((d_1/d_k)²)` 억제(Prop. 3). Case B에서 1.00 → 0.17로 끄고 Case C에서 0.41까지 내렸다가 1.20으로 되살린다. **오류 이후를 마스킹하는 방식은 31.85로 최악**.

4. **차선 대비 0.6B +20.86% / 1.7B +18.50%, 0.6B가 AIME 2025 average@32 26.13%.** 1.7B는 4B teacher 성능의 **69.8%** 회수(OPD 58.9%), 0.6B SOD(24.22)가 1.7B Vanilla(17.10)를 앞선다. **average@32는 32회 샘플링의 평균 정답률**이지 pass@1도 pass@32도 아니다 — 보수적 지표다.

5. **0.6B에서는 SFT(8.97)와 GRPO(11.32)가 Vanilla(12.16)보다도 나쁘다.** static demonstration과 sparse outcome reward는 sub-billion TIR에 부적합하며, GRPO는 entropy가 붕괴하며 **multi-step tool 호출 자체를 포기**한다. SLM 에이전트를 만든다면 dense 감독은 선택이 아니라 전제다.

6. **teacher는 클수록 좋다는 통념이 깨진다.** 0.6B student에 teacher를 4B → 14B로 키우면 **OPD는 평균 정확도가 오히려 떨어진다**(capacity gap 확대). SOD는 반대로 14B에서 양쪽 스케일 모두 개선 — **재가중 장치 없이 큰 teacher를 붙이는 것은 역효과**다.

7. **비용은 사실상 중립.** 0.6B에서 SOD는 OPD보다 **3.5% 빠르고**(1052.3s vs 1090.5s/step, 총 35.1h vs 36.4h) 1.7B는 +4.9% 오버헤드뿐. 추가 하이퍼파라미터는 **δ = 0.2 하나**, 8×H20에서 0.6B/1.7B가 2~3일. VeRL + Open-AgentRL 기반이라 기존 OPD 파이프라인에 step 분할 로직만 얹으면 된다.

---

## 7. 원문 블로그 대비 갱신점

| # | 원문 블로그(2025.10) 주장 | SOD의 판정 | 근거 |
|---|---|---|---|
| ① | OPD = dense × on-policy, **두 함정 동시 회피** | **조건부** — 제3의 함정 추가 | TIR에서는 dense 신호 자체가 오염된다. 저-overlap 상태에서 gradient SNR → 0(Prop. 2). on-policy라는 사실만으로 감독의 신뢰성이 보장되지 않는다 |
| ② | `advantage = −reverse_kl` **한 줄이면 끝** | **확장(비용은 유지)** | 한 줄로는 부족하지만 추가분이 매우 싸다 — step 분할 + K개(3~7) 스칼라 누적곱, 추가 forward 0회. 0.6B에서는 오히려 3.5% 빠름 |
| ③ | **per-token 균일 처리로 충분하다** | **정면 반박 (가장 핵심)** | uniform weighting = 1.7B avg **34.70 vs SOD 42.98**(−8.28pt, −19.3%). GRPO 없는 순수 비교에서도 naive OPD 33.79 → SOD w/o GRPO 40.78 = **+20.69%** |
| ④ | 벌점이 **forking token에 잘 몰린다** | **TIR에서는 성립 안 함** | 단일 턴에서는 벌점이 갈림길에 몰렸지만 TIR에서는 **tool 오류 이후 오염 구간에 몰린다**. 그 구간은 teacher 자신이 불확실(H 2.14, 78% 토큰 H>1.0)해 벌점 위치가 정보를 주지 못한다 |
| ⑤ | **partial rollout 가능** | **호환 유지** | w_k는 d_1..d_k만 쓰는 causal 양이라 prefix까지만으로 정의된다. 다만 논문이 직접 실험하지는 않았다 |
| ⑥ | **단일 턴 수학 추론만** 다룸, 멀티턴/에이전트 미다룸 | **정면으로 메움 (핵심 기여)** | multi-turn TIR, **최대 16 tool-call 턴**, Python 코드 인터프리터. tool observation이 만드는 이산 state transition을 이론(Prop. 1)과 실험(Fig. 1a) 양쪽으로 규명 |
| ⑦ | **작은 모델도 잘 된다** | **한정 인정 + 정밀화** | 0.6B에서 SFT(8.97)·GRPO(11.32)는 Vanilla(12.16) 미달. OPD는 20.04로 유효하지만 **재가중 없이는 잠재력의 상당 부분을 잃는다**(SOD 24.22). "작은 모델도 된다"가 **"dense + 신뢰도 변조가 있어야 된다"로 좁혀진다** |

### 다른 후속 연구와의 관계

| 연구 | 제기한 문제 | SOD의 답 |
|---|---|---|
| **Rethinking OPD** (2604.13016) | overlap token이 확률질량의 97~99%이므로 long-horizon 확장에 의문 | **구성적 답변**. overlap 붕괴(ρ_t → 0)가 바로 실패 원인임을 형식화하고 **step 단위 신뢰도 변조로 long-horizon을 실제로 굴린다**. 동시에 의문 자체도 확증 — 1.7B에서 vanilla OPD는 중반 peak 후 열화(Fig. 4d) |
| **Revisiting OPD** (2603.25562) | 실패 모드 3종 + truncated reverse-KL로 +19.8% | 같은 "OPD 신호를 무조건 믿지 말라" 계열. Revisiting은 **token-level 절단**, SOD는 **step-level 연속 가중**. SOD의 hard masking ablation(31.85, 최악)은 **절단형 처리를 TIR에 그대로 쓰면 위험**함을 시사 |
| **Demystifying OPD** (2604.08527) | length inflation / repetition saturation, Stable-OPD +7.2% | 둘 다 안정성을 다루나 원인이 다르다 — SOD의 불안정은 **외부 tool observation이 주입한 OOD state**. SOD 쪽에서는 tool 턴 수가 오히려 **줄어든다** |
| **ExOPD** (2602.12125) | OPD = dense KL-constrained RL의 특수 케이스, reward scaling으로 teacher 초월 | SOD는 초월이 아니라 **회수율**(58.9% → 69.8%)을 다룬다. 다만 Obs 5의 "큰 teacher가 OPD에서 역효과"는 **teacher 상한 논의 자체가 capacity gap에 종속**됨을 보여준다 |
| **SDFT** (2601.19897) | demonstration-conditioned 자기 teacher | SOD의 OPSD_gt / OPSD_hint baseline이 같은 계열. TIR에서는 **1.7B 29.07 / 28.11로 외부 teacher OPD(36.27) 미달** — self-distillation은 tool 실행 실패를 스스로 교정할 근거가 없다 |

---

[← 후속 연구 정리](./opd_follow_up_research.md) · [원문 요약](./on_policy_distillation.md)
