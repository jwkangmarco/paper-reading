# Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes

> **Venue**: Preprint. Under review. (2026.04.27, v2)
> **Authors**: Yuqian Fu\*, Haohuan Huang\* (CASIA 멀티모달 인공지능시스템 국가중점연구실 / UCAS 인공지능학원), Kaiwen Jiang (CASIA / UCAS), Jiacai Liu (Fudan University), Zhuo Jiang (Independent Researcher), Yuanheng Zhu†, Dongbin Zhao† (CASIA / UCAS)
> **arXiv**: 2603.25562v2 · Code / Blog 공개 (\*공동 1저자, †교신저자)

**한 줄 정의**: OPD의 표준 구현인 **sampled-token log-ratio 한 개**가 왜 부서지는지를 이론(bias-variance)·현상(실패 모드 3종)으로 해부하고, 그 자리를 **teacher top-K 지지집합 위의 truncated reverse-KL**로 바꿔 multi-task 수학 평균 **+19.8%** 를 얻은 논문.

---

## 1. Background

### OPD 실무 적용의 현황

- OPD는 이제 연구 아이디어가 아니라 **제품 파이프라인의 구성요소**다. Thinking Machines Lab 블로그(2025), Qwen3(2025), MiMo-V2-Flash(2026), GLM-5(2026)가 모두 student rollout 위 supervision 혹은 그 변형을 보고했다. 매력의 근원은 ① student가 실제로 방문하는 prefix 위에서 배우고(on-policy) ② teacher logprob이 token마다 dense feedback을 준다는 것 — 그것도 상대적으로 싸게.
- 그런데 **현재 대부분의 LLM 구현은 token-level 추정기**를 쓴다. 이것이 sequence-level reverse-KL에 대해 biased라는 것은 알려져 있으나, 왜 그럼에도 그 선택이 옳은지는 정식으로 정리되지 않았다. 동시에 실무 보고들은 반대 신호도 낸다. Gu et al.(2024)은 **repetition 같은 열화 출력**을, Ko et al.(2026)·Jin et al.(2026b)은 **sampled-token OPD의 entropy collapse**를, Zhao et al.(2026)·DeepSeek-AI(2026)는 **full-vocabulary distillation이 더 낫다**는 것을 보고했다 — one-token 정식화가 teacher 정보를 남긴다는 뜻이다.

### 기존 방법의 한계

| 방법 | 신호 형태 | 비용 | 문제점 |
|---|---|---|---|
| Sequence-level reverse-KL (MiniLLM 계열) | 시퀀스 log-ratio × return-to-go | 중간 | 각 token 업데이트가 **미래 reward 전부와 결합** → worst-case 분산 **O(T⁴)** |
| **Sampled-token OPD** (블로그 표준 레시피) | 뽑힌 token 1개의 log-ratio | **최저** | 신호가 **한 점 추정**. 긴 rollout에서 prefix가 teacher의 전형적 support를 벗어나면 부서진다 |
| Full-vocabulary logit distillation | 전체 vocab 분포 매칭 | **최고** | 긴 응답 × 큰 vocab에서 logit materialize 비용이 금지적 |
| Top-K KL (EMA-PG, SDPO, OpenClaw-RL) | top-K + tail 보정 | 중간 | 효율은 얻지만 sampled token을 지지집합에 명시적으로 넣지 않는 경우가 많다 |
| **이 논문 (LSM)** | teacher top-K 지지집합 위 renormalized reverse-KL | **낮음** (K = 32) | **sequence-level의 분산과 full-vocab의 비용 사이.** 지역성은 유지하되 신호를 한 점이 아니라 **작은 분포 비교**로 만든다 |

---

## 2. Motivation

### 핵심 통찰 1: token-level OPD는 "틀린 근사"가 아니라 "의도된 거래"다

- token-level 추정기는 sequence-level reverse-KL에 대해 **명백히 biased**다 — 미래 reward 결합항을 통째로 버린다. 그런데 그 결합항이야말로 분산의 원천이고, 시퀀스 길이 T에 대해 worst-case 분산 상한이 **O(T²) vs O(T⁴)** 로 갈린다.
- 즉 "블로그가 discount 0을 쓴 것"은 편의가 아니라 **long-horizon 학습을 위한 정당한 설계 선택**이다. 저자들은 이것을 명시적 원리로 승격시킨다: *keep supervision token-level to control variance.* 그렇다면 질문은 하나로 좁혀진다 — **token-level의 분산 이점은 유지하면서, 그 지역 신호를 덜 부서지게 만들 수 없나?**

### 핵심 통찰 2: 문제는 "token-level"이 아니라 "sampled-token"이다

- 표준 구현은 token-level을 **sampled-token 비교**로 구현한다. 각 스텝에서 뽑힌 token 하나의 teacher-student log-probability 차이가 업데이트 전부를 결정한다. 이 한 점 추정은 세 갈래로 무너진다.
  1. **objective-level**: 그 한 token의 신호가 심하게 **불균형**하다 (대부분 음수).
  2. **objective-level**: student가 drift한 prefix 위에서 **teacher 안내 자체가 신뢰할 수 없어진다** — repetition loop인데도 teacher가 국소적으로 동의한다.
  3. **implementation-level**: **tokenizer / special token 불일치**가 의미적으로 옳은 출력에 벌점을 준다.
- 세 가지 모두 "한 token에만 supervision을 걸었다"는 사실에서 증폭된다. 분포 비교였다면 한 token의 이상값에 업데이트 전체가 끌려가지 않는다.

---

## 3. Contributions

1. **OPD의 이론적 trade-off 정리**: token-level OPD가 sequence-level reverse-KL에 대해 biased임을 기댓값 차이로 명시하고, worst-case 분산 상한이 **O(T²) vs O(T⁴)** 로 갈림을 유도. long-horizon post-training에서 token-level이 선호되는 이유에 정식 근거를 준다.
2. **γ를 0→1로 보간하는 controlled synthetic study**: discounted return-to-go 추정기로 두 추정기 사이를 연속적으로 잇고, **γ가 클수록 gradient variance가 높고 정책이 목표로 수렴하지 못하고 drift**함을 3개 seed에서 확인.
3. **sampled-token OPD 실패 모드 3종의 경험적 규명**: 불균형 supervision / student prefix 위 신뢰 불가 teacher / tokenizer·special-token mismatch. 각각에 대한 진단 그림과 수치 증거 제시.
4. **teacher top-K local support matching (LSM) 제안**: truncated reverse-KL + support-set renormalization + top-p rollout + special-token masking. 지역 업데이트의 효율을 유지하면서 한 token 의존을 제거.
5. **single-task 수학 + multi-task(agentic + reasoning) 양쪽 검증**: multi-task 수학 평균 34.8 → 41.7 (**+19.8%**), single-task 평균 36.4 → 41.7, 학습 dynamics(gradient norm, clipping-boundary fraction)도 개선. 동시에 **teacher matching이 불완전한 프록시임을 명시** — Appendix H의 reward-hacking 사례집(repetition loop, over-continuation, 깨진 비영어 출력에도 teacher가 높은 확률).

---

## 4. Method

### 4.1 이론 분석 — bias vs variance

OPD의 목적함수와 그 gradient, 그리고 각 decoding step t의 prefix context c_t = (x, y_<t) 위에서 정의되는 score / per-token reward는 다음과 같다.

```
J_OPD(θ) = E_{x~D} [ D_KL( π_θ(· | x) || q(· | x) ) ]        (π_θ = student, q = teacher)
∇_θ J_OPD(θ) = E_{x, y~π_θ(·|x)} [ ( log π_θ(y|x) − log q(y|x) ) ∇_θ log π_θ(y|x) ]

s_t = ∇_θ log π_θ(y_t | c_t)                     (score 항)
r_t = log [ π_θ(y_t | c_t) / q(y_t | c_t) ]      (per-token log-ratio)

autoregressive 분해:
  log π_θ(y|x) − log q(y|x) = Σ_{t'=1..T} r_{t'} ,   ∇_θ log π_θ(y|x) = Σ_{t=1..T} s_t

sequence-level 추정기:
  ĝ_seq = Σ_{t=1..T} ( Σ_{t'=1..T} r_{t'} ) s_t                     ... (1)
token-level 추정기 (즉시항만 유지 — 실제 LLM 구현):
  ĝ_tok = Σ_{t=1..T} r_t s_t                                        ... (2)
```

t' < t 인 항은 기댓값이 0이다 (r_{t'}는 step t 이전 prefix에만 의존하고, E[s_t | x, y_<t] = Σ_{y_t} π_θ(y_t|c_t) ∇_θ log π_θ(y_t|c_t) = 0). 따라서 `E[ĝ_seq] = E[ Σ_t ( Σ_{t'=t..T} r_{t'} ) s_t ]` 라는 causal reward-to-go 형태가 되고, 여기서 **각 token 업데이트가 궤적의 모든 미래 reward와 결합**되어 있음이 드러난다.

#### Bias와 worst-case variance — 정식 진술 (Appendix D)

bounded reward / bounded gradient 가정 (|r_t| ≤ B_r, ||s_t|| ≤ B_s 인 상수가 존재).

```
[Bias, D.1]
  ĝ_seq = Σ_{t=1..T} r_t s_t  +  Σ_{t=1..T} Σ_{t'=t+1..T} r_{t'} s_t
  E[ĝ_seq] − E[ĝ_tok] = E[ Σ_{t=1..T} Σ_{t'=t+1..T} r_{t'} s_t ]
  → 버려진 것은 정확히 future-reward coupling 항 전체 ⇒ token-level은 일반적으로 biased

[Variance, D.2]
  token-level    : ||ĝ_tok|| ≤ Σ_t |r_t| ||s_t|| ≤ T B_r B_s
                   E||ĝ_tok||² ≤ T² B_r² B_s²  ⇒  Var(ĝ_tok) = O(T²)
  sequence-level : R = Σ_t r_t, S = Σ_t s_t, ĝ_seq = R·S, |R| ≤ T B_r, ||S|| ≤ T B_s
                   ||ĝ_seq|| ≤ T² B_r B_s, E||ĝ_seq||² ≤ T⁴ B_r² B_s² ⇒ Var(ĝ_seq) = O(T⁴)
  (Var(X) ≤ E||X||² 사용)

[보간, 식 (3)]
  ĝ_γ = Σ_{t=1..T} ( Σ_{t'=t..T} γ^(t'−t) r_{t'} ) s_t ,  γ ∈ [0,1]
  γ = 0 → token-level OPD (블로그의 discount 0) ,  γ = 1 → causal sequence-level
```

> **결론**: sequence-level은 정확한 궤적 수준 목적함수에 더 가깝지만, score 항 하나하나를 많은 미래 reward와 곱하기 때문에 worst-case 분산 증가율이 **quadratic → quartic** 으로 뛴다. 논문 스스로 "보수적 논증(conservative)"이라고 밝히지만, long-horizon에서 왜 강한 reward coupling이 문제가 되는지를 포착한다.

#### controlled synthetic study (Appendix E)

| 항목 | 설정 |
|---|---|
| 환경 / 전이 | 1차원 연속 제어, **두 task가 서로 거울상**(left: +2 시작 −3 목표 / right: −2 시작 +3 목표). `s_{t+1} = s_t + δ`, `δ ~ N(μ, σ)` |
| 정책 | 3-layer MLP, 약 **4K 파라미터**. 입력 3차원 (task identity, 현재 위치, 정규화된 time step), 출력은 Gaussian action의 평균·표준편차 |
| 절차 / 스윕 | 두 task의 teacher를 REINFORCE로 따로 학습 → **alternating-task OPD**로 공유 student에 distill. γ ∈ {0, 0.25, 0.5, 0.75, 1.0}, 2000 iteration, seed 42 / 43 / 2026 |
| 분산 추정 | batch B = 64 궤적을 M = 8 micro-batch로 분할, 출력층 파라미터의 gradient g_m로 `Var(g) = (1/M) Σ_m ||g_m − ḡ||²` (정성적 프록시로만 사용) |

**관측**: 모든 설정에서 초기에 큰 variance spike가 나지만 **γ가 클수록 후반까지 높은 variance에 머문다.** 여러 run에서 **γ = 0.75 / 1.0의 variance가 작은 γ 대비 1~수 자릿수 높게 유지**된다. state visitation heatmap을 보면 **γ = 0 (token-level)은 seed와 무관하게 두 task 모두 목표 상태로 이동**하고, 중간 γ는 비슷하나 궤적이 퍼지며(diffuse), **γ = 1.0에서는 정책이 목표 방향으로 일관되게 움직이지 못하고 drift하며 sub-optimal 영역에 정착**한다.

> 이론의 O(T²) vs O(T⁴)와 toy의 "γ↑ → variance↑ → drift"가 같은 그림을 가리킨다. 이것이 논문이 나머지 전부를 token-level 위에서 전개하는 근거다.

### 4.2 실패 모드 3종

세 실패 모드 모두 **student = Qwen2.5-7B-Instruct, teacher = OpenThinker3-7B**(Qwen2.5-7B-Instruct 기반 SFT 모델) 로 돌린 수학 추론 sampled-token OPD 실험에서 관측되었다.

#### (i) 심하게 불균형한 token-level supervision

step t의 업데이트를 결정하는 것은 단 하나의 값 `log q(y_t | c_t) − log π_θ(y_t | c_t)` 이다.

- student가 뽑은 token에 student가 teacher보다 높은 확률을 주면 **음의 reward**가 된다. 그런데 그 token은 애초에 student가 뽑은 token이므로 **음수가 기본값**이다. Figure 2(첫 iteration의 teacher 확률 vs student 확률 산점도)에서 **대다수의 sampled token이 음의 reward**를 받는다 — 점들이 y = x 아래쪽에 몰려 있다.
- 결과: 최적화가 **국소적으로 양수인 소수 token 집합에 지배**된다. 그 소수는 고빈도 filler와 짧은 연결어처럼 **국소 점수는 좋지만 궤적 품질 기여는 미미한** token이기 쉽다.

#### (ii) student-generated prefix 위에서 신뢰할 수 없는 teacher 안내

- sampled-token OPD의 전제는 "student가 뽑은 token에 대한 teacher 확률이 궤적 품질의 유용한 프록시"라는 것인데, 이 전제는 **student에게는 흔하지만 teacher에게는 드문 prefix**에서 무너진다. Figure 3: student가 **repetition loop**에 빠졌는데도 반복 token들에서 teacher 확률이 **0.9~1.0 부근으로 student와 나란히** 유지된다 → sampled-token OPD가 이 행동을 벌하지 못한다.
- Appendix H의 사례집이 이를 확장한다. **over-continuation**(정답이 사실상 나온 뒤에도 `implies` 같은 generic filler와 연결어에 큰 질량이 실려 멈추지 않는다), **hesitation loop**(`wait` 반복·문장부호 위주 연속이 궤적이 비생산적으로 바뀐 뒤에도 국소적으로 보상받는다), **degenerate drift**(깨진 비영어 텍스트로 흘러간 뒤에도 많은 token이 높은 teacher 확률을 받는다).
- 저자들의 가설은 두 증폭 요인이다. ① **sharp teacher distribution** — 사소한 teacher-student 불일치가 큰 log-ratio를 만든다 ② **긴 rollout을 따라 커지는 teacher-student divergence**. Figure 4가 이를 뒷받침한다 — teacher−student log-probability gap의 분포를 token position 구간별(0 ~ 16k)로 그리면 **뒤쪽 position bucket일수록 lower tail이 넓고 극단값이 많다** → 긴 horizon의 student rollout에서 teacher 신호가 더 noisy하다.

#### (iii) tokenizer / special-token mismatch

sampled-token OPD는 student가 생성한 **정확히 그 token**을 teacher 분포로 평가한다. 두 모델의 분절 규약이 다르면 같은 raw text가 다르게 쪼개진다 (Figure 5).

```
Raw text     : <think>  ...  Final answer is boxed{7}  <EOS>
Student 분절 : '<' , 'think' , '>'   ...   '<|im_end|>'
Teacher 기대 : '<th' , 'ink' , '>'   ...   (다른 종료 규약)

실제 logprob:  '<'          → student −0.07 , teacher −19.16
               '<|im_end|>' → student −0.00 , teacher −58.71
  ⇒ log q(<) ≪ log π_θ(<) ,  log q(<|im_end|>) ≪ log π_θ(<|im_end|>)
```

- 의미적으로는 **완전히 옳은 출력**인데 log-ratio가 19~59 수준의 거대한 벌점을 만든다. supervision이 단일 token에 걸려 있으므로 이 왜곡이 그대로 reward 신호를 오염시킨다. 이것은 objective가 아니라 **implementation-level 문제**이며, 그래서 baseline에도 독립적으로 적용 가능한 수정(masking)이 존재한다.

### 4.3 제안 기법 — teacher top-K local support matching (LSM)

출발점은 prefix c_t에서의 full-vocabulary reverse-KL이고, sampled-token OPD는 그것의 **Monte Carlo 근사 한 점**이다. 논문은 그 한 점을 **teacher가 정의한 지역 지지집합 위의 분포 비교**로 대체한다.

```
full-vocab reverse-KL:
  L_full(c_t) = Σ_{v ∈ V} π_θ(v | c_t) · log [ π_θ(v | c_t) / q(v | c_t) ]          ... (4)
sampled-token OPD (= 그것의 1-sample MC 근사):
  L_sample(c_t, y_t) = log [ π_θ(y_t | c_t) / q(y_t | c_t) ] ,  y_t ~ π_θ(· | c_t)  ... (5)

제안: prompt x마다 student inference policy로 그룹 {o_1, ..., o_G} 샘플, prefix c_{i,t} = (x, y_{i,<t})
  teacher 지지집합 : S(c_{i,t}) = TopK_q(c_{i,t})   (그 prefix에서 teacher 확률 상위 K개) ... (6)
  지지집합 내부 재정규화:
    π̂_θ(v | c_{i,t}) = π_θ(v | c_{i,t}) / Σ_{u ∈ S(c_{i,t})} π_θ(u | c_{i,t})
    q̂(v | c_{i,t})   = q(v | c_{i,t})   / Σ_{u ∈ S(c_{i,t})} q(u | c_{i,t})         ... (7)
```

#### Training Objective

```
L_LSM = E_{x, {o_i} ~ π_θ,infer} [
          (1 / Σ_i |o_i|) · Σ_{i=1..G} Σ_{t=1..|o_i|} Σ_{v ∈ S(c_{i,t})}
              π̂_θ(v | c_{i,t}) · log [ π̂_θ(v | c_{i,t}) / q̂(v | c_{i,t}) ]
        ]                                                                       ... (8)
```

sampled-token OPD 대비 **한 점 추정 → 같은 prefix에서 teacher가 지지하는 후보들에 대한 분포 비교**로 바뀌어, 업데이트가 더 이상 sampled token 하나의 log-ratio 부호·크기로 결정되지 않는다. full-vocabulary KL 대비로는 **훨씬 싸고**(K = 32) token-level 업데이트의 지역성이 유지되므로 §4.1의 분산 이점을 잃지 않는다. 저자들은 이것이 **truncated objective**임을 Appendix A에서 솔직히 명시한다 — 지지집합 밖 token은 gradient 기여를 받지 못하므로 full-vocab reverse-KL 대비 bias가 있고, "장점이나 단점이 아니라 추정기의 성질"로 서술한다.

#### 구성요소 ↔ 실패 모드 대응표 (실전 안정화 3종 포함)

| 구성요소 | 내용 | 겨냥하는 실패 모드 | 근거 / 없으면 |
|---|---|---|---|
| **teacher top-K local support matching** (truncated reverse-KL) | 식 (8). K = 32개 teacher-지지 후보 위 분포 비교 | **(i) 불균형 supervision** — 한 token 부호 의존 제거 | single-task 평균 36.4 → 41.7, multi-task 수학 평균 34.8 → 41.7 |
| **Support-set renormalization** | 지지집합 내부 logit에 **별도 softmax**. gradient가 지지집합 밖으로 직접 전파되지 않는다 | LSM 자체의 수치 안정성 (전제조건) | 없으면 teacher/student의 지지집합 내부 확률질량이 직접 비교 불가 → **rapid collapse** (Fig 6a) |
| **Top-p rollout sampling** | rollout을 top-p = 0.9로 생성. 제약 없는 샘플링은 아주 낮은 확률 token을 만들고 그 prefix 위 teacher 신호가 덜 유익해진다 | **(ii) 신뢰 불가 teacher 안내** — 궤적을 typical continuation 근처에 유지 | Table 3: top-K 단독 17.7 → top-K + top-p **23.6** |
| **Special-token masking** | 호환되지 않는 tokenization 규약이 만드는 false negative를 마스킹. **model-agnostic한 가장 단순한 교정** | **(iii) tokenizer / special-token mismatch** | baseline 36.4 → 40.7 (**+4.3점**), LSM은 41.7 → 41.5 (거의 무영향) |

> 마지막 행이 이 논문의 가장 깔끔한 논증이다. **masking이 baseline은 크게 살리지만 LSM은 거의 바꾸지 않는다** → LSM이 이미 tokenizer mismatch에 둔감하다는 직접 증거이자, 성능 향상이 mismatch 처리만으로 설명되지 않는다는 증거. (masking은 원칙적으로 multi-token marker 병합·등가 tokenization 평균으로도 다룰 수 있으나, 저자들은 tokenizer-specific 처방을 피했다.)

---

## 5. Experiments

### 5.1 Setup / Dataset

| | Math (§4.2, §4.3) | ALFWorld (§4.3) | WebShop (Appendix G.1) |
|---|---|---|---|
| Student | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct | Qwen2.5-1.5B-Instruct |
| Teacher | OpenThinker3-7B | GiGPO-Qwen2.5-7B-It-ALFWorld | GiGPO-Qwen2.5-1.5B-It-WebShop |
| 학습 데이터 | DAPO-Math-17K (영어 부분) | ALFWorld multi-turn agentic | WebShop |
| 평가 | Math500 / AIME24 / AIME25 / Minerva / OlympiadBench (pass@1, 일부 avg@32) | success rate | task score / success rate |
| Max prompt / response | 2048 / 16384 | 2048 / 512 | 2048 / 512 |
| Max turns | – | 30 | 15 |

- **세 가지 세팅**: ① single-task 수학 추론 ② 수학과 agentic task를 **batch 단위로 번갈아** 학습하는 multi-task (400 step = **수학 200 update + ALFWorld 200 update**) ③ 더 작은 student(1.5B)의 single-task agentic.

### 5.2 Implementation Details

| 항목 | 값 (Math / ALFWorld / WebShop 공통, 명시된 것 제외) |
|---|---|
| 프레임워크 / 하드웨어 | verl + **verl-agent** / 노드 1대, **NVIDIA H100 × 8** |
| Rollout | group size **8**, top-p **0.9**, temperature 1 |
| Evaluation | top-p 0.9, temperature 1 |
| **Teacher top-K** | **32** (세 세팅 모두 동일) |
| Optimizer / LR / warmup | AdamW / **2 × 10⁻⁶** / 0 |
| Batch / mini-batch | 128 / 64 |
| Total training steps | **400** (Math, ALFWorld) / 60 (WebShop) |

> **중요한 각주**: rollout **top-p는 LSM에서만 쓰이고 sampled-token OPD baseline에는 적용되지 않았다**. 따라서 main table의 비교는 "objective + rollout 정책" 묶음 단위의 비교이며, top-p만 떼어낸 기여는 Table 3의 ablation에서 확인해야 한다 (baseline 20.4 → baseline+top-p 21.6).

### 5.3 Main Results

#### (a) Single-task 수학 추론 (Table 1, pass@1)

| Method | Math500 | AIME24 | AIME25 | Minerva | OlympiadBench | **Avg.** |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-It (student 초기값) | 68.2 | 13.3 | 0.0 | 26.5 | 32.9 | 28.2 |
| OpenThinker3-7B (**teacher**) | 92.2 | 53.3 | 40.0 | 39.0 | 55.6 | **56.0** |
| Sampled-token OPD | 80.0 | 10.0 | 16.7 | 32.4 | 43.1 | 36.4 |
| Sampled-token OPD w/ mask | 81.4 | **26.7** | 16.7 | 34.2 | **44.7** | 40.7 |
| **Ours w/o mask** | 80.4 | 23.3 | **26.7** | 34.2 | 43.9 | **41.7** |
| **Ours w/ mask** | **82.0** | 23.3 | 23.3 | **34.9** | 43.9 | 41.5 |

- sampled-token OPD도 student를 28.2 → 36.4로 끌어올리긴 하지만 teacher 56.0에 **한참 못 미친다.** 여기에 masking만 얹어도 36.4 → **40.7** → tokenizer mismatch가 실패의 **실질적 일부**임이 증명된다. 나아가 LSM 두 변형이 **masked baseline까지 넘어선다**(41.7 / 41.5 vs 40.7) → 이득이 mismatch 처리에서만 오는 것이 아니라 **더 강한 분포 수준 distillation 신호**에서 온다. 반대로 LSM에서 masking의 영향은 41.7 → 41.5로 미미 → LSM이 one-token supervision보다 tokenizer mismatch에 덜 민감하다.

#### (b) Multi-task: ALFWorld + 수학 추론 (Table 2)

| Method | **ALFWorld** | MATH500 | AIME24 | AIME25 | Minerva | OlympiadBench | **Reasoning Avg.** |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-It | 21.9 | 68.2 | 13.3 | 0.0 | 26.5 | 32.9 | 28.2 |
| GiGPO-Qwen2.5-7B-It-ALFWorld (agentic teacher) | 95.3 | – | – | – | – | – | – |
| OpenThinker3-7B (math teacher) | – | 92.2 | 53.3 | 40.0 | 39.0 | 55.6 | 56.0 |
| Sampled-token OPD | 90.6 | 74.8 | 13.3 | 13.3 | 32.1 | 40.5 | 34.8 |
| Sampled-token OPD w/ mask | 93.8 | 76.0 | 20.0 | 13.3 | 33.5 | 40.4 | 36.6 |
| **Ours w/o mask** | 95.3 | **82.0** | **33.3** | 16.7 | 32.7 | **44.0** | **41.7** |
| **Ours w/ mask** | **97.7** | 79.0 | 20.0 | 16.7 | **34.6** | 42.5 | 38.6 |

- **논문의 헤드라인 +19.8% 는 여기서 나온다**: 수학 평균 34.8 → 41.7, 상대 개선 (41.7 − 34.8) / 34.8 = **+19.8%**. 그러면서 ALFWorld는 90.6 → 95.3으로 오히려 상승. masked LSM은 **ALFWorld 최고점 97.7**(agentic teacher 95.3마저 상회)을 얻지만 수학 이득의 일부를 반납한다(41.7 → 38.6).
- 해석: **LSM은 reasoning 쪽에 특히 유효**하다 — sampled-token 신호가 prefix drift에 더 크게 노출되는 쪽이 reasoning이기 때문. 반면 **masking은 이 run에서 trade-off를 agentic 쪽으로 밀어낸다.** 평가 성능 외에 최적화 dynamics도 일관되게 개선된다 (§5.4 (c)).

#### (c) 작은 student의 agentic setting (Table A2, WebShop / Qwen2.5-1.5B-Instruct)

| Method | Task score | Success rate |
|---|---|---|
| Qwen2.5-1.5B-It | 12.7 | 2.3 |
| GiGPO-Qwen2.5-1.5B-Webshop (**teacher**) | 81.9 | 66.4 |
| Sampled-token OPD | 73.0 | 50.0 |
| **Ours** | **75.1** | **57.8** |

- success rate **50.0 → 57.8** (+7.8점) — 경향이 math/ALFWorld 밖으로도 전이된다. 다만 teacher(66.4)와의 격차는 여전히 크고, 저자들도 이를 인정한다. 이 세팅에는 masking 변형이 없는데, **teacher가 같은 base 모델에서 RL로 만들어져 tokenizer/special-token mismatch가 지배 요인이 아니기 때문**이다 — 실패 모드 (iii)의 원인 진단과 정확히 일치하는 통제.

### 5.4 Ablation Study

#### (a) 구성요소 분해 (Table 3, single-task, AIME24 avg@32)

| Method | AIME24 avg@32 |
|---|---|
| Qwen2.5-7B-Instruct (student 초기값) | 10.0 |
| OpenThinker3-7B (teacher) | 63.3 |
| Sampled-token OPD | 20.4 |
| Sampled-token OPD + top-p | 21.6 |
| **Teacher top-K (단독)** | **17.7** |
| **Teacher top-K + top-p** | **23.6** |

**읽는 법이 중요하다.** **teacher top-K 단독은 baseline보다 나쁘다**(17.7 < 20.4) — "top-K 비교로 바꾸기만 하면 된다"는 명제는 거짓이다. rollout policy가 안정 영역에 머물러야 top-K가 살아나며, top-p를 더하면 17.7 → **23.6**(+5.9점)으로 뒤집힌다. **동일한 top-p 조건에서 비교**하면 21.6 → 23.6(**+2.0점**)이 objective 변경 자체의 순수 기여이고, top-p 단독 기여는 baseline 위에서 20.4 → 21.6(+1.2점)이다.

**설계 선택별 민감도(Figure 6)**: **Support renormalization**은 제거하면 **급속 붕괴(rapid collapse)** — 필수 요소다. **Support size K**는 아주 작으면 학습을 해치지만 충분히 크면 정확한 값에 민감하지 않다(main은 K = 32). **Rollout top-p**가 완전히 unconstrained면 학습이 불안정해진다.

#### (b) 학습 dynamics (Appendix G.2, Figure A4~A6)

| 지표 | LSM vs sampled-token OPD |
|---|---|
| Training reward / eval 곡선 | **학습 전 구간에서** 개선 (마지막 checkpoint에서만이 아니다), single-task·multi-task 모두 |
| Gradient norm / clipping-boundary fraction | 둘 다 **더 낮다** (충분한 policy entropy는 유지). Response length는 더 **짧다** |
| sampled token 위 teacher-student logprob gap | **0에 더 가까워진다** — baseline이 쓰는 진단 지표로 봐도 정렬이 개선 |
| special-token masking의 효과 | **baseline의 clipping-boundary fraction을 초·중반에 크게 낮춘다.** LSM에는 minor effect → masking은 baseline용 응급처치이고 LSM에게는 거의 필요 없다 |

#### (c) 지지집합 구성 변형 5종 (Table 4 / Table A3)

지지집합을 어떻게 만드는지, sampled token을 어떻게 포함시키는지에 따라 5개 변형이 있다. Variant 1~3은 지지집합 위 renormalized reverse-KL, Variant 4~5는 renormalization 대신 truncated head 항 + importance-weighted sampled-token tail 보정(EMA-PG 방식, full-vocab reverse-KL에 대해 unbiased).

| Variant | single-task AIME24 avg@32 | single-task Avg. | multi-task ALFWorld | multi-task Reasoning Avg. |
|---|---|---|---|---|
| **V1 teacher top-K (기본)** | **23.6** | 41.7 | 95.3 | **41.7** |
| V2 student top-K + sampled (renorm) | 22.3 | 41.9 | 95.3 | 28.4 |
| V3 teacher top-K + sampled (renorm) | 22.4 | **42.9** | 94.5 | 26.9 |
| V4 student top-K + sampled (EMA-PG) | 21.1 | 41.2 | 93.8 | 36.2 |
| V5 teacher top-K + sampled (EMA-PG) | 20.7 | 38.1 | **96.1** | 33.7 |

- **single-task에서는 셋 다 비슷하다** (V3가 pass@1 평균 최고 42.9, V1이 avg@32 최고 23.6). 그런데 **multi-task에서 순위가 완전히 갈린다** — 기본형 41.7 vs 나머지 26.9~36.2로 수학 벤치마크가 크게 무너진다. **EMA-PG 변형은 unbiasedness라는 이론적 동기에도 불구하고 경험적으로 더 낫지 않다.**
- 저자들의 결론은 신중하다: "**지지집합 구성이 중요하다는 증거로 다루되, 변형 간 순위를 과잉해석하지 말라**" — partial ablation이자 예비 탐색이라고 명시.

---

## 6. Key Takeaways

1. **token-level OPD는 "부정확한 근사"가 아니라 "정당한 거래"다.** sequence-level reverse-KL에 대해 biased인 것은 맞지만(버려지는 항이 정확히 future-reward coupling), worst-case 분산 상한이 **O(T²) vs O(T⁴)** 로 갈린다. 16k token 응답을 다루는 long-horizon post-training에서 이 차이는 결정적이다. toy 실험에서도 **γ = 0.75~1.0의 gradient variance가 작은 γ보다 1~수 자릿수 높게 유지**되고, **γ = 1.0에서는 정책이 목표로 수렴하지 못하고 drift**한다.

2. **부서지는 것은 token-level이 아니라 sampled-token이다.** 실패는 세 갈래로 온다 — ① 첫 iteration 산점도에서 **대다수 sampled token이 음의 reward**를 받아 최적화가 소수의 filler token에 지배된다 ② **repetition loop에서도 teacher가 국소적으로 동의**한다 ③ tokenizer 불일치가 **의미적으로 옳은 출력에 log-ratio 19~59의 벌점**을 붙인다(`'<'`: student −0.07 / teacher −19.16, `'<|im_end|>'`: student −0.00 / teacher −58.71).

3. **teacher signal의 신뢰도는 rollout이 길어질수록 떨어진다.** teacher−student logprob gap 분포를 token position 구간별로 보면 **뒤쪽 bucket일수록 lower tail이 넓고 극단값이 많다**(Figure 4, 0~16k). 증폭 요인은 sharp teacher distribution과 누적되는 teacher-student divergence — "on-policy이므로 항상 유효한 신호"라는 직관에 대한 실증적 반례다.

4. **top-K로 바꾸는 것만으로는 오히려 나빠진다.** teacher top-K 단독은 AIME24 avg@32에서 **17.7로 baseline 20.4보다 낮다.** top-p rollout을 더해야 **23.6**으로 뒤집힌다. 동일 top-p 조건에서 objective 변경의 순수 기여는 21.6 → 23.6 (+2.0점). **rollout 정책과 objective는 함께 설계해야 한다.**

5. **핵심 성과는 multi-task 수학 평균 34.8 → 41.7 (+19.8%)이며, ALFWorld도 90.6 → 95.3으로 동반 상승**한다. single-task에서는 36.4 → 41.7. 작은 student(1.5B)의 WebShop success rate도 50.0 → 57.8. 다만 **teacher(single-task 56.0, WebShop 66.4)와의 격차는 여전히 크게 남는다.**

6. **masking의 비대칭이 진단을 확증한다.** special-token masking은 baseline을 36.4 → **40.7**(+4.3점)로 크게 살리지만 LSM은 41.7 → 41.5로 거의 바꾸지 않는다. 학습 dynamics에서도 masking은 **baseline의 clipping-boundary fraction만 크게 낮춘다.** 즉 LSM은 이미 tokenizer mismatch에 둔감하고, 성능 이득은 mismatch 처리가 아닌 **분포 수준 신호** 자체에서 온다.

7. **teacher matching은 task success의 불완전한 프록시다.** Appendix H는 over-continuation(정답 후 계속 추론), hesitation loop(`wait` 반복), degenerate drift(깨진 비영어 출력)에서 **teacher가 여전히 높은 확률을 주는** 사례를 모았다. 이 논문은 reward hacking이 OPD에서 **실제로 일어난다**고 명시한다. 더 나은 지역 supervision은 distillation 문제의 일부일 뿐이며, 저자들은 rollout 제어·distribution shift 처리·teacher uncertainty 활용·**outcome-verifiable reward와의 결합**을 남은 과제로 든다.

---

## 7. 원문 블로그 대비 갱신점

| # | 블로그(2025.10.27) 주장 | 이 논문의 판정 | 근거 |
|---|---|---|---|
| ① | **OPD = dense × on-policy, 두 함정 동시 회피** | **유지** | 전제로 삼는다. 다만 "dense = 정확"이 아님을 추가 — dense한데 **틀린 방향으로 dense**할 수 있다 |
| ② | **`advantages = -reverse_kl` 한 줄이면 구현 끝** | **🔴 정면 반박** | 그 한 줄이 곧 sampled-token 추정이고, 실패 모드 3종의 공통 원인이다. 실제로 필요한 것은 **top-K 지지집합 + 내부 renormalization + top-p rollout + special-token masking** 네 가지. renormalization을 빼면 **rapid collapse**, top-p를 빼면 top-K가 **baseline보다 나빠진다**(17.7 < 20.4) |
| ③ | **discount 0 덕분에 안정화 장치 불필요** | **⚠️ 절반 강화 / 절반 반박** | 전반부는 **블로그보다 강한 근거로 승격**: discount 0(= token-level)은 편의가 아니라 **O(T²) vs O(T⁴)** 분산 상한으로 정당화되며 toy 실험이 이를 재현한다. 후반부는 반박: discount 0은 **variance는 잡지만 signal quality는 못 잡는다**. renormalization·top-p·masking이 실제로 필요했다 |
| ④ | **reverse KL의 mode-seeking이 장점** | **유지 + 정밀화** | reverse-KL 정식화를 그대로 채택. 다만 실제 최적화되는 것은 full-vocab reverse-KL이 아니라 **그것의 1-sample Monte Carlo 근사**임을 명시하고, 그 근사가 문제의 근원이라고 지적. 관련 문헌도 정리 — Jin et al.(2026a)은 teacher entropy가 높으면 순수 reverse-KL OPD가 brittle해진다고 보고 |
| ⑤ | **reverse KL은 unhackable** | **🔴 반박** | Appendix H가 **reward-hacking 사례집**이다. repetition loop(Fig 3), over-continuation, `wait` 반복, 깨진 비영어 출력에도 teacher가 높은 확률을 준다. Appendix A: "**teacher matching remains an imperfect proxy for task success**" — 국소적으로 teacher가 선호하는 연속이 전체 궤적에는 무익하거나 해로울 수 있다 |
| ⑥ | **teacher가 성능 상한** | **지지 (오히려 강화)** | 상한이 아니라 **도달조차 못 하는 먼 목표**. single-task 최고 student 41.7 vs teacher 56.0. WebShop success 57.8 vs teacher 66.4. 저자들은 "teacher와의 눈에 띄는 격차가 남는다"고 명시하고, **teacher와 student가 크게 다를 때는 더 나은 지역 supervision만으로 부족**하다고 결론 |
| ⑦ | **어떤 open-weight teacher든 쓸 수 있다** | **⚠️ 조건부** | tokenizer/special-token 규약이 어긋나면 **의미적으로 옳은 출력에 −19.16, −58.71 수준의 벌점**이 붙는다. 이 실험의 teacher(OpenThinker3-7B)는 student와 **같은 Qwen2.5-7B-Instruct 계열**인데도 문제가 발생했다. 역으로 teacher가 **같은 base에서 RL로 만들어진** WebShop 세팅에서는 masking 변형 자체가 불필요했다 → **teacher 선택은 vocabulary·special-token 규약까지 확인해야 한다** |
| ⑧ | **샘플 1개짜리 per-token 추정으로 충분하다** | **🔴 정면 반박 (논문의 본론)** | "충분하다"의 근거는 기댓값 unbiasedness였으나, **첫 iteration 산점도에서 대다수 token이 음의 reward**를 받고 최적화가 소수 filler에 지배된다. K = 32 지지집합 위 분포 비교로 바꾸면 single-task 36.4 → 41.7, multi-task 수학 34.8 → 41.7(**+19.8%**). full-vocab을 쓰지 않고도 **한 점 → 32점**만으로 회복된다는 것이 실무적 핵심 |

### 후속 연구들 사이에서의 위치

| 논문 | 이 논문과의 관계 |
|---|---|
| ExOPD (2602.12125) | **상보적**. ExOPD는 ⑥(teacher 상한)을 공격하고, 이 논문은 ②·⑧(추정기 품질)을 공격한다 |
| Rethinking OPD (2604.13016) | **강하게 상보적**. "overlap token이 확률질량 97~99%, 작은 공유 token 집합에 신호가 집중된다"는 관찰이 **top-K 지지집합이 왜 작아도 되는가**에 대한 독립적 설명이 된다 |
| Demystifying OPD (2604.08527) | **같은 계열, 다른 증상**. 이쪽은 "신호가 틀리는 문제", 저쪽은 "학습이 발산하는 문제"(length inflation / truncation collapse). LSM이 **response length를 줄인다**는 dynamics 관측이 저쪽 문제와 맞닿는다 |
| SOD (2605.07725) | **동일 증상의 agentic 판본**(SLM tool call 오류 연쇄). 이 논문의 WebShop(1.5B) 결과가 접점 |
| SDFT (2601.19897) | **미래 방향으로만 접점**. Appendix B가 continual learning을 "distribution shift·teacher staleness·근사 오차 누적을 동시에 압박하는 testbed"로 지목 |

> **읽는 순서 제안**: 블로그로 개념 → Rethinking OPD로 teacher 선정 → **이 논문으로 objective와 rollout 정책을 확정** → Demystifying OPD로 발산 대비. 특히 **코드를 쓰기 직전**에 이 논문의 §4.3 대응표와 Table 3(구성요소 ablation)을 다시 볼 것.

---

[← 후속 연구 정리](./opd_follow_up_research.md) · [원문 요약](./on_policy_distillation.md)
