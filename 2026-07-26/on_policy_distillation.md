# On-Policy Distillation

> **Venue**: Thinking Machines Lab — Connectionism blog (2025.10.27)
> **Authors**: Kevin Lu (Thinking Machines Lab)
> **Platform**: Tinker / Tinker Cookbook (오픈 레시피 공개, 약 75줄)
> **Link**: https://thinkingmachines.ai/blog/on-policy-distillation/

**한 줄 정의**: student가 직접 rollout을 샘플링하고, teacher가 그 rollout의 **모든 token을** 채점한다.
RL의 on-policy 관련성 + distillation의 dense 신호를 한 자리에 모은 post-training 기법.

---

## 1. Background

### LLM post-training의 현황

- 전문 모델은 세 단계로 만들어진다.
  - **Pre-training** — 일반 능력 (언어 사용, 폭넓은 추론, 세계 지식) / 웹 코퍼스
  - **Mid-training** — 도메인 지식 (코드, 의료 DB, 사내 문서) / 도메인 코퍼스
  - **Post-training** — 목표 행동 (instruction following, 수학 풀이, chat) ← **OPD의 자리**
- 잘 학습된 작은 모델은 자기 도메인에서 큰 범용 모델을 이긴다. 실무적 이점이 크다.
  - 로컬 배포 (프라이버시·보안), 지속 업데이트 (재학습이 가볍다), 추론 비용 절감 (서빙 단가)
- 따라서 관건은 **마지막 단계를 어떤 방법으로 하느냐**이다.

### post-training을 가르는 두 축

- **축 1 — sampling**: 누가 만든 시퀀스로 배우나 (off-policy = 남의 시퀀스 / on-policy = student 자신의 rollout)
- **축 2 — reward signal**: 피드백이 얼마나 촘촘한가 (dense = O(N) bits/episode / sparse = O(1) bits/episode)

| | **Sparse** — O(1) bits/episode | **Dense** — O(N) bits/episode |
|---|---|---|
| **Off-policy** (남이 만든 시퀀스) | 남의 시퀀스에 점수 하나 — episode당 신호 최소 | **SFT / off-policy distillation**<br>teacher 시퀀스에 cross-entropy |
| **On-policy** (student 자신의 rollout) | **RL (RLVR: GRPO / DAPO)**<br>자기 rollout이지만 신호는 정오답 하나 | **On-policy distillation**<br>student rollout 위에서 token마다 teacher가 채점 |

> OPD는 비어 있던 새 칸이 아니라, **RL과 SFT가 각각 반쪽씩 갖고 있던 두 장점을 한 칸에 모은 것**이다.

### 기존 방법의 한계

| 방법 | 신호 밀도 | 분포 | 문제점 |
|---|---|---|---|
| SFT / off-policy distillation | dense — O(N) bits | 남의 것 | compounding error, 스타일만 모방 |
| RL (GRPO / DAPO) | sparse — O(1) bits | 자기 것 | credit assignment 부재, search에 compute 낭비 |
| **On-policy distillation** | **dense — O(N) bits** | **자기 것** | **두 함정 모두 회피** |

### 체스 비유 (원문)

| 방식 | 비유 | 무엇이 부족한가 |
|---|---|---|
| on-policy RL | 코치 없이 대국만 두기 | 승패는 내 수에서 나오지만 판당 1 bit, 어느 수가 결정적이었는지 모름 |
| off-policy distillation | 그랜드마스터 기보 관전 | 수는 훌륭하지만, 그 판면은 초보인 내가 실전에서 만날 상태가 아님 |
| **On-policy distillation** | **내 대국의 내 수마다 등급** | 코치가 내가 둔 수를 "blunder"부터 "brilliant"까지 등급 매김 |

### 계보 — 고전 KD의 두 갈래

| 갈래 | 방식 | 특성 |
|---|---|---|
| **Logit distillation** | 각 위치에서 teacher의 다음 token 분포 전체(soft label)를 맞춤 | 정보량 최대, 그러나 teacher forward가 학습 루프 안에 있어야 하고 저장·전송이 무거움 |
| **Sequence (sample) distillation** | teacher가 샘플링한 시퀀스를 정답으로 SFT | 시퀀스 샘플은 teacher 분포의 unbiased 추정 → 기대값으로 같은 목적함수. 데이터셋을 한 번 만들어 공개·재사용 (업계 표준) |

- sequence distillation 성공 사례는 **전부 off-policy**: Alpaca(instruction following), OpenThoughts-3 / DeepSeek-R1-distill(수학·과학 추론), UltraChat(multi-turn chat)
- 주의: "unbiased 추정"은 **teacher 분포 위에서의 목적함수 추정**이 unbiased라는 뜻이지, student가 방문할 상태를 다룬다는 뜻이 아니다. 추정 문제와 분포 문제는 별개다.

### 선행 연구와의 관계

| 설계 항목 | MiniLLM (2023) | GKD (2023) | **On-policy distillation (2025)** |
|---|---|---|---|
| 샘플링 분포 | teacher-mixed (0.2p + 0.8q) | λ 혼합 (최적은 λ=1) | **순수 student (λ=1)** |
| 목적함수 | reverse KL + 누적 return R_t | 선택형 divergence (직접 미분) | **per-token reverse KL, discount 0** |
| 최적화 | policy gradient + 안정화 3종 | supervised 식 gradient | **advantage = −KL 로 기존 RL loss 재사용** |
| 추가 장치 | length norm, clipping, LM loss | 없음 | **없음** |

- GKD의 핵심 발견: 요약·번역·수학 전부에서 **λ=1(순수 on-policy)이 최고**. RLHF의 reference 정규화 항을 teacher로 바꾸면 distillation이 RL 파이프라인에 올라탄다.
- **discount 0 이 단순화의 핵심**: MiniLLM의 누적 return R_t를 버리고 즉시 KL만 남기면 고분산의 원천이 사라지고, 그것을 누르던 안정화 장치도 불필요해진다.
- 같은 시기 **Qwen3**가 이 2단계(off-policy → on-policy distillation)를 제품 파이프라인에 넣었으나 **loss 수식과 KL 방향은 비공개**. 이 글의 기여는 그 빈칸을 구체 레시피로 채워 공개 재현한 것.

---

## 2. Motivation

### 핵심 통찰 1: off-policy의 함정 — 한 번 벗어나면 계속 벗어난다

teacher가 자주 가는 상태에서만 배우면, student는 **자기 실수에서 복구하는 법을 배우지 못한다.**

- **Compounding error**: student가 teacher는 하지 않을 초기 실수를 하면, 학습에서 관측한 상태로부터 점점 더 멀어진다.
- **DAgger (Ross et al. 2010)** — behavior cloning의 기대 오차는 horizon T에 대해 **O(T²)**. student가 자기 정책으로 방문한 상태에서 expert 라벨을 받으면 **O(uT)** 로 줄어든다.
- **Exposure bias (Bengio et al. 2015)** — 학습은 정답 prefix(teacher forcing), 추론은 자기 생성 prefix. 조건 분포가 어긋나 복구를 못 배운다.
- **False Promise (Gudibande et al. 2023)** — 말투·형식은 그럴듯해지지만 **사실성·능력 격차는 그대로**. 사람 평가자도 처음엔 속는다.

### 핵심 통찰 2: RL의 함정 — episode당 O(1) bits

on-policy라 분포 문제는 없지만, 피드백이 너무 성기다.

- rollout이 수천 token이어도 배우는 건 스칼라 하나. "21"이라는 오답을 낸 rollout에서 student가 배우는 것은 **"이 rollout 전체가 나빴다"** 뿐이다.
  - 연산 순서를 틀렸는지, 산수를 틀렸는지 모른다 → **credit assignment 부재**
- GRPO / DAPO도 마찬가지. group 상대 비교로 만든 advantage를 **시퀀스 전체에 똑같이 바른다.**
- 그 결과 RL은 compute의 대부분을 gradient 업데이트가 아니라 **search**(rollout을 뽑고 어쩌다 좋은 전략에 걸리기)에 쓴다.

### 핵심 통찰 3: dense feedback이 낫다는 근거는 이미 있었다

- **Let's Verify Step by Step (Lightman et al. 2023)** — process supervision(스텝별 채점)이 outcome supervision(최종 답 채점)보다 낫다.
  - 고정 generator에서 best-of-1860 선택 정확도 **78.2% vs 72.4%**
- 다만 process reward model(PRM)은 **라벨링 비용이 크다.**
- **OPD의 답**: 이미 있는 teacher 모델의 logprob이 곧 **공짜 per-token 채점기**다. PRM을 따로 학습하거나 라벨링할 필요가 없다.

---

## 3. Contributions

1. **비어 있던 사분면을 채우는 정식화**: post-training을 (sampling 축 × reward density 축)으로 정리하고, "dense × 자기 분포" 칸이 OPD임을 명확히 함.
2. **per-token reverse KL을 advantage로 쓰는 구체 레시피 공개**: Qwen3 테크리포트가 비공개로 남긴 loss 수식과 KL 방향을 채워 공개 재현. Tinker Cookbook 약 75줄, distillation 전용 loss 없이 기존 RL의 `train_step` 재사용.
3. **수학 추론에서 RL 대비 압도적 비용 효율 입증**: AIME'24 **74.4% (OPD, 1,800 GPU h) vs 67.6% (RL, 17,920 GPU h)** — 약 1/10 비용으로 더 높은 점수. 회계 기준에 따라 **9~30배** 절감.
4. **personalization에서 파괴적 망각 복구 입증**: 사내 문서 mid-training으로 무너진 instruction-following을, **mid-training 이전의 자기 자신을 teacher로** 삼아 지식 손실 없이 복구 (IF-eval 79% → 83%).
5. **continual learning 도구로서의 성질 제시**: teacher가 고정된 OPD는 항상 on-policy로 남기 때문에, 자기 샘플 SFT와 달리 시간이 지나도 열화되지 않는다.

---

## 4. Method

### 4.1 한 문장 정의와 목적함수

student가 rollout을 뽑고, teacher가 그 rollout의 **모든 token을** 채점한다.

```
per-token reverse KL:

KL(π_θ || π_teacher)
  = E_{x ~ π_θ} [ log π_θ(x_{t+1} | x_1..t) − log π_teacher(x_{t+1} | x_1..t) ]

reward_t = − KL_t
advantage_t = − ( log π_θ(x_t | x_<t) − log π_teacher(x_t | x_<t) )
```

핵심 설계 3가지:

| 설계 | 내용 |
|---|---|
| **기대값이 student 분포 위** | x ~ π_θ 에서 잡힌다 — 이것이 "reverse"이자 동시에 "on-policy". student가 실제로 가는 상태에서만 차이를 잰다. |
| **샘플 하나로 계산** | student가 뽑은 token에서 `log π_θ − log π_teacher` 값 하나. teacher에게는 "이 token에 네 logprob이 얼마냐"만 물으면 된다. |
| **discount = 0** | 각 시점에서 바로 다음 token의 즉시 KL만 최적화. 미래 token에 미칠 영향(return)은 계산에 넣지 않는다. |

- student가 teacher와 완전히 같아지면 **KL = 0**.
- full distribution을 요구하는 logit distillation과 달리 **teacher forward pass 1번**이면 된다.

> **discount 0인데 장기 전략을 어떻게 배우나?**
> 각 상태에서 teacher의 조건부 분포를 따라가면 궤적 전체가 teacher의 장기 전략을 재현한다.
> 미래 크레딧은 teacher가 이미 자기 정책에 녹여놨다.

### 4.2 RL 기계에 그대로 태운다

새 loss를 만들지 않는다 — **per-token advantage를 −reverse KL로 세팅할 뿐**이다.

| 단계 | 내용 |
|---|---|
| 1. teacher client 초기화 | 샘플링은 하지 않는다. logprob 계산만 하면 되므로 sampling client 하나. |
| 2. rollout 샘플링 | RL과 완전히 동일. student logprob은 이때 이미 나온다. |
| 3. reward 계산 | teacher에게 같은 token 시퀀스의 logprob을 묻고, 차이를 advantage에 −로 넣는다. |
| 4. 학습 | 기존 RL의 importance-sampling loss를 그대로 호출한다. |

```python
teacher_client = service_client.create_sampling_client(
    base_model=teacher_config.base_model)

trajectories = do_group_rollout(student_client, builder)
sampled_logprobs = trajectories["logprobs"]

teacher_logprobs = teacher_client.compute_logprobs(traj)
reverse_kl = sampled_logprobs - teacher_logprobs
trajectories["advantages"] = -reverse_kl

training_client.forward_backward(
    trajectories, loss_fn="importance_sampling")
```

```
importance_sampling loss = clip 없는 per-token policy gradient

loss = -(exp(target_lp - sampling_lp) * advantages).sum()
```

- Tinker cookbook 실제 구현은 **약 75줄**. distillation 전용 loss는 없고 RL의 `train_step`을 그대로 import 한다.
- **RL 구현 위에서는 사실상 한 줄 변경** (KL 정규화를 쓰는 RL이라면 더더욱).
- verl 대응: 2 = rollout + old_log_prob, 3 = teacher 서버 logprob 조회 + advantage 세팅, 4 = actor update.

#### advantage의 부호

| 상황 | KL | advantage | 결과 |
|---|---|---|---|
| student는 잘 뽑는데 teacher는 안 뽑을 token (자신만의 나쁜 습관) | 큼 | 큰 음수 | 확률을 깎는다 |
| teacher도 그 상태에서 뽑았을 token | ≈ 0 | ≈ 0 | 건드리지 않는다 |
| 샘플 1개짜리 추정이라 token 단위로 음수 KL이 나온 경우 | 음수 | 양수 | 오히려 강화된다 (기대값으로만 ≥ 0) |

#### reverse KL의 세 성질

1. **Mode-seeking** — 능력이 부족한 student에게는 "teacher의 스타일 전부"보다 **"teacher의 한 가지 좋은 전략"** 이 낫다. forward KL처럼 여러 mode 사이로 퍼지지 않는다.
2. **Unhackable** — 학습된 reward model은 프록시라서 허점을 파고들 수 있다. reverse KL이 낮다는 것 **자체가** "teacher 관점에서 바람직한 행동"이라 프록시와 목표가 일치한다.
3. **Exposure bias 감소** — 자기가 만든 prefix 위에서 배우므로 학습 분포와 추론 분포가 일치한다.

> RL 자체가 (KL penalty 형태에서) sequence-level reverse KL 계열을 최적화한다.
> OPD는 **같은 기하학을 per-token으로 촘촘하게 만든 것**이다.

### 4.3 실제 채점 예시 — 벌점은 forking token에 몰린다

- 문제: SimpleBench "프라이팬 위 각얼음 개수" — 정답은 B. 0 (얼음은 녹는다)
- student: Qwen3-4B-Instruct-2507 — 물리 맥락을 무시하고 **순수 산수 문제로** 풀었다
- teacher: Qwen3-235B-A22B-Instruct-2507 — 이 문제를 맞히는 모델

| 관측 | 해석 |
|---|---|
| **벌점이 큰 token** | 잘못된 풀이 방향으로 갈라져 들어가는 문구의 시작점 — "단순 계산 문제다"라고 결정해버리는 지점 (**forking token**, Wang et al. 2025) |
| **최종 오답 token은 벌점이 거의 없다** | 앞의 잘못된 추론을 다 조건으로 넣고 보면 그 오답은 완전히 예측 가능하다 — teacher조차 같은 token을 뽑는다 |

> sparse RL이라면 이 rollout 전체에 벌점 하나를 발랐을 것이다.
> OPD는 **"어디서 갈라졌는지"** 를 정확히 짚는다.

### 4.4 왜 싼가 — 비용 구조 네 가지

| # | 이유 | 설명 |
|---|---|---|
| 1 | **teacher는 forward pass 1번** | 비싼 샘플링은 작은 student가 하고, 큰 teacher는 이미 뽑힌 token의 logprob만 계산 |
| 2 | **partial rollout 가능** | reward가 시퀀스 끝에 걸려 있지 않으니, 끝까지 안 뽑아도 그 지점까지의 모든 token이 학습 신호를 만든다 |
| 3 | **별도 reward model이 없다** | instruction-tuned open-weight 모델이면 무엇이든 `compute_logprobs`만으로 teacher가 된다 |
| 4 | **작은 batch로도 된다** | episode당 O(N) bits를 주니 gradient 노이즈가 작다. sparse RL은 큰 batch로 노이즈를 눌러야 한다 |

> RL은 episode가 끝나야 reward가 나온다 — OPD는 그 제약이 없다.

#### 학습 vs 추론

| 단계 | 과정 |
|---|---|
| **학습** | student가 prompt에 대해 rollout 샘플링 → teacher가 같은 token 시퀀스의 logprob 계산 → advantage = −(student logprob − teacher logprob) → importance-sampling loss로 student 업데이트 |
| **추론** | student 단독. teacher는 배포에 전혀 관여하지 않는다 |

---

## 5. Experiments

### 5.1 Dataset

| | 실험 ① 수학 추론 | 실험 ② Personalization |
|---|---|---|
| Student 초기화 | Qwen3-8B-Base | Qwen3-8B (이미 RL post-trained) |
| Teacher | Qwen3-32B (Qwen3-8B이 더 좋기도 함) | **mid-training 이전의 자기 자신 (Qwen3-8B)** |
| 학습 데이터 | OpenThoughts-3 (teacher 생성 수학 추론 400k) | mid-train: 사내 문서 / OPD: **Tulu3 chat prompt** |
| 평가 | AIME'24, GPQA-Diamond | 사내 QA eval, IF-eval |

### 5.2 Implementation Details

- 공통 초기화: OpenThoughts-3 400k SFT (AIME'24 **60%**)
- OPD 학습량: 약 **150 step**, **77k prompts × 4 samples/prompt**
- LoRA 실험: rank 32 (personalization에서는 rank 32~256)
- 구현: Tinker Cookbook, `loss_fn="importance_sampling"`

### 5.3 Main Results — 실험 ① 수학 추론

Qwen3 Technical Report Table 21 — **같은 SFT 초기화 위에 마지막 단계만 바꾼** 비교.

| 방법 | AIME'24 | GPQA-Diamond | GPU hours |
|---|---|---|---|
| off-policy distillation (SFT) | 55.0% | 55.6% | 미보고 |
| + Reinforcement learning | 67.6% | 61.3% | 17,920 |
| **+ On-policy distillation** | **74.4%** | **63.3%** | **1,800** |

> **RL의 약 1/10 비용으로 더 높은 점수.** 이 결과가 저자들이 재현에 나선 계기다.

#### 비용 회계 — 무엇을 비용에 넣느냐에 따라 9~30배

FLOPs 기준. GPU 병렬화가 잘 되는 logprob 계산의 실제 비용은 이보다 낮다.

| 방법 | AIME'24 | Teacher FLOPs | Student FLOPs | CE vs SFT-2M |
|---|---|---|---|---|
| 초기화: SFT-400K | 60% | 8.5 × 10²⁰ | 3.8 × 10²⁰ | – |
| SFT-2M (외삽) | ~70% | 3.4 × 10²¹ | 1.5 × 10²¹ | 1× |
| Reinforcement learning | 68% | – | – | ≈1× |
| **On-policy distillation** | **70%** | **8.4 × 10¹⁹** | **8.2 × 10¹⁹** | **9–30×** |

| 배수 | 회계 기준 |
|---|---|
| **9×** | SFT 데이터셋이 이미 있다고 칠 때. off-policy의 teacher 생성 비용은 안 세고, OPD의 teacher logprob 비용은 센다 |
| **≈18×** | GPU hours 기준. teacher logprob 계산은 병렬화가 잘 돼서 FLOPs 수치보다 실제 시간이 덜 든다 |
| **30×** | 새 태스크라 데이터셋부터 만들 때. teacher 샘플링 비용까지 off-policy 쪽에 온전히 계상하는 경우 |

- off-policy distillation은 400k prompt에서 60%, **log-linear scaling**을 따른다. 외삽하면 70% 도달에 약 **2M prompt** 필요.
- RL(17,920 GPU h)은 대략 2M off-policy prompt와 비슷한 compute 비용.
- OPD는 **150 step (77k prompts)** 만에 70% 도달.
- RL 행의 68%는 회계 표의 반올림 표기로 67.6%와 같은 값.

#### LoRA 보너스

| 조건 | full FT 대비 격차 |
|---|---|
| 대용량·고배치 SFT 후 LoRA (rank 32) | **−13%** |
| + On-policy distillation 후 | **−6%** |

작은 batch로도 학습되는 영역이라 LoRA가 full FT와 대등해진다.

### 5.4 Main Results — 실험 ② Personalization

**목표**: 도메인 지식(사내 문서) + post-train된 행동(instruction following)을 **둘 다** 가진 어시스턴트.

#### 문제: 파괴적 망각 (catastrophic forgetting)

| 모델 | 사내 QA | IF-eval |
|---|---|---|
| 원본 Qwen3-8B | 18% | 85% |
| + mid-train (문서 100%) | 43% | **45%** |
| + mid-train (문서 70% / chat 30%) | 36% | **79%** |
| **+ mid-train (70%) + On-policy distillation** | **41%** | **83%** |

- **어떤 혼합 비율로도 IF-eval 원래 성능을 유지하지 못했다.** background chat 데이터를 섞어도 마찬가지.
- **LoRA(rank 32~256)로 업데이트를 제약해도 실패** — "LoRA Learns Less and Forgets Less"를 재확인. 지식은 덜 배우면서 post-training 행동은 여전히 잊는다.

#### 해법: OPD로 행동 복구 — teacher는 과거의 자기 자신

- mid-training이 끝난 모델에, **mid-training 이전 버전(Qwen3-8B)을 teacher로** OPD.
- prompt는 **Tulu3**만 사용. **사내 문서는 전혀 쓰지 않는다.**
- 결과: IF-eval **79% → 83%** (원본 85%에 근접) 하면서 사내 QA는 **36% → 41%** 로 오히려 상승.

> 언어모델 자신을 reward model로 쓰는 셈이다 — 고확률 행동이 곧 보상.

### 5.5 Ablation / 추가 분석

#### (a) RL과의 직접 비교 — dense supervision의 효율

1. Qwen3-8B-Base를 DeepMath로 RL 학습
2. 그 RL 학습된 모델을 다시 base 모델에 distill

| 항목 | 결과 |
|---|---|
| 동등 성능 도달 속도 | RL 대비 **약 7~10배 빠름** (동일 아키텍처 기준) |
| 총 compute 환산 | **50~100배 절감** (문맥 길이·batch size 요구 차이 반영) |

> **해석**: RL은 rollout search로 전략을 탐색한다. distillation은 중간 전략들을 건너뛰고 **최종 의미 전략만** 직접 배운다.

#### (b) 데이터 재사용 — 단일 prompt 실험

- 수학 prompt **단 하나**로 20 step 연속 학습, step당 256 rollout (**총 5,120 시퀀스**)
- 결과: **teacher 성능에 근접**

> 답을 외우는 것이 아니라 **분포를 배운다**는 증거. RL과 달리 데이터 재사용이 효과적으로 가능하다.

#### (c) continual learning 관점 — 자기 샘플 SFT는 왜 안 되나

| 방식 | 이론상 KL | 실제 결과 |
|---|---|---|
| 모델 자신의 샘플로 SFT | 0 | **IF-eval 열화** — "0보다 큰 어떤 학습률에서도 성능이 떨어진다" |
| **OPD (teacher 고정)** | — | **항상 on-policy로 남는다** |

- 이유: 유한 batch는 분포 편차를 갖고, 학습이 진행되면 그 데이터는 점차 **off-policy가 된다.**
- 반면 teacher가 고정된 OPD는 student가 어디로 움직이든 그 지점에서 다시 채점하므로 on-policy성이 유지된다 → **continual learning에 매우 유망한 도구.**

#### (d) 왜 RL은 느린가 — semantic strategy search 가설

- RL은 파라미터 공간을 탐색하는 것이 아니라, **과거에 찾은 전략을 조금씩 변형하며 운으로 새 전략에 "걸려 넘어진다".**
- 일단 전략이 발견되고 나면 distillation은 지름길이다. student는 중간 전략들을 다 거칠 필요 없이 **최종 전략만** 배우면 된다.
- 비유: 연구는 답을 찾는 데 오래 걸리지만, 찾아낸 결과를 자연어로 가르치는 것은 훨씬 쉽다. (반복 연습이 필요한 스포츠 같은 skill과는 다르다.)

---

## 6. Key Takeaways

1. **OPD = dense × on-policy.** RL의 on-policy 관련성(자기 상태 분포에서 학습)과 distillation의 dense 신호(token당 피드백)를 한 칸에 모은다. 두 함정(compounding error, credit assignment 부재)을 **동시에** 회피한다.

2. **구현은 사실상 한 줄.** 새 loss가 필요 없다. per-token advantage를 `−reverse_kl`로 세팅하고 기존 RL의 importance-sampling loss를 그대로 호출한다. Tinker Cookbook 기준 **약 75줄**.

3. **비용 효율이 압도적.** AIME'24에서 **74.4% / 1,800 GPU h (OPD)** vs **67.6% / 17,920 GPU h (RL)** — 약 **1/10 비용에 더 높은 점수**. FLOPs 회계 기준으로도 **9~30배** 절감.

4. **벌점은 forking token에 몰린다.** 최종 오답 token이 아니라 **추론이 갈라지는 지점**에 큰 KL이 붙는다. sparse RL이 rollout 전체에 벌점 하나를 바르는 것과 대비되는, credit assignment의 실질적 해결.

5. **파괴적 망각의 실용적 해법.** 사내 문서 mid-training으로 IF-eval이 85%→79%(문서 100%면 45%)까지 무너진 모델을, **mid-training 이전의 자기 자신을 teacher로** 삼아 **83%까지 복구**하면서 사내 QA는 36%→41%로 오히려 상승. **복구 단계에 사내 문서를 전혀 쓰지 않는다**는 점이 핵심.

6. **분포를 배우지 답을 외우지 않는다.** 단일 prompt에 20 step × 256 rollout(5,120 시퀀스)만으로 teacher 성능에 근접 — RL과 달리 **데이터 재사용이 가능**하다. teacher가 고정되어 항상 on-policy로 남으므로 **continual learning 도구**로서도 유망하다.

7. **reverse KL은 unhackable하다.** 학습된 reward model과 달리 reward가 "teacher 분포와의 거리 그 자체"라 프록시와 목표가 원리적으로 일치한다. reward hacking이 성립하지 않는다.

---

## 참고: 후속 연구

→ [`opd_follow_up_research.md`](./opd_follow_up_research.md) 참조 (2026.07 기준 후속 연구 정리 및 읽기 순서)
