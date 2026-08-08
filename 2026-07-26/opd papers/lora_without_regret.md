# LoRA Without Regret

> **Venue**: Thinking Machines Lab — Connectionism blog (2025.09.29)
> **Authors**: John Schulman and Thinking Machines Lab
> **DOI**: `10.64434/tml.20250929`
> **Link**: https://thinkingmachines.ai/blog/lora/

**한 줄 정의**: LoRA는 "full fine-tuning의 열등한 근사"가 아니다. **두 조건**(① **all layers**, 특히 MLP/MoE layer에 적용 ② **not capacity constrained** — capacity 제약에 걸리지 않을 것)만 지키면 **sample efficiency와 ultimate performance가 FullFT와 동일**하고, FLOPs는 **2/3**만 쓴다. 그리고 그 **"low-regret regime"** 이 **대부분의 post-training 시나리오를 덮는다.**

![LoRA Without Regret cover](../assets/lora-cover.svg)

> 📌 **이 노트의 위치**: 같은 팀의 [`on_policy_distillation.md`](on_policy_distillation.md)가 §5.3 "LoRA 보너스"와 §5.5에서 이 글을 **두 번 인용**한다 — "large dataset·large batch size SFT에서 LoRA가 뒤처진다"는 예측의 출처가 여기다. 그 예측이 왜 성립하고 **언제 성립하지 않는지**가 이 글의 본론이다.

> 📖 원문을 절 순서 그대로 따라 읽는 한국어 정독본은 [`lora_without_regret_ko.md`](lora_without_regret_ko.md)에 별도로 정리했다.

---

## 1. Background

### 왜 PEFT인가 — 규모의 비대칭

- 오늘날의 leading language model은 **1조 개 이상의 parameter**를 **수십 조 token**으로 pre-training한다. base model 성능은 scale과 함께 계속 좋아지며, 이 수조 개는 **글로 쓰인 인간 지식의 모든 패턴을 학습·표현하는 데 실제로 필요하다.**
- 그런데 **post-training은 정반대다.** dataset이 훨씬 작고, 지식의 도메인과 행동의 범위도 좁다.

> **핵심 직관**: **기가비트 또는 메가비트** 분량의 training data에서 온 update를 표현하는 데 **테라비트의 weight**를 쓰는 것은 낭비로 보인다.

이 직관이 **PEFT**(parameter-efficient fine-tuning)를 낳았다 — 훨씬 작은 parameter 집합만 update해 큰 network를 조정한다.

### LoRA의 정의

PEFT의 대표 주자가 **LoRA**(low-rank adaptation)다. 원본 model의 각 weight matrix(가중치 행렬) W를 다음으로 대체한다.

```
W' = W + γ · B·A

  B, A : 합쳐도 W보다 parameter가 훨씬 적은 두 matrix
  γ    : constant scaling factor
```

즉 LoRA는 **fine-tuning이 가하는 update의 low-dimensional representation**(저차원 표현)을 만든다.

### LoRA를 쓰는 세 가지 운영상 이유

성능 이야기 이전에, 실무에서 LoRA를 선호하는 이유가 따로 있다. **이 셋만으로도 2021년 원 논문 이후의 인기 상승이 설명된다.**

| 이유 | 내용 |
|---|---|
| **Multi-tenant serving** | adapter(A·B matrix)만 학습하고 원본 weight는 그대로 두므로, **하나의 inference server가 여러 adapter(= 여러 model 버전)를 메모리에 올려두고 batch로 동시에 sampling**할 수 있다. vLLM·SGLang 같은 현대 inference engine이 이 기능을 구현하고 있다 (Punica, Chen & Ye et al. 2023) |
| **Layout size for training** | FullFT는 원본 weight와 **함께 optimizer state를 저장**해야 하고, 그것도 대개 **더 높은 precision**으로 저장한다. 그래서 FullFT는 **같은 model에서 sampling하는 것보다 대개 an order of magnitude 더 많은 accelerator**를 요구하고, 결과적으로 **layout 자체가 달라진다.** LoRA는 학습하는 weight가 훨씬 적고 메모리도 훨씬 덜 쓰므로 **sampling용 layout보다 아주 조금만 큰 layout에서 학습**할 수 있다. 이것이 학습을 더 접근 가능하게, 그리고 대개 더 효율적으로 만든다 |
| **Ease of loading and transfer** | 저장할 weight가 적으니 adapter의 셋업과 머신 간 전송이 **빠르고 쉽다** |

> **layout 각주 상세**: 학습에는 weight 외에 **모든 weight에 대한 gradient와 optimizer moment**를 저장해야 하고, 이 변수들은 inference용 weight 저장 precision(bfloat16 이하)보다 **높은 precision(float32)** 으로 저장되는 경우가 많다. 이 두 배수가 곱해져서 "an order of magnitude 차이"가 나온다.

### 문헌이 답하지 못한 것

위 세 이유는 **인기**를 설명하지만 **성능**은 설명하지 않는다. 그리고 문헌은 여기서 불명확하다.

| 합의된 것 | 합의되지 않은 것 |
| --- | --- |
| **pre-training을 닮은 setting** — LoRA parameter의 storage limit을 넘어서는 아주 큰 dataset — 에서 LoRA가 underperform한다는 것 ("LoRA Learns Less and Forgets Less", Biderman et al. 2024) | post-training에서 전형적인 dataset size에서는 LoRA가 **essential information을 저장할 충분한 capacity**를 갖는다. 그런데 **이 사실은 sample efficiency와 compute efficiency에 대해 아무것도 보장하지 않는다** |

> **이 글이 던지는 질문**: *"can LoRA match the performance of full fine-tuning, and if so, under which conditions?"* — LoRA가 full fine-tuning의 성능에 필적할 수 있는가, 할 수 있다면 **어떤 조건에서**인가?

그리고 답은: **몇 가지 key detail만 제대로 잡으면, LoRA는 FullFT와 동일한 sample efficiency로 학습하고 동일한 ultimate performance에 도달한다.**

---

## 2. Motivation

### 핵심 통찰 1: 기존 실험들은 "capacity"와 "적용 위치"를 분리하지 않았다

이 글이 이전 LoRA 실험들과 **다르게 한 두 가지**가 방법론의 핵심이다.

| 이전 연구 | 이 글 |
|---|---|
| 특정 dataset·task에 집중 | **training set size와 LoRA parameter 수 사이의 일반적 관계**를 조사 |
| sampling-based eval(정확도 등) | **log loss를 측정.** 같은 일반성 목표에서 나온 선택 — log loss는 training step과 training parameter의 범위에 걸쳐 **clean result와 scaling law**를 준다 |

> **왜 log loss인가**: 정확도 같은 sampling-based eval은 노이즈가 크고 task에 종속된다. log loss는 연속적이고, **learning curve의 모양 자체가 정보**다 — 어느 지점에서 LoRA가 minimum-loss curve에서 **fall off**하는지를 볼 수 있다. §5.1에서 보게 될 "capacity가 소진되는 지점"은 이 측정 방식이 아니면 보이지 않는다.

### 핵심 통찰 2: capacity 부족은 "floor"가 아니라 "training efficiency 저하"로 나타난다

이것이 이 글에서 가장 반직관적인 관찰이다.

> dataset이 LoRA capacity를 초과하면 LoRA가 FullFT에 underperform한다. 그런데 그 방식이 **"더 내려갈 수 없는 뚜렷한 loss floor에 도달하는 것"이 아니다.** 대신 LoRA는 **model capacity와 dataset size의 관계에 의존하는 worse training efficiency**를 낳는다.

즉 capacity가 다 차면 **멈추는 게 아니라 느려진다.** loss는 계속 내려가지만 minimum-loss curve에서 벗어난다. 이 구분이 실무에서 중요한 이유는, "floor에 부딪혔다"면 rank를 올리는 것 외에 방법이 없지만 "느려졌다"면 **학습을 더 돌리는 것으로도 일부 만회**되기 때문이다.

### 핵심 통찰 3: RL은 애초에 capacity가 거의 필요 없다

information-theoretic argument(정보이론 논증)가 실험보다 먼저 있었다. 저자들은 이 결과를 **anticipate하고 확인했다.**

```
supervised learning : O(number of tokens) bits per episode
policy gradient RL  : O(1) bits per episode      ← advantage scalar 하나

episode가 수천 token이면
→ RL은 학습 시 token당 supervised learning보다 약 1000배 적은 정보를 흡수한다
```

그렇다면 **RL에 필요한 capacity는 극히 작아야 하고, rank 1로도 충분해야 한다.** §5.4가 이것을 확인한다.

> 이 논증은 [`on_policy_distillation.md`](on_policy_distillation.md) §1의 "bits/episode" 프레임과 **같은 뿌리**다. 같은 팀의 두 글이 같은 정보량 관점을 서로 다른 결론에 쓴다 — OPD는 "그래서 dense가 필요하다", 이 글은 "그래서 RL에는 capacity가 거의 필요 없다".

---

## 3. Contributions

1. **"low-regret regime"의 특성 규명**: dataset size와 LoRA parameter 수의 관점에서, LoRA가 FullFT와 유사하게 동작하는 regime을 characterize한다. **이 regime이 대부분의 post-training 시나리오를 덮는다.**
2. **두 조건의 확립**: ① LoRA를 **all layers**, 특히 parameter 대부분을 차지하는 **MLP/MoE layer**에 적용할 것 ② **not capacity constrained**일 것.
3. **batch size 취약성 발견**: 일부 시나리오에서 LoRA는 large batch size에 FullFT보다 **less tolerant**다. 그리고 이 penalty는 **rank를 올려도 mitigate되지 않는다** — **product-of-matrices parametrization**(BA, 곱행렬 파라미터화) 자체의 property다.
4. **RL에서의 완전한 동등성**: rank 1에서도 FullFT와 동일한 성능. information-theoretic argument로 예측하고 실험으로 확인.
5. **hyperparameter 지형 정리**: 4개 hyperparameter 중 **2개가 redundant**임을 parametrization invariance로 증명하고, `1/r` prefactor가 왜 optimal LR을 rank에 대해 approximately independent로 만드는지 설명. **optimal LR은 FullFT의 10배**라는 경험 법칙 제시.
6. **compute efficiency 우위**: forward–backward pass당 FLOPs가 FullFT의 **약 2/3**.

---

## 4. Method

### 4.1 실험 설계

성능 차이가 **suboptimal learning rate라는 confound**에서 오는 것이 아님을 보장하는 것이 설계의 핵심이다.

| 항목 | 내용 |
|---|---|
| **rank 범위** | **1 ~ 512** — **three orders of magnitude**를 훑고 FullFT와 비교 |
| **LR 처리** | suboptimal learning rate에서 오는 potential confound를 없애기 위해 **각 experimental condition마다 LR을 sweep**. **constant learning rate schedule**(warmup·cooldown 없음) |
| **model** | **Llama 3** 계열, **Qwen3** 계열 — **mixture of experts(MoE) model 포함** |
| **SL dataset** | **Tulu3**(instruction following), **OpenThoughts3**(reasoning). 두 셋은 **scope·structure·application이 크게 달라** 결과의 generality를 뒷받침한다 |
| **RL task** | mathematical reasoning task, **answer correctness as the reward** |

> **plot 읽는 법**: 아래 그림들에서 rank마다 colored line 하나가 그려지는데, 그 선은 **각 training step에서 모든 learning rate에 대해 pointwise minimum을 취한 것**이다. 즉 "이 rank가 낼 수 있는 최선"의 포락선이다.

### 4.2 LoRA parametrization

```
W' = W + (α / r) · B·A

  r     : LoRA rank
  α     : LoRA scaling factor  — 이 글의 전 실험에서 α = 32 (다른 구현들의 standard practice)
  A, B  : rank r 의 LoRA weight matrix
```

**초기화 (Huggingface `peft` 표준, Hu et al. 제안)**

| 대상 | 초기화 |
|---|---|
| A | scale `1/√d_in` 의 **uniform distribution** |
| B | **zero initialization** |
| LR | A와 B에 **동일한 값** |
| α | **32** |

> 저자들은 **이 hyperparameter를 개선하지 못했다**(unable to improve on these hyperparameters)고 명시한다.

#### MoE layer 처리

MoE 실험에서는 **각 expert마다 separate LoRA를 학습**하고, 각각의 rank를 `total rank / number of active experts`(Qwen3 MoE의 경우 8)로 두었다.

> **왜 이렇게 나누나**: 이 scaling이 **MoE layer에서도 "LoRA parameter / FullFT parameter" 비율을 다른 layer와 동일하게** 유지하기 때문이다. 그래야 MoE와 dense를 같은 축에서 비교할 수 있다.

### 4.3 RL 알고리즘

```
objective = Σ_t  ( p_learner / p_sampler ) · Adv_t        # importance sampling correction
```

- **basic policy gradient + importance sampling correction** (원문 각주: *"Your Efficient RL Framework Secretly Brings You Off-Policy RL Training"*)
- **GRPO-like centering scheme**: 문제당 multiple completion을 sampling하고 **subtract the mean reward per group** (Shao et al. 2024)
- **base model 선택의 이유**: MATH·GSM 실험에 **Llama-3.1-8B**를 썼다. Qwen2.5·Qwen3는 **tech report가 밝힌 대로 math 성능을 끌어올리는 data로 pretrain된 것으로 알려져 있어**, RL 중에만 학습된 것이 무엇인지 측정하기 어렵기 때문이다.

### 4.4 FLOPs 회계 — 왜 2/3인가

**표기**

```
W ∈ R^{N×N}   weight matrix
x ∈ R^N       input vector
y = Wx ∈ R^N  output vector
x̄, ȳ ∈ R^N    backward pass에서 계산되는 x·y에 대한 loss의 gradient
W̄ ∈ R^{N×N}   W에 대한 loss의 gradient
```

**FullFT의 연산**

```
Forward
  y = Wx                    N² multiply–adds

Backward
  x̄ = Wᵀ ȳ                  N² multiply–adds
  W̄ += x ȳᵀ                 N² multiply–adds
                            ─────────────────
  합계                        3N²
```

> forward pass가 N², backward pass가 2N² → 합 **3N²**. **training은 forward-only inference의 3배 FLOPs**를 쓴다.

**LoRA의 연산**

W를 `W + BA`로 대체한다 (`B ∈ R^{N×R}`, `A ∈ R^{R×N}`, `R ≪ N`).

```
Ā, B̄ 만 update하므로 → 세 번째 step(W̄ update, N²)이 much cheaper operation으로 대체된다

A와 B는 N·R matrix → 각각의 full forward-backward가 3NR      → 둘 합쳐 6NR
Wx 와 x̄ 의 forward-backward는 그대로 필요 (FullFT의 첫 두 step)  → 2N²
                                                          ────────────
합계                                                        2N² + 6NR

R ≪ N 이면  →  3N² 의 2/3 보다 slightly more
```

| | FullFT | LoRA |
|---|---|---|
| pass당 multiply-adds | **3N²** | **2N² + 6NR ≈ (2/3) × 3N²** |

> **결과**: training step이 아니라 **FLOPs에 대해** 성능을 plot하면 LoRA가 FullFT보다 **clear advantage**를 보인다.
> **단서**: 이 분석은 **attention에 쓰이는 FLOPs를 omit**한다 — long-context setting에서는 그 몫이 significant할 수 있다.

---

## 5. Experiments

### 5.1 LoRA rank — capacity가 소진되는 지점

Tulu3와 OpenThoughts3 subset에서 **single epoch** 학습. dataset·model size마다 rank와 learning rate를 sweep.

![LoRA training curves for various ranks](../assets/lora-rank-curves.svg)
*Tulu3와 OpenThoughts3에서 rank별 LoRA training curve. FullFT와 high-rank LoRA는 **loss가 step 수의 logarithm에 linear하게 감소**하는 유사한 learning curve를 보인다. low-rank LoRA는 **adapter가 runs out of capacity하면 minimum-loss curve에서 fall off**한다. 아래쪽(1B model)에서는 high-rank LoRA가 한 dataset에서 FullFT보다 낫고 다른 dataset에서는 못하다 — training dynamics나 generalization behavior의 차이로 인한 **random variation**이 있을 수 있다.*

**관찰**

| | 내용 |
|---|---|
| FullFT · high-rank LoRA | loss가 **log(steps)에 linear**하게 감소하는 유사한 learning curve |
| medium/low-rank LoRA | **rank와 correlate되는 어떤 step threshold**에서 minimum-loss curve를 fall off한다 |
| 직관 | **adapter가 runs out of capacity하면 학습이 느려진다.** 그 capacity는 rank가 결정한다 |

#### LR sweep — 각 rank의 optimal LR을 실제로 덮었는가

![Learning rate versus final loss](../assets/lora-lr-vs-loss.svg)
*Tulu3에서 rank별 learning rate 대비 final loss. **minimum loss는 high-rank LoRA와 FullFT가 거의 같다. optimal LR은 LoRA 쪽이 10배 높다.***

| 발견 | 수치 |
|---|---|
| FullFT의 optimal LR | high-rank LoRA보다 **a factor of 10만큼 낮다** (Biderman et al. 2024 Figure S1이 sampling eval로 유사한 10x ratio를 발견) |
| rank 간 optimal LR | **거의 같다** — 다만 약간의 rank dependence는 있다. **rank=1이 high-rank보다 optimal LR이 낮다.** rank 4와 512 사이에서는 **a factor of less than 2**로 변한다 |

### 5.2 Batch size effects — rank로는 못 고치는 gap

OpenThoughts3의 **10,000-example subset**을 사용.

![Batch size effects](../assets/lora-batch-size.svg)
*LoRA vs FullFT의 batch size effect. **왼쪽**: batch size별 learning curve — large batch size에서 LoRA(dashed)와 FullFT(solid) 사이에 **persistent gap**이 보인다. **오른쪽**: batch size의 함수로 그린 final loss — LoRA가 batch size 증가에 **larger penalty**를 지불한다.*

| 관찰 | 내용 |
|---|---|
| large batch size | LoRA와 FullFT learning curve 사이에 **persistent gap**. 시간이 지나도 좁혀지지 않는다 |
| smaller batch size(32) | gap이 **더 작고 시간이 지나며 shrink한다** |
| batch size를 키울수록 | loss gap이 **increasingly diverging** |

**가장 중요한 단서 — rank로 해결되지 않는다**

> large batch에서의 learning gap은 **rank에 의존하지 않는 것으로 보이며, 오히려 a property of LoRA**로 보인다. likely reason은 **product-of-matrices parametrization(BA)이 이 dataset에서 full matrix(W)보다 less favorable optimization dynamics**를 갖기 때문이다.

**그러나 실무적 함의는 완화된다**

> **LoRA와 FullFT 모두 smaller batch size에서 best loss를 달성**하므로, 이 gap이 실무에서 그렇게 중요하지 않을 수 있다.

> 💡 **[`on_policy_distillation.md`](on_policy_distillation.md)와 연결되는 지점.** OPD 글 §5.3이 "large batch size의 large-scale SFT에서 LoRA가 full FT에 뒤처지는 것도 여기서 보인다(LoRA Without Regret의 예측과 일치)"라고 쓴 것의 근거가 이 절이다. 그리고 **OPD가 LoRA gap을 −13% → −6%로 좁히는 이유**도 여기서 나온다 — OPD는 episode당 O(N) bits를 주므로 **smaller batch size로도 학습되고**, LoRA가 불리해지는 조건(large dataset·large batch size) 자체를 피해 간다.

### 5.3 Layers Where LoRA Is Applied — 이 글에서 가장 실무적인 발견

원 논문(Hu et al.)은 **attention matrix에만** LoRA를 적용할 것을 권했고 이후 많은 논문이 그것을 따랐다. 최근 추세는 all layers 적용이다.

**결과: all layers, 특히 MLP(MoE 포함)에 적용할 때 far better result를 얻었다. 사실 MLP-only 위에 attention matrix를 추가해도 no additional benefit이다.**

![Attention-only vs MLP-only LoRA](../assets/lora-layers-dense-moe.svg)
*Attention-only LoRA는 MLP-only LoRA보다 **significantly underperform**하고, LoRA-on-MLP 위에 얹어도 성능을 **further improve하지 못한다**. 이 효과는 dense model(Llama-3.1-8B)과 sparse MoE(Qwen3-30B-A3B-Base) **양쪽에서** 성립한다.*

#### parameter 수 때문이 아니다 — 통제 실험

attention-only가 underperform하는 것이 **단지 parameter가 적어서**가 아님을 parameter 수를 맞춰 확인한다.

| LoRA configuration | Params |
|---|---|
| mlp, rank=256 | 0.49B |
| **attn, rank=256** | **0.25B** |
| all, rank=256 | 0.70B |
| **mlp, rank=128** | **0.24B** |

> *Parameter counts for LoRA on Llama-3.1-8B*

**볼드 처리된 두 행이 이 표의 요점이다.** `attn rank=256`(0.25B)과 `mlp rank=128`(0.24B)은 **parameter 수가 거의 같은데**, attention-only 쪽이 underperform한다. 즉 **어디에 붙이느냐가 얼마나 붙이느냐보다 중요하다.**

#### 두 개의 additional setting에서 재확인

![Learning rate vs final loss or reward by layer](../assets/lora-layers-lr-sweep.svg)
*LoRA를 어느 layer에 적용하는지를 바꿔가며 측정한 learning rate 대비 final loss 또는 reward.*

1. OpenThoughts3 small subset에서의 supervised learning (rank=256)
2. **MATH dataset에서의 reinforcement learning**

두 setting 모두에서 **attention-only LoRA가 MLP-only LoRA보다 underperform한다** (MLP-only는 MLP+attention과 유사한 성능).

> **선행 연구와의 관계**: QLoRA 논문도 유사하게 attention-only가 MLP나 MLP+attention보다 나쁘다는 것을 발견했다. 다만 QLoRA는 `MLP+attention > MLP > attention` 순으로 봤고, **이 글은 앞의 둘이 roughly equal**하다고 본다. Biderman et al. (2024)도 attention-only가 MLP-only 위에 no additional benefit을 준다는 유사한 결과를 얻었다.

#### 왜 all layers여야 하는가 — eNTK 설명

작은 양의 fine-tuning에서 일어나는 일의 approximation으로 **empirical neural tangent kernel(eNTK)** 을 보는 관점 (Malladi et al. 2022).

```
eNTK는 gradient의 dot product에 기반한다:

  g_i  = ∂/∂θ  log p(token_i | prefix_i)
  K(i, j) = g_i · g_j
```

**결과적으로 parameter가 가장 많은 layer가 kernel에 the most influence를 준다.** 그리고 Malladi et al.은 **all layers를 학습할 때 LoRA의 eNTK가 full fine-tuning의 eNTK와 approximately the same**이라는 것을 지적한다. 따라서

```
LoRA training  ≈  eNTK(LoRA)  ≈  eNTK(FullFT)  ≈  FullFT
```

> **결정적 단서**: approximation `eNTK(LoRA) ≈ eNTK(FullFT)`는 **dot product를 구성하는 parameter 대부분을 담은 layer에 LoRA를 적용할 때만 성립한다.** attention에만 붙이면 이 사슬이 첫 고리에서 끊어진다. **그래서 tiny-data regime에서도** attention-only는 slower learning을 보인다.

### 5.4 Reinforcement learning — rank 1로 충분하다

**핵심 발견: policy gradient algorithm을 돌릴 때 LoRA는 rank가 1만큼 낮아도 FullFT의 learning performance에 fully match한다.**

![LR vs final reward on GSM and MATH](../assets/lora-rl-gsm-math.svg)
*grade school math(GSM, 왼쪽)와 MATH(오른쪽)에서 RL을 돌렸을 때의 learning rate 대비 final reward(accuracy).*

> LoRA는 **wider range of performant learning rates**를 보이고, FullFT(black line)와 **same peak performance**에 도달한다 — 적어도 RL의 noisiness가 허용하는 precision limit 안에서.

#### information-theoretic argument — 숫자로

§2 통찰 3의 논증을 실험 수치로 구체화한다.

```
MATH 실험:  ~10,000 problems × 32 samples per problem

각 completion이 a single bit of information를 준다고 가정하면
  → 전체 training process가 흡수해야 하는 정보량 = 320,000 bits

Llama-3.1-8B의 rank-1 LoRA parameter 수 = 3M
  (모든 weight matrix에 대해 rank·d_in (matrix A) + rank·d_out (matrix B) 를 합산)

3M  vs  320,000 bits  →  almost 10 times
```

> **rank-1에서도 LoRA는 학습 중 제공되는 모든 정보를 흡수하기에 more than enough capacity를 갖는다.**

**또 하나의 point of comparison — DeepSeek-R1-Zero**

```
DeepSeek-R1-Zero 학습 규모:
  10,400 steps × 32 unique questions × 16 samples = 5.3M episodes
                                                   → 5.3M bits

5.3M bits < low-rank LoRA의 parameter 수
  → 저자들의 예측: "we predict that the results can be replicated with LoRA"
```

#### 대규모 재확인 — DeepMath

MATH보다 much larger하고 일반적으로 더 어려운 문제를 담은 **DeepMath** dataset에서 **Qwen3-8b-base**로 larger-scale 실험. 실험 속도를 위해 training·evaluation sample을 **8192 tokens**로 제한했다 (backtracking과 reasoning은 가능하지만, longer chain-of-thought 대비 성능은 제한된다).

![DeepMath experiments](../assets/lora-rl-deepmath.svg)
*Qwen3-8b-base로 DeepMath에서 수행한 실험. **왼쪽**: rank별·FullFT의 learning curve — 각 setting에서 final performance가 가장 높은 best learning rate를 표시. **오른쪽**: learning rate 대비 final performance. 이전 math 실험과 마찬가지로 **LoRA가 near-optimal learning rate의 wider peak를 보인다.***

![DeepMath AIME and CoT length](../assets/lora-rl-deepmath-aime-cot.svg)
*DeepMath 실험의 additional plot. **왼쪽**: training set보다 어려운 AIME test set의 benchmark score. **오른쪽**: training step에 따른 chain-of-thought(CoT) length — **a sign of learning to reason**으로 볼 수 있다.*

| 관찰 | 내용 |
|---|---|
| 학습 진행 | 각 setting에서 optimal learning rate를 고르면, **서로 다른 size의 LoRA와 full fine-tuning이 almost identical하게** 진행된다 |
| held-out generalization | **AIME 2024·2025**의 held-out problem에서도 유사한 결과 |
| **qualitative behavior** | LoRA와 full fine-tuning run **양쪽 모두** backtracking·self-verification·in-context exploration 같은 **advanced reasoning behavior를 발달**시키며, 이는 **CoT length의 증가**로 관측된다 |

> 마지막 항목이 중요하다. LoRA가 단지 점수를 맞추는 게 아니라 **같은 종류의 능력을 획득**한다는 증거다.

### 5.5 Setting LoRA hyperparameters — 4개가 아니라 2개다

LoRA 채택의 barrier 하나는 FullFT와 다른 optimal hyperparameter를 골라야 한다는 점이다. **이 문제가 보기만큼 daunting하지 않다**는 것이 이 절의 요지다.

#### (a) Optimal learning rate and rank — `1/r` prefactor가 rank independence를 만든다

**`1/r` scaling factor가 optimal learning rate를 rank에 대해 approximately independent로 만든다.** 사실 a stronger condition이 성립한다 — **학습 초기에는 rank와 무관하게 learning curve가 exactly the same이다.**

> 이 효과가 워낙 striking해서, 저자들은 **서로 다른 rank의 learning curve가 너무 가까워 rank parameter가 무시되는 bug가 있는 줄 알고 걱정했다**고 적고 있다.

![Rank independence early in training](../assets/lora-rank-independence.svg)
*동일한 learning rate에서 rank별 learning curve의 **학습 초기** 차이. **왼쪽**은 learning curve, **오른쪽**은 rank 16과 256의 차이 — 시간이 지나며 커진다. 특이하게도 **처음 몇 step에서는 (아주 작지만) negative**여서 그 부분은 plot에서 빠져 있다.*

**왜 그런가 — first training update의 expected update 논증**

```
LoRA product BA 를 r 개의 rank-1 outer product의 합으로 본다:

  BA = Σ_{i=1..r} b_i a_iᵀ = Σ_{i=1..r} Δ_i        (Δ_i := b_i a_iᵀ)

여기서
  ∂Loss/∂Δ_i 는 모든 i 에 대해 같다
  그러나 gradient ∂Loss/∂b_i 와 ∂Loss/∂a_i 는 initialization에 의존한다
    (예: ∂Loss/∂b_i 는 a_i 에 의존)

a_i 와 b_i 의 initialization이 rank에 의존하지 않으므로
  → E[Δ_i] 는 모든 i 에 대해 같고 rank에 의존하지 않는다

first step of training에서 각 항의 expected update는 equal하고 independent of the rank다.
따라서 (1/r)·Σ_{i=1..r} Δ_i 는 same expectation을 갖는 r 개 항의 sample average일 뿐이고,
그 average의 expectation — 즉 adapter (1/r)BA 의 change — 은 rank에 의존하지 않는다.
```

> **단서**: short training regime에서는 optimal LR도 rank independent다. 그러나 §5.1 Figure 2에서 봤듯 **longer-training regime에서는 optimal LR에 some rank-dependence가 생긴다.**

#### (b) Parametrization invariances — 4개 중 2개는 redundant다

LoRA에 적용 가능한 hyperparameter는 potentially **4개**다.

| # | hyperparameter | 의미 |
|---|---|---|
| 1 | **α** | `α / r` 에 등장하는 scale factor |
| 2 | **LR_A** | down-projection matrix A의 learning rate |
| 3 | **LR_B** | up-projection matrix B의 learning rate |
| 4 | **init_A** | matrix A의 initialization scale (random initialization의 경우 A 초기 원소들의 standard deviation). **B는 zero로 initialize되므로 init_B는 정의할 필요가 없다** |

4개를 모두 튜닝해야 한다면 overwhelming하다. **그러나 invariances in the training dynamics 때문에 이 중 2개는 redundant고, learning behavior는 2개로 결정된다.**

```
Adam 으로 학습하고 ε = 0 일 때, optimization process는 다음 two-parameter transformation에 invariant다.

p, q > 0 에 대해:

  α      →  (1 / (p·q)) · α
  init_A →  p · init_A
  LR_A   →  p · LR_A
  LR_B   →  q · LR_B
```

> ε > 0 으로도 확장할 수 있다. gradient가 `1/q` 만큼 scale되므로 그만큼 보정해 주면 된다.

**4개 중 two degrees of freedom이 learning process에 영향을 주지 않으므로 남는 것은 2D parameter space다.** straightforward interpretation을 갖는 basis 하나를 고르면:

| basis 축 | 의미 |
|---|---|
| **α · init_A · LR_B** | **scale of initial updates**, 즉 **learning curve의 initial slope**를 결정한다. B가 zero로 initialize되므로 **LR_A와 A에 대한 initial update는 irrelevant하다** |
| **init_A / LR_A** | Adam이 매 step A의 원소를 approximately `LR_A` 만큼 update하므로, 이 **timescale parameter**는 **A가 initial state에서 significantly 벗어나는 데 걸리는 step 수**를 결정한다 |

**기존 제안들을 이 basis로 다시 읽으면**

| 제안 | 이 basis에서의 해석 |
|---|---|
| **LoRA+** (Hayou et al. 2024) — A와 B에 different LR, B에 higher rate | **LR_B를 올리는 것 = `init_A/LR_A`를 올리는 것** → A가 longer timescale로 변한다 |
| **Unsloth's LoRA Hyperparameter Guide** — high-rank LoRA에 higher α (예: `1/r` scaling 회피) | 이것도 **`init_A/LR_A`를 올리는 것과 equivalent**하다. α를 올리면 same update size를 얻기 위해 LR_A와 LR_B를 낮춰야 하고, 그러면 **LR_A가 init_A에 비해 작아진다** |

> **두 제안이 사실은 같은 축을 움직이고 있었다**는 것이 이 정리의 값이다.

#### (c) Optimal learning rates for LoRA vs. FullFT — 10x

> **실험 결과, LoRA의 optimal LR은 같은 application에서 FullFT의 것보다 consistently 10x 높다 — supervised learning과 reinforcement learning 양쪽 모두에서.** 이는 performance(loss 또는 reward)를 learning rate에 대해 그린 **every U-shaped plot에서 나타난다.** 덕분에 FullFT의 hyperparameter를 LoRA로 옮기기가 더 straightforward해진다.

adequate theoretical explanation은 아직 없다. optimal LoRA LR이 rank에 invariant라는 사실과 full-rank LoRA가 FullFT와 directly comparable하다는 사실에서 유도를 시도하면 **`hidden size / (2·α)`** 라는 LR ratio가 나오는데, 이는 **base model과 무관하게 10으로 고정**되는 empirical result와 맞지 않는다.

**empirical analysis — 14개 model sweep**

Llama와 Qwen **14개 model**에 대해 Tulu3에서 LoRA와 FullFT 양쪽의 LR sweep을 수행하고, 다음 functional form을 fit했다.

```
LR = M_LoRA · ( 2000 / hidden_size ) ^ (model_pow + LoRA_pow)

  M_LoRA      : LoRA를 쓸 때 적용되는 multiplier (FullFT면 1)
  model_pow   : model source(Llama / Qwen)마다 따로 계산되는 exponent adjustment
  LoRA_pow    : LoRA에 대한 additional exponent adjustment
  hidden_size : model residual stream의 dimension
```

- predicted learning rate의 score는 sweep data에 대한 **linear interpolation으로 loss를 예측**하고, **14개 문제에 대해 predicted loss를 합산**해 매겼다.
- **결과**: LoRA의 multiplier **9.8**, Qwen3와 Llama에 대해 서로 다른 hidden_size dependence. 그러나 **LoRA LR은 FullFT LR과 hidden_size에 대해 same dependence**를 가졌다 — 즉 optimization이 찾은 값은 **`LoRA_pow = 0`**.

> 실무 함의: **FullFT에서 튜닝한 LR을 그대로 10x 해서 LoRA에 쓰면 된다.** hidden_size에 따른 별도 보정은 필요 없다.

#### (d) Learning rates in short and long runs — implicit schedule

LoRA의 typical initialization은 **effective learning rate 변화의 implicit schedule**을 만든다.

```
학습 시작:  B = 0 으로 initialize
            → B가 very small인 동안, A의 change는 원본 network weight에 더해지는
              adapter BA 에 negligible effect를 준다

학습 진행:  B가 커진다
            → A에 대한 update가 network output에 bigger impact를 주기 시작
            → B가 A의 scale에 근접하면서 effective learning rate가 학습 과정에서 증가
```

실제로 Tulu3와 OpenThoughts dataset의 full training run이 끝날 무렵, **B matrix가 A matrix보다 larger spectral norm**을 갖게 되었다.

**함의: short training run에서는 optimal LR을 더 높게 잡아야 한다.**

| training run 길이 | FullFT 대비 optimal multiplier |
|---|---|
| **short run** (anecdotal evidence로 **~100 steps 이하**) | **약 15x** |
| long run | 앞서 말한 **10x**로 수렴 |

---

## 6. Discussion

### 6.1 두 조건 — 그리고 그 사이의 논리

LoRA가 full fine-tuning과 유사하게 동작하는 **두 조건**을 다시 정리하면:

| # | 조건 |
|---|---|
| **①** | LoRA가 network의 **all layers**, 특히 parameter 대부분을 담은 **MLP/MoE layer**에 적용될 것 |
| **②** | **not capacity constrained**일 것 — number of trainable parameters가 학습해야 할 정보량을 초과할 것. 이 정보량은 **dataset size로 추정**할 수 있다 |

> **두 조건이 시간축에서 어떻게 맞물리나**:
> **①이 만족되면 학습의 맨 처음에 FullFT와 similar learning dynamics를 얻는다.**
> **그리고 ②에 따라, capacity limit에 도달하기 시작할 때까지 LoRA는 계속 FullFT처럼 보인다.**

즉 ①은 **출발점**을 맞추고 ②는 **얼마나 오래 유지되는지**를 정한다. 하나라도 어긋나면 gap이 생긴다.

### 6.2 supervised learning과 RL은 얼마나 capacity가 필요한가

#### 2 bits per parameter

선행 연구(Allen-Zhu & Li 2024, *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws*)는 **neural network가 parameter당 2 bits를 저장할 수 있다**는 것을 보였다.

> **단서**: 이 결과는 **long-training limit에서 흡수되는 maximum amount of information**에 관한 것이지, compute efficiency나 rate of learning에 관한 것이 아니다.

#### 실제 문제의 정보량 추정은 어렵다

2-bits-per-parameter 결과는 **precise amount of information을 담도록 cleverly constructed된 synthetic dataset**에 의존했다. realistic learning problem의 information content를 추정하는 것은 그만큼 straightforward하지 않다.

```
classic observation:
  log-loss를 최소화할 때, first epoch of training 동안 측정된 total log-loss가
  dataset의 description length 를 측정한다
  = dataset을 memorize하는 데 필요한 bit 수의 upper bound

LLM dataset은 대개 token당 around 1 bit (0.69 nats) — dataset과 model size에 따라 다름
```

> **단서**: 이 추정은 **dataset을 perfectly memorize하는 데 필요한 capacity**를 재는 것이라, test data의 log-loss를 줄이는 **"generalizable" learning에 실제로 필요한 capacity를 overestimate**한다. supervised learning의 capacity requirement와 그것이 number of trainable parameters와 어떻게 상호작용하는지를 측정하는 것은 **an open question for future work**다.

#### RL의 1 bit per episode — 그리고 그 한계

> RL에 대해서는 **policy gradient algorithm이 roughly 1 bit of information per episode를 학습**한다고 주장했다. episode 끝에 single reward value 하나뿐이기 때문이다.

**중요한 자기 제한**: 이것은 **a fundamental property of RL이 아니다.** 다른 알고리즘은 각 episode에서 훨씬 많이 배울 수 있다. 예컨대 **model-based RL**은 observation을 예측하도록 agent를 학습시켜 world model을 만들며, **episode당 더 많은 정보를 extract할 잠재력**이 있다. **1-bit-per-episode 주장은 policy gradient algorithm에만 narrowly 적용될 수 있다.**

#### information theory로 날을 세우면

```
episode(trajectory τ + final reward)를 unknown reward function R 에 대한 정보를 주는
message(= noisy channel)로 본다.
current policy와 training history에 condition을 걸고, policy gradient estimator와 R 사이의
mutual information(상호정보량)을 본다.

REINFORCE update:  G = S · Adv,     S = ∇ log p_θ(τ)

  S 는 history가 주어지면 R 과 independent다
  → R 에 의존하는 유일한 component는 scalar advantage 뿐이다

data processing inequality 에 의해:

  I(G ; R | history)  ≤  I((S, Adv) ; R | history)
                      =  I(Adv ; R | S, history)
                      ≤  H(Adv)

advantage 를 B 개의 bin으로 quantize하면  H(Adv) ≲ log(B)

→ episode당 gleaned되는 useful information의 bit 수는 O(1) 이고, model size와 무관하다
```

이 bit들은 **우리가 discrete set of reward functions(equivalently, optimal-policy classes) 중 어디에 있는지**를 알려준다. 이 mutual information 분석은 optimization algorithm의 theoretical analysis에서 쓰이는 것과 같은 방식이다 (Raginsky & Rakhlin 2009).

> **단서**: 이 추정은 training이 흡수하는 정보량의 **upper bound**다. 실제로 학습되는 양은 policy initialization과 기타 details에 의존한다. 예컨대 **reward를 전혀 받지 못하는 policy로 initialize하면 advantage의 entropy는 log(B)가 아니라 0이고, 아무것도 학습하지 못한다.**

### 6.3 Open questions

| # | 질문 |
|---|---|
| 1 | **LoRA 성능 예측의 sharpening** — FullFT와 match하는 precise condition. equal performance regime을 roughly characterize하고 required capacity를 token·episode 단위로 추정할 수는 있지만, **아직 accurate forecast는 못 한다** |
| 2 | **theoretical understanding의 부족** — LoRA learning rate와 training dynamics에 대한 이론이 limited하다. **LoRA와 FullFT learning rate의 ratio(10x)를 설명하는 fuller theory**가 가치 있을 것 |
| 3 | **LoRA variants** — PiSSA(Meng, Wang & Zhang 2024) 같은 variant가 이 글의 methodology로 측정하면 어떻게 나오는가 |
| 4 | **MoE layer 적용 방식** — LoRA를 MoE layer에 적용하는 various option이 있다. 각각의 성능과, 대형 MoE에 중요한 **tensor parallelism·expert parallelism과의 compatibility**에 대한 investigation이 필요 |

### 6.4 Closing thoughts

- Thinking Machines는 fine-tuning의 힘을 **널리 접근 가능하고 특정 요구에 쉽게 customizable하게** 만드는 것을 목표로 LoRA에 관심을 둔다.
- 실용적 쓰임 외에도, LoRA 연구는 **model capacity, dataset complexity, sample efficiency**에 대한 더 깊은 탐구로 이어졌다. **learning speed와 performance가 capacity에 어떻게 의존하는지를 보는 것이 머신러닝의 근본 질문을 연구하는 렌즈**가 된다.

---

## 7. Key Takeaways

1. **LoRA는 조건부로 FullFT와 동등하다 — 그리고 그 조건이 대부분의 post-training을 덮는다.** ① all layers(특히 MLP/MoE layer)에 적용 ② not capacity constrained. 이 둘만 지키면 **sample efficiency와 ultimate performance가 같다.** 저자들은 이 regime을 **"low-regret regime"** 이라 부른다.

2. **어디에 붙이느냐가 얼마나 붙이느냐보다 중요하다.** `attn rank=256`(0.25B)이 `mlp rank=128`(0.24B)에게 진다 — **parameter 수가 거의 같은데도.** 원 LoRA 논문의 "attention matrix에만" 권고는 **틀렸다**. eNTK 관점이 이유를 설명한다: `eNTK(LoRA) ≈ eNTK(FullFT)`는 **parameter 대부분을 담은 layer에 적용할 때만** 성립한다.

3. **RL은 rank 1로 충분하다 — 이론이 먼저 anticipate했다.** policy gradient는 episode당 O(1) bits만 준다. MATH 실험 전체가 흡수해야 할 정보는 **320,000 bits**인데 rank-1 LoRA는 이미 **3M parameter**로 almost 10 times 여유다. DeepSeek-R1-Zero의 5.3M episodes(= 5.3M bits)조차 low-rank LoRA parameter 수보다 적어, **LoRA로 replicate 가능할 것**이라고 예측한다.

4. **capacity가 다 차면 멈추는 게 아니라 느려진다.** loss가 **distinct floor에 닿는 것이 아니라**, model capacity와 dataset size의 관계에 의존하는 **worse training efficiency**로 나타난다. low-rank curve가 minimum-loss curve에서 **fall off**하는 모양이 그것이다.

5. **batch size 취약성은 rank로 고칠 수 없다.** large batch size에서의 gap은 **rank에 의존하지 않으며 a property of LoRA**로 보인다 — product-of-matrices parametrization(BA)이 full matrix(W)보다 less favorable optimization dynamics를 갖기 때문. 다만 **양쪽 모두 smaller batch size에서 best loss를 내므로** 실무 영향은 제한적일 수 있다.

6. **hyperparameter는 4개가 아니라 2개다.** `(α, LR_A, LR_B, init_A)`는 two-parameter transformation에 invariant라 실제 degrees of freedom은 2다. interpretable한 basis는 **`α·init_A·LR_B`(initial slope of the learning curve)** 와 **`init_A/LR_A`(A의 change timescale)**. 이 basis로 보면 **LoRA+와 Unsloth의 권고가 사실은 같은 축**을 움직이고 있다.

7. **`1/r` prefactor 덕분에 optimal LR이 rank에 approximately independent다.** 학습 초기에는 **rank와 무관하게 learning curve가 exactly the same이다** — 저자들이 bug를 의심할 정도로. first step의 expected update `E[Δ_i]`가 rank에 의존하지 않기 때문이다. rank 4~512 사이에서 optimal LR은 **a factor of less than 2**로 변한다.

8. **FullFT LR에 10을 곱하면 된다.** supervised·RL 양쪽에서, **every U-shaped plot에서** consistently 나타난다. 14개 Llama·Qwen model sweep에서 fit된 multiplier는 **9.8**이고, `LoRA_pow = 0` — **hidden_size에 따른 별도 보정이 필요 없다.** 단, **~100 steps 이하의 short run에서는 15x**가 낫다 (B가 0에서 시작해 effective learning rate가 학습 중 증가하는 implicit schedule 때문).

9. **compute도 싸다 — pass당 FLOPs의 2/3.** FullFT는 `3N²`(forward N² + backward 2N²), LoRA는 `2N² + 6NR ≈ (2/3)·3N²`. **training step이 아니라 FLOPs에 대해 plot하면 LoRA가 clear advantage**다. (attention FLOPs는 이 분석에서 omit — long-context setting에서는 significant할 수 있다)

10. **LoRA와 FullFT는 같은 종류의 능력을 얻는다.** DeepMath 실험에서 양쪽 모두 **backtracking·self-verification·in-context exploration** 같은 advanced reasoning behavior를 발달시키고, **CoT length 증가**로 관측된다. 점수만 같은 것이 아니다.

---

## 8. `on_policy_distillation.md`와의 연결

같은 팀·같은 블로그의 두 글이고, OPD 글이 이 글을 **두 번 인용**한다. 두 글을 같이 읽을 때 맞물리는 지점을 정리한다.

| OPD 글의 서술 | 이 글에서의 근거 |
|---|---|
| §5.3 *"large batch size의 large-scale SFT에서 LoRA가 full FT에 뒤처지는 것도 여기서 보인다(**LoRA Without Regret의 예측과 일치**)"* | **§5.2 batch size effects** + **§5.1 capacity 소진**. 두 조건(large dataset = capacity 초과, large batch size = parametrization 취약성)이 **동시에** 걸리는 지점이다 |
| §5.3 LoRA 보너스 — SFT 직후 **−13%** → OPD 이후 **−6%** | OPD가 **smaller batch size로도 학습되므로**(episode당 O(N) bits) §5.2의 large batch size penalty를 **회피**한다. 즉 LoRA가 불리해지는 조건 자체를 밟지 않는다 |
| §5.5 (a) RL 비교에서 **LoRA rank 128** 사용 — *"LoRA Without Regret과 동일한 절차"* | **§5.4**. RL에서는 rank 1로도 충분하다는 것이 이 글의 결론이므로, rank 128은 넉넉한 선택이다 |
| §5.4 personalization에서 **LoRA(rank 32~256)로도 catastrophic forgetting을 못 막았다** | **이 글의 범위 밖**이다. 이 글은 "**성능이 같은가**"를 묻지 "**덜 잊는가**"를 묻지 않는다. 두 결과는 서로 다른 질문에 대한 답이며, OPD 글도 §5.3에서 이를 명시적으로 경고한다 |

**두 글이 공유하는 정보량 프레임 — 그리고 서로 다른 결론**

```
공통 전제:  policy gradient RL 은 O(1) bits per episode
            supervised / dense signal 은 O(N) bits per episode

OPD 글의 결론:   그러니 O(1) 은 부족하다 → dense signal(per-token KL)로 가자
이 글의 결론:    그러니 O(1) 이면 capacity도 O(1) 이면 된다 → RL 에는 rank 1 로 충분하다
```

> 같은 관찰에서 한 글은 **"signal을 늘리자"**, 다른 글은 **"parameter를 줄이자"** 로 간다. 모순이 아니라 **같은 비대칭의 양쪽 끝**이다.

**아직 아무도 안 한 조합**: OPD는 O(N) bits per episode를 준다. 그렇다면 **OPD에 필요한 LoRA capacity는 RL보다 크고 SFT보다 작아야** 하는데, 이 글의 methodology(rank sweep × log loss)로 **OPD의 capacity requirement를 측정한 실험은 없다.** OPD 글의 "−13% → −6%"는 rank 32 한 점의 관측일 뿐이다.

---

## 9. 인용

```bibtex
@article{schulman2025lora,
  author  = {John Schulman and Thinking Machines Lab},
  title   = {LoRA Without Regret},
  journal = {Thinking Machines Lab: Connectionism},
  year    = {2025},
  note    = {https://thinkingmachines.ai/blog/lora/},
  doi     = {10.64434/tml.20250929},
}
```

---

[← On-Policy Distillation (같은 블로그)](on_policy_distillation.md) · [원문 구조 한국어 정독본](lora_without_regret_ko.md) · [OPD 후속 연구 정리](opd_follow_up_research.md)
