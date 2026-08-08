# LoRA Without Regret — 원문 구조 한국어 정독본

> **원문**: [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) — John Schulman, in collaboration with others at Thinking Machines · Thinking Machines Lab: Connectionism, 2025.09.29 · DOI `10.64434/tml.20250929`

![LoRA Without Regret cover](../assets/lora-cover.svg)

---

## 들어가며


**PEFT**(parameter efficient fine-tuning) — 훨씬 작은 parameter 집합만 update해서 큰 network를 조정하는 방식이다.

**대표적인 PEFT 기법이 LoRA**(low-rank adaptation)다. LoRA는 원본 model의 각 weight matrix `W`를 다음으로 대체한다.

```
W' = W + γ · B·A

  B, A : 합쳐도 W보다 parameter 수가 훨씬 적은 두 matrix
  γ    : constant scaling factor
```

효과적으로 LoRA는 **fine-tuning이 가하는 update의 low-dimensional representation**을 만든다.

### full fine-tuning(이하 FullFT) 대신 LoRA를 쓰는 운영상의 이유

LoRA는 post-training의 비용과 속도에서 이점이 있을 수 있다. 그리고 그와 별개로, FullFT보다 선호할 만한 **운영상의 이유가 셋** 있다. 셋 모두 **"원본 weight를 건드리지 않는다"** 는 하나의 성질에서 나온다.

| #   | 이유                               | 내용                                                                                                                                                                                                                                                                                                   |
| --- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Multi-tenant serving**         | adapter(A·B matrix)만 학습하고 원본 weight는 그대로 두므로, **하나의 inference server가 여러 adapter(= 서로 다른 model version)를 메모리에 올려두고 batch 방식으로 동시에 sampling**할 수 있다<br>📎 *Punica: Multi-Tenant LoRA Serving* (Chen, Ye et al., 2023) — vLLM·SGLang 같은 현대 inference engine이 이 기능을 구현하고 있다                             |
| 2   | **Layout size for training**     | FullFT는 원본 weight와 **함께 optimizer state를 저장**해야 하고 그것도 종종 더 높은 precision으로 저장한다 → **같은 model에서 sampling할 때보다 an order of magnitude 더 많은 accelerator**를 요구하고, 따라서 **layout 자체가 달라진다**<br>LoRA는 학습 weight 수도 메모리도 훨씬 적으므로 **sampling용 layout보다 아주 조금만 큰 layout에서 학습**할 수 있다 → 학습이 더 접근 가능하고 종종 더 효율적이다 |
| 3   | **Ease of loading and transfer** | 저장할 weight가 적으니 adapter의 셋업과 머신 간 전송이 빠르고 쉽다                                                                                                                                                                                                                                                         |


![Multi-tenant serving](../assets/lora-multi-tenant-serving.svg)


### 성능에 대해 불명확하다

- **합의된 부분**: **pre-training을 닮은 setting** — 즉 LoRA parameter의 storage limit을 초과하는 아주 큰 dataset — 에서는 LoRA가 underperform한다. (📎 *LoRA Learns Less and Forgets Less*, Biderman et al., 2024)
- **합의되지 않은 부분**: post-training에서 전형적인 dataset size라면 LoRA는 **essential information을 저장할 충분한 capacity**를 갖는다. **그러나 이 사실은 sample efficiency와 compute efficiency에 대해 아무것도 보장하지 않는다.**

 질문은 이것이다.

> "can LoRA match the performance of full fine-tuning, and if so, under which conditions?"

그리고 답은 이렇다. **몇 가지 key detail만 제대로 잡으면, LoRA는 FullFT와 동일한 sample efficiency로 학습하고 동일한 ultimate performance에 도달한다.**

### LoRA의 주요 hyperparameter와 calibration

여기서 말하는 "몇 가지 key detail"이 무엇인지 먼저 한 표로 정리한다. 각 항목의 근거는 뒤 절에서 다룬다.

| hyperparameter           | 무엇인가                                | calibration 방법                                                                                                                                                                                                                                                                 | 근거                                                                                                                              |
| ------------------------ | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| **rank `r`**             | adapter의 **capacity**를 결정           | **"학습할 정보량 < trainable param × 2 bits"** 를 만족하는 최소 rank. dataset 정보량은 `데이터셋 token 수 × 약 1 bit`로 상한 추정. **진단이 더 실용적이다** — learning curve가 FullFT의 **log-linear decay에서 이탈하기 시작하면 capacity 부족 신호**이므로 rank를 올린다. **RL은 episode당 약 1 bit**라 요구 capacity가 극히 낮아 **small rank로 충분** | [LoRA rank](#lora-rank) · [How much capacity…](#how-much-capacity-is-needed-by-supervised-and-reinforcement-learning)           |
| **`α` (scaling factor)** | `W' = W + (α/r)·B·A` 의 scale factor | **32 고정. 튜닝 대상이 아니다.** `1/r` prefactor가 **optimal LR을 rank에 무관하게** 만들어 주므로 rank를 바꿔도 LR을 다시 잡을 필요가 없다. rank를 올릴 때 α를 함께 올리는 관행(Unsloth 등)은 사실상 **`init_A/LR_A` 축을 움직이는 것과 등가**                                                                                                 | [Optimal learning rate and rank](#optimal-learning-rate-and-rank) · [Parametrization invariances](#parametrization-invariances) |
| **batch size**           | 한 step에 보는 example 수                | **작게 — 32 부근.** LoRA는 large batch에 FullFT보다 less tolerant이고, 이 penalty는 **rank를 올려도 사라지지 않는다**(product-of-matrices parametrization 고유의 성질). 다만 **FullFT도 작은 batch에서 best loss**이므로 실무상 제약은 크지 않다                                                                               | [Batch size effects](#batch-size-effects)                                                                                       |


---

## What matters for LoRA

원문은 LoRA가 FullFT의 efficiency에 match하는 조건을 규명하기 위해 일련의 supervised fine-tuning  실험을 수행했다. 이를 위해 **이전 LoRA 실험들과 두 가지를 다르게** 했다.

1. 특정 dataset·task에 집중하는 대신, **training set size와 LoRA parameter 수 사이의 일반적 관계**를 조사했다.
2. supervised learning에서 sampling-based eval 대신 **log loss**를 측정했다. 같은 generality 목표에서 나온 선택이다.
   > "Log loss measurement gives clean results and scaling laws over ranges of training steps and training parameters."

### 발견 요약 (원문의 5개 bullet)

| #   | 발견                                                                                                                                                                                                                                                                                                   |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | small-to-medium-sized instruction-tuning·reasoning dataset에서의 supervised fine-tuning에서는 **LoRA가 full fine-tuning과 동일하게 동작한다**                                                                                                                                                                        |
| 2   | **LoRA capacity를 초과하는 dataset**에서는 LoRA가 FullFT에 underperform한다. 다만 그 방식이 **loss가 더 못 내려가는 distinct floor에 도달하는 것이 아니라**, model capacity와 dataset size의 관계에 의존하는 **worse training efficiency**로 나타난다                                                                                                 |
| 3   | 일부 시나리오에서 LoRA는 **large batch size에 FullFT보다 less tolerant**다 — 어느 지점을 넘어 batch size가 커지면 loss에서 더 큰 penalty를 지불한다. **이 penalty는 LoRA rank를 올려도 mitigate되지 않는다.** 이는 **product-of-matrices parametrization의 property**이며, 이 parametrization은 원본 weight matrix를 직접 최적화하는 것과 다른 training dynamics를 갖는다 |


---

## Methods and results

실험 setup의 주요 사항:

| 항목                     | 내용                                                                                                                                                                                      |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **rank**               | **1 ~ 512**, 즉 **three orders of magnitude**에 걸쳐 변화시키고 full fine-tuning과 비교                                                                                                             |
| **learning rate**      | suboptimal LR에서 오는 **potential confound를 제거**하기 위해 **각 experimental condition마다 LR을 sweep**. **constant learning rate schedule** 사용(warmup·cooldown 없음)                                 |
| **model**              | **Llama 3** 계열 (📎 Dubey et al., 2024), **Qwen3** 계열 (📎 Qwen Team, 2025) — **mixture of experts(MoE) model 포함**                                                                        |
| **supervised dataset** | **Tulu3** (📎 Ivison et al., 2024, instruction following)와 **OpenThoughts3** (📎 Guha et al., 2025, reasoning). 두 dataset은 **scope·structure·application이 크게 달라** 결과의 generality를 뒷받침한다 |
|                        |                                                                                                                                                                                         |

#### supervised dataset — 규모·포맷·용도

|                 | **Tulu3**                                                  | **OpenThoughts3**                                                                           |
| --------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **성격**          | instruction following (general chat·math·code·safety 혼합)   | reasoning (long chain-of-thought)                                                           |
| **규모**          | SFT mixture 기준 **약 940k example**                          | **약 1.2M만 example** (math 85만 / code 25만 / science 10만)                                     |
| **example당 길이** | **짧다** — 일반적인 대화 턴 수준                                      | **길다** — 수천 token 규모의 추론 trace                                                              |
| **데이터 포맷**      | multi-turn **chat message 배열**. loss는 **assistant turn에만** | **문제 → 긴 CoT → 최종 답** 한 덩어리. teacher model(QwQ-32B)의 추론 trace를 distill한 것                   |
| **원문에서의 사용**    | **전체를 1 epoch** 학습 (rank sweep, LR sweep)                  | **subset만 사용** — layer 비교는 rank=256의 small subset, batch size 실험은 **10,000-example subset** |

```
Tulu3 (짧은 instruction following)
  [{"role":"user","content":"..."},
   {"role":"assistant","content":"..."}]        ← 이 부분에만 loss

OpenThoughts3 (긴 reasoning trace)
  문제 → "<think> ... 수천 token의 풀이·backtracking ... </think>" → 최종 답
```


---

### LoRA rank


## 관찰 1:  FFT 와 high rank lora는 log-linear decay But **medium·low-rank LoRA는 step이 커지면서 saturation 되기 시작한다. 

- FullFT와 high-rank LoRA는 **loss가 log(number of steps)에 linear하게 감소**하는 유사한 learning curve를 보인다. 
- **medium·low-rank LoRA는 rank와 correlate되는 어떤 step threshold에서 minimum-loss learning curve를 fall off**한다. 
- step 이 진행됨에 따라서 성능 차이는 벌어진다.
	- 직관적으로: **adapter가 runs out of capacity하면 학습이 느려진다.** 그리고 그 capacity는 rank가 결정한다.
- Tulu3 data에서는 작은 step에서는 lora > FFT 이다.

Tulu3 dataset 전체와 OpenThoughts3의 subset에 대해 **single epoch** 학습했다. 각 dataset과 model size마다 **LoRA rank와 learning rate를 sweep**했다.

> **plot 읽는 법**: 아래 그림에서 rank마다 colored line이 하나씩 그려지는데, 그 선은 **각 training step에서 모든 learning rate에 대해 pointwise minimum을 취해** 얻은 것이다.

![LoRA training curves for various ranks](../assets/lora-rank-curves.svg)

## 관찰2: learning rate effect
다음으로 각 rank에 대해 sweep이 실제로 best learning rate를 덮었는지 확인하기 위해 loss가 LR에 따라 어떻게 변하는지를 본다.

![Learning rate versus final loss](../assets/lora-lr-vs-loss.svg)

- **FullFT의 optimal learning rate는 high-rank LoRA보다 a factor of 10만큼 낮다.**
  > 📎 Biderman et al. (2024), Figure S1 — sampling eval 기반 실험에서 유사한 **10x ratio**를 발견했다.
- **optimal LR은 서로 다른 rank의 LoRA run에서 대체로 비슷해 보인다.** (이에 대한 theoretical explanation은 아래 [Optimal learning rate and rank](#optimal-learning-rate-and-rank)에서 제시된다.) 다만 **some rank dependence는 있다** — **rank=1이 higher-rank LoRA보다 optimal LR이 낮다.** rank 4와 rank 512 사이에서 optimal LR은 **a factor of less than 2**로 변한다.

---

## 관찰3: Lora에서는 작은 batch sizse가 좋다.(32 정도)

일부 setting에서 **LoRA는 FullFT보다 large batch size에 less tolerant**임을 발견했다. 그리고 **performance gap은 batch size가 커질수록 커지며, 이는 rank와 무관하다.** 이 실험에는 OpenThoughts3의 **10,000-example subset**을 사용했다.

![Batch size effects](../assets/lora-batch-size.svg)
- 왼쪽 plot: 
	- example이 커지면 성능이 좋아진다.
	- rank와 무관하게 batch size 32 가 제일 성능이 좋다.
- 오른 쪽 plot: batch size가 커질 수록 lora 와 FFT 간의 성능 갭은 벌어진다.

