# Qwen-Scope — 목차 (Table of Contents)

> 논문: *Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models* (Qwen Team, 2026-04-30)
> 리소스:
> - https://huggingface.co/collections/Qwen/qwen-scope
> - https://modelscope.cn/collections/Qwen/Qwen-Scope

---

## 한 줄 요약

Qwen3 / Qwen3.5 (dense + MoE, 7개 변종)에 대해 학습한 **14개 SAE(sparse auto-encoder) 그룹**을 오픈소스로 공개하고, SAE feature를 단순 post-hoc 분석 도구가 아니라 **모델 개발(steering, evaluation, data, post-training)을 위한 재사용 가능한 representation-level interface**로 활용하는 4가지 응용을 제시.

## 4대 응용 축

1. **Inference-time Steering** — feature 방향으로 언어/개념/선호 제어 (가중치 변경 X)
2. **Evaluation Analysis** — feature coverage를 벤치마크 redundancy / 능력 커버리지의 proxy로 사용
3. **Data-centric Workflows** — 다국어 toxicity 분류 + safety-oriented 데이터 합성
4. **Post-Training Optimization** — SFT(언어 누설 억제) + RL(반복 억제)에 SAE feature 신호를 주입

---

## 목차 (논문 §)

### §1 Introduction (p.3)
- LLM 내부의 불투명성 → mechanistic interpretability → SAE
- 기존 SAE 워크플로우는 post-hoc 분석에 머무름 → **개발 인터페이스로 격상** 주장
- Qwen-Scope = 14 SAE groups × 7 model variants (Qwen3 + Qwen3.5, dense + MoE) + 4 applications

### §2 Training in Practice (p.4)
- §2.1 Why Sparse Auto-Encoders? (p.4)
- §2.2 Training in Practice (p.4)

### §3 Application: Steering with SAEs during Inference (p.5)
- §3.1 What is Steering? (p.5)
- §3.2 How to Identify Features for Steering (p.5)
- §3.3 Case Studies of SAE Steering (p.6)

### §4 Application: Evaluation (p.7)
- §4.1 SAE Feature Extraction (p.7)
- §4.2 Benchmark Redundancy (p.7)
- §4.3 Inter-Benchmark Similarity Analysis (p.10)

### §5 Application: Data Classification (p.12)
- §5.1 SAE-Based Toxicity Classifier (p.12)
  - §5.1.1 Toxic Feature Discovery (p.12)
  - §5.1.2 Rule-Based Classification with Selected Features (p.13)
- §5.2 Cross-Lingual Generalization of Toxic Features (p.14)
  - §5.2.1 Shared Toxic Structure Across Languages (p.14)
  - §5.2.2 Transfer of English-Discovered Features (p.14)
- §5.3 Toward Efficient and Practical Classification (p.15)
  - §5.3.1 Layer Selection and Multi-Layer Composition (p.15)
  - §5.3.2 Data Efficiency of Feature Discovery (p.16)

### §6 Application: Data Synthesis (p.17)
- §6.1 Feature-Driven Safety Data Synthesis (p.17)
  - §6.1.1 Target Feature Discovery (p.17)
  - §6.1.2 Data Synthesis from Feature Descriptions (p.18)
- §6.2 Toward Controllable Safety Post-Training (p.19)
  - §6.2.1 Training and Evaluation Setup (p.19)
  - §6.2.2 Coverage Efficiency of Feature-Driven Synthesis (p.20)
  - §6.2.3 Results with Synthetic Data (p.20)

### §7 Application: Supervised Fine-tuning (p.22)
- §7.1 Unexpected Code-Switching (p.22)
- §7.2 Feature Analysis (p.23)
- §7.3 Method (p.23)
- §7.4 Main Results (p.24)

### §8 Application: Reinforcement Learning (p.25)
- §8.1 Feature Analysis (p.25)
- §8.2 Method (p.26)
- §8.3 Experimental Setting (p.27)
- §8.4 Main Results (p.28)

### §9 Conclusion (p.29)
- §9.1 Summary (p.29)
- §9.2 Exploring Directions (p.29)
- §9.3 Social Impact (p.29)

---

## 읽기 계획 (제안)

- **Phase 1 — 기반 이해:** §1, §2 (SAE가 무엇이고, Qwen-Scope는 어떻게 학습됐는가)
- **Phase 2 — 4대 응용 walkthrough:**
  - §3 Steering / §4 Evaluation / §5 Data Classification / §6 Data Synthesis
  - §7 SFT (code-switching) / §8 RL (repetition)
- **Phase 3 — 종합:** §9 + 한계·사회적 영향, DeepSeek-V4 와의 포지셔닝 비교

DeepSeek-V4 와 동일하게 각 하위 섹션별로 `N.M-...md` walkthrough 파일을 만들어 가는 방식 권장.
