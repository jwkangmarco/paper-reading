# 2026년 공개 한국어 Transformer Encoder 조사

> **조사일**: 2026-10-28
> **범위**: 2026-01-01 이후 Hugging Face / GitHub에 공개된 **한국어 사전학습 encoder base 모델** (파인튜닝 파생 모델 제외)
> **방법**: HF API로 `ko` 태그 + `fill-mask` / `feature-extraction` / `sentence-similarity` 모델을 생성일순 수집, 이름 검색(korean, kobert, ko-roberta 등)·국내 주요 조직 계정 스캔, `gh search repos`(2026년 이후 생성), 웹 검색. 후보는 모델 카드를 직접 읽어 base 여부를 판별했다.

**한 줄 결론**: 2026년에 **새로 사전학습된 실용급 한국어 encoder는 Kiwi COCO-LM 하나**이고, 나머지는 연구용·소형이다. 2026년의 한국어 encoder 생태계는 대부분 SKT **A.X-Encoder-base(2025-07)** 를 백본으로 한 임베딩·리랭커·sparse 파인튜닝이다. GitHub에서 독립적으로 발견된 신규 모델은 없었다(관련 저장소는 모두 HF 카드에서 링크된 것).

---

## 1. 2026년 이후 공개된 한국어 encoder base 모델

| 모델 | 공개 | 크기 / 구조 | 학습 | 비고 |
|---|---|---|---|---|
| [kiwi-farm/kiwi-coco-lm-base](https://huggingface.co/kiwi-farm/kiwi-coco-lm-base) (+[generator](https://huggingface.co/kiwi-farm/kiwi-coco-lm-base-generator)) | 2026-09-24 | 161M, ModernBERT 22층, 8K ctx, 64K vocab | ModernBERT-base 가중치 재사용 + COCO-LM(RTD + Corrective LM + Seq Contrastive) 한·영 추가 학습. 한국어 토큰은 kiwipiepy 형태소 경계 Byte-level BPE | **가장 실용적.** KLUE 평균 82.34 (A.X 82.91, KF-DeBERTa 82.89와 동급), GLUE 평균 83.02 (비교군 1위). RTX 4070 Ti SUPER 1장, 51일, Muon + stochastic rounding BF16. Apache-2.0 |
| [alice-noa-chan/nuri-base](https://huggingface.co/alice-noa-chan/nuri-base) / [nuri-base-moe](https://huggingface.co/alice-noa-chan/nuri-base-moe) | 2026-10-24 | 302M / MoE 527M (활성 302M), ko·en·ja | 양방향 encoder + recurrent latent core + masked readout (custom code) | 연구용. frozen 한국어 주제분류 85.16 vs KLUE-RoBERTa-base 84.38 (95% CI 겹침). 한국어 STS Spearman 62.4로 mE5-small 75.8보다 낮음. [GitHub](https://github.com/alice-noa-chan/nuri) |
| [HPLT/hplt_gpt_bert_base_3_0_kor_Hang](https://huggingface.co/HPLT/hplt_gpt_bert_base_3_0_kor_Hang) | 2026-02-11 | 24층, hidden 640, 32K vocab | GPT-BERT (MLM + CLM 혼합), HPLT 3.0 한국어 단일언어 데이터로 from scratch | 한국어 벤치마크 결과 없음. 다운로드 거의 없음. 대조군용 |
| [cadazar/han2han-pt](https://huggingface.co/cadazar/han2han-pt) | 2026-10-05 | 169M, **encoder-decoder** | 자모·문자 임베딩 융합 + 형태소 인지 denoising, 35B 토큰 | 한자·한글 표기 불변 표현, 고문헌 특화. Findings of EMNLP 2026. encoder 출력(640d)을 임베딩으로 제공 |
| [alice-noa-chan/sion-1.5-base](https://huggingface.co/alice-noa-chan/sion-1.5-base) | 2026-10-13 | 274M, encoder-decoder | ko·ja·en span corruption | 번역 모델의 기반. 저자 스스로 "동급 공개 모델보다 약하다"고 명시 |
| [prismdata/bert-ko-pretrained](https://huggingface.co/prismdata/bert-ko-pretrained) | 2026-02-13 | hidden 256 소형 BERT | MLM | 장난감 규모 (PPL 39). GPL-3.0 |

### 한국어를 포함하는 2026년 다국어 encoder (대조군 후보)

- [BidirLM-*-Base](https://huggingface.co/BidirLM/BidirLM-1B-Base) (2026-04) — Gemma-3 기반 양방향화, 270M~1.7B, `ko` 포함
- [granite-embedding-311m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2) (2026-04) — ModernBERT 기반 **임베딩** 모델 (base 아님)
- LFM2.5-Encoder, UltraBERT — 한국어를 지원 언어로 명시하지 않아 제외

### 제외한 모델

| 분류 | 예시 |
|---|---|
| A.X-Encoder 파인튜닝 | KURE-v2, KURE-Reranker-nano, KURE-Sparse-nano, dragonkue/colbert-ko-0.1b, yjoonjang/colbert-ko-en-v2, sewoong/korean-neural-sparse-encoder, NGA-KR/ko-embed-v0, NomaDamas/KoJev-v0 |
| vocab 확장만 | hwp0725/roberta-korean-hanja-stdict |
| 생성 파인튜닝 | solonsophy/kf-deberta-gen (KF-DeBERTa에 MDLM 방식 chat 튜닝) |
| LLM 기반 임베딩 | KURE-v3, BCCard/MoAI-Embedding, sionic-ai/comsat-embed-ko-8b 등 |

**참고**: [SCRIPT](https://arxiv.org/abs/2604.12377) (ACL 2026 Findings)는 새 모델이 아니라, 기존 한국어 PLM에 자모(subcharacter) 구성 정보를 주입하는 **model-agnostic 모듈**이다. 추가 사전학습 없이 NLU/NLG 성능을 올린다고 보고한다. 코드: [SungHo3268/SCRIPT](https://github.com/SungHo3268/SCRIPT).

---

## 2. 모델 비교: KLUE TC/STS/NLI · KoBEST · 모델 크기

**출처마다 평가 프로토콜이 달라 숫자를 한 표에 섞지 않았다.** 2026년 신규 모델 중 KLUE 수치가 공개된 것은 Kiwi COCO-LM뿐이고, **KoBEST 수치를 공개한 2026년 모델은 없다.**

- **임베딩 차원**: 출력 벡터 차원 (hidden size)
- **임베딩 파라미터**: 토큰 임베딩 행렬 크기 = vocab × hidden

### 표 A. KLUE TC/STS/NLI + 모델 크기 (점수 출처: Kiwi COCO-LM 카드, 동일 조건 평가)

| 모델                   | 공개      | 전체 파라미터 | 임베딩 차원 | vocab / 임베딩 파라미터 | 레이어  | 최대 길이  | YNAT (Macro-F1) | KLUE-STS (Pearson / F1) | KLUE-NLI (Acc) |
| -------------------- | ------- | ------- | ------ | ---------------- | ---- | ------ | --------------- | ----------------------- | -------------- |
| **Kiwi COCO-LM**     | 2026-09 | 161M    | 768    | 64K / 49M        | 22   | 8,192  | 86.15           | 92.23 / 86.90           | **87.87**      |
| A.X-Encoder-base     | 2025-07 | 149M    | 768    | 50K / 38M        | 22   | 16,384 | 85.77           | 92.58 / 86.33           | 87.37          |
| KF-DeBERTa-base      | 2023    | ≈186M*  | 768    | 130K / 100M      | 12   | 512    | **86.53**       | 92.63 / **87.45**       | 87.73          |
| KLUE-RoBERTa-base    | 2021    | 111M    | 768    | 32K / 25M        | 12   | 512    | 85.99           | 92.68 / 85.89           | 85.63          |
| KoELECTRA-base-v3    | 2020    | ≈113M*  | 768    | 35K / 27M        | 12   | 512    | 86.05           | **93.13** / 86.23       | 87.23          |
| mmBERT-base          | 2025    | 307M    | 768    | 256K / 197M      | 22   | 8,192  | 84.24           | 90.51 / 81.58           | 83.30          |
| ModernBERT-base      | 2024    | 150M    | 768    | 50K / 39M        | 22   | 8,192  | 77.05           | 76.63 / 69.62           | 67.40          |
| Nuri-base            | 2026-10 | 302M    | 1,024  | 65.5K / 67M      | 14+2 |        |                 |                         |                |
| HPLT GPT-BERT-ko     | 2026-02 | ≈149M*  | 640    | 32.8K / 21M      | 24   |        |                 |                         |                |
| Han2Han-pt (enc-dec) | 2026-10 | 169M†   | 640    | 38.4K / 25M      |      |        |                 |                         |                |

### 표 B. KLUE + KoBEST + 모델 크기 (점수 출처: A.X-Encoder 카드)

| 모델 | 전체 파라미터 | 임베딩 차원 | YNAT (Acc) | STS (F1) | NLI (Acc) | BoolQ | COPA | SentiNeg | WiC | HellaSwag | KoBEST 평균(4개) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A.X-Encoder-base | 149M | 768 | 86.50 | **84.80** | **87.00** | **84.50** | **78.70** | **96.00** | **80.80** | | **85.50** |
| KF-DeBERTa-base | ≈186M* | 768 | **87.00** | 84.30 | 86.10 | 81.30 | 76.50 | 94.70 | 80.50 | | 83.25 |
| KLUE-RoBERTa-base | 111M | 768 | 86.48 | 84.57 | 84.53 | 72.04 | 65.14 | 90.39 | 78.19 | | 76.44 |
| Kiwi COCO-LM | 161M | 768 | | | | | | | | | |

KoBEST 점수는 모두 F1이다.

**수치 출처**
- 파라미터와 차원: HF safetensors 메타데이터와 각 모델 `config.json`. mmBERT는 모델 카드 값, Nuri는 카드에 적힌 구조.
- `*`: safetensors 메타데이터가 없어 fp32 가중치 파일 크기 / 4로 추정. MLM 헤드가 포함되어 약간 클 수 있다.
- `†`: 카드 표기값. 저장된 가중치 기준으로는 189M.

### 해석 시 주의

- **같은 모델도 출처마다 점수가 다르다.** KF-DeBERTa의 KLUE-NLI는 표 A에서 87.73, 표 B에서 86.10, 자체 카드에서는 88.37이다. 시드 수, 하이퍼파라미터 탐색, dev와 test 중 무엇을 썼는지가 출처마다 다르다. **출처가 다른 표의 숫자끼리 비교하면 안 된다.**
- 표 A 안에서는 상위 4개 모델(Kiwi, A.X, KF-DeBERTa, KoELECTRA)의 차이가 대부분 1점 이내다. 단일 실행 결과라면 오차 범위 안이다.
- **본체 크기는 두 무리로 나뉜다.** 기준 모델들은 임베딩 차원이 모두 768이라, 크기 차이는 사실상 vocab에서 나온다. 임베딩 행렬을 빼면 ModernBERT 계열(Kiwi, A.X, mmBERT, ModernBERT)은 약 110M, 12층 BERT 계열(KF-DeBERTa, KLUE-RoBERTa, KoELECTRA)은 약 86M이다.
- KF-DeBERTa는 전체 파라미터의 절반 이상이 130K vocab 임베딩이다. 본체만 보면 KLUE-RoBERTa와 같은 크기다.
- mmBERT도 전체의 64%가 다국어 vocab 임베딩이다. 307M이지만 한국어에서는 사실상 base급으로 봐야 한다.

---

## 3. 한국어 NLI 벤치마크

| 데이터셋 | 구성 | 규모 / 레이블 | 용도 |
|---|---|---|---|
| **KLUE-NLI** | 한국어 원문 문장 6종을 바탕으로 사람이 직접 작성 | 약 2.5만 쌍, 3분류 | 표준. KLUE 논문에 따르면 평가자 간 일치도가 MNLI와 KorNLI보다 높다 |
| **[KorNLI](https://huggingface.co/datasets/kakaobrain/kor_nli)** (Kakao Brain, Findings of EMNLP 2020) | train은 SNLI와 MNLI를 기계번역, dev와 test는 XNLI를 사람이 번역 | train 약 94만 쌍, dev 약 2.5천, test 약 5천, 3분류 | KLUE-NLI 다음 표준. 평가는 사람이 번역한 test로 한다 |
| [multilingual-NLI-26lang-2mil7](https://huggingface.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7) | MNLI, ANLI, FEVER 등을 26개 언어로 기계번역 (한국어 포함) | 언어당 약 10만 쌍 | 학습 데이터 보강용. 평가용으로는 부적합 |

- **XNLI에는 한국어가 없다.** XNLI는 15개 언어로 구성되며, KorNLI의 dev와 test가 바로 그 XNLI 셋을 한국어로 옮긴 것이다.
- **KoBEST에는 NLI 과제가 없다.** BoolQ(지문 기반 예/아니오 판단)와 COPA(원인·결과 추론)를 함의 판단의 보조 지표로 쓸 수 있다.
- **KorSTS**는 KorNLI와 함께 공개된 문장 유사도 데이터셋이다. NLI는 아니지만 보통 함께 평가한다.
- **한국어 adversarial NLI나 HANS류 진단 셋은 찾지 못했다** (HF 데이터셋 검색, 웹 검색 모두 결과 없음).
- 권장: **KLUE-NLI(dev)와 KorNLI(test)를 함께 보고**한다. 원문 기반과 번역 기반 사이의 점수 차이로 번역체 강건성도 확인할 수 있다.

---

## 4. 벤치마크 제안

**비교 기준 모델**: KLUE-RoBERTa-base, KF-DeBERTa-base, **A.X-Encoder-base**(현재 사실상 기준), mmBERT-base. 여기에 위 후보를 붙인다.

| 축 | 벤치마크 | 보는 것 |
|---|---|---|
| ① 기본 NLU (파인튜닝) | **KLUE**: YNAT, STS, NLI, NER, RE, DP, MRC (+ KorNLI test) | 표준 비교. Kiwi 카드 표와 바로 비교 가능 |
| ② 추론·상식 | **KoBEST**: BoolQ, COPA, WiC, HellaSwag, SentiNeg | KLUE가 포화된 상태에서 모델 간 차이를 드러냄 |
| ③ 표현 품질 (frozen) | KLUE-YNAT/NLI에 linear probe, KorSTS에 zero-shot 코사인 유사도 | 파인튜닝 없이 사전학습 표현 자체의 품질 확인. 구조가 다른 모델(Nuri 등)도 공정하게 비교 가능 |
| ④ 검색 (임베딩 backbone) | 같은 레시피로 대조학습한 뒤 **MTEB 한국어 subset** (Ko-StrategyQA, MIRACL-ko, Belebele-ko 등) | 실제 쓰임새가 임베딩이라면 가장 중요. backbone만 바꾸고 학습 레시피는 고정 |
| ⑤ 긴 문맥 | 긴 문서 MRC/분류 (KorQuAD 2.0 등)를 512·2K·8K 길이로 잘라 비교 | ModernBERT 계열의 8K/16K 문맥이 실제로 이득인지 확인 |
| ⑥ 한·영 겸용 | GLUE 일부 (SST-2, MNLI, STS-B) | 이중언어 모델의 영어 성능 저하 정도 |
| ⑦ 효율 | 어절당 토큰 수, 희귀 음절 UNK 비율, 처리량 (seq/s, 512·8K) | 서비스 비용. [korean-tokenizer-analysis](https://github.com/bab2min/korean-tokenizer-analysis) 재사용 가능 |

**공정한 비교를 위한 조건**: 모델마다 같은 하이퍼파라미터 탐색 범위(LR 3~4개, epoch 고정)를 쓰고, seed 3~5개의 평균±표준편차로 보고한다.

**우선순위**: 실용 목적이라면 **A.X-Encoder vs Kiwi COCO-LM vs KF-DeBERTa vs mmBERT**를 ①②④⑦ 위주로 먼저 돌린다. 특히 Kiwi COCO-LM의 **KoBEST 점수가 비어 있어** 직접 평가가 필요하다. Nuri, HPLT, Han2Han은 ③ frozen probe 정도만 확인하면 충분하다.

---

## Sources

- [kiwi-farm/kiwi-coco-lm-base](https://huggingface.co/kiwi-farm/kiwi-coco-lm-base)
- [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base)
- [kakaobank/kf-deberta-base](https://huggingface.co/kakaobank/kf-deberta-base)
- [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)
- [alice-noa-chan/nuri-base](https://huggingface.co/alice-noa-chan/nuri-base), [EVALUATION.md](https://huggingface.co/alice-noa-chan/nuri-base/blob/main/EVALUATION.md)
- [HPLT/hplt_gpt_bert_base_3_0_kor_Hang](https://huggingface.co/HPLT/hplt_gpt_bert_base_3_0_kor_Hang)
- [cadazar/han2han-pt](https://huggingface.co/cadazar/han2han-pt)
- [alice-noa-chan/sion-1.5-base](https://huggingface.co/alice-noa-chan/sion-1.5-base)
- [SCRIPT (arXiv 2604.12377)](https://arxiv.org/abs/2604.12377)
- [KLUE (arXiv 2105.09680)](https://arxiv.org/abs/2105.09680)
- [KorNLI and KorSTS (arXiv 2004.03289)](https://arxiv.org/abs/2004.03289)
- [kakaobrain/kor_nli](https://huggingface.co/datasets/kakaobrain/kor_nli)
- [MoritzLaurer/multilingual-NLI-26lang-2mil7](https://huggingface.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7)
