# CLAUDE.md

## Project Overview
- 마르코가 개인적으로 관심있는 학술 논문는 읽고 요약하는 프로젝트

## Repository Structure
```
{yyyy-mm-dd}/                 요약을 작성한 날짜 폴더
  {논문_약칭}.md               논문 요약 (snake_case)
  assets/                     해당 폴더 문서들이 참조하는 이미지
tools/extract_figs.py         PDF에서 Figure/Table 자동 추출
```
- **PDF는 `.gitignore` 대상**(`*.pdf`)이라 추적되지 않는다. 로컬에만 두고 쓴다.
- 이미지는 문서와 같은 날짜 폴더의 `assets/` 에 두고 `./assets/...` 상대경로로 참조한다.
- 관련 스킬: `paper-summary`(요약 생성), `paper-comparison`(논문 간 비교)

## Conventions
- 논문 요약은 마크다운으로 작성
- LaTeX 수식은 plain text로 변환하여 가독성 확보
- 논문 간 비교 분석 문서도 작성
- 커밋 메시지는 영어로 간결하게 작성

## Figure 추출
- 논문 PDF의 Figure/Table은 **`tools/extract_figs.py` 로 자동 추출**한다. 사용자에게 스크린샷을 요청하지 않는다.
  ```bash
  python3 tools/extract_figs.py <pdf> <pages> <dpi> <outdir> <prefix>
  # 예) python3 tools/extract_figs.py paper.pdf 1-35 150 2026-08-02/assets mixlm
  ```
- 캡션 전문이 `{outdir}/figures.json` 에 함께 저장되므로 마크다운 캡션에 활용한다.
- 삽입 형식은 `<img src="./assets/{파일명}" width="480~560">` + 바로 아래 `> **Figure N**: 설명`.
- 추출 결과는 눈으로 확인한다. 캡션 침범·잘림이 있으면 dpi를 올려 재추출한다.
- 상세 절차는 `paper-summary` 스킬 참조.

## Language
- 논문 요약 및 코드 관련 내용: 영어
- 사용자와의 대화: 한국어 선호
