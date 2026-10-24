# qd-rec-stocks

Sakana AI의 [QD-Rec](https://github.com/SakanaAI/QD-Rec)을 미국 주식 아이디어 발굴에 응용하는 개인 연구 프로젝트.
과거 시점 백테스트로 "QD-Rec으로 주식 후보를 추천해봤다"는 블로그 글을 쓰는 것이 목표다. 자동매매나 매매 권유가 아니다.

- 설계: `docs/design.md` (현재 설계만 적는다. 설계가 바뀌면 이 파일을 고친다)
- 결과: `reports/backtest500.md`, 블로그 초안: `docs/blog-draft.md`
- 원 논문: arXiv 2610.23470

## 환경

- Windows, PowerShell, `uv`, Python 3.10+
- 리모트: `origin` = 비공개 저장소(sunyeul/qd-rec-stocks), `upstream` = SakanaAI/QD-Rec
- 원본 업데이트: `git fetch upstream && git merge upstream/main`
- LLM은 OpenRouter 경유(`.env`의 `OPENROUTER_API_KEY`). `.env`는 커밋하지 않는다. 계정 충전 잔액이 키 한도보다 작으면 잔액이 상한이다.

## 구조

- `ingest/`: 과거 날짜 t의 스냅샷 재구성(`backfill.py`). 사용법은 `ingest/README.md`
- `apps/stocks/`: 종목 카드, 제안·반론·숫자 검증·Jev 심사, QD-Rec 루프 연결
- `portfolio/build.py`: 아카이브 → 상위 20, 섹터당 3
- `evaluation/`: t 이후 수익, 리포트(A~E), 수익 원천 분류
- `scripts/backtest.py`: 전체 실행(`--step 1`이면 날짜 하나씩)
- `shared/`(원본 코어)는 최소한으로만 수정한다.

## 코드 규칙

- **누출 방지가 최우선이다.**
  - 스냅샷 날짜 t의 정보 기준 시각은 t일 뉴욕 16:00(장 마감).
  - 종목 카드 등 생성 쪽 코드는 `ingest.reader`(`load_asof`, `load_news_asof`)로만 데이터를 읽는다.
  - t 이후 가격이 필요한 코드는 `evaluation/`에만 두고, 수익률은 t 종가부터 잰다.
  - 모든 모델은 학습 마감일이 스냅샷보다 앞서야 한다. 모델을 바꾸면 `scripts/backtest.py`의 `DATES`도 다시 정한다.
- 누출 관련 로직을 바꾸면 `tests/test_ingest.py`에 테스트를 추가한다.
- 본 실행 전에 10회짜리 시험으로 비용과 구성 요소(숫자 검증 실패, 점수 분포, 카드 길이)를 먼저 확인한다.
- 테스트: `uv run python -m pytest tests -q`

## 현재 상태 (2026-10-24)

- S&P 500, 2026-04-30 하루만 실행하고 멈췄다(QD-Rec, Novelty 각 300회). 나머지 날짜는 블로그의 "다음 단계"로 넘겼다.
- 결과 요약과 한계는 `docs/design.md`의 "결과", "한계".
