# ingest: 시점 t 데이터 수집 (PoC A단계)

매일 미국 장 마감 후 S&P 100 종목의 가격 지표, 뉴스, (주 1회) 재무 지표를 저장한다.
LLM은 쓰지 않으므로 비용이 들지 않는다.

## 준비

```
uv add yfinance lxml
uv add --dev pytest
uv run python -m pytest tests -q        # 누출 방지 테스트 (네트워크 불필요)
```

## 실행

```
uv run python -m ingest.run_daily --limit 5     # 5종목으로 시험
uv run python -m ingest.run_daily               # 전체
```

- 스냅샷 날짜 t = 마감이 확정된 가장 최근 미국 거래일(뉴욕 시간 16:30 이후).
- 이미 있는 날짜는 건너뛴다. 다시 받으려면 `--overwrite`.
- 재무 지표는 최근 7일 안에 받은 게 없을 때만 받는다(`--force-fundamentals`로 강제).
- 다른 종목 목록을 쓰려면 `--universe-file my.csv` (`ticker` 열 필수).

## 저장 구조

```
data/snapshots/
├─ universe/snapshot_date=2026-10-23/data.parquet       # 그날의 유니버스 구성
├─ prices/snapshot_date=.../data.parquet                # 수익률·변동성·거래대금, SPY 포함
├─ fundamentals/snapshot_date=.../data.parquet          # PER, PBR, 성장률, 사업 요약 등
└─ news/snapshot_date=.../data.parquet                  # 기사 제목·요약·발행 시각
```

## 누출 방지 규칙

- 정보 기준 시각(cutoff)은 t일 뉴욕 16:00(장 마감). 그 뒤 발행된 뉴스는 다음 스냅샷 몫이다.
- 종목 카드를 만드는 코드는 반드시 `ingest.reader`만 쓴다. `load_asof(source, t)`와
  `load_news_asof(t)`는 t 이후 데이터를 절대 돌려주지 않으며, `tests/test_ingest.py`가 이를 검사한다.
- t 이후 가격이 필요한 평가 코드는 `evaluation/`에 따로 두고, 수익률은 t 종가부터 잰다.

## 매일 자동 실행 (Windows 작업 스케줄러)

PowerShell에서 한 번만 실행한다. 매일 07:00(JST)은 뉴욕 전날 저녁이라 마감 데이터가 확정된 뒤다.

```
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$PWD\scripts\run_ingest.ps1`""
$trigger = New-ScheduledTaskTrigger -Daily -At 7:00
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable
Register-ScheduledTask -TaskName "qd-rec-stocks ingest" -Action $action -Trigger $trigger -Settings $settings
```

- 저장소 루트에서 실행해야 `$PWD`가 맞다. 로그는 `data\logs\`에 남는다.
- PC가 꺼져 있던 날의 뉴스·재무는 나중에 되살릴 수 없다(가격은 `--date`로 다시 받을 수 있음).
  빠지는 날이 잦으면 항상 켜진 환경으로 옮기는 걸 고려한다.

## 알려진 한계

- yfinance는 비공식 API라 응답 형식이나 한도가 예고 없이 바뀔 수 있다. 실패한 종목은 `error` 열에 남는다.
- 거래소 휴장일은 따로 다루지 않는다. 휴장일 스냅샷의 가격 지표는 직전 거래일과 같고, `last_price_date`로 확인할 수 있다.
- `dividend_yield`는 yfinance 버전에 따라 단위(비율/퍼센트)가 다를 수 있어 원값 그대로 저장한다.
