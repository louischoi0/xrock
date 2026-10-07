# xrock

암호화폐 마진(무기한 선물) 자동매매 시스템입니다. C++20과 CMake를 사용합니다. 첫 대상 거래소는 Bybit입니다.

## 레이어

| # | 레이어 | 위치 | 내용 |
|---|---|---|---|
| 1 | 데이터 | `src/data` | `Store`(백엔드 교체 가능: `kds`, `file`, `sqlite`), 수집기 2종(`backfill`: API 과거 데이터, `Collector`: 실시간), `ReplayFeed`(시간순 k-way merge), `BarAggregator` |
| 2 | 백테스트 | `src/backtest` | `SimGateway`(지연, 수수료, 슬리피지, 펀딩, 격리마진 청산), 이벤트 재생 루프 |
| 3 | 리서치 | `src/research` | 파라미터 그리드와 베이스라인 벤치마크를 병렬 실행하고 샤프 지수 순으로 정렬 |
| 4 | 체결 | `src/execution` | `LiveRunner`: 거래소 IO 스레드 → 큐 → 단일 엔진 스레드, 종료 시 미체결 주문 취소 |
| 5 | 결과 분석 | `src/analytics` | `Journal`(주문, 체결, 펀딩, 자산곡선 CSV), `Metrics`(수익률, MDD, 샤프, 승률, 수수료) |

공유 모듈
- `src/core`: 타입, 이벤트, 설정(INI + `${ENV}`), 시계, 인터페이스(`MarketDataFeed`, `OrderGateway`)
- `src/exchange`: 거래소 어댑터. `[exchange] name`으로 선택합니다(`bybit`, `synthetic`).
- `src/strategy`: `Strategy`/`Context` API, 예제 전략(`ma_cross` 바 기반, `trade_flow` 틱+타이머 기반, `buy_hold` 베이스라인)
- `src/engine`: 백테스트와 실거래가 공유하는 런타임(`Engine`, `Portfolio`, `RiskManager`)

핵심 원칙: 전략, 엔진, 리스크 코드는 백테스트와 실거래에서 같습니다. 달라지는 것은 시계(`SimClock`/`WallClock`), 게이트웨이(`SimGateway`/거래소), 이벤트 공급원뿐입니다.

의존 방향(CMake 타깃으로 강제합니다)

```
engine    -> strategy, data, analytics -> core
backtest  -> engine
research  -> backtest
execution -> engine, exchange -> core
```

## conf에서 선택하는 항목

| 키 | 값 |
|---|---|
| `[exchange] name` | `bybit` \| `synthetic` |
| `[exchange] env` | `mainnet` \| `testnet` \| `demo` (mainnet 실거래는 `[live] allow_mainnet = true`가 있어야 시작합니다) |
| `[data] backend` | `kds` \| `file` \| `sqlite` (`[data.kds] host/port/table_prefix`, `[data.file] root`, `[data.sqlite] path`). 기본값은 `kds`이며 프로젝트가 직접 호스팅하는 서버를 사용합니다(아래 "KDS 서버" 참고). `conf/demo.conf`는 서버 없이 실행되도록 `sqlite`를 사용합니다. |
| `[strategy] name` | `ma_cross` \| `trade_flow` \| `buy_hold`. 파라미터는 `[strategy.<name>]`에 둡니다. |

확장 지점
- 거래소 추가: `exchange/<venue>/`에 `MarketDataFeed`와 `OrderGateway`를 구현하고 `exchange/registry.cpp`에 분기를 추가합니다.
- DB 추가: `data::Store`를 구현하고 `data/store.cpp`의 `make_store()`에 분기를 추가합니다.
- 전략 추가: `strategy/<name>.cpp`를 작성하고 `strategy/registry.cpp`와 `CMakeLists.txt`에 한 줄씩 추가합니다.

## 데이터 수집

| 명령 | 컴포넌트 | 소스(Bybit) | 비고 |
|---|---|---|---|
| `xrock fetch` | `data::backfill` + `HistoricalSource` | trade: `public.bybit.com` 일별 아카이브, funding: `/v5/market/funding/history` | `[fetch] start/end/streams`. 이미 저장된 날짜는 건너뛰므로 다시 실행하면 이어서 받습니다. |
| `xrock collect` | `data::Collector` + `MarketDataFeed` | 공개 WebSocket: `publicTrade`, `orderbook.N`, `tickers` | 로컬 L2 오더북 유지, `[market] book_interval` 간격으로 저장, 10초마다 flush, 자동 재연결 |

과거 오더북은 아직 받을 수 없으므로 실시간으로만 수집합니다(TODO: ob500 아카이브).

## ML 파이프라인 (학습 전 단계 인터페이스)

`src/ml`(L3)과 `python/xrock_ml`로 구성됩니다. 같은 피처 이름 목록과 샘플링 규칙을 쓰면 항상 같은 값이 나오도록 하는 것이 핵심 원칙입니다.

| 단계 | 구성요소 | 내용 |
|---|---|---|
| 피처 | `ml::FeatureSet` | `ret_<d> vol_<d> flow_<d> logvol_<d> rate_<d> spread imb_<n> funding`. 변동성 모델 `rv_<d>_<n> ewmavol_<d>_<n> garch_<d>`는 d 간격 그리드 수익률 기준의 구간당 σ를 냅니다. 시점 t의 값은 t 이전 이벤트만 사용합니다. |
| 레이블 | `ml::DatasetBuilder` | `fwd_<h> = log(mid(t+h)/mid(t))`. 데이터 공백이 `max_gap`을 넘으면 그 샘플은 버립니다. |
| 장벽 레이블 | `[dataset] barriers`, `barrier_vol` | `tb_<i>_long/short`: 익절 `k_tp·σ_h`, 손절 `k_sl·σ_h`, 시간 장벽 h에서의 로그 수익률입니다(`σ_h = σ·√(h/d)`). 경로는 샘플 간격 고가·저가로 판정하고, 한 간격에서 둘 다 닿으면 손절로 처리합니다. |
| 분할 | `ml::Splits` | walk-forward + embargo(가장 긴 horizon 이상인지 검사합니다) + 최종 holdout |
| 내보내기 | `xrock dataset` | `X.npy`, `Y.npy`, `ts.npy`, `meta.json`(xrock-dataset/1) |
| Python | `xrock_ml.load_dataset` | fold별 마스크와 holdout 마스크를 제공합니다. `save_linear_model`, `save_onnx_model_header`로 모델 파일을 씁니다. `onnx_export.linear_to_onnx`는 실행기 검증용 참조 그래프를 만듭니다. |
| 모델 | `ml::load_model` | `xrock-model/1` JSON 파일이 피처 계약을 함께 담습니다. `linear` 타입과 `onnx` 타입(ONNX Runtime 1.30.0, CPU, 단일 스레드)을 지원합니다. ONNX 그래프는 입력 float32 `[1, F]`(원시 피처, 정규화는 그래프 안에서 수행), 출력 `[1, O]` 형태여야 하고, 모델을 로드할 때 이 형태를 검사합니다. |
| 평가 | `xrock model-eval <model.json>` | fold와 holdout별로 IC, rank IC, 적중률, 예측 분위별 평균 수익률을 계산합니다. |
| 시퀀스 | `ml::SampleWindow`, `Dataset.windows(T)` / `window_X` | 모델 파일의 `"window": T`이면 입력은 `[1, T, F]`(오래된 행 먼저)입니다. 표본 간격만큼 연속된 샘플만 하나의 창으로 묶고, 간격이 끊기면 다시 시작합니다. 오프라인에서 만든 창은 엔진에도 항상 같은 행으로 존재합니다. |
| 파라미터 실행 | 전략 `barrier` | 격자 방식(출력 `tb_*` = 장벽 조합별 기대 수익 → 최선 조합·방향 선택) 또는 직접 방식(출력 `side, k_tp, k_sl, hold`)을 씁니다. 진입 기준은 `min_edge`입니다. 손절·익절·시간 청산은 전략이 관리합니다. 장벽 설정은 모델 파일에 함께 담깁니다. |
| 결정 기록 | `Context::log_decision` → `decisions.csv` | 열은 `ts, symbol, kind, reason, order_id, values(k=v;...)`입니다. `barrier` 전략은 enter/open/exit/rejected/skip(`log_skips`)을, `model` 전략은 rebalance/rejected를 남깁니다. `xrock report`가 유형·사유별 집계를 출력하고, `xrock_ml.journal.barrier_trades()`가 결정과 체결을 주문 id로 이어 거래 단위 표로 만듭니다. |
| 실행 | 전략 `model` | `trade = false`이면 섀도 모드로, 주문 없이 `predictions.csv`만 기록합니다. |

Python 환경: `python3 -m venv --without-pip .venv`, `.venv/bin/python var/bootstrap/get-pip.py`, `.venv/bin/python -m pip install -r python/requirements.txt`. 이 머신에는 ensurepip가 없어서 이 방법을 씁니다. 실행할 때는 `PYTHONPATH=python`을 지정합니다.

## KDS 서버

xrock 데이터는 프로젝트가 직접 호스팅하는 KDS(ckdbs) 서버에 저장합니다.

| 항목 | 위치 |
|---|---|
| 바이너리 | 루트의 `kds_server` (ckdbs `25c58497` Release 빌드, `scripts/build-kds.sh`로 재빌드) |
| 서버 설정 | `conf/kds.conf` (포트 25432, 데이터 `var/kds/xrock.db`) |
| 제어 | `scripts/kds.sh start\|stop\|restart\|status\|log` (stop은 `STOP`으로 정상 종료합니다) |
| 점검 | `build/xrock db ping`, `build/xrock db sql "SHOW TABLES"` |

agent용 상세 문서는 `.claude/docs/ckdbs/`에 있습니다.

## 빌드와 실행

의존성은 시스템 패키지를 먼저 찾고, 없으면 CMake가 소스에서 직접 받아 빌드합니다(IXWebSocket, simdjson, zlib, SQLite). ONNX Runtime은 SHA256 해시로 고정한 공식 빌드 파일을 받아 씁니다(`-DXROCK_WITH_ONNX=OFF`로 끌 수 있습니다). 따라서 root 권한이 필요 없습니다. 필수 요소는 C/C++ 컴파일러, CMake, OpenSSL, GTest입니다.

```sh
cmake -S . -B build && cmake --build build -j
(cd build && ctest)
scripts/kds.sh start                      # xrock fetch / collect / backtest 전에 실행

# 오프라인 데모: synthetic 거래소 -> 수집 -> 백테스트 / 리서치 / 리포트
./build/xrock collect  -c conf/demo.conf
./build/xrock backtest -c conf/demo.conf
./build/xrock research -c conf/demo.conf
./build/xrock report results/backtest-ma_cross -c conf/demo.conf
```

## 다음 단계

1. Bybit `Gateway`: private WS(order, execution, position)와 주문 REST를 구현합니다. 서명과 주문 본문은 이미 구현되어 있습니다.
2. 과거 오더북(ob500 아카이브)을 가져오고, 실시간 수집에서 연결이 끊긴 구간을 아카이브로 보충합니다.
3. `SimGateway`: 부분 체결, 큐 포지션, mark price 기반 청산, 교차마진, inverse 계약을 지원합니다.
4. 리서치: walk-forward 및 out-of-sample 분할을 추가합니다.
5. 실거래: 지갑과 포지션을 동기화하고, 저널을 스트리밍으로 기록하고, 종료 시 청산 옵션을 추가합니다. 미체결 주문의 노출도 리스크에 반영합니다.
