# future_dev

期货量化开发目录（TqSdk / 天勤量化）。

## 依赖安装（清华源）

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
```

## 凭据配置

快期账号密码通过环境变量或 `.env` 文件注入，**不写入代码**：

```bash
cp .env.example .env   # 然后填入自己的账号
```

`.env` 已被 `.gitignore` 排除。

## 数据下载

拉取上期所白银主连（`KQ.m@SHFE.ag`）的历史 K 线并落盘为 CSV：

```bash
python download_silver_main_tqsdk.py
```

输出到 `silver_main_data/`。三个周期（15m / 1h / 4h）各自从 TqSdk 取
`data_length=10000` 根（实测 TqSdk 3.10.2 接受该值，未触发单序列上限报错），
随后**自动丢弃每个周期末尾尚未走完的 forming bar**，再按“bar 闭合覆盖”把
三周期对齐到完全相同的窗口：

- 共同起点 = 三周期第一根 bar start 的最大值
- 共同终点 = 三周期最后一根 bar 的 bar_end（start + 周期时长）的最小值
- 每个周期只保留 `bar_start >= 共同起点` 且 `bar_end <= 共同终点` 的行

这样三份数据覆盖的是同一段“所有周期均已完整闭合”的时间区间，避免某周期多带
一根尚未闭合的尾巴、或把起点前的不完整段算进来（多周期策略/回测按时间戳对齐用）。

落盘前还会做 15m→1h、1h→4h 的跨周期聚合一致性校验：用高周期实际相邻
timestamp 作 bucket 边界（期货时段不连续，不用简单 resample），且最后一根
higher bar 也按名义结束时间校验、缺失低周期数据的 higher bar 直接记 error；
要求 `empty = 0` 且 `mismatches = 0` 才写盘。

| 文件 | 周期 | 根数 | 共同闭合窗口（最新运行 2026-09-05） |
| --- | --- | --- | --- |
| `silver_main_15m.csv` | 15 分钟 | 9996 | 2025-07-23 09:00 → 2026-09-05 00:00 |
| `silver_main_1h.csv`  | 1 小时  | 2975 | 同 15m |
| `silver_main_4h.csv`  | 4 小时  | 1082 | 同 15m |

> 注：本下载是固定长度（每个周期 10000 根）的**滑动尾部窗口**，随运行时刻
> 前移，上表日期为某次运行快照；但三周期之间的“共同闭合窗口”口径恒定、彼此
> 严格一致，不随运行时刻漂移。

### CSV 列

`datetime`（北京时间）、`datetime_ns`（原始纳秒时间戳，int64）、`symbol`、
`timeframe`、`open`、`high`、`low`、`close`、`volume`、`open_oi`、`close_oi`。

编码为 `utf-8-sig`，Excel 直接打开不乱码。

## 主连复权

> ⚠️ 以下复权产物（`silver_main_data/adjusted/`、`rollover_owner_*.csv`、
> `contract_closes_*.csv`）是基于**旧行情窗口**（2025-10-15 起）与旧对齐口径
> （仅按 bar start 截断到 15m 区间）生成的；本轮 raw 行情已改为**按 bar 闭合覆盖
> 对齐**的三周期同窗口，**这些派生数据暂不可用**，待下一轮审查并重建
> `build_continuous.py` 后再行刷新。

`KQ.m@` 主连是多个真实月份合约按时间段拼接而成的，换月处会混入合约价差。
`build_continuous.py` 把它转成前复权连续序列：

```bash
python build_continuous.py --refresh   # 联网重建归属表（需快期账号）
python build_continuous.py             # 用缓存离线重算
```

输出在 `silver_main_data/adjusted/`，额外带 `contract`（该 bar 的真实合约）
和 `adj_factor`（复权因子）两列。每周期还会生成 `rollover_report_{tf}.csv`
（换月明细，已入库）与 `rollover_owner_{tf}.csv` / `contract_closes_{tf}.csv`
（重建缓存，可删，重新 `python build_continuous.py --refresh` 即得）。

构成图（2025-10 ~ 2026-09，共 5 次换月）：

| 合约 | 区间 | 换月价差 |
| --- | --- | --- |
| ag2512 | 2025-10-15 → 2025-11-14 | +0.194% |
| ag2602 | 2025-11-14 → 2025-12-29 | −0.022% |
| ag2604 | 2025-12-29 → 2026-03-06 | −0.952% |
| ag2606 | 2026-03-06 → 2026-05-18 | +0.167% |
| ag2608 | 2026-05-18 → 2026-07-17 | +0.170% |
| ag2610 | 2026-07-17 → 2026-09-04 | — |

**这段区间的实际影响很小**：15m 年化波动率 16.15% → 16.13%，单次换月价差均
不超过 1%。若策略对换月跳空敏感仍应使用复权序列，但不必高估这个修正的量级。

### 换月检测为什么不用持仓量

TqSdk 换月时会把新合约的 `open_oi` 接成旧合约的 `close_oi`，导致
`open_oi[t] == close_oi[t-1]` 在全序列恒成立，该信号完全不可用。
单 bar 持仓量变化率能抓到大部分换月（+30%~+57% 离群），但会漏掉
2025-12-29 那次（仅 4.8%，淹没在噪声里）。因此采用逐 bar 精确匹配合约，可审计。

### 注意事项

- `volume` / `open_oi` / `close_oi` 未做复权（成交量与持仓量不可按比例缩放）。
- 脚本已自动丢弃每个周期末尾未走完的 forming bar（`drop_unclosed_tail`），
  落盘数据均为已闭合的完整 K 线，可直接用于回测。
- 历史数据只覆盖到 TqSdk 服务器提供的深度，更早的数据需用本地数据库。
- 复权因子依赖缓存的归属表；换月后需重新 `--refresh`。
