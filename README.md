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

输出到 `silver_main_data/`：

| 文件 | 周期 | 根数 | 名义覆盖 |
| --- | --- | --- | --- |
| `silver_main_15m.csv` | 15 分钟 | 8000 | 约 11 个月 |
| `silver_main_1h.csv` | 1 小时 | 2000 | 同上 |
| `silver_main_4h.csv` | 4 小时 | 500 | 同上 |

三个周期按相同名义时间跨度对齐：`8000 × 15m = 2000 × 1h = 500 × 4h`。
8000 是 TqSdk 单个 K 线序列的请求上限。

### CSV 列

`datetime`（北京时间）、`datetime_ns`（原始纳秒时间戳，int64）、`symbol`、
`timeframe`、`open`、`high`、`low`、`close`、`volume`、`open_oi`、`close_oi`。

编码为 `utf-8-sig`，Excel 直接打开不乱码。

### 注意事项

- 主连序列未做复权，换月处存在价格跳空，跨月回测需自行处理。
- 序列最后一根 K 线可能是当前未走完的 bar。
- 历史数据只覆盖到 TqSdk 服务器提供的深度，更早的数据需用 `TqBacktest` 或本地数据库。
