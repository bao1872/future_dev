# future_dev

沪银期货离线量化策略研究项目。

## 项目定位

这是一个 **研究阶段** 的量化项目，不是生产交易系统。

核心工作流固定为：

```text
TqSdk
  ↓
拉取并验证行情
  ↓
当前有效离线行情（CSV）
  ↓
Panji canonical indicators
  ↓
策略研究 / 参数实验
  ↓
Streamlit + Plotly
```

### 当前边界

- 固定研究品种：上期所白银主连 `KQ.m@SHFE.ag`
- 时间周期：`15m / 1h / 4h`
- TqSdk 只负责行情获取
- 策略研究只读取离线行情
- Streamlit 是唯一研究 UI
- 不做 dataset 版本化，只维护一套当前有效离线行情
- 不做生产工程化

完整治理规则见 [`AGENTS.md`](AGENTS.md)。

---

## 目录结构

```text
future_dev/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── app.py                         # Streamlit 首页
├── pages/
│   ├── 1_Data.py                  # 离线行情状态 / 校验
│   ├── 2_Chart.py                 # K线 + canonical 指标可视化
│   ├── 3_Strategy_Lab.py          # 策略实验入口
│   └── 4_Results.py               # 研究结果浏览
│
├── market_data/
│   ├── config.py                  # 固定品种 / 周期 / 路径
│   ├── offline_store.py           # 策略端唯一离线行情读取入口
│   ├── validation.py              # 离线行情 targeted validation
│   └── tqsdk_source.py            # TqSdk 刷新入口，委托现有 downloader
│
├── research/
│   ├── indicator_adapter.py       # canonical 指标薄适配层
│   ├── experiment_store.py        # 极简实验结果记录
│   ├── experiments.md             # 研究假设人工纪要
│   └── results/                   # JSON 实验结果
│
├── strategies/
│   ├── README.md                  # 策略契约
│   ├── registry.py                # 明确注册策略，不做插件框架
│   └── template_strategy.py       # 模板，不默认注册
│
├── scripts/
│   ├── check_project.py           # 检查核心资产是否完整
│   ├── restore_core.py            # 全新目录时恢复当前已审查核心源码
│   └── refresh_data.py            # 刷新当前离线行情
│
│   # 以下为现有项目已验证/已有资产，脚手架围绕它们工作：
├── panji_indicators.py
├── download_silver_main_tqsdk.py
├── build_continuous.py
├── visualize_smc_momentum_tqsdk.py
└── silver_main_data/
```

> 重要：治理目标不是把已有脚本强行搬进新目录。现有 canonical / downloader 继续作为已建立资产；新的接口层只是管理访问边界。

---

## 安装

```bash
pip install -r requirements.txt
```

配置 TqSdk / 快期凭据：

```bash
cp .env.example .env
```

然后填写：

```text
TQ_USER=...
TQ_PASSWORD=...
```

`.env` 不进入 Git。

---

## 启动 Streamlit

```bash
streamlit run app.py
```

当前工作台包含：

### Data

- 查看 15m / 1h / 4h 当前离线行情
- 起止时间、bar 数量
- 基础数据校验
- 跨周期聚合校验

### Chart

- Candlestick
- DSA VWAP（直接消费 canonical 实现）
- SMC BOS / CHoCH
- Order Blocks
- Momentum / SQZMOM

所有指标先在完整离线历史上计算，再裁剪显示窗口，避免因先 `tail()` 导致结构状态失真。

### Strategy Lab

- 从 `strategies/registry.py` 选择已注册策略
- 选择研究时间区间
- 调整参数
- 执行离线策略
- 保存研究结果

### Results

- 浏览 `research/results/*.json`
- 比较实验的时间区间、参数和结果摘要

---

## 行情刷新

推荐：

```bash
python scripts/refresh_data.py
```

它委托现有 `download_silver_main_tqsdk.py`。

现有 downloader 已负责：

- forming bar 剔除
- 三周期共同闭合窗口
- timestamp / OHLC / volume / OI sanity
- 15m -> 1h 聚合一致性
- 1h -> 4h 聚合一致性

策略代码不应该自己调用 TqSdk。

---

## 离线数据原则

当前不做 dataset 版本化。

研究只维护：

```text
silver_main_data/silver_main_15m.csv
silver_main_data/silver_main_1h.csv
silver_main_data/silver_main_4h.csv
```

实验可复现性记录：

- `data_start`
- `data_end`
- strategy
- parameters
- result summary
- Git SHA（可选）

这足够满足当前固定品种研究。

---

## Canonical indicator 原则

`panji_indicators.py` 是 Panji 指标计算 SSOT。

默认不得为策略实验重写：

- DSA
- SMC
- BOS / CHoCH
- Order Block lifecycle
- Momentum / SQZMOM

策略负责组合它们，不负责重新定义它们。

---

## 策略代码约定

策略模块暴露：

```python
NAME = "..."
DESCRIPTION = "..."
DEFAULT_PARAMS = {...}

def run(data: dict[str, pd.DataFrame], params: dict) -> dict:
    ...
```

策略输入是离线 DataFrame：

```python
{
    "15m": df_15m,
    "1h": df_1h,
    "4h": df_4h,
}
```

策略禁止：

- import / login TqSdk
- 下载行情
- 覆盖原始 CSV
- 修改 canonical indicator 全局默认值
- 实盘下单

---

## 当前连续合约复权提醒

现有 README 曾明确指出 `silver_main_data/adjusted/` 与部分 rollover 派生产物存在旧窗口/旧对齐口径问题。

本脚手架默认 **不使用 adjusted 数据做策略研究**。

只有在 `build_continuous.py` 针对当前 raw 数据重新审查、重建并验证后，才应把 adjusted 数据接入正式回测入口。

---

## 明确不做的事情

当前阶段默认不引入：

- Docker
- CI/CD
- GitHub Actions
- FastAPI
- 数据库
- Redis
- 微服务
- 部署环境
- 全量回归测试体系
- provider factory
- dataset versioning

出现真实需求时再增加。


---

## 如果你是删除旧目录后全新替换

脚手架本身不重新生成 canonical 核心代码。运行：

```bash
python scripts/restore_core.py
python scripts/check_project.py
python scripts/refresh_data.py
streamlit run app.py
```

核心恢复固定到当前审查基线 commit `3ee6010f9a182038bca667e924081162b79b4c0c`。
