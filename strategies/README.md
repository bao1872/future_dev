# Strategies

这里管理 **交易假设**，不管理指标定义，也不管理行情下载。

## Strategy contract

策略模块建议暴露：

```python
NAME = "my_strategy"
DESCRIPTION = "..."
DEFAULT_PARAMS = {...}


def run(data: dict[str, pd.DataFrame], params: dict) -> dict:
    ...
```

其中 `data` 来自 `market_data.offline_store.load_bundle()`。

策略不得：

- import TqSdk
- 登录行情源
- 下载数据
- 覆盖 `silver_main_data/`
- 修改 `panji_indicators.py` 的 canonical 语义
- 下实盘订单

新增策略后，在 `strategies/registry.py` 显式注册。

当前阶段故意不做动态 plugin discovery；真实出现大量策略后再决定是否需要。
