# Scaffold installation

这套交付有两种使用方式。

## 方式 A：推荐——覆盖当前仓库

把压缩包内容覆盖到当前 `future_dev` 根目录。

保留现有：

- `panji_indicators.py`
- `download_silver_main_tqsdk.py`
- `build_continuous.py`
- `visualize_smc_momentum_tqsdk.py`
- `silver_main_data/`

这是最安全的方式：脚手架只增加治理和研究结构，不重写已验证核心。

完成后：

```bash
python scripts/check_project.py
pip install -r requirements.txt
streamlit run app.py
```

## 方式 B：整个旧目录删除后全新放置

脚手架不会伪造 canonical 代码。先恢复当前已经审查过的核心文件：

```bash
python scripts/restore_core.py
python scripts/check_project.py
```

`restore_core.py` 固定从以下当前仓库 commit 下载核心源码：

```text
3ee6010f9a182038bca667e924081162b79b4c0c
```

然后重新拉取当前离线行情：

```bash
cp .env.example .env
# 填写 TQ_USER / TQ_PASSWORD
python scripts/refresh_data.py
```

最后：

```bash
streamlit run app.py
```

> 不建议用 AI 临时“重新生成” `panji_indicators.py` 来补文件，这会破坏 canonical SSOT。
