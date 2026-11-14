# IA Monitor

IA（Initial Amount）保证金计算与组合风险分析系统。项目覆盖从全股票基础数据到组合 IA 及归因剖析的四层流程，并提供本地上传、参数配置、实时计算和 Excel 导出页面。

## 功能概览

- **L1 全股票基础数据**：行情、申万一级行业、股本、指数成分、波动率、新股、停牌/风险警示和重点监控标记。
- **L2 组合风险输入**：支持 6 位纯数字代码自动补齐 `.SH/.SZ` 后缀与同代码多笔同向持仓自动轧差合并；计算单票市值、组合总市值、申万一级行业与双创微盘集中度、股本持仓占比等中间量。
- **L3 单券 IA**：计算四大调整项、六大限制项（主控限制项定位）、特殊标的（停牌/风险警示/波动率缺失）100% 兜底、规则保证金、`upper_im` 双算法保证金、最终保证金、调整后 IA 及全组合边际 IA。
- **L4 组合 IA 与归因拆解**：汇总组合总保证金，以两种等价方法计算组合 IA，并精确拆解正常层基础保证金、各项加点、限制项 100% 超限及 `upper_im` 修正对组合 IA 的贡献。
- **本地页面**：上传 CSV/XLSX 组合、在线增删/编辑持仓、一键切换「演示参数」与「标准规则参数」、查看结果并导出格式化 Excel。

## 四层计算口径

### L1：基础数据层

- 波动率窗口为 **20、40、90 个交易日**，由前复权日收盘价（`CPS=2`）计算日对数收益率，再以样本标准差乘以 `√252` 年化。
- `total_float_shares` 使用 iFinD `ths_total_float_shares_stock`，单位为股。
- `is_new_stock=1` 表示上市日起 **20 个交易日内**，含上市当日。
- `is_cc_micro=1` 表示不属于沪深300、中证500或中证1000。
- 20 日日均成交额由日线成交额序列在本地平均（停牌日留空不计入）。

### L2：组合风险输入层

输入单边组合（纯多头或纯空头）及 L1 数据，输出第三层所需的全部中间量：

- `market_value`：单票市值，`abs(shares) × price`（同标的同向多笔持仓自动合并汇总）。
- `total_mv`：组合总市值。
- `weight_in_portfolio`：个股集中度。
- `industry_mv`、`industry_weight`、`weight_in_industry`：申万一级行业市值、行业集中度及个股市值占行业比例。
- `cc_micro_mv`、`cc_micro_weight`、`weight_in_cc_micro`：双创微盘市值、集中度及个股市值占板块比例。
- `total_share_ratio`、`float_share_ratio`：持仓占总股本、流通股本比例。
- 20 日日均成交额、指数成分标记及停牌/风险警示状态标记。

本层**不使用阈值，不计算限制项金额**。

### L3：单券 IA 层

调整项包括个股集中度、申万一级行业集中度、90 日波动率和双创微盘集中度。限制项包括：

- `restriction1`：流动性超限。
- `restriction2`：个股集中度超限，上证50成分股豁免。
- `restriction3`：总股本超限。
- `restriction4`：流通股本超限。
- `restriction5`：申万一级行业集中度超限，按个股行业贡献度分摊。
- `restriction7`：双创微盘超限，按个股板块贡献度分摊。

各限制项取最大值，不相加：

```text
restr_mv = MAX(r1, r2, r3, r4, r5, r7)
rule_margin = (market_value - restr_mv) × maintenance_im + restr_mv
```

- **特殊标的 100% 兜底**：停牌股（`is_suspended=1`）、风险警示股（`is_risk_warning=1`）或 90 日波动率缺失标的（如次新股），自动按 **100%** 计收保证金并标注原因。
- **`upper_im` 比较规则**：支持银河规则 1（持仓数 `<阈值` 取 `MAX(A, B)`，`>=阈值` 取 `A`）、少于阈值取小、以及东证规则 2（持仓数 `<阈值` 直接取 `B`，`>=阈值` 取 `MIN(A, B)`）。
- **单券调整后 IA 与边际 IA**：

```text
单券调整后 IA 比率 = 单券最终保证金 ÷ 单票市值
单券边际 IA = Δ组合总保证金 ÷ Δ单票市值（包含推高行业/双创集中度的跨标的连带效应）
```

### L4：组合 IA 与归因拆解层

页面与导出文件同时展示以下两种等价方法及 IA 构成归因：

```text
方法一：组合 IA = Σ单券最终保证金 ÷ 组合总市值
方法二：组合 IA = Σ(单券调整后 IA 比率 × 单票市值 ÷ 组合总市值)
```

## 环境准备

要求：

- Python >= 3.10
- `uv`
- 可用的 iFinD HTTP API `refresh_token`

安装：

```bash
uv venv
uv pip install -e .
cp config.example.yaml config.yaml
```

在 `config.yaml` 中填写 `ifind_refresh_token`，或设置环境变量：

```bash
export IFIND_REFRESH_TOKEN="你的 refresh_token"
```

## 启动计算页面

```bash
uv run python src/webapp.py --host 127.0.0.1 --port 8000
```

浏览器访问：

```text
http://127.0.0.1:8000
```

三级智能数据寻址：

1. **第一级**：优先从本地 L1 SQLite 快照库（`security_master_daily`）毫秒级读取；
2. **第二级**：匹配内置演示数据与单券级内存缓存（`_SECURITY_CACHE`）；
3. **第三级**：仅对缺失股票差量调用 iFinD HTTP API（基础指标合并请求，指数成分独立缓存）。

## 每日全股票流水线

```bash
uv run python src/pipeline.py --date 2026-11-09 --config config.yaml --ifind-mode http
```

也可使用每日脚本：

```bash
./run_daily.sh
```

## 测试

```bash
uv run python -m pytest tests/ -q
```

测试覆盖波动率向量化计算、交易日历、缓存、代码规范化与同向持仓轧差、集中度、单券调整与限制、`upper_im` 双算法、特殊标的 100% 兜底、《IA Sample.html》标准测试案例（TC01~TC13、TC16）、L1 数据库三级寻址、文件上传和 Excel 导出。
