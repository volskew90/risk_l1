# IA Monitor

IA（Initial Amount）保证金计算与组合风险分析系统。项目覆盖从全股票基础数据到组合 IA 的四层流程，并提供本地上传、参数配置、计算和 Excel 导出页面。

## 功能概览

- **L1 全股票基础数据**：行情、行业、股本、指数成分、波动率、新股和重点监控标记。
- **L2 组合风险输入**：单票市值、组合总市值、行业与双创微盘集中度、股本持仓占比等中间量。
- **L3 单券 IA**：计算调整项、限制项、规则保证金、`upper_im` 保证金、最终保证金及调整后 IA。
- **L4 组合 IA**：汇总组合总保证金，并以两种等价方法计算组合 IA。
- **本地页面**：上传 CSV/XLSX 组合、编辑股价和股数、填写阈值、查看结果并导出 Excel。

## 四层计算口径

### L1：基础数据层

- 波动率窗口为 **20、40、90 个交易日**，由前复权日收盘价计算日对数收益率，再以样本标准差乘以 `√252` 年化。
- `total_float_shares` 使用 iFinD `ths_total_float_shares_stock`，单位为股。
- `is_new_stock=1` 表示上市日起 **20 个交易日内**，含上市当日。
- `is_cc_micro=1` 表示不属于沪深300、中证500或中证1000。
- 20 日日均成交额由日线成交额序列在本地平均。

### L2：组合风险输入层

输入单边组合及 L1 数据，输出第三层所需的全部中间量：

- `market_value`：单票市值，`abs(shares) × price`。
- `total_mv`：组合总市值。
- `weight_in_portfolio`：个股集中度。
- `industry_mv`、`industry_weight`、`weight_in_industry`：行业市值、行业集中度及个股市值占行业比例。
- `cc_micro_mv`、`cc_micro_weight`、`weight_in_cc_micro`：双创微盘市值、集中度及个股市值占板块比例。
- `total_share_ratio`、`float_share_ratio`：持仓占总股本、流通股本比例。
- 20 日日均成交额及指数成分标记。

本层**不使用阈值，不计算限制项金额**。

### L3：单券 IA 层

调整项包括个股集中度、行业集中度、90 日波动率和双创微盘集中度。限制项包括：

- `restriction1`：流动性超限。
- `restriction2`：个股集中度超限，上证50成分股豁免。
- `restriction3`：总股本超限。
- `restriction4`：流通股本超限。
- `restriction5`：行业集中度超限，按个股行业贡献度分摊。
- `restriction7`：双创微盘超限，按个股板块贡献度分摊。

各限制项取最大值，不相加：

```text
restr_mv = MAX(r1, r2, r3, r4, r5, r7)
rule_margin = (market_value - restr_mv) × maintenance_im + restr_mv
```

维持保证金率由全局基础保证金率、框架合约加点及调整项计算，并封顶 100%。组合股票数少于页面设定阈值时，再计算 `market_value × upper_im`，按页面选择与规则保证金取大或取小。

```text
单券调整后 IA 比率 = 单券最终保证金 ÷ 单票市值
```

### L4：组合 IA 层

页面同时展示以下两种方法：

```text
方法一：组合 IA = Σ单券最终保证金 ÷ 组合总市值
方法二：组合 IA = Σ(单券调整后 IA 比率 × 单票市值 ÷ 组合总市值)
```

两种方法数学上等价，差额应为 0。

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

环境变量优先于配置文件。真实 token 不应提交到 GitHub。

## 启动计算页面

```bash
./.venv/bin/python src/webapp.py --host 127.0.0.1 --port 8000
```

浏览器访问：

```text
http://127.0.0.1:8000
```

页面操作：

1. 上传组合 CSV/XLSX，或使用内置演示组合。
2. 检查并修改股价、股数和阈值。
3. 点击“计算组合”。
4. 查看单券和组合结果。
5. 点击“导出 Excel（个股IA / 组合IA）”。

上传文件格式：

| 列名 | 必需 | 说明 |
|---|---|---|
| `sec_code` | 是 | 带交易所后缀的证券代码，如 `600519.SH` |
| `sec_name` | 否 | 证券名称；缺省时使用证券代码 |
| `shares` | 是 | 持仓股数；空头使用负数 |
| `price` | 是 | 股价，必须大于 0 |

仅支持纯多头或纯空头单边组合，不支持同一文件中混合多空。非演示股票会通过第一层数据和 iFinD API 补齐；首次计算需要等待接口返回，之后会使用进程内缓存。

演示文件：

- `data/demo_portfolio.csv`
- `data/demo_security_master.csv`

## Excel 导出

导出文件包含两个 sheet：

- **个股IA**：中间量、调整项、限制项、规则与 `upper_im` 保证金、最终保证金和调整后 IA。
- **组合IA**：组合总市值、总保证金、两种组合 IA、方法差额及本次计算参数。

## 每日全股票流水线

```bash
./.venv/bin/python src/pipeline.py --date 2026-11-09 --config config.yaml --ifind-mode http
```

也可使用每日脚本：

```bash
./run_daily.sh
```

主要配置：

| 配置项 | 说明 | 当前值 |
|---|---|---|
| `windows` | 波动率窗口 | `[20, 40, 90]` |
| `annual_factor` | 年化因子 | `252` |
| `min_obs_ratio` | 最小有效样本比例 | `0.6` |
| `price_lookback_days` | 行情回看长度 | `130` |
| `db_path` | SQLite 数据库路径 | `./data/risk_l1.sqlite` |
| `output_dir` | 导出目录 | `./output` |
| `basic_data_refresh_weekday` | 基础数据全量刷新日，0 为周一 | `0` |

数据库主要表：

- `security_master_daily`：全股票每日快照。
- `security_static`：行业、上市日期、股本等低频缓存。
- `price_history`：前复权收盘价历史。
- `monitor_event`：重点监控事件。
- `pipeline_run`：流水线运行记录。

## 测试

```bash
./.venv/bin/python -m pytest tests/ -q
```

测试覆盖波动率、交易日历、缓存、集中度、单券调整与限制、`upper_im`、组合 IA、文件上传和 Excel 导出。

## 项目结构

```text
risk_l1/
├── config.example.yaml       # 安全配置模板
├── config.yaml               # 本地配置，含私密信息，不提交
├── data/                     # 演示组合；运行数据库默认不提交
├── src/
│   ├── pipeline.py           # L1 每日流水线
│   ├── concentration.py      # L2 组合风险输入
│   ├── margin.py             # L3/L4 IA 计算
│   ├── webapp.py             # 本地页面与上传/导出 API
│   └── static/ia_calculator.html
├── tests/
├── output/                   # 运行结果，默认不提交
└── run_daily.sh
```

## 数据与密钥安全

`.gitignore` 默认排除本地配置、虚拟环境、SQLite 数据库、日志、输出文件和缓存。提交前应检查：

```bash
git status --short
git diff --cached
```

不要将真实 iFinD token、账号密码、生产持仓或交易数据库上传到公开仓库。页面中的默认阈值和演示组合仅用于功能验证，实际业务参数以框架合约和风控配置为准。
