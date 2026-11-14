#!/usr/bin/env python3
"""本地 L2-L4 保证金计算页面，仅使用 Python 标准库提供 HTTP 服务。"""
import argparse
import base64
import binascii
import io
import json
import logging
import math
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import calc_vol, concentration, db, margin, trade_cal
from src.fetch_ifind import IFindClient

logger = logging.getLogger(__name__)

PAGE = ROOT / "src" / "static" / "ia_calculator.html"
PORTFOLIO = ROOT / "data" / "demo_portfolio.csv"
SECURITY = ROOT / "data" / "demo_security_master.csv"
CONFIG = ROOT / "config.yaml"
CONFIG_EXAMPLE = ROOT / "config.example.yaml"
DB_PATH = ROOT / "data" / "risk_l1.sqlite"

# 单券级缓存: {sec_code: row_dict}
_SECURITY_CACHE = {}
# 指数成分股缓存: {(index_code, trade_date): set(codes)}
_INDEX_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _json_ready(value):
    """递归将 NumPy/Pandas 类型及非有限浮点数转换为标准 JSON 可序列化对象"""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_demo():
    """加载内置演示组合与演示第一层证券基础数据"""
    portfolio = pd.read_csv(PORTFOLIO, dtype={"sec_code": str})
    security = pd.read_csv(SECURITY, dtype={"sec_code": str})
    return portfolio, security


def parse_portfolio_file(filename, encoded):
    """解析上传的 CSV/XLSX 组合文件，支持 6 位代码补齐后缀与同标的同向持仓自动合并"""
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("上传文件编码无效") from exc
    if not raw or len(raw) > 10_000_000:
        raise ValueError("上传文件为空或超过 10MB")
    suffix = Path(filename).suffix.lower()
    source = io.BytesIO(raw)
    try:
        if suffix == ".csv":
            try:
                df = pd.read_csv(source, dtype={"sec_code": str})
            except UnicodeDecodeError:
                source.seek(0)
                df = pd.read_csv(source, dtype={"sec_code": str}, encoding="gb18030")
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(source, dtype={"sec_code": str})
        else:
            raise ValueError("仅支持 CSV、XLSX 文件")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"无法解析上传文件: {exc}") from exc

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in ["sec_code", "shares", "price"] if c not in df.columns]
    if missing:
        raise ValueError(f"组合文件缺少必需列: {missing}")

    df = df.copy()
    df["sec_code"] = df["sec_code"].apply(concentration.normalize_sec_code)
    if "sec_name" not in df.columns:
        df["sec_name"] = df["sec_code"]
    else:
        df["sec_name"] = df["sec_name"].fillna(df["sec_code"]).astype(str).str.strip()
        empty_mask = df["sec_name"] == ""
        df.loc[empty_mask, "sec_name"] = df.loc[empty_mask, "sec_code"]

    consolidated = concentration.consolidate_positions(df, merge_duplicates=True)
    return consolidated[["sec_code", "sec_name", "shares", "price"]].to_dict(
        orient="records"
    )


def _load_config() -> dict:
    """加载配置文件，若 config.yaml 不存在则回退至 config.example.yaml"""
    target = CONFIG if CONFIG.exists() else CONFIG_EXAMPLE
    if not target.exists():
        return {}
    with target.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_trade_date() -> str:
    """
    解析最新可用交易日：
    1) 优先从本地 L1 SQLite 数据库读取最新快照交易日；
    2) 若本地尚未生成 SQLite 数据库或表为空，自动回退至交易日历最近交易日。
    """
    if DB_PATH.exists():
        conn = db.get_conn(str(DB_PATH))
        try:
            row = conn.execute(
                "SELECT MAX(trade_date) FROM security_master_daily"
            ).fetchone()
            if row and row[0]:
                return str(row[0])
        except Exception:
            pass
        finally:
            conn.close()

    today_str = datetime.now().strftime("%Y-%m-%d")
    recent = trade_cal.recent_trade_dates(today_str, 2)
    if recent:
        # 若当前时间早于 16:30（当日日线可能尚未完结），取上一交易日
        if len(recent) >= 2 and recent[-1] == today_str and datetime.now().hour < 16:
            return recent[-2]
        return recent[-1]
    return today_str


def _load_from_l1_db(codes: list) -> dict:
    """
    第一级寻址：从本地 L1 SQLite 数据库（security_master_daily）读取已落库的最新快照数据。
    返回 {sec_code: row_dict}
    """
    if not DB_PATH.exists() or not codes:
        return {}
    conn = db.get_conn(str(DB_PATH))
    try:
        row = conn.execute(
            "SELECT MAX(trade_date) FROM security_master_daily"
        ).fetchone()
        if not row or not row[0]:
            return {}
        trade_date = str(row[0])
        placeholders = ",".join("?" * len(codes))
        df = pd.read_sql_query(
            f"SELECT * FROM security_master_daily WHERE trade_date = ? AND sec_code IN ({placeholders})",
            conn,
            params=(trade_date, *codes),
        )
        if df.empty:
            return {}
        found = {}
        required_non_null = [
            "ind_sw_l1_name",
            "total_shares",
            "total_float_shares",
            "avg_amt_20d",
        ]
        for _, r in df.iterrows():
            if any(pd.isna(r.get(c)) for c in required_non_null):
                continue
            c = str(r["sec_code"])
            found[c] = {
                "sec_code": c,
                "sec_name": str(r.get("sec_name") or c),
                "board": str(
                    r.get("board")
                    or (
                        "科创板"
                        if c.startswith("688")
                        else "创业板"
                        if c.startswith(("300", "301"))
                        else "主板"
                    )
                ),
                "ind_sw_l1_code": r.get("ind_sw_l1_code"),
                "ind_sw_l1_name": r.get("ind_sw_l1_name"),
                "total_shares": float(r["total_shares"]),
                "total_float_shares": float(r["total_float_shares"]),
                "avg_amt_20d": float(r["avg_amt_20d"]),
                "vol_90d": float(r["vol_90d"]) if pd.notna(r.get("vol_90d")) else np.nan,
                "is_sse50": int(r.get("is_sse50") or 0),
                "is_csi300": int(r.get("is_csi300") or 0),
                "is_csi500": int(r.get("is_csi500") or 0),
                "is_csi1000": int(r.get("is_csi1000") or 0),
                "is_cc_micro": int(r.get("is_cc_micro") or 0),
                "is_suspended": int(r.get("is_suspended") or 0),
                "is_risk_warning": int(r.get("is_risk_warning") or 0),
                "close": None,
                "as_of_date": trade_date,
            }
        return found
    except Exception as exc:
        logger.warning("Failed to query L1 SQLite snapshot: %s", exc)
        return {}
    finally:
        conn.close()


def _fetch_security_data(codes: list) -> pd.DataFrame:
    """
    第三级寻址：对本地库与缓存未命中的差量股票调用 iFinD HTTP API 补齐。
    - 基础数据使用 l1_only=True 单请求批量拉取；
    - 指数成分使用 _INDEX_CACHE 缓存，避免每次重复拉取四大指数成分股；
    - 90 日波动率遵循 min_obs_ratio=0.6（允许停牌缺口，次新股样本不足时由 L3 按 100% 兜底）。
    """
    trade_date = _resolve_trade_date()
    config = _load_config()
    client = IFindClient(
        mode="http",
        refresh_token=config.get("ifind_refresh_token"),
        retry_times=int(config.get("retry_times", 3)),
        retry_backoff=float(config.get("retry_backoff", 2)),
    )
    basic_df = client.get_basic_data(codes, trade_date, l1_only=True)
    daily_df = client.get_daily_basic(codes, trade_date)
    if basic_df.empty or daily_df.empty:
        raise RuntimeError(f"iFinD 未返回证券基础数据: {codes}")
    basic = basic_df.set_index("sec_code")
    daily = daily_df.set_index("sec_code")
    prices = client.get_daily_close(codes, trade_date, lookback=130)

    index_codes = {
        "is_sse50": "000016.SH",
        "is_csi300": "000300.SH",
        "is_csi500": "000905.SH",
        "is_csi1000": "000852.SH",
    }
    index_sets = {}
    for flag, index_code in index_codes.items():
        cache_key = (index_code, trade_date)
        with _CACHE_LOCK:
            cached_members = _INDEX_CACHE.get(cache_key)
        if cached_members is not None:
            index_sets[flag] = cached_members
            continue
        members = set(client.get_index_components(index_code, trade_date))
        if not members:
            raise RuntimeError(f"指数成分为空: {index_code}")
        with _CACHE_LOCK:
            _INDEX_CACHE[cache_key] = members
        index_sets[flag] = members

    min_obs_ratio = float(config.get("min_obs_ratio", 0.6))
    rows = []
    for code in codes:
        if code not in basic.index or code not in daily.index:
            raise RuntimeError(f"iFinD 缺少证券基础数据: {code}")
        series = prices.get(code, [])
        vols, _, _ = calc_vol.calc_volatility(
            series, windows=[90], min_obs_ratio=min_obs_ratio
        )
        valid_prices = [float(v) for v in series if pd.notna(v) and float(v) > 0]
        is_susp = 1 if (not series or pd.isna(series[-1]) or float(series[-1] or 0) <= 0) else 0
        b, d = basic.loc[code], daily.loc[code]
        row = {
            "sec_code": code,
            "sec_name": code,
            "board": "科创板"
            if code.startswith("688")
            else "创业板"
            if code.startswith(("300", "301"))
            else "主板",
            "ind_sw_l1_code": b.get("ind_sw_l1_code"),
            "ind_sw_l1_name": b.get("ind_sw_l1_name"),
            "total_shares": b.get("total_shares"),
            "total_float_shares": b.get("total_float_shares"),
            "avg_amt_20d": d.get("avg_amt_20d"),
            "vol_90d": vols[90] if vols.get(90) is not None else np.nan,
            "close": valid_prices[-1] if valid_prices else np.nan,
            "is_cc_micro": 0,
            "is_suspended": is_susp,
            "is_risk_warning": 0,
            "as_of_date": trade_date,
        }
        for flag, members in index_sets.items():
            row[flag] = int(code in members)
        row["is_cc_micro"] = int(
            not (row["is_csi300"] or row["is_csi500"] or row["is_csi1000"])
        )
        rows.append(row)

    security = pd.DataFrame(rows)
    required = ["ind_sw_l1_name", "total_shares", "total_float_shares", "avg_amt_20d"]
    if security[required].isna().any().any():
        raise RuntimeError("上传组合存在未补齐的第一层数据")
    return security


def get_security_data(codes) -> pd.DataFrame:
    """
    三级智能寻址获取第一层证券数据：
    1) 内置 Demo 股票集快速匹配 & 单券内存缓存 (_SECURITY_CACHE)
    2) 本地 L1 SQLite 最新快照表 (security_master_daily)
    3) 差量调用 iFinD HTTP API 并写回单券缓存
    """
    norm_codes = sorted({concentration.normalize_sec_code(c) for c in codes})
    _, demo_security = load_demo()
    demo_map = {
        str(r["sec_code"]): r.to_dict() for _, r in demo_security.iterrows()
    }

    # 若全部属于内置演示股票，直接返回演示快照
    if set(norm_codes).issubset(demo_map.keys()):
        return demo_security[demo_security["sec_code"].isin(norm_codes)].copy()

    collected = {}
    missing = []

    # 1. 先查单券级内存缓存
    with _CACHE_LOCK:
        for c in norm_codes:
            if c in _SECURITY_CACHE:
                collected[c] = _SECURITY_CACHE[c].copy()
            else:
                missing.append(c)

    # 2. 查本地 L1 SQLite 最新快照库
    if missing:
        db_hits = _load_from_l1_db(missing)
        if db_hits:
            with _CACHE_LOCK:
                for c, row_dict in db_hits.items():
                    _SECURITY_CACHE[c] = row_dict.copy()
                    collected[c] = row_dict.copy()
            missing = [c for c in missing if c not in db_hits]

    # 3. 查 Demo 表兜底
    if missing:
        for c in list(missing):
            if c in demo_map:
                collected[c] = demo_map[c].copy()
                missing.remove(c)

    # 4. 仅对真正缺失的股票调用 iFinD API 差量拉取
    if missing:
        fetched_df = _fetch_security_data(missing)
        with _CACHE_LOCK:
            for _, r in fetched_df.iterrows():
                c = str(r["sec_code"])
                row_dict = r.to_dict()
                _SECURITY_CACHE[c] = row_dict.copy()
                collected[c] = row_dict.copy()

    ordered_rows = [collected[c] for c in norm_codes if c in collected]
    return pd.DataFrame(ordered_rows)


def calculate(payload):
    """执行 L2 -> L3 -> L4 组合保证金与 IA 全链路计算"""
    positions = payload.get("positions") or []
    if not positions:
        raise ValueError("组合不能为空")
    portfolio = pd.DataFrame(positions)
    required = {"sec_code", "sec_name", "shares", "price"}
    if not required.issubset(portfolio.columns):
        raise ValueError("持仓必须包含 sec_code、sec_name、shares、price")

    portfolio = concentration.consolidate_positions(portfolio, merge_duplicates=True)

    security = get_security_data(portfolio["sec_code"].astype(str).tolist())
    # 若持仓名称为空或等于代码，且第一层数据有中文名称，自动补全显示名
    sec_name_map = {
        str(r["sec_code"]): str(r["sec_name"])
        for _, r in security.iterrows()
        if pd.notna(r.get("sec_name")) and str(r.get("sec_name")).strip()
    }
    portfolio["sec_name"] = portfolio.apply(
        lambda r: sec_name_map.get(r["sec_code"], r["sec_name"])
        if (not r["sec_name"] or r["sec_name"] == r["sec_code"])
        else r["sec_name"],
        axis=1,
    )

    l1 = security.drop(columns=["sec_name", "close", "as_of_date"], errors="ignore")
    merged = portfolio.merge(l1, on="sec_code", how="left", validate="one_to_one")
    if merged["ind_sw_l1_name"].isna().any():
        bad = merged.loc[merged["ind_sw_l1_name"].isna(), "sec_code"].tolist()
        raise ValueError(f"第一层证券数据不包含: {bad}")
    l2, industry_df, l2_summary = concentration.calc_concentration(merged)
    l3 = margin.calculate_stock_margin(l2, payload.get("params") or {})
    summary = margin.calculate_portfolio_ia(l3)
    summary["top_industry"] = l2_summary.get("top_industry", "")
    summary["top_industry_weight"] = l2_summary.get("top_industry_weight", 0.0)
    summary["cc_micro_weight"] = l2_summary.get("cc_micro_weight", 0.0)
    return {
        "summary": summary,
        "rows": l3.to_dict(orient="records"),
        "industries": industry_df.to_dict(orient="records"),
        "as_of_date": str(security["as_of_date"].iloc[0]),
    }


def build_export(rows, summary, params):
    """构建包含『个股IA』与『组合IA』两个 Sheet 的专业格式化 Excel"""
    detail = pd.DataFrame(rows)
    columns = {
        "sec_code": "证券代码",
        "sec_name": "证券名称",
        "ind_sw_l1_name": "申万一级行业",
        "shares": "持仓数量",
        "price": "股价",
        "market_value": "单票市值",
        "weight_in_portfolio": "个股集中度",
        "industry_mv": "行业市值",
        "industry_weight": "行业集中度",
        "cc_micro_mv": "双创微盘市值",
        "cc_micro_weight": "双创微盘集中度",
        "total_share_ratio": "总股本持仓占比",
        "float_share_ratio": "流通股本持仓占比",
        "avg_amt_20d": "20日日均成交额",
        "vol_90d": "90日波动率",
        "stock_adj": "个股调整",
        "industry_adj": "行业调整",
        "vol_adj": "波动率调整",
        "star_adj": "双创微盘调整",
        "adjustment_sum": "调整项之和",
        "maintenance_im": "维持保证金率",
        "restriction1": "r1流动性超限",
        "restriction2": "r2个股集中度超限",
        "restriction3": "r3总股本超限",
        "restriction4": "r4流通股本超限",
        "restriction5": "r5行业超限",
        "restriction7": "r7双创微盘超限",
        "restr_mv": "restr_mv",
        "active_restriction": "主控限制项",
        "special_reason": "特殊状态",
        "rule_margin_requirement": "规则保证金",
        "upper_margin_requirement": "upper_im保证金",
        "margin_requirement": "最终保证金",
        "adjusted_ia": "调整后IA",
        "marginal_ia": "边际IA",
    }
    detail = detail[[c for c in columns if c in detail.columns]].rename(columns=columns)
    summary_items = [
        ("持仓数量", summary.get("n_positions", 0), "只"),
        ("组合总市值", summary.get("total_market_value", 0.0), "元"),
        ("正常层市值", summary.get("normal_market_value", 0.0), "元"),
        ("超限层市值（100%计收）", summary.get("restricted_market_value", 0.0), "元"),
        ("总保证金占用", summary.get("total_margin_requirement", 0.0), "元"),
        ("组合IA（总保证金/总市值）", summary.get("portfolio_ia", 0.0), "比例"),
        ("组合IA（个券IA市值加权）", summary.get("portfolio_ia_weighted", 0.0), "比例"),
        ("两种IA方法差额", summary.get("ia_method_difference", 0.0), "比例"),
    ]
    breakdown = summary.get("ia_breakdown") or {}
    if breakdown:
        summary_items.extend(
            [
                ("IA归因：基础保证金率贡献", breakdown.get("base_im_contrib", 0.0), "比例"),
                ("IA归因：个股集中度加点贡献", breakdown.get("stock_adj_contrib", 0.0), "比例"),
                ("IA归因：行业集中度加点贡献", breakdown.get("industry_adj_contrib", 0.0), "比例"),
                ("IA归因：90日波动率加点贡献", breakdown.get("vol_adj_contrib", 0.0), "比例"),
                ("IA归因：双创微盘加点贡献", breakdown.get("star_adj_contrib", 0.0), "比例"),
                ("IA归因：限制项100%超限贡献", breakdown.get("restriction_contrib", 0.0), "比例"),
                ("IA归因：upper_im规则修正", breakdown.get("upper_im_contrib", 0.0), "比例"),
            ]
        )
    summary_items.append(("计算参数", "以下为本次输入", ""))
    summary_items.extend(
        (f"参数：{key}", value, "") for key, value in (params or {}).items()
    )
    summary_df = pd.DataFrame(summary_items, columns=["项目", "数值", "单位/说明"])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        detail.to_excel(writer, sheet_name="个股IA", index=False)
        summary_df.to_excel(writer, sheet_name="组合IA", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for column in ws.columns:
                letter = column[0].column_letter
                width = min(
                    max(
                        len(str(c.value)) if c.value is not None else 0
                        for c in column
                    )
                    + 2,
                    42,
                )
                ws.column_dimensions[letter].width = max(width, 12)
        ws = writer.sheets["个股IA"]
        ws.auto_filter.ref = ws.dimensions
        percent_names = {
            "个股集中度",
            "行业集中度",
            "双创微盘集中度",
            "总股本持仓占比",
            "流通股本持仓占比",
            "90日波动率",
            "个股调整",
            "行业调整",
            "波动率调整",
            "双创微盘调整",
            "调整项之和",
            "维持保证金率",
            "调整后IA",
            "边际IA",
        }
        for cell in ws[1]:
            if cell.value in percent_names:
                for row_cells in ws.iter_cols(
                    min_col=cell.column, max_col=cell.column, min_row=2
                ):
                    for item in row_cells:
                        item.number_format = "0.00%"
    return output.getvalue()


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, value):
        self._send(
            status,
            json.dumps(_json_ready(value), ensure_ascii=False),
            "application/json; charset=utf-8",
        )

    def do_GET(self):
        if self.path in {"/", "/index.html"}:
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if self.path == "/api/demo":
            portfolio, security = load_demo()
            self._json(
                200,
                {
                    "positions": portfolio.to_dict(orient="records"),
                    "params": margin.default_margin_params(),
                    "standard_params": margin.standard_margin_params(),
                    "as_of_date": str(security["as_of_date"].iloc[0]),
                },
            )
            return
        self._json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path not in {"/api/calculate", "/api/upload", "/api/export"}:
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 15_000_000:
                raise ValueError("请求体为空或过大")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/api/upload":
                positions = parse_portfolio_file(
                    payload.get("filename") or "",
                    payload.get("content") or "",
                )
                self._json(200, {"positions": positions})
                return
            if self.path == "/api/export":
                content = build_export(
                    payload.get("rows") or [],
                    payload.get("summary") or {},
                    payload.get("params") or {},
                )
                self._send(
                    200,
                    content,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                return
            self._json(200, calculate(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description="IA L2-L4 本地计算页面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"IA calculator: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
