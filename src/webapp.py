#!/usr/bin/env python3
"""本地 L2-L4 保证金计算页面，仅使用 Python 标准库提供 HTTP 服务。"""
import argparse
import base64
import binascii
import io
import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import calc_vol, concentration, db, margin
from src.fetch_ifind import IFindClient

PAGE = ROOT / "src" / "static" / "ia_calculator.html"
PORTFOLIO = ROOT / "data" / "demo_portfolio.csv"
SECURITY = ROOT / "data" / "demo_security_master.csv"
CONFIG = ROOT / "config.yaml"
DB_PATH = ROOT / "data" / "risk_l1.sqlite"
_SECURITY_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _json_ready(value):
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
    portfolio = pd.read_csv(PORTFOLIO, dtype={"sec_code": str})
    security = pd.read_csv(SECURITY, dtype={"sec_code": str})
    return portfolio, security


def parse_portfolio_file(filename, encoded):
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
    if "sec_name" not in df.columns:
        df["sec_name"] = df["sec_code"]
    df = df[["sec_code", "sec_name", "shares", "price"]].copy()
    df["sec_code"] = df["sec_code"].astype(str).str.strip()
    if df["sec_code"].duplicated().any():
        raise ValueError("组合文件存在重复证券代码")
    return df.to_dict(orient="records")


def _latest_trade_date():
    conn = db.get_conn(str(DB_PATH))
    try:
        row = conn.execute("SELECT MAX(trade_date) FROM security_master_daily").fetchone()
        if not row or not row[0]:
            raise RuntimeError("第一层数据库没有可用交易日")
        return row[0]
    finally:
        conn.close()


def _fetch_security_data(codes):
    trade_date = _latest_trade_date()
    with CONFIG.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    client = IFindClient(
        mode="http", refresh_token=config.get("ifind_refresh_token"),
        retry_times=int(config.get("retry_times", 3)),
        retry_backoff=float(config.get("retry_backoff", 2)),
    )
    basic = client.get_basic_data(codes, trade_date).set_index("sec_code")
    daily = client.get_daily_basic(codes, trade_date).set_index("sec_code")
    prices = client.get_daily_close(codes, trade_date, lookback=130)
    index_codes = {
        "is_sse50": "000016.SH", "is_csi300": "000300.SH",
        "is_csi500": "000905.SH", "is_csi1000": "000852.SH",
    }
    index_sets = {}
    for flag, index_code in index_codes.items():
        members = set(client.get_index_components(index_code, trade_date))
        if not members:
            raise RuntimeError(f"指数成分为空: {index_code}")
        index_sets[flag] = members
    rows = []
    for code in codes:
        if code not in basic.index or code not in daily.index:
            raise RuntimeError(f"iFinD 缺少证券基础数据: {code}")
        series = prices.get(code, [])
        vols, ns, _ = calc_vol.calc_volatility(series, windows=[90])
        valid_prices = [float(v) for v in series if pd.notna(v) and float(v) > 0]
        if ns.get(90) != 90 or not valid_prices:
            raise RuntimeError(f"{code} 的90日行情数据不足")
        b, d = basic.loc[code], daily.loc[code]
        row = {
            "sec_code": code, "sec_name": code,
            "board": "科创板" if code.startswith("688") else
                     "创业板" if code.startswith(("300", "301")) else "主板",
            "ind_sw_l1_code": b.get("ind_sw_l1_code"),
            "ind_sw_l1_name": b.get("ind_sw_l1_name"),
            "total_shares": b.get("total_shares"),
            "total_float_shares": b.get("total_float_shares"),
            "avg_amt_20d": d.get("avg_amt_20d"),
            "vol_90d": vols[90], "close": valid_prices[-1],
            "is_cc_micro": 0, "as_of_date": trade_date,
        }
        for flag, members in index_sets.items():
            row[flag] = int(code in members)
        row["is_cc_micro"] = int(not (
            row["is_csi300"] or row["is_csi500"] or row["is_csi1000"]
        ))
        rows.append(row)
    security = pd.DataFrame(rows)
    required = ["ind_sw_l1_name", "total_shares", "total_float_shares", "avg_amt_20d"]
    if security[required].isna().any().any():
        raise RuntimeError("上传组合存在未补齐的第一层数据")
    return security


def get_security_data(codes):
    codes = sorted(set(codes))
    _, demo_security = load_demo()
    if set(codes).issubset(set(demo_security["sec_code"])):
        return demo_security[demo_security["sec_code"].isin(codes)].copy()
    key = tuple(codes)
    with _CACHE_LOCK:
        cached = _SECURITY_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    security = _fetch_security_data(codes)
    with _CACHE_LOCK:
        _SECURITY_CACHE[key] = security.copy()
    return security


def calculate(payload):
    positions = payload.get("positions") or []
    if not positions:
        raise ValueError("组合不能为空")
    portfolio = pd.DataFrame(positions)
    required = {"sec_code", "sec_name", "shares", "price"}
    if not required.issubset(portfolio.columns):
        raise ValueError("持仓必须包含 sec_code、sec_name、shares、price")
    if portfolio["sec_code"].duplicated().any():
        raise ValueError("组合存在重复证券代码")
    portfolio["shares"] = pd.to_numeric(portfolio["shares"], errors="coerce")
    portfolio["price"] = pd.to_numeric(portfolio["price"], errors="coerce")
    if portfolio[["shares", "price"]].isna().any().any():
        raise ValueError("股数和股价必须为数值")
    if (portfolio["shares"] == 0).any() or (portfolio["price"] <= 0).any():
        raise ValueError("股数不能为 0，股价必须大于 0")
    if (portfolio["shares"] > 0).any() and (portfolio["shares"] < 0).any():
        raise ValueError("仅支持纯多头或纯空头组合")
    portfolio["abs_shares"] = portfolio["shares"].abs()
    portfolio["market_value"] = portfolio["abs_shares"] * portfolio["price"]

    security = get_security_data(portfolio["sec_code"].astype(str).tolist())
    l1 = security.drop(columns=["sec_name", "close", "as_of_date"], errors="ignore")
    merged = portfolio.merge(l1, on="sec_code", how="left", validate="one_to_one")
    if merged["ind_sw_l1_name"].isna().any():
        bad = merged.loc[merged["ind_sw_l1_name"].isna(), "sec_code"].tolist()
        raise ValueError(f"第一层证券数据不包含: {bad}")
    l2, _, _ = concentration.calc_concentration(merged)
    l3 = margin.calculate_stock_margin(l2, payload.get("params") or {})
    summary = margin.calculate_portfolio_ia(l3)
    return {
        "summary": summary,
        "rows": l3.to_dict(orient="records"),
        "as_of_date": str(security["as_of_date"].iloc[0]),
    }


def build_export(rows, summary, params):
    detail = pd.DataFrame(rows)
    columns = {
        "sec_code": "证券代码", "sec_name": "证券名称", "shares": "持仓数量",
        "price": "股价", "market_value": "单票市值", "weight_in_portfolio": "个股集中度",
        "industry_mv": "行业市值", "industry_weight": "行业集中度",
        "cc_micro_mv": "双创微盘市值", "cc_micro_weight": "双创微盘集中度",
        "total_share_ratio": "总股本持仓占比", "float_share_ratio": "流通股本持仓占比",
        "avg_amt_20d": "20日日均成交额", "vol_90d": "90日波动率",
        "stock_adj": "个股调整", "industry_adj": "行业调整", "vol_adj": "波动率调整",
        "star_adj": "双创微盘调整", "adjustment_sum": "调整项之和",
        "maintenance_im": "维持保证金率", "restriction1": "r1流动性超限",
        "restriction2": "r2个股集中度超限", "restriction3": "r3总股本超限",
        "restriction4": "r4流通股本超限", "restriction5": "r5行业超限",
        "restriction7": "r7双创微盘超限", "restr_mv": "restr_mv",
        "rule_margin_requirement": "规则保证金", "upper_margin_requirement": "upper_im保证金",
        "margin_requirement": "最终保证金", "adjusted_ia": "调整后IA",
    }
    detail = detail[[c for c in columns if c in detail.columns]].rename(columns=columns)
    summary_items = [
        ("持仓数量", summary["n_positions"], "只"),
        ("组合总市值", summary["total_market_value"], "元"),
        ("总保证金占用", summary["total_margin_requirement"], "元"),
        ("组合IA（总保证金/总市值）", summary["portfolio_ia"], "比例"),
        ("组合IA（个券IA市值加权）", summary["portfolio_ia_weighted"], "比例"),
        ("两种IA方法差额", summary["ia_method_difference"], "比例"),
    ]
    summary_items.append(("计算参数", "以下为本次输入", ""))
    summary_items.extend((f"参数：{key}", value, "") for key, value in (params or {}).items())
    summary_df = pd.DataFrame(summary_items, columns=["项目", "数值", "单位/说明"])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        detail.to_excel(writer, sheet_name="个股IA", index=False)
        summary_df.to_excel(writer, sheet_name="组合IA", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for column in ws.columns:
                letter = column[0].column_letter
                width = min(max(len(str(c.value)) if c.value is not None else 0 for c in column) + 2, 42)
                ws.column_dimensions[letter].width = max(width, 12)
        ws = writer.sheets["个股IA"]
        ws.auto_filter.ref = ws.dimensions
        percent_names = {"个股集中度", "行业集中度", "双创微盘集中度", "总股本持仓占比",
                         "流通股本持仓占比", "90日波动率", "个股调整", "行业调整", "波动率调整",
                         "双创微盘调整", "调整项之和", "维持保证金率", "调整后IA"}
        for cell in ws[1]:
            if cell.value in percent_names:
                for row_cells in ws.iter_cols(min_col=cell.column, max_col=cell.column, min_row=2):
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
        self._send(status, json.dumps(_json_ready(value), ensure_ascii=False),
                   "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in {"/", "/index.html"}:
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if self.path == "/api/demo":
            portfolio, security = load_demo()
            self._json(200, {
                "positions": portfolio.to_dict(orient="records"),
                "params": margin.default_margin_params(),
                "as_of_date": str(security["as_of_date"].iloc[0]),
            })
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
                    payload.get("filename") or "", payload.get("content") or "",
                )
                self._json(200, {"positions": positions})
                return
            if self.path == "/api/export":
                content = build_export(
                    payload.get("rows") or [], payload.get("summary") or {},
                    payload.get("params") or {},
                )
                self._send(
                    200, content,
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
