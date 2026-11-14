import base64
import io
import os
import tempfile

import pandas as pd

from src import db, webapp


def _encoded_csv():
    raw = (webapp.ROOT / "data" / "demo_portfolio.csv").read_bytes()
    return base64.b64encode(raw).decode("ascii")


def test_parse_uploaded_csv():
    rows = webapp.parse_portfolio_file("demo_portfolio.csv", _encoded_csv())
    assert len(rows) == 8
    assert {"sec_code", "sec_name", "shares", "price"} == set(rows[0])


def test_parse_uploaded_xlsx():
    source = io.BytesIO()
    pd.DataFrame(
        [
            {"sec_code": "600519.SH", "shares": 100, "price": 1444.99},
        ]
    ).to_excel(source, index=False)
    encoded = base64.b64encode(source.getvalue()).decode("ascii")
    rows = webapp.parse_portfolio_file("portfolio.xlsx", encoded)
    assert rows[0]["sec_code"] == "600519.SH"
    assert rows[0]["sec_name"] == "600519.SH"


def test_parse_uploaded_csv_with_duplicate_and_bare_code():
    """上传 CSV 包含无后缀代码及同名同向多笔持仓时自动规范化并合并"""
    csv_text = "sec_code,shares,price\n600519,100,1400\n600519.SH,100,1600\n"
    encoded = base64.b64encode(csv_text.encode("utf-8")).decode("ascii")
    rows = webapp.parse_portfolio_file("dup.csv", encoded)
    assert len(rows) == 1
    assert rows[0]["sec_code"] == "600519.SH"
    assert rows[0]["shares"] == 200
    assert rows[0]["price"] == 1500.0


def test_l1_db_lookup_in_webapp(monkeypatch):
    """验证 WebApp 优先从本地 L1 SQLite 快照表读取非 Demo 股票，无需发起网络请求"""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    os.unlink(path)
    db.init_db(path)
    try:
        snap = pd.DataFrame(
            [
                {
                    "trade_date": "2026-11-09",
                    "sec_code": "600036.SH",
                    "sec_name": "招商银行",
                    "board": "主板",
                    "ind_sw_l1_code": "S48",
                    "ind_sw_l1_name": "银行",
                    "total_shares": 25219845601.0,
                    "total_float_shares": 20628944429.0,
                    "avg_amt_20d": 3500000000.0,
                    "vol_90d": 0.25,
                    "is_sse50": 1,
                    "is_csi300": 1,
                    "is_csi500": 0,
                    "is_csi1000": 0,
                    "is_cc_micro": 0,
                    "is_suspended": 0,
                    "is_risk_warning": 0,
                }
            ]
        )
        db.save_snapshot(snap, path)
        monkeypatch.setattr(webapp, "DB_PATH", webapp.Path(path))
        with webapp._CACHE_LOCK:
            webapp._SECURITY_CACHE.pop("600036.SH", None)

        res = webapp.calculate(
            {
                "positions": [
                    {
                        "sec_code": "600036",
                        "sec_name": "",
                        "shares": 10000,
                        "price": 38.0,
                    }
                ],
                "params": {},
            }
        )
        assert res["summary"]["n_positions"] == 1
        assert res["rows"][0]["sec_name"] == "招商银行"
        assert res["rows"][0]["ind_sw_l1_name"] == "银行"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_export_has_stock_and_portfolio_sheets():
    portfolio, _ = webapp.load_demo()
    result = webapp.calculate(
        {
            "positions": portfolio.to_dict(orient="records"),
            "params": {},
        }
    )
    content = webapp.build_export(result["rows"], result["summary"], {})
    xls = pd.ExcelFile(io.BytesIO(content))
    assert xls.sheet_names == ["个股IA", "组合IA"]
    stocks = pd.read_excel(xls, sheet_name="个股IA")
    portfolio = pd.read_excel(xls, sheet_name="组合IA")
    assert len(stocks) == 8
    assert "调整后IA" in stocks.columns
    assert "边际IA" in stocks.columns
    assert "主控限制项" in stocks.columns
    assert "两种IA方法差额" in portfolio["项目"].tolist()
