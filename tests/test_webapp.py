import base64
import io

import pandas as pd

from src import webapp


def _encoded_csv():
    raw = (webapp.ROOT / "data" / "demo_portfolio.csv").read_bytes()
    return base64.b64encode(raw).decode("ascii")


def test_parse_uploaded_csv():
    rows = webapp.parse_portfolio_file("demo_portfolio.csv", _encoded_csv())
    assert len(rows) == 8
    assert {"sec_code", "sec_name", "shares", "price"} == set(rows[0])


def test_parse_uploaded_xlsx():
    source = io.BytesIO()
    pd.DataFrame([
        {"sec_code": "600519.SH", "shares": 100, "price": 1444.99},
    ]).to_excel(source, index=False)
    encoded = base64.b64encode(source.getvalue()).decode("ascii")
    rows = webapp.parse_portfolio_file("portfolio.xlsx", encoded)
    assert rows[0]["sec_code"] == "600519.SH"
    assert rows[0]["sec_name"] == "600519.SH"


def test_export_has_stock_and_portfolio_sheets():
    portfolio, _ = webapp.load_demo()
    result = webapp.calculate({
        "positions": portfolio.to_dict(orient="records"), "params": {},
    })
    content = webapp.build_export(result["rows"], result["summary"], {})
    xls = pd.ExcelFile(io.BytesIO(content))
    assert xls.sheet_names == ["个股IA", "组合IA"]
    stocks = pd.read_excel(xls, sheet_name="个股IA")
    portfolio = pd.read_excel(xls, sheet_name="组合IA")
    assert len(stocks) == 8
    assert "调整后IA" in stocks.columns
    assert "两种IA方法差额" in portfolio["项目"].tolist()
