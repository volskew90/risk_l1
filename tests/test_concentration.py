"""
L2 组合集中度单元测试
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from src import concentration, db


@pytest.fixture()
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    os.unlink(path)
    db.init_db(path)
    yield path
    os.unlink(path)


def _make_snapshot(path):
    snap = pd.DataFrame(
        [
            {
                "trade_date": "2026-09-24",
                "sec_code": "600519.SH",
                "sec_name": "贵州茅台",
                "ind_sw_l1_code": "801120",
                "ind_sw_l1_name": "食品饮料",
                "is_cc_micro": 0,
                "total_shares": 1000000.0,
                "total_float_shares": 800000.0,
                "avg_amt_20d": 50000000.0,
                "is_sse50": 1,
            },
            {
                "trade_date": "2026-09-24",
                "sec_code": "000858.SZ",
                "sec_name": "五粮液",
                "ind_sw_l1_code": "801120",
                "ind_sw_l1_name": "食品饮料",
                "is_cc_micro": 0,
                "total_shares": 2000000.0,
                "total_float_shares": 1500000.0,
                "avg_amt_20d": 40000000.0,
                "is_sse50": 0,
            },
            {
                "trade_date": "2026-09-24",
                "sec_code": "601318.SH",
                "sec_name": "中国平安",
                "ind_sw_l1_code": "801790",
                "ind_sw_l1_name": "非银金融",
                "is_cc_micro": 0,
                "total_shares": 10000000.0,
                "total_float_shares": 8000000.0,
                "avg_amt_20d": 60000000.0,
                "is_sse50": 1,
            },
            {
                "trade_date": "2026-09-24",
                "sec_code": "300001.SZ",
                "sec_name": "特锐德",
                "ind_sw_l1_code": "801730",
                "ind_sw_l1_name": "电力设备",
                "is_cc_micro": 1,
                "total_shares": 1000000.0,
                "total_float_shares": 500000.0,
                "avg_amt_20d": 20000000.0,
                "is_sse50": 0,
                "is_suspended": 1,
                "is_risk_warning": 0,
            },
        ]
    )
    db.save_snapshot(snap, path)


def test_load_portfolio(tmp_path):
    f = tmp_path / "p.csv"
    f.write_text(
        "sec_code,sec_name,shares,price\n600519.SH,贵州茅台,100,1500.0\n",
        encoding="utf-8",
    )
    df = concentration.load_portfolio(str(f))
    assert df.iloc[0]["market_value"] == 150000.0

    short = tmp_path / "short.csv"
    short.write_text(
        "sec_code,sec_name,shares,price\n600519.SH,贵州茅台,-100,1500.0\n",
        encoding="utf-8",
    )
    df = concentration.load_portfolio(str(short))
    assert df.iloc[0]["abs_shares"] == 100.0
    assert df.iloc[0]["market_value"] == 150000.0

    mixed = tmp_path / "mixed.csv"
    mixed.write_text(
        "sec_code,sec_name,shares,price\n"
        "600519.SH,贵州茅台,100,1500.0\n"
        "000858.SZ,五粮液,-100,100.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        concentration.load_portfolio(str(mixed))


def test_normalize_code_and_consolidate_duplicates(tmp_path):
    """6位无后缀代码自动补齐 .SH/.SZ，且同向重复代码自动合并轧差"""
    dup_file = tmp_path / "dup.csv"
    dup_file.write_text(
        "sec_code,sec_name,shares,price\n"
        "600519,贵州茅台,100,1400.0\n"
        "600519.sh,,200,1550.0\n"
        "000858,五粮液,500,80.0\n",
        encoding="utf-8",
    )
    df = concentration.load_portfolio(str(dup_file))
    assert len(df) == 2
    mt = df.set_index("sec_code").loc["600519.SH"]
    assert mt["sec_name"] == "贵州茅台"
    assert mt["shares"] == 300
    assert mt["market_value"] == 100 * 1400.0 + 200 * 1550.0
    assert mt["price"] == pytest.approx(1500.0)
    assert "000858.SZ" in df["sec_code"].values


def test_concentration_weights(tmp_db):
    _make_snapshot(tmp_db)
    portfolio = pd.DataFrame(
        [
            {
                "sec_code": "600519.SH",
                "sec_name": "贵州茅台",
                "shares": 100,
                "price": 1500.0,
                "market_value": 150000.0,
            },
            {
                "sec_code": "000858.SZ",
                "sec_name": "五粮液",
                "shares": 1000,
                "price": 100.0,
                "market_value": 100000.0,
            },
            {
                "sec_code": "601318.SH",
                "sec_name": "中国平安",
                "shares": 1000,
                "price": 50.0,
                "market_value": 50000.0,
            },
            # 不在快照里的股票 -> 未知行业
            {
                "sec_code": "999999.XH",
                "sec_name": "测试股",
                "shares": 100,
                "price": 500.0,
                "market_value": 50000.0,
            },
        ]
    )
    df = concentration.attach_industry(portfolio, tmp_db, "2026-09-24")
    df, industry, summary = concentration.calc_concentration(df)

    # 权重合计 = 1
    assert abs(df["weight_in_portfolio"].sum() - 1) < 1e-9
    assert abs(industry["weight_in_portfolio"].sum() - 1) < 1e-9

    # 食品饮料合计 25 万 / 35 万
    food = industry.set_index("ind_sw_l1_name").loc["食品饮料"]
    assert abs(food["weight_in_portfolio"] - 250000 / 350000) < 1e-9

    # 行业内占比：五粮液占食品饮料 40%
    wly = df.set_index("sec_code").loc["000858.SZ"]
    assert abs(wly["weight_in_industry"] - 100000 / 250000) < 1e-9

    # 行业内占比合计 = 1（逐行业验证）
    for ind_name, grp in df.groupby("ind_sw_l1_name"):
        assert abs(grp["weight_in_industry"].sum() - 1) < 1e-9, ind_name

    # 未知行业兜底（is_cc_micro 也按 0 处理，不贡献双创微盘占比）
    assert df.set_index("sec_code").loc["999999.XH", "ind_sw_l1_name"] == "未知"
    assert df.set_index("sec_code").loc["999999.XH", "is_cc_micro"] == 0

    # 摘要：第一大行业 = 食品饮料，HHI 正确
    assert summary["top_industry"] == "食品饮料"
    expected_hhi = sum((w / 350000) ** 2 for w in (250000, 50000, 50000))
    assert abs(summary["industry_hhi"] - expected_hhi) < 1e-9


def test_cc_micro_weight(tmp_db):
    """双创微盘占比：组合内双创微盘股的市值占比"""
    _make_snapshot(tmp_db)
    portfolio = pd.DataFrame(
        [
            {
                "sec_code": "600519.SH",
                "sec_name": "贵州茅台",
                "shares": 100,
                "price": 1500.0,
                "market_value": 150000.0,
            },
            {
                "sec_code": "601318.SH",
                "sec_name": "中国平安",
                "shares": 1000,
                "price": 50.0,
                "market_value": 50000.0,
            },
            # 快照中标记 is_cc_micro=1
            {
                "sec_code": "300001.SZ",
                "sec_name": "特锐德",
                "shares": 1000,
                "price": 100.0,
                "market_value": 100000.0,
            },
        ]
    )
    df = concentration.attach_industry(portfolio, tmp_db, "2026-09-24")
    df, industry, summary = concentration.calc_concentration(df)

    assert summary["cc_micro_market_value"] == 100000.0
    assert abs(summary["cc_micro_weight"] - 100000 / 300000) < 1e-9


def test_restriction_inputs(tmp_db):
    """第二层输出第三层限制项所需的全部中间量，但不使用阈值"""
    _make_snapshot(tmp_db)
    portfolio = pd.DataFrame(
        [
            {
                "sec_code": "600519.SH",
                "sec_name": "贵州茅台",
                "shares": 100,
                "price": 1500.0,
                "market_value": 150000.0,
            },
            {
                "sec_code": "601318.SH",
                "sec_name": "中国平安",
                "shares": 1000,
                "price": 50.0,
                "market_value": 50000.0,
            },
            {
                "sec_code": "300001.SZ",
                "sec_name": "特锐德",
                "shares": 1000,
                "price": 100.0,
                "market_value": 100000.0,
            },
        ]
    )
    df = concentration.attach_industry(portfolio, tmp_db, "2026-09-24")
    df, _, _ = concentration.calc_concentration(df)
    row = df.set_index("sec_code").loc["300001.SZ"]

    assert row["total_mv"] == 300000.0
    assert row["industry_mv"] == 100000.0
    assert abs(row["industry_weight"] - 100000 / 300000) < 1e-9
    assert row["cc_micro_mv"] == 100000.0
    assert row["weight_in_cc_micro"] == 1.0
    assert row["total_share_ratio"] == 1000 / 1000000
    assert row["float_share_ratio"] == 1000 / 500000
    assert row["avg_amt_20d"] == 20000000.0
    assert row["is_sse50"] == 0
    assert row["is_suspended"] == 1
    assert not any(c.startswith("restriction") for c in df.columns)


def test_latest_snapshot_date_default(tmp_db):
    _make_snapshot(tmp_db)
    portfolio = pd.DataFrame(
        [
            {
                "sec_code": "601318.SH",
                "sec_name": "中国平安",
                "shares": 100,
                "price": 50.0,
                "market_value": 5000.0,
            },
        ]
    )
    df = concentration.attach_industry(portfolio, tmp_db)  # 不指定日期 -> 最新
    assert df.iloc[0]["ind_sw_l1_name"] == "非银金融"
    assert df.iloc[0]["snapshot_date"] == "2026-09-24"


if __name__ == "__main__":
    import pytest as _pt

    raise SystemExit(_pt.main([__file__, "-q"]))
