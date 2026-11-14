"""
L3 单券保证金与 L4 组合 IA 单元测试
包含原有单元测试以及《IA Sample.html》标准测试案例（TC01~TC13, TC16）验收测试。
"""
import numpy as np
import pandas as pd
import pytest

from src import margin


def _row(**overrides):
    row = {
        "sec_code": "600000.SH",
        "price": 10.0,
        "abs_shares": 1000.0,
        "market_value": 10000.0,
        "total_mv": 100000.0,
        "weight_in_portfolio": 0.1,
        "industry_mv": 20000.0,
        "industry_weight": 0.2,
        "weight_in_industry": 0.5,
        "cc_micro_mv": 0.0,
        "cc_micro_weight": 0.0,
        "weight_in_cc_micro": 0.0,
        "total_shares": 1000000.0,
        "total_float_shares": 900000.0,
        "avg_amt_20d": 1000000.0,
        "vol_90d": 0.3,
        "is_sse50": 0,
        "is_csi300": 1,
        "is_cc_micro": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _params(**overrides):
    params = {
        "upper_position_limit": 1,
        "position_weight_limit": 1.0,
    }
    params.update(overrides)
    return params


def _std_params(**overrides):
    """标准业务规则参数（用于 TC01~TC16 验收）"""
    p = margin.standard_margin_params()
    p["upper_position_limit"] = 1
    p.update(overrides)
    return p


def test_base_margin_without_adjustments_or_restrictions():
    result = margin.calculate_stock_margin(_row(), _params()).iloc[0]
    assert result["maintenance_im"] == 0.25
    assert result["restr_mv"] == 0.0
    assert result["margin_requirement"] == 2500.0
    assert result["adjusted_ia"] == 0.25
    assert result["active_restriction"] == "无"


def test_photo_volatility_adjustment():
    result = margin.calculate_stock_margin(
        _row(vol_90d=0.6621),
        _params(vol_adj_ratio=0.7),
    ).iloc[0]
    assert result["vol_adj"] == 0.0435
    assert result["maintenance_im"] == 0.2935
    assert result["margin_requirement"] == 2935.0


def test_restrictions_take_max_not_sum():
    result = margin.calculate_stock_margin(
        _row(avg_amt_20d=1000.0),
        _params(position_weight_limit=0.01, liquidity_threshold=1.0),
    ).iloc[0]
    assert result["restriction1"] == 9000.0
    assert result["restriction2"] == 9000.0
    assert result["restr_mv"] == 9000.0
    assert result["margin_requirement"] == 9250.0
    assert "r1:流动性" in result["active_restriction"]
    assert "r2:个股集中度" in result["active_restriction"]


def test_sse50_is_exempt_from_position_weight_restriction():
    result = margin.calculate_stock_margin(
        _row(is_sse50=1),
        _params(position_weight_limit=0.01),
    ).iloc[0]
    assert result["restriction2"] == 0.0


def test_share_restrictions_use_share_units():
    result = margin.calculate_stock_margin(
        _row(abs_shares=20000.0, market_value=200000.0),
        _params(),
    ).iloc[0]
    assert result["restriction3"] == 100000.0
    assert result["restriction4"] == 20000.0


@pytest.mark.parametrize("compare,expected", [("max", 8000.0), ("min", 2500.0)])
def test_upper_im_comparison(compare, expected):
    result = margin.calculate_stock_margin(
        _row(),
        _params(upper_position_limit=20, upper_compare=compare),
    ).iloc[0]
    assert result["rule_margin_requirement"] == 2500.0
    assert result["upper_margin_requirement"] == 8000.0
    assert result["margin_requirement"] == expected


def test_portfolio_ia_methods_are_identical():
    one = _row()
    two = _row(
        sec_code="000001.SZ",
        market_value=30000.0,
        total_mv=100000.0,
        weight_in_portfolio=0.3,
        industry_mv=50000.0,
        industry_weight=0.5,
        weight_in_industry=0.6,
        vol_90d=0.7,
    )
    result = margin.calculate_stock_margin(
        pd.concat([one, two], ignore_index=True),
        _params(),
    )
    summary = margin.calculate_portfolio_ia(result)
    assert summary["total_market_value"] == 40000.0
    assert summary["portfolio_ia"] == pytest.approx(summary["portfolio_ia_weighted"])
    assert summary["ia_method_difference"] == pytest.approx(0.0)
    assert "ia_breakdown" in summary


# ===========================================================================
# 《IA Sample.html》标准测试案例验收（TC01 ~ TC13, TC16）
# ===========================================================================


def test_tc01_normal_stock_zero_trigger():
    """TC01: 常规标的·零触发 -> 维持保证金率 25.00%，保证金 125.00"""
    df = _row(
        sec_code="600000.SH",
        market_value=500.0,
        total_mv=10000.0,
        weight_in_portfolio=0.05,
        industry_mv=800.0,
        industry_weight=0.08,
        weight_in_industry=0.625,
        vol_90d=0.30,
    )
    res = margin.calculate_stock_margin(df, _std_params()).iloc[0]
    assert res["maintenance_im"] == pytest.approx(0.25)
    assert res["margin_requirement"] == pytest.approx(125.00, abs=0.01)


def test_tc02_stock_concentration_non_dual():
    """TC02: 个股集中度（非双创，阈值 10%，比例 0.7）-> 保证金率 28.50%，保证金 427.50"""
    df = _row(
        sec_code="600000.SH",
        market_value=1500.0,
        total_mv=10000.0,
        weight_in_portfolio=0.15,
        industry_mv=900.0,
        industry_weight=0.09,
        vol_90d=0.35,
    )
    res = margin.calculate_stock_margin(df, _std_params()).iloc[0]
    assert res["stock_adj"] == pytest.approx(0.035)
    assert res["maintenance_im"] == pytest.approx(0.285)
    assert res["margin_requirement"] == pytest.approx(427.50, abs=0.01)


def test_tc03_stock_concentration_dual_non300():
    """TC03: 个股集中度（双创非沪深300，阈值 5%，比例 0.7）-> 保证金率 32.00%，保证金 480.00"""
    df = _row(
        sec_code="688001.SH",
        is_csi300=0,
        market_value=1500.0,
        total_mv=10000.0,
        weight_in_portfolio=0.15,
        industry_mv=900.0,
        industry_weight=0.09,
        vol_90d=0.35,
    )
    res = margin.calculate_stock_margin(df, _std_params()).iloc[0]
    assert res["stock_adj"] == pytest.approx(0.07)
    assert res["maintenance_im"] == pytest.approx(0.32)
    assert res["margin_requirement"] == pytest.approx(480.00, abs=0.01)


def test_tc04_industry_concentration():
    """TC04: 行业集中度（阈值 20%，比例 0.4）-> 保证金率 32.50%，保证金 487.50"""
    df = _row(
        sec_code="600000.SH",
        market_value=1500.0,
        total_mv=10000.0,
        weight_in_portfolio=0.15,
        industry_mv=3000.0,
        industry_weight=0.30,
        weight_in_industry=0.50,
        vol_90d=0.35,
    )
    res = margin.calculate_stock_margin(df, _std_params()).iloc[0]
    assert res["industry_adj"] == pytest.approx(0.04)
    assert res["maintenance_im"] == pytest.approx(0.325)
    assert res["margin_requirement"] == pytest.approx(487.50, abs=0.01)


def test_tc05_volatility_adjustment():
    """TC05: 90日波动率（阈值 60%，比例 0.4）-> 保证金率 38.50%，保证金 577.50"""
    df = _row(
        sec_code="600000.SH",
        market_value=1500.0,
        total_mv=10000.0,
        weight_in_portfolio=0.15,
        industry_mv=900.0,
        industry_weight=0.09,
        vol_90d=0.85,
    )
    res = margin.calculate_stock_margin(df, _std_params()).iloc[0]
    assert res["vol_adj"] == pytest.approx(0.10)
    assert res["maintenance_im"] == pytest.approx(0.385)
    assert res["margin_requirement"] == pytest.approx(577.50, abs=0.01)


def test_tc06_cc_micro_concentration():
    """TC06: 双创/微盘板块集中度（阈值 30%，比例 0.4）-> 保证金率 33.30%，保证金 499.50"""
    df = _row(
        sec_code="002001.SZ",
        is_csi300=0,
        is_cc_micro=1,
        market_value=1500.0,
        total_mv=10000.0,
        weight_in_portfolio=0.15,
        industry_mv=900.0,
        industry_weight=0.09,
        cc_micro_mv=4200.0,
        cc_micro_weight=0.42,
        weight_in_cc_micro=1500.0 / 4200.0,
        vol_90d=0.35,
    )
    res = margin.calculate_stock_margin(df, _std_params()).iloc[0]
    assert res["star_adj"] == pytest.approx(0.048)
    assert res["maintenance_im"] == pytest.approx(0.333)
    assert res["margin_requirement"] == pytest.approx(499.50, abs=0.01)


def test_tc07_four_adjustments_combined():
    """TC07: 四项调整叠加 -> 维持保证金率 59.50%，保证金 1190.00"""
    df = _row(
        sec_code="300001.SZ",
        is_csi300=0,
        is_cc_micro=1,
        market_value=2000.0,
        total_mv=10000.0,
        weight_in_portfolio=0.20,
        industry_mv=3500.0,
        industry_weight=0.35,
        weight_in_industry=2000.0 / 3500.0,
        cc_micro_mv=4500.0,
        cc_micro_weight=0.45,
        weight_in_cc_micro=2000.0 / 4500.0,
        vol_90d=0.90,
    )
    res = margin.calculate_stock_margin(df, _std_params()).iloc[0]
    assert res["adjustment_sum"] == pytest.approx(0.345)
    assert res["maintenance_im"] == pytest.approx(0.595)
    assert res["margin_requirement"] == pytest.approx(1190.00, abs=0.01)


def test_tc08_maintenance_im_cap_100pct():
    """TC08: 加点封顶 100% -> 维持保证金率 100%，保证金 3000.00"""
    df = _row(
        sec_code="688001.SH",
        is_csi300=0,
        is_cc_micro=1,
        market_value=3000.0,
        total_mv=10000.0,
        weight_in_portfolio=0.30,
        industry_mv=6000.0,
        industry_weight=0.60,
        weight_in_industry=0.50,
        cc_micro_mv=6000.0,
        cc_micro_weight=0.60,
        weight_in_cc_micro=0.50,
        vol_90d=1.50,
    )
    res = margin.calculate_stock_margin(df, _std_params()).iloc[0]
    assert res["maintenance_im"] == pytest.approx(1.0)
    assert res["margin_requirement"] == pytest.approx(3000.00, abs=0.01)


def test_tc09_restriction1_liquidity():
    """TC09: 限制项 r1 流动性超限 -> restr_mv=700, 保证金=928.00"""
    df = _row(
        sec_code="600000.SH",
        market_value=1500.0,
        total_mv=10000.0,
        weight_in_portfolio=0.15,
        industry_mv=900.0,
        industry_weight=0.09,
        avg_amt_20d=800.0,
        vol_90d=0.35,
    )
    res = margin.calculate_stock_margin(df, _std_params()).iloc[0]
    assert res["restriction1"] == pytest.approx(700.0)
    assert res["restr_mv"] == pytest.approx(700.0)
    assert res["margin_requirement"] == pytest.approx(928.00, abs=0.01)


def test_tc10_restriction2_and_5_take_max():
    """TC10: 限制项 r2 与 r5 取 MAX 而非相加 -> restr_mv=1090.91, 保证金=2836.36"""
    df = _row(
        sec_code="600000.SH",
        market_value=4000.0,
        total_mv=10000.0,
        weight_in_portfolio=0.40,
        industry_mv=5500.0,
        industry_weight=0.55,
        weight_in_industry=4000.0 / 5500.0,
        vol_90d=0.35,
    )
    res = margin.calculate_stock_margin(df, _std_params()).iloc[0]
    assert res["maintenance_im"] == pytest.approx(0.60)
    assert res["restriction2"] == pytest.approx(1000.0)
    assert res["restriction5"] == pytest.approx(1090.9091, abs=0.01)
    assert res["restr_mv"] == pytest.approx(1090.9091, abs=0.01)
    assert res["margin_requirement"] == pytest.approx(2836.36, abs=0.01)


def test_tc12_restriction7_cc_micro():
    """TC12: 限制项 r7 微盘+双创超限 -> restr_mv=50.00, 保证金=225.00"""
    df = _row(
        sec_code="002001.SZ",
        is_csi300=0,
        is_cc_micro=1,
        market_value=550.0,
        total_mv=10000.0,
        weight_in_portfolio=0.055,
        industry_mv=550.0,
        industry_weight=0.055,
        weight_in_industry=1.0,
        cc_micro_mv=5500.0,
        cc_micro_weight=0.55,
        weight_in_cc_micro=0.10,
        vol_90d=0.35,
    )
    res = margin.calculate_stock_margin(df, _std_params()).iloc[0]
    assert res["maintenance_im"] == pytest.approx(0.35)
    assert res["restriction7"] == pytest.approx(50.0)
    assert res["margin_requirement"] == pytest.approx(225.00, abs=0.01)


def test_tc13_upper_im_dual_rules():
    """TC13: upper_im 双算法取数（规则1银河 vs 规则2东证）"""
    # 构造 6 只股票组合，其中第 1 只 A 票市值 3000，规则保证金率 60%（A = 1800），upper_im = 80%（B = 2400）
    rows = []
    for i in range(6):
        rows.append(
            _row(
                sec_code=f"60000{i}.SH",
                market_value=3000.0 if i == 0 else 1400.0,
                total_mv=10000.0,
                weight_in_portfolio=0.40 if i == 0 else 0.14,
                industry_mv=5500.0 if i == 0 else 1400.0,
                industry_weight=0.55 if i == 0 else 0.14,
                weight_in_industry=1.0,
                vol_90d=0.35,
                is_sse50=1,
            ).iloc[0]
        )
    df6 = pd.DataFrame(rows)

    # 规则 1：6 只 < 20 只 -> 取 MAX(1800, 2400) = 2400
    r1 = margin.calculate_stock_margin(
        df6,
        _std_params(
            upper_position_limit=20,
            upper_compare="max",
            industry_weight_limit=1.0,
        ),
    ).iloc[0]
    assert r1["rule_margin_requirement"] == pytest.approx(1800.0)
    assert r1["upper_margin_requirement"] == pytest.approx(2400.0)
    assert r1["margin_requirement"] == pytest.approx(2400.0)

    # 规则 2（东证）：6 只 >= 5 只 -> 取 MIN(1800, 2400) = 1800
    r2_many = margin.calculate_stock_margin(
        df6,
        _std_params(
            upper_position_limit=5,
            upper_compare="rule2",
            industry_weight_limit=1.0,
        ),
    ).iloc[0]
    assert r2_many["margin_requirement"] == pytest.approx(1800.0)

    # 规则 2（东证）：若组合仅 4 只 (< 5 只) -> 直接取 B = 2400
    r2_few = margin.calculate_stock_margin(
        df6.iloc[:4].copy(),
        _std_params(
            upper_position_limit=5,
            upper_compare="rule2",
            industry_weight_limit=1.0,
        ),
    ).iloc[0]
    assert r2_few["margin_requirement"] == pytest.approx(2400.0)


def test_tc16_special_stocks_100pct_fallback():
    """TC16: 停牌股、风险警示股(ST)、波动率缺失次新股自动按 100% 计收兜底，不崩溃"""
    df_susp = _row(market_value=1000.0, is_suspended=1)
    res_susp = margin.calculate_stock_margin(df_susp, _std_params()).iloc[0]
    assert res_susp["is_special_100"] == 1
    assert "停牌股" in res_susp["special_reason"]
    assert res_susp["maintenance_im"] == pytest.approx(1.0)
    assert res_susp["margin_requirement"] == pytest.approx(1000.0)

    df_st = _row(market_value=500.0, is_risk_warning=1)
    res_st = margin.calculate_stock_margin(df_st, _std_params()).iloc[0]
    assert res_st["is_special_100"] == 1
    assert "风险警示" in res_st["special_reason"]
    assert res_st["margin_requirement"] == pytest.approx(500.0)

    df_no_vol = _row(market_value=800.0, vol_90d=np.nan)
    res_no_vol = margin.calculate_stock_margin(df_no_vol, _std_params()).iloc[0]
    assert res_no_vol["is_special_100"] == 1
    assert "波动率缺失" in res_no_vol["special_reason"]
    assert res_no_vol["margin_requirement"] == pytest.approx(800.0)
