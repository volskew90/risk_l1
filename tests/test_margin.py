"""
L3 单券保证金与 L4 组合 IA 单元测试
包含原有单元测试以及新规《IA_Margin_Rule.html》全部 15 个标准测试案例（TC01~TC15，人民币元口径）验收测试。
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


def _rule_params(**overrides):
    """新规《IA_Margin_Rule.html》标准参数（用于 TC01~TC15 验收）"""
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
# 《IA_Margin_Rule.html》全部 15 个标准测试案例验收（TC01 ~ TC15，人民币元）
# ===========================================================================


def test_tc01_normal_stock_zero_trigger():
    """TC01: 常规标的·零触发 -> 市值 5,000,000，费率 25.00%，IA 1,250,000.00 元"""
    df = _row(
        sec_code="600000.SH",
        market_value=5_000_000.0,
        total_mv=100_000_000.0,
        weight_in_portfolio=0.05,
        industry_mv=8_000_000.0,
        industry_weight=0.08,
        weight_in_industry=0.625,
        avg_amt_20d=1_000_000_000.0,
        total_shares=1_000_000_000.0,
        total_float_shares=1_000_000_000.0,
        vol_90d=0.30,
    )
    res = margin.calculate_stock_margin(df, _rule_params()).iloc[0]
    assert res["maintenance_im"] == pytest.approx(0.25)
    assert res["margin_requirement"] == pytest.approx(1_250_000.00, abs=0.01)


def test_tc02_stock_concentration_non_dual():
    """TC02: 个股集中度（非双创，阈值 10%，比例 0.7，15%未触发r2）-> 费率 28.50%，IA 4,275,000.00 元"""
    df = _row(
        sec_code="600000.SH",
        market_value=15_000_000.0,
        total_mv=100_000_000.0,
        weight_in_portfolio=0.15,
        industry_mv=15_000_000.0,
        industry_weight=0.15,
        avg_amt_20d=1_000_000_000.0,
        total_shares=1_000_000_000.0,
        total_float_shares=1_000_000_000.0,
        vol_90d=0.35,
    )
    res = margin.calculate_stock_margin(df, _rule_params()).iloc[0]
    assert res["stock_adj"] == pytest.approx(0.035)
    assert res["restriction2"] == pytest.approx(0.0)
    assert res["maintenance_im"] == pytest.approx(0.285)
    assert res["margin_requirement"] == pytest.approx(4_275_000.00, abs=0.01)


def test_tc03_stock_concentration_dual_non300():
    """TC03: 个股集中度（双创非沪深300，阈值 5%，比例 0.7）-> 费率 32.00%，IA 4,800,000.00 元"""
    df = _row(
        sec_code="688001.SH",
        is_csi300=0,
        is_cc_micro=1,
        market_value=15_000_000.0,
        total_mv=100_000_000.0,
        weight_in_portfolio=0.15,
        industry_mv=15_000_000.0,
        industry_weight=0.15,
        cc_micro_mv=15_000_000.0,
        cc_micro_weight=0.15,
        weight_in_cc_micro=1.0,
        avg_amt_20d=1_000_000_000.0,
        total_shares=1_000_000_000.0,
        total_float_shares=1_000_000_000.0,
        vol_90d=0.35,
    )
    res = margin.calculate_stock_margin(df, _rule_params()).iloc[0]
    assert res["stock_adj"] == pytest.approx(0.07)
    assert res["maintenance_im"] == pytest.approx(0.32)
    assert res["margin_requirement"] == pytest.approx(4_800_000.00, abs=0.01)


def test_tc04_industry_concentration():
    """TC04: 行业集中度（阈值 20%，比例 0.4）-> 费率 32.50%，IA 4,875,000.00 元"""
    df = _row(
        sec_code="600000.SH",
        market_value=15_000_000.0,
        total_mv=100_000_000.0,
        weight_in_portfolio=0.15,
        industry_mv=30_000_000.0,
        industry_weight=0.30,
        weight_in_industry=0.50,
        avg_amt_20d=1_000_000_000.0,
        total_shares=1_000_000_000.0,
        total_float_shares=1_000_000_000.0,
        vol_90d=0.35,
    )
    res = margin.calculate_stock_margin(df, _rule_params()).iloc[0]
    assert res["industry_adj"] == pytest.approx(0.04)
    assert res["maintenance_im"] == pytest.approx(0.325)
    assert res["margin_requirement"] == pytest.approx(4_875_000.00, abs=0.01)


def test_tc05_volatility_adjustment():
    """TC05: 90日波动率（阈值 60%，比例 0.4）-> 费率 38.50%，IA 5,775,000.00 元"""
    df = _row(
        sec_code="600000.SH",
        market_value=15_000_000.0,
        total_mv=100_000_000.0,
        weight_in_portfolio=0.15,
        industry_mv=15_000_000.0,
        industry_weight=0.15,
        avg_amt_20d=1_000_000_000.0,
        total_shares=1_000_000_000.0,
        total_float_shares=1_000_000_000.0,
        vol_90d=0.85,
    )
    res = margin.calculate_stock_margin(df, _rule_params()).iloc[0]
    assert res["vol_adj"] == pytest.approx(0.10)
    assert res["maintenance_im"] == pytest.approx(0.385)
    assert res["margin_requirement"] == pytest.approx(5_775_000.00, abs=0.01)


def test_tc06_cc_micro_concentration():
    """
    TC06: 双创/微盘板块集中度（阈值 30%，比例 0.4，仅对双创或微盘且非1800成分股触发）：
    - A 票为双创且非 1800 成分股 -> stock_adj=7.00%, star_adj=4.80% -> 费率 36.80%，IA 5,520,000.00 元
    - 反例：主板蓝筹股（is_cc_micro=0）在同一组合中 star_adj 仍为 0
    """
    df_a = _row(
        sec_code="688001.SH",
        is_csi300=0,
        is_cc_micro=1,
        market_value=15_000_000.0,
        total_mv=100_000_000.0,
        weight_in_portfolio=0.15,
        industry_mv=15_000_000.0,
        industry_weight=0.15,
        cc_micro_mv=42_000_000.0,
        cc_micro_weight=0.42,
        weight_in_cc_micro=15.0 / 42.0,
        avg_amt_20d=1_000_000_000.0,
        total_shares=1_000_000_000.0,
        total_float_shares=1_000_000_000.0,
        vol_90d=0.35,
    )
    res_a = margin.calculate_stock_margin(df_a, _rule_params()).iloc[0]
    assert res_a["stock_adj"] == pytest.approx(0.07)
    assert res_a["star_adj"] == pytest.approx(0.048)
    assert res_a["maintenance_im"] == pytest.approx(0.368)
    assert res_a["margin_requirement"] == pytest.approx(5_520_000.00, abs=0.01)

    # 反例：主板 1800 成分股在 42% 板块集中度下 star_adj 为 0
    df_blue = df_a.copy()
    df_blue["sec_code"] = "600000.SH"
    df_blue["is_csi300"] = 1
    df_blue["is_cc_micro"] = 0
    df_blue["weight_in_cc_micro"] = 0.0
    res_blue = margin.calculate_stock_margin(df_blue, _rule_params()).iloc[0]
    assert res_blue["star_adj"] == pytest.approx(0.0)


def test_tc07_four_adjustments_combined():
    """TC07: 四项调整叠加（个股 15% 恰好不触发 r2）-> 费率 56.00%，IA 8,400,000.00 元"""
    df = _row(
        sec_code="300001.SZ",
        is_csi300=0,
        is_cc_micro=1,
        market_value=15_000_000.0,
        total_mv=100_000_000.0,
        weight_in_portfolio=0.15,
        industry_mv=35_000_000.0,
        industry_weight=0.35,
        weight_in_industry=15.0 / 35.0,
        cc_micro_mv=45_000_000.0,
        cc_micro_weight=0.45,
        weight_in_cc_micro=15.0 / 45.0,
        avg_amt_20d=1_000_000_000.0,
        total_shares=1_000_000_000.0,
        total_float_shares=1_000_000_000.0,
        vol_90d=0.90,
    )
    res = margin.calculate_stock_margin(df, _rule_params()).iloc[0]
    assert res["restr_mv"] == pytest.approx(0.0)
    assert res["adjustment_sum"] == pytest.approx(0.31)
    assert res["maintenance_im"] == pytest.approx(0.56)
    assert res["margin_requirement"] == pytest.approx(8_400_000.00, abs=0.01)


def test_tc08_maintenance_im_cap_100pct():
    """TC08: 加点封顶 100%（不触发限制项，纯测试费率封顶）-> 费率 100.00%，IA 15,000,000.00 元"""
    df = _row(
        sec_code="688001.SH",
        is_csi300=0,
        is_cc_micro=1,
        market_value=15_000_000.0,
        total_mv=100_000_000.0,
        weight_in_portfolio=0.15,
        industry_mv=40_000_000.0,
        industry_weight=0.40,
        weight_in_industry=15.0 / 40.0,
        cc_micro_mv=50_000_000.0,
        cc_micro_weight=0.50,
        weight_in_cc_micro=15.0 / 50.0,
        avg_amt_20d=1_000_000_000.0,
        total_shares=1_000_000_000.0,
        total_float_shares=1_000_000_000.0,
        vol_90d=2.00,
    )
    res = margin.calculate_stock_margin(df, _rule_params()).iloc[0]
    assert res["restr_mv"] == pytest.approx(0.0)
    assert res["adjustment_sum"] == pytest.approx(0.79)
    assert res["maintenance_im"] == pytest.approx(1.0)
    assert res["margin_requirement"] == pytest.approx(15_000_000.00, abs=0.01)


def test_tc09_restriction1_liquidity():
    """TC09: 限制项 r1 流动性超限 -> restr_mv=7,000,000, IA=9,280,000.00 元"""
    df = _row(
        sec_code="600000.SH",
        market_value=15_000_000.0,
        total_mv=100_000_000.0,
        weight_in_portfolio=0.15,
        industry_mv=15_000_000.0,
        industry_weight=0.15,
        avg_amt_20d=8_000_000.0,
        total_shares=1_000_000_000.0,
        total_float_shares=1_000_000_000.0,
        vol_90d=0.35,
    )
    res = margin.calculate_stock_margin(df, _rule_params()).iloc[0]
    assert res["restriction1"] == pytest.approx(7_000_000.0)
    assert res["restr_mv"] == pytest.approx(7_000_000.0)
    assert res["margin_requirement"] == pytest.approx(9_280_000.00, abs=0.01)


def test_tc10_restriction2_and_5_take_max():
    """TC10: 限制项 r2 (15%) 与 r5 (40%) 取 MAX -> r2=3,000,000, r5=6,000,000, IA=11,592,000.00 元"""
    df = _row(
        sec_code="600000.SH",
        market_value=18_000_000.0,
        total_mv=100_000_000.0,
        weight_in_portfolio=0.18,
        industry_mv=60_000_000.0,
        industry_weight=0.60,
        weight_in_industry=0.30,
        avg_amt_20d=1_000_000_000.0,
        total_shares=1_000_000_000.0,
        total_float_shares=1_000_000_000.0,
        vol_90d=0.35,
    )
    res = margin.calculate_stock_margin(df, _rule_params()).iloc[0]
    assert res["maintenance_im"] == pytest.approx(0.466)
    assert res["restriction2"] == pytest.approx(3_000_000.0)
    assert res["restriction5"] == pytest.approx(6_000_000.0)
    assert res["restr_mv"] == pytest.approx(6_000_000.0)
    assert res["margin_requirement"] == pytest.approx(11_592_000.00, abs=0.01)
    assert res["adjusted_ia"] == pytest.approx(0.644)


def test_tc11_restriction3_and_4_shares():
    """TC11: 限制项 r3 (1%) / r4 (2%) 股本超限（人民币直算）-> r4=2,000,000, IA=4,000,000.00 元"""
    df = _row(
        sec_code="600000.SH",
        price=2.0,
        abs_shares=5_000_000.0,
        market_value=10_000_000.0,
        total_mv=100_000_000.0,
        weight_in_portfolio=0.10,
        industry_mv=10_000_000.0,
        industry_weight=0.10,
        weight_in_industry=1.0,
        total_shares=1_000_000_000.0,
        total_float_shares=200_000_000.0,
        avg_amt_20d=1_000_000_000.0,
        vol_90d=0.30,
    )
    res = margin.calculate_stock_margin(df, _rule_params()).iloc[0]
    assert res["restriction3"] == pytest.approx(0.0)
    assert res["restriction4"] == pytest.approx(2_000_000.0)
    assert res["restr_mv"] == pytest.approx(2_000_000.0)
    assert res["margin_requirement"] == pytest.approx(4_000_000.00, abs=0.01)
    assert res["adjusted_ia"] == pytest.approx(0.40)


def test_tc12_restriction7_cc_micro():
    """TC12: 限制项 r7 微盘+双创超限 -> restr_mv=500,000, IA=2,250,000.00 元"""
    df = _row(
        sec_code="002001.SZ",
        is_csi300=0,
        is_cc_micro=1,
        market_value=5_500_000.0,
        total_mv=100_000_000.0,
        weight_in_portfolio=0.055,
        industry_mv=10_000_000.0,
        industry_weight=0.10,
        weight_in_industry=0.55,
        cc_micro_mv=55_000_000.0,
        cc_micro_weight=0.55,
        weight_in_cc_micro=0.10,
        avg_amt_20d=1_000_000_000.0,
        total_shares=1_000_000_000.0,
        total_float_shares=1_000_000_000.0,
        vol_90d=0.35,
    )
    res = margin.calculate_stock_margin(df, _rule_params()).iloc[0]
    assert res["maintenance_im"] == pytest.approx(0.35)
    assert res["restriction7"] == pytest.approx(500_000.0)
    assert res["margin_requirement"] == pytest.approx(2_250_000.00, abs=0.01)


def test_tc13_upper_im_dual_rules():
    """TC13: upper_im 双算法取数（规则1券商A: 24,000,000.00 元 vs 规则2券商B: 18,000,000.00 元）"""
    rows = []
    for i in range(6):
        rows.append(
            _row(
                sec_code=f"60000{i}.SH",
                market_value=30_000_000.0 if i == 0 else 14_000_000.0,
                total_mv=100_000_000.0,
                weight_in_portfolio=0.40 if i == 0 else 0.14,
                industry_mv=55_000_000.0 if i == 0 else 14_000_000.0,
                industry_weight=0.55 if i == 0 else 0.14,
                weight_in_industry=1.0,
                avg_amt_20d=1_000_000_000.0,
                total_shares=1_000_000_000.0,
                total_float_shares=1_000_000_000.0,
                vol_90d=0.35,
                is_sse50=1,
            ).iloc[0]
        )
    df6 = pd.DataFrame(rows)

    # 规则 1：6 只 < 20 只 -> 取 MAX(18,000,000, 24,000,000) = 24,000,000.00
    r1 = margin.calculate_stock_margin(
        df6,
        _rule_params(
            upper_position_limit=20,
            upper_compare="max",
            industry_weight_limit=1.0,
        ),
    ).iloc[0]
    assert r1["rule_margin_requirement"] == pytest.approx(18_000_000.0)
    assert r1["upper_margin_requirement"] == pytest.approx(24_000_000.0)
    assert r1["margin_requirement"] == pytest.approx(24_000_000.0)

    # 规则 2（券商B）：6 只 >= 5 只 -> 取 MIN(18,000,000, 24,000,000) = 18,000,000.00
    r2_many = margin.calculate_stock_margin(
        df6,
        _rule_params(
            upper_position_limit=5,
            upper_compare="rule2",
            industry_weight_limit=1.0,
        ),
    ).iloc[0]
    assert r2_many["margin_requirement"] == pytest.approx(18_000_000.0)

    # 规则 2（券商B）：若组合仅 4 只 (< 5 只) -> 直接取 B = 24,000,000.00
    r2_few = margin.calculate_stock_margin(
        df6.iloc[:4].copy(),
        _rule_params(
            upper_position_limit=5,
            upper_compare="rule2",
            industry_weight_limit=1.0,
        ),
    ).iloc[0]
    assert r2_few["margin_requirement"] == pytest.approx(24_000_000.0)


def test_tc14_contract_add_on_only_on_base_im():
    """
    TC14: 框架合约加点（仅作用于 base_im）：
    - 场景一（上浮比例 10%）：adj_base_im = 27.50% -> 正常层 31.00%，IA = 4,650,000.00 元
    - 场景二（绝对值 +3.00%）：adj_base_im = 28.00% -> 正常层 31.50%，IA = 4,725,000.00 元
    """
    df = _row(
        sec_code="600000.SH",
        market_value=15_000_000.0,
        total_mv=100_000_000.0,
        weight_in_portfolio=0.15,
        industry_mv=15_000_000.0,
        industry_weight=0.15,
        avg_amt_20d=1_000_000_000.0,
        total_shares=1_000_000_000.0,
        total_float_shares=1_000_000_000.0,
        vol_90d=0.35,
    )
    res_rel = margin.calculate_stock_margin(
        df,
        _rule_params(contract_add_on_type="relative", contract_add_on=0.10),
    ).iloc[0]
    assert res_rel["base_im_after_add_on"] == pytest.approx(0.275)
    assert res_rel["maintenance_im"] == pytest.approx(0.31)
    assert res_rel["margin_requirement"] == pytest.approx(4_650_000.00, abs=0.01)

    res_abs = margin.calculate_stock_margin(
        df,
        _rule_params(contract_add_on_type="absolute", contract_add_on=0.03),
    ).iloc[0]
    assert res_abs["base_im_after_add_on"] == pytest.approx(0.28)
    assert res_abs["maintenance_im"] == pytest.approx(0.315)
    assert res_abs["margin_requirement"] == pytest.approx(4_725_000.00, abs=0.01)


def test_tc15_special_stocks_100pct_fallback_and_rule2_immunity():
    """
    TC15: 特殊标的（停牌股 10,000,000 元、风险警示 ST、未知标的、波动率缺失）按 100% 计收兜底，
    且即便在 upper_im 规则 2 (MIN(A, 80%)) 下也绝不会被降为 80%！
    """
    df_susp = _row(market_value=10_000_000.0, is_suspended=1)
    res_susp = margin.calculate_stock_margin(
        df_susp,
        _rule_params(upper_compare="rule2", upper_position_limit=1),
    ).iloc[0]
    assert res_susp["is_special_100"] == 1
    assert "停牌股" in res_susp["special_reason"]
    assert res_susp["maintenance_im"] == pytest.approx(1.0)
    assert res_susp["margin_requirement"] == pytest.approx(10_000_000.0)
    assert res_susp["adjusted_ia"] == pytest.approx(1.0)

    df_unknown = _row(market_value=5_000_000.0, ind_sw_l1_name="未知")
    res_unknown = margin.calculate_stock_margin(
        df_unknown,
        _rule_params(upper_compare="rule2", upper_position_limit=1),
    ).iloc[0]
    assert res_unknown["is_special_100"] == 1
    assert "不在A股标的表中" in res_unknown["special_reason"]
    assert res_unknown["margin_requirement"] == pytest.approx(5_000_000.0)

    df_no_vol = _row(market_value=8_000_000.0, vol_90d=np.nan)
    res_no_vol = margin.calculate_stock_margin(df_no_vol, _rule_params()).iloc[0]
    assert res_no_vol["is_special_100"] == 1
    assert "波动率缺失" in res_no_vol["special_reason"]
    assert res_no_vol["margin_requirement"] == pytest.approx(8_000_000.0)
