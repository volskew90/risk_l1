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


def test_base_margin_without_adjustments_or_restrictions():
    result = margin.calculate_stock_margin(_row(), _params()).iloc[0]
    assert result["maintenance_im"] == 0.25
    assert result["restr_mv"] == 0.0
    assert result["margin_requirement"] == 2500.0
    assert result["adjusted_ia"] == 0.25


def test_photo_volatility_adjustment():
    result = margin.calculate_stock_margin(
        _row(vol_90d=0.6621), _params(vol_adj_ratio=0.7),
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


def test_sse50_is_exempt_from_position_weight_restriction():
    result = margin.calculate_stock_margin(
        _row(is_sse50=1), _params(position_weight_limit=0.01),
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
        _row(), _params(upper_position_limit=20, upper_compare=compare),
    ).iloc[0]
    assert result["rule_margin_requirement"] == 2500.0
    assert result["upper_margin_requirement"] == 8000.0
    assert result["margin_requirement"] == expected


def test_portfolio_ia_methods_are_identical():
    one = _row()
    two = _row(sec_code="000001.SZ", market_value=30000.0, total_mv=100000.0,
               weight_in_portfolio=0.3, industry_mv=50000.0, industry_weight=0.5,
               weight_in_industry=0.6, vol_90d=0.7)
    result = margin.calculate_stock_margin(
        pd.concat([one, two], ignore_index=True), _params(),
    )
    summary = margin.calculate_portfolio_ia(result)
    assert summary["total_market_value"] == 40000.0
    assert summary["portfolio_ia"] == pytest.approx(summary["portfolio_ia_weighted"])
    assert summary["ia_method_difference"] == pytest.approx(0.0)
