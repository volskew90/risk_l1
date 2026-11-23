"""
独立情景与敏感性分析模块（Scenario Analysis）单元测试
对齐《IA_Margin_Rule.html》新规（r2=15%）与选项 B（行业/板块由 N 只等权股票平分联动）。
"""
import io
import pandas as pd
import pytest

from src import scenario_app, webapp


def test_generate_step_values_forward_and_backward():
    """测试正向递增与反向递减步进序列生成"""
    fwd = scenario_app.generate_step_values(0.10, 0.10, steps=9, min_val=0.0, max_val=1.0)
    assert len(fwd) == 10
    assert fwd[0] == pytest.approx(0.10)
    assert fwd[-1] == pytest.approx(1.00)

    bwd = scenario_app.generate_step_values(1.00, -0.10, steps=9, min_val=0.0, max_val=1.0)
    assert len(bwd) == 10
    assert bwd[0] == pytest.approx(1.00)
    assert bwd[-1] == pytest.approx(0.10)


def test_theoretical_stock_weight_new_rule_r2_15pct():
    """
    验证新规（r2=15%）下架空基准模式扫描个股集中度：
    1) 起点（10%）未超阈值时，起始 IA 严格等于系统 base_im (25.00%)；
    2) 在 20% 处：触发个股加点 (20%-10%)*0.7 = 7%（维持保证金率 32%），
       且触发新规 r2 个股超限 (20%-15%)/20% = 25%，
       最终 IA = 75% * 32% + 25% * 100% = 49.00%！
    3) 在 50% 处：维持保证金率 = 25% + 28%(个股) + 12%(行业) = 65%，
       r2 超限占比 = (50%-15%)/50% = 70%，r5 超限占比 = (50%-40%)/50% = 20%，取 MAX 为 70%，
       最终 IA = 30% * 65% + 70% * 100% = 89.50%！
    """
    res_normal = scenario_app.run_theoretical_scenario(
        {
            "factor": "stock_weight",
            "archetype": "normal",
            "start": 0.10,
            "step": 0.10,
            "steps": 9,
            "compare_all_archetypes": True,
        }
    )
    rows = res_normal["rows"]
    assert len(rows) == 10

    # Step 0 (10%): 未超 10% 起征点，未超 15% r2 -> 严格等于 25.00%
    r10 = rows[0]
    assert r10["factor_value"] == pytest.approx(0.10)
    assert r10["final_ia"] == pytest.approx(0.25)

    # Step 1 (20%): 维持保证金率 32%，r2 超限占比 (20%-15%)/20% = 25% -> IA = 49.00%
    r20 = rows[1]
    assert r20["maintenance_im"] == pytest.approx(0.32)
    assert r20["r2_ratio"] == pytest.approx(0.25)
    assert r20["final_ia"] == pytest.approx(0.49)

    # Step 4 (50%): 维持保证金率 65%，r2 超限占比 70% -> IA = 89.50%
    r50 = rows[4]
    assert r50["maintenance_im"] == pytest.approx(0.65)
    assert r50["r2_ratio"] == pytest.approx(0.70)
    assert r50["r5_ratio"] == pytest.approx(0.20)
    assert r50["restr_ratio"] == pytest.approx(0.70)
    assert r50["final_ia"] == pytest.approx(0.895)

    # 验证 5 类架空标的对比曲线已同时生成
    comp = res_normal["comparison_series"]
    assert set(comp.keys()) == {
        "normal",
        "dual_non300",
        "dual_1800",
        "main_micro",
        "sse50",
    }


def test_theoretical_industry_weight_option_b_n_stocks_split():
    """
    验证选项 B：扫描行业集中度时，支持设定『该行业由 N 只等权股票平分 (n_stocks_in_group)』。
    当 N=2（行业由 2 只股票平分）且行业集中度 = 100% 时：
    每只股票个股集中度 = 50%，同步触发个股加点与 r2 超限，最终 IA 达 92.20%（显著高于 N=20 分散时的 82.80%）。
    """
    res_n2 = scenario_app.run_theoretical_scenario(
        {
            "factor": "industry_weight",
            "archetype": "normal",
            "start": 0.20,
            "step": 0.40,
            "steps": 2,
            "n_stocks_in_group": 2,
        }
    )
    r100_n2 = res_n2["rows"][-1]
    assert r100_n2["industry_weight"] == pytest.approx(1.00)
    assert r100_n2["stock_weight"] == pytest.approx(0.50)
    assert r100_n2["final_ia"] == pytest.approx(0.955)

    res_n20 = scenario_app.run_theoretical_scenario(
        {
            "factor": "industry_weight",
            "archetype": "normal",
            "start": 0.20,
            "step": 0.40,
            "steps": 2,
            "n_stocks_in_group": 20,
        }
    )
    r100_n20 = res_n20["rows"][-1]
    assert r100_n20["stock_weight"] == pytest.approx(0.05)
    assert r100_n20["final_ia"] == pytest.approx(0.828)


def test_theoretical_volatility_clean_baseline():
    """验证架空模式下扫描 90 日波动率：<= 60% 时 IA 恒为 25.00%，超过 60% 后按斜率线性增加"""
    res = scenario_app.run_theoretical_scenario(
        {
            "factor": "vol_90d",
            "archetype": "normal",
            "start": 0.30,
            "step": 0.15,
            "steps": 4,
        }
    )
    rows = res["rows"]
    assert rows[0]["final_ia"] == pytest.approx(0.25)
    assert rows[1]["final_ia"] == pytest.approx(0.25)
    assert rows[2]["final_ia"] == pytest.approx(0.25)
    assert rows[3]["final_ia"] == pytest.approx(0.31)
    assert rows[4]["final_ia"] == pytest.approx(0.37)


def test_portfolio_scenario_and_export():
    """验证基于组合的情景推演模式及 Excel 导出"""
    portfolio, _ = webapp.load_demo()
    res = scenario_app.run_portfolio_scenario(
        {
            "positions": portfolio.to_dict(orient="records"),
            "target_sec_code": "688057.SH",
            "factor": "stock_weight",
            "start": 0.10,
            "step": 0.20,
            "steps": 4,
            "exempt_capacity_limits": True,
        }
    )
    rows = res["rows"]
    assert len(rows) == 5
    assert rows[-1]["industry_weight"] > rows[0]["industry_weight"]
    assert rows[-1]["cc_micro_weight"] > rows[0]["cc_micro_weight"]
    assert rows[-1]["portfolio_ia"] > rows[0]["portfolio_ia"]

    excel_bytes = scenario_app.build_scenario_export(res, {})
    xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    assert xls.sheet_names == ["情景分析步进明细", "分析参数配置"]
    df_sheet = pd.read_excel(xls, sheet_name="情景分析步进明细")
    assert len(df_sheet) == 5
