"""
独立情景与敏感性分析模块（Scenario Analysis）单元测试
"""
import io
import pandas as pd
import pytest

from src import scenario_app, webapp


def test_generate_step_values_forward_and_backward():
    """测试正向递增与反向递减步进序列生成"""
    # 正向：10% 起，每次 +10%，走 9 步 -> 10%, 20%, ..., 100%（共 10 个点）
    fwd = scenario_app.generate_step_values(0.10, 0.10, steps=9, min_val=0.0, max_val=1.0)
    assert len(fwd) == 10
    assert fwd[0] == pytest.approx(0.10)
    assert fwd[-1] == pytest.approx(1.00)

    # 反向：100% 起，每次 -10%，走 9 步 -> 100%, 90%, ..., 10%
    bwd = scenario_app.generate_step_values(1.00, -0.10, steps=9, min_val=0.0, max_val=1.0)
    assert len(bwd) == 10
    assert bwd[0] == pytest.approx(1.00)
    assert bwd[-1] == pytest.approx(0.10)


def test_theoretical_stock_weight_clean_start_and_physical_coupling():
    """
    验证架空基准模式（Clean Theoretical Mode）下扫描个股集中度：
    1) 起点（10%）未超阈值时，无关项全为 0，起始 IA 严格等于系统 base_im (25.00%)；
    2) 随着个股集中度从 10% 升至 100%，必然同步推高行业集中度（若为双创微盘则同步推高双创微盘集中度）。
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

    # Step 0 (10%): 未超 10% 起征点，行业 10% 未超 20%，无超限 -> 严格等于 25.00%
    r10 = rows[0]
    assert r10["factor_value"] == pytest.approx(0.10)
    assert r10["industry_weight"] == pytest.approx(0.10)
    assert r10["cc_micro_weight"] == pytest.approx(0.0)
    assert r10["final_ia"] == pytest.approx(0.25)

    # Step 1 (20%): 个股加点 (20%-10%)*0.7 = 7%，行业恰好 20% 未超 -> IA = 32.00%
    r20 = rows[1]
    assert r20["stock_adj"] == pytest.approx(0.07)
    assert r20["industry_adj"] == pytest.approx(0.0)
    assert r20["final_ia"] == pytest.approx(0.32)

    # Step 2 (30%): 个股加点 14% + 联动行业加点 (30%-20%)*0.4 = 4% -> IA = 43.00%
    r30 = rows[2]
    assert r30["industry_weight"] == pytest.approx(0.30)
    assert r30["stock_adj"] == pytest.approx(0.14)
    assert r30["industry_adj"] == pytest.approx(0.04)
    assert r30["final_ia"] == pytest.approx(0.43)

    # Step 4 (50%): 维持保证金率 = 25% + 28%(个股) + 12%(行业) = 65%
    # 同时触发 r2 个股超限 (50%-30%)/50% = 40% 与 r5 行业超限 (50%-40%)/50% = 20%，取 MAX 为 40%
    # 最终 IA = 60% * 65% + 40% * 100% = 79.00%
    r50 = rows[4]
    assert r50["maintenance_im"] == pytest.approx(0.65)
    assert r50["r2_ratio"] == pytest.approx(0.40)
    assert r50["r5_ratio"] == pytest.approx(0.20)
    assert r50["restr_ratio"] == pytest.approx(0.40)
    assert r50["final_ia"] == pytest.approx(0.79)

    # 验证 4 类架空标的对比曲线已同时生成
    comp = res_normal["comparison_series"]
    assert set(comp.keys()) == {"normal", "dual_non300", "main_micro", "sse50"}
    # 在 10% 处，双创非300 (5% 起征) 的 IA 应为 28.50%，高于普通股的 25.00%
    assert comp["dual_non300"]["values"][0] == pytest.approx(0.285)
    # 在 40% 处，上证50豁免 r2 且恰好在 r5 (40%) 边界上无超限，IA 低于触发 r2 的普通股
    assert comp["sse50"]["values"][3] < comp["normal"]["values"][3]


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
    # 30%, 45%, 60% 均未超 60% 阈值，IA 恒为 25.00%
    assert rows[0]["final_ia"] == pytest.approx(0.25)
    assert rows[1]["final_ia"] == pytest.approx(0.25)
    assert rows[2]["final_ia"] == pytest.approx(0.25)
    # 75%: 25% + (75%-60%)*0.4 = 31.00%
    assert rows[3]["final_ia"] == pytest.approx(0.31)
    # 90%: 25% + (90%-60%)*0.4 = 37.00%
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
    # 随着 688057.SH（科创板双创微盘股）个股集中度从 10% 升至 90%，其所属行业集中度与组合双创集中度必然同步上升
    assert rows[-1]["industry_weight"] > rows[0]["industry_weight"]
    assert rows[-1]["cc_micro_weight"] > rows[0]["cc_micro_weight"]
    assert rows[-1]["portfolio_ia"] > rows[0]["portfolio_ia"]

    excel_bytes = scenario_app.build_scenario_export(res, {})
    xls = pd.ExcelFile(io.BytesIO(excel_bytes))
    assert xls.sheet_names == ["情景分析步进明细", "分析参数配置"]
    df_sheet = pd.read_excel(xls, sheet_name="情景分析步进明细")
    assert len(df_sheet) == 5
