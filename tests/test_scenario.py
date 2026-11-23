"""
独立情景与敏感性分析模块（Scenario Analysis）全方位单元测试套件
覆盖：
  1) 正向与反向步进序列生成器；
  2) 新规 r2=15% 下的 5 类架空标的个股集中度步进曲线；
  3) 各调整比例（stock_adj_ratio / industry_adj_ratio / vol_adj_ratio / cc_micro_adj_ratio）对斜率的单调性影响；
  4) 各限制项阈值（r1 / r2 / r3 / r4 / r5 / r7）与 disable_restrictions 纯调整项斜率模式；
  5) 选项 B：行业/板块集中度下 N 只等权股票平分联动；
  6) 模式二：真实组合全因子（7 项因子）调仓联动与 Excel 导出。
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
    - 10% 处：未超 10% 起征点，未超 15% r2 -> 严格等于 25.00%
    - 20% 处：维持保证金率 32%，r2 超限占比 (20%-15%)/20% = 25% -> IA = 49.00%
    - 50% 处：维持保证金率 65%，r2 超限占比 70% -> IA = 89.50%
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
    assert rows[0]["final_ia"] == pytest.approx(0.25)
    assert rows[1]["maintenance_im"] == pytest.approx(0.32)
    assert rows[1]["r2_ratio"] == pytest.approx(0.25)
    assert rows[1]["final_ia"] == pytest.approx(0.49)
    assert rows[4]["maintenance_im"] == pytest.approx(0.65)
    assert rows[4]["restr_ratio"] == pytest.approx(0.70)
    assert rows[4]["final_ia"] == pytest.approx(0.895)

    comp = res_normal["comparison_series"]
    assert set(comp.keys()) == {
        "normal",
        "dual_non300",
        "dual_1800",
        "main_micro",
        "sse50",
    }
    # dual_1800（双创但属中证500/1000）：在 10% 处因 5% 起征触发 stock_adj=3.5% -> 28.5%
    assert comp["dual_1800"]["values"][0] == pytest.approx(0.285)
    # 在 40% 处，dual_non300（非1800）比 dual_1800 多出一项 star_adj (40%-30%)*0.4 = 4%
    assert comp["dual_non300"]["values"][3] > comp["dual_1800"]["values"][3]


def test_stock_adj_ratio_changes_slope_significantly():
    """
    验证修改个股调整比例 (stock_adj_ratio) 会显著改变维持保证金率与最终 IA 的斜率：
    1) 在开启限制项时：ratio=0.2 vs ratio=0.7 vs ratio=1.2 在 20%~70% 各点的 IA 严格单调递增；
    2) 在勾选 disable_restrictions=True（纯观察调整项斜率）时：final_ia 完全等于 maintenance_im，呈现精确线性斜率。
    """
    r_low = scenario_app.run_theoretical_scenario(
        {
            "factor": "stock_weight",
            "archetype": "normal",
            "start": 0.10,
            "step": 0.10,
            "steps": 5,
            "params": {"stock_adj_ratio": 0.2},
        }
    )["rows"]
    r_mid = scenario_app.run_theoretical_scenario(
        {
            "factor": "stock_weight",
            "archetype": "normal",
            "start": 0.10,
            "step": 0.10,
            "steps": 5,
            "params": {"stock_adj_ratio": 0.7},
        }
    )["rows"]
    r_high = scenario_app.run_theoretical_scenario(
        {
            "factor": "stock_weight",
            "archetype": "normal",
            "start": 0.10,
            "step": 0.10,
            "steps": 5,
            "params": {"stock_adj_ratio": 1.2},
        }
    )["rows"]

    # 在 20%, 30%, 40% 处，随 stock_adj_ratio 增大，maintenance_im 和 final_ia 严格增大
    for idx in [1, 2, 3]:
        assert (
            r_low[idx]["maintenance_im"]
            < r_mid[idx]["maintenance_im"]
            < r_high[idx]["maintenance_im"]
        )
        assert (
            r_low[idx]["final_ia"]
            < r_mid[idx]["final_ia"]
            < r_high[idx]["final_ia"]
        )

    # 验证 disable_restrictions=True 时，restr_ratio 恒为 0，final_ia 严格等于线性加点后的 maintenance_im
    r_pure_slope = scenario_app.run_theoretical_scenario(
        {
            "factor": "stock_weight",
            "archetype": "normal",
            "start": 0.10,
            "step": 0.10,
            "steps": 5,
            "disable_restrictions": True,
            "params": {"stock_adj_ratio": 0.5, "industry_adj_ratio": 0.0},
        }
    )["rows"]
    for step_idx, row in enumerate(r_pure_slope):
        assert row["restr_ratio"] == pytest.approx(0.0)
        expected_im = min(0.25 + step_idx * 0.10 * 0.5, 1.0)
        assert row["final_ia"] == pytest.approx(expected_im)


def test_theoretical_industry_weight_option_b_n_stocks_split():
    """
    验证选项 B：扫描行业集中度时，设定『该行业由 N 只等权股票平分 (n_stocks_in_group)』。
    - N=1 时等同于单票集中度，在 100% 处 final_ia = 100.00%；
    - N=2 时单票集中度 = 50%，在 100% 处 final_ia = 95.50%；
    - N=20 时单票分散（5%），在 100% 处 final_ia = 82.80%。
    """
    res_n1 = scenario_app.run_theoretical_scenario(
        {
            "factor": "industry_weight",
            "archetype": "normal",
            "start": 0.20,
            "step": 0.40,
            "steps": 2,
            "n_stocks_in_group": 1,
        }
    )
    assert res_n1["rows"][-1]["final_ia"] == pytest.approx(1.00)

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
    assert res_n2["rows"][-1]["stock_weight"] == pytest.approx(0.50)
    assert res_n2["rows"][-1]["final_ia"] == pytest.approx(0.955)

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
    assert res_n20["rows"][-1]["stock_weight"] == pytest.approx(0.05)
    assert res_n20["rows"][-1]["final_ia"] == pytest.approx(0.828)


def test_all_seven_factors_in_theoretical_and_portfolio_modes():
    """遍历测试全部 7 个分析因子在架空基准模式与组合模式下的计算正确性"""
    portfolio, _ = webapp.load_demo()
    positions = portfolio.to_dict(orient="records")

    for factor_key, meta in scenario_app.FACTOR_META.items():
        # 1. 架空基准模式（使用 normal 普通股，在默认起点 <= 10% 时均处于阈值以内）
        theo_res = scenario_app.run_theoretical_scenario(
            {
                "factor": factor_key,
                "archetype": "normal",
                "start": meta["default_start"],
                "step": meta["default_step"],
                "steps": 5,
                "n_stocks_in_group": 2,
            }
        )
        t_rows = theo_res["rows"]
        assert len(t_rows) == 6
        # 第 0 步均在阈值以内，起始 IA 严格等于 25%
        assert t_rows[0]["final_ia"] == pytest.approx(0.25)
        # 最后一步均已超过阈值，最终 IA 严格大于 25%
        assert t_rows[-1]["final_ia"] > 0.25
        # 单调不减
        for i in range(1, len(t_rows)):
            assert t_rows[i]["final_ia"] >= t_rows[i - 1]["final_ia"] - 1e-9

        # 2. 组合推演模式
        port_res = scenario_app.run_portfolio_scenario(
            {
                "positions": positions,
                "target_sec_code": "688057.SH",
                "factor": factor_key,
                "start": meta["default_start"],
                "step": meta["default_step"],
                "steps": 4,
            }
        )
        p_rows = port_res["rows"]
        assert len(p_rows) == 5
        assert p_rows[-1]["target_stock_ia"] >= p_rows[0]["target_stock_ia"] - 1e-9


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


def test_upper_im_rule2_and_rule1_in_theoretical_mode():
    """
    验证架空模式下切换 upper_compare（none / rule2 / max / min）与修改 upper_im 值：
    - rule2（券商B规则2）：曲线从 25% -> 49% -> 71.5% 爬升，超过 upper_im (80%) 后封顶在 80%，绝非全程 80% 死线！
    - 当把 upper_im 改为 60% 时，rule2 的封顶高度随之变为 60%！
    - max（券商A规则1）：前半段由 80% 托底，后半段超过 80% 后继续上升至 100%！
    """
    res_rule2_80 = scenario_app.run_theoretical_scenario(
        {
            "factor": "stock_weight",
            "archetype": "normal",
            "start": 0.10,
            "step": 0.10,
            "steps": 9,
            "upper_enabled": True,
            "respect_upper_compare": True,
            "params": {"upper_compare": "rule2", "upper_im": 0.80},
        }
    )["rows"]
    ias_80 = [r["final_ia"] for r in res_rule2_80]
    assert ias_80[0] == pytest.approx(0.25)
    assert ias_80[1] == pytest.approx(0.49)
    assert ias_80[2] == pytest.approx(0.715)
    assert ias_80[3] == pytest.approx(0.80)
    assert ias_80[-1] == pytest.approx(0.80)

    # 修改 upper_im 为 60% (0.60)，封顶值立即变为 0.60
    res_rule2_60 = scenario_app.run_theoretical_scenario(
        {
            "factor": "stock_weight",
            "archetype": "normal",
            "start": 0.10,
            "step": 0.10,
            "steps": 9,
            "upper_enabled": True,
            "respect_upper_compare": True,
            "params": {"upper_compare": "rule2", "upper_im": 0.60},
        }
    )["rows"]
    ias_60 = [r["final_ia"] for r in res_rule2_60]
    assert ias_60[0] == pytest.approx(0.25)
    assert ias_60[1] == pytest.approx(0.49)
    assert ias_60[2] == pytest.approx(0.60)
    assert ias_60[-1] == pytest.approx(0.60)

    # 券商A规则1 (max)：前半段 80% 保底，后半段超过 80% 后继续涨到 100%
    res_rule1 = scenario_app.run_theoretical_scenario(
        {
            "factor": "stock_weight",
            "archetype": "normal",
            "start": 0.10,
            "step": 0.10,
            "steps": 9,
            "upper_enabled": True,
            "respect_upper_compare": True,
            "params": {"upper_compare": "max", "upper_im": 0.80},
        }
    )["rows"]
    ias_r1 = [r["final_ia"] for r in res_rule1]
    assert ias_r1[0] == pytest.approx(0.80)
    assert ias_r1[3] == pytest.approx(0.8275)
    assert ias_r1[-1] == pytest.approx(1.00)


def test_every_single_parameter_field_responsiveness():
    """
    逐一测试全部核心风控参数字段（Field）在其对应的分析维度下修改数值时，
    计算输出的 IA 曲线必须发生显著且符合方向的改变（确保没有任何一个字段假死或失效）。
    """
    cases = [
        # (factor, archetype, param_key, val_low, val_high, extra_params)
        ("stock_weight", "normal", "base_im", 0.20, 0.35, {}),
        ("stock_weight", "normal", "contract_add_on", 0.0, 0.10, {"contract_add_on_type": "absolute"}),
        ("stock_weight", "normal", "stock_weight_threshold_other", 0.15, 0.05, {}),
        ("stock_weight", "dual_non300", "stock_weight_threshold_dual", 0.12, 0.02, {}),
        ("stock_weight", "normal", "stock_adj_ratio", 0.3, 1.0, {}),
        ("stock_weight", "normal", "position_weight_limit", 0.30, 0.12, {}),
        ("industry_weight", "normal", "industry_weight_threshold", 0.35, 0.10, {}),
        ("industry_weight", "normal", "industry_adj_ratio", 0.2, 0.8, {}),
        ("industry_weight", "normal", "industry_weight_limit", 0.60, 0.25, {}),
        ("cc_micro_weight", "dual_non300", "cc_micro_weight_threshold", 0.45, 0.15, {}),
        ("cc_micro_weight", "dual_non300", "cc_micro_adj_ratio", 0.2, 0.8, {}),
        ("cc_micro_weight", "dual_non300", "cc_micro_weight_limit", 0.70, 0.35, {}),
        ("vol_90d", "normal", "vol_90d_threshold", 0.80, 0.40, {}),
        ("vol_90d", "normal", "vol_adj_ratio", 0.2, 0.8, {}),
        ("liquidity_ratio", "normal", "liquidity_threshold", 1.50, 0.50, {}),
        ("total_share_ratio", "normal", "total_share_limit", 0.02, 0.005, {}),
        ("float_share_ratio", "normal", "float_share_limit", 0.04, 0.01, {}),
    ]

    for factor, arch, p_key, v_low_ia, v_high_ia, extra in cases:
        meta = scenario_app.FACTOR_META[factor]
        r_low = scenario_app.run_theoretical_scenario(
            {
                "factor": factor,
                "archetype": arch,
                "start": meta["default_start"],
                "step": meta["default_step"],
                "steps": meta["default_steps"],
                "n_stocks_in_group": 10,
                "params": {**extra, p_key: v_low_ia},
            }
        )["rows"]
        r_high = scenario_app.run_theoretical_scenario(
            {
                "factor": factor,
                "archetype": arch,
                "start": meta["default_start"],
                "step": meta["default_step"],
                "steps": meta["default_steps"],
                "n_stocks_in_group": 10,
                "params": {**extra, p_key: v_high_ia},
            }
        )["rows"]
        sum_low = sum(r["final_ia"] for r in r_low)
        sum_high = sum(r["final_ia"] for r in r_high)
        assert sum_high > sum_low + 1e-4, f"Field {p_key} under {factor}/{arch} did not change IA!"

    # 18. contract_add_on_type (relative vs absolute when contract_add_on > 0)
    r_rel = scenario_app.run_theoretical_scenario(
        {
            "factor": "stock_weight",
            "archetype": "normal",
            "start": 0.05,
            "step": 0.05,
            "steps": 2,
            "params": {"base_im": 0.25, "contract_add_on": 0.10, "contract_add_on_type": "relative"},
        }
    )["rows"]
    r_abs = scenario_app.run_theoretical_scenario(
        {
            "factor": "stock_weight",
            "archetype": "normal",
            "start": 0.05,
            "step": 0.05,
            "steps": 2,
            "params": {"base_im": 0.25, "contract_add_on": 0.10, "contract_add_on_type": "absolute"},
        }
    )["rows"]
    assert r_rel[0]["final_ia"] == pytest.approx(0.275)
    assert r_abs[0]["final_ia"] == pytest.approx(0.35)

    # 19 & 20. upper_compare (none / rule2 / max / min) & upper_im
    r_min = scenario_app.run_theoretical_scenario(
        {
            "factor": "stock_weight",
            "archetype": "normal",
            "start": 0.10,
            "step": 0.20,
            "steps": 4,
            "upper_enabled": True,
            "respect_upper_compare": True,
            "params": {"upper_compare": "min", "upper_im": 0.70},
        }
    )["rows"]
    assert r_min[0]["final_ia"] == pytest.approx(0.25)
    assert r_min[-1]["final_ia"] == pytest.approx(0.70)

    # 21. upper_position_limit (组合模式：8 只股票组合下阈值设为 5 vs 15)
    portfolio, _ = webapp.load_demo()
    positions = portfolio.to_dict(orient="records")
    p_lim5 = scenario_app.run_portfolio_scenario(
        {
            "positions": positions,
            "target_sec_code": "688057.SH",
            "factor": "stock_weight",
            "start": 0.10,
            "step": 0.10,
            "steps": 2,
            "upper_enabled": True,
            "respect_upper_compare": True,
            "params": {"upper_compare": "max", "upper_im": 0.80, "upper_position_limit": 5},
        }
    )["rows"]
    p_lim15 = scenario_app.run_portfolio_scenario(
        {
            "positions": positions,
            "target_sec_code": "688057.SH",
            "factor": "stock_weight",
            "start": 0.10,
            "step": 0.10,
            "steps": 2,
            "upper_enabled": True,
            "respect_upper_compare": True,
            "params": {"upper_compare": "max", "upper_im": 0.80, "upper_position_limit": 15},
        }
    )["rows"]
    assert p_lim15[0]["portfolio_ia"] > p_lim5[0]["portfolio_ia"] + 0.10

    # 22. liquidity_cap & exempt_capacity_limits (组合模式下关闭容量豁免并调整 liquidity_cap)
    p_cap_high = scenario_app.run_portfolio_scenario(
        {
            "positions": positions,
            "target_sec_code": "688057.SH",
            "factor": "stock_weight",
            "start": 0.10,
            "step": 0.10,
            "steps": 2,
            "exempt_capacity_limits": False,
            "params": {"liquidity_cap": 2_000_000_000.0},
        }
    )["rows"]
    p_cap_low = scenario_app.run_portfolio_scenario(
        {
            "positions": positions,
            "target_sec_code": "688057.SH",
            "factor": "stock_weight",
            "start": 0.10,
            "step": 0.10,
            "steps": 2,
            "exempt_capacity_limits": False,
            "params": {"liquidity_cap": 100.0},
        }
    )["rows"]
    assert p_cap_low[0]["target_stock_ia"] > p_cap_high[0]["target_stock_ia"] + 0.05

    # 23. disable_restrictions & compare_all_archetypes
    r_with_restr = scenario_app.run_theoretical_scenario(
        {
            "factor": "stock_weight",
            "archetype": "normal",
            "start": 0.10,
            "step": 0.20,
            "steps": 4,
            "disable_restrictions": False,
            "compare_all_archetypes": True,
        }
    )
    r_no_restr = scenario_app.run_theoretical_scenario(
        {
            "factor": "stock_weight",
            "archetype": "normal",
            "start": 0.10,
            "step": 0.20,
            "steps": 4,
            "disable_restrictions": True,
            "compare_all_archetypes": True,
        }
    )
    assert len(r_with_restr["comparison_series"]) == 5
    assert r_no_restr["rows"][1]["restr_ratio"] == 0.0
    assert r_with_restr["rows"][1]["final_ia"] > r_no_restr["rows"][1]["final_ia"] + 0.20


