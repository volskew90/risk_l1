#!/usr/bin/env python3
"""
独立情景与敏感性分析模块（Scenario Analysis）
完全独立于原有 webapp.py，提供两大分析模式：
  1) Tab 1 - 架空基准模式（Clean Theoretical Mode）：
     不依赖既有组合，除当前分析变量必然物理包含/联动的项外，其余所有无关指标均假设在阈值内（0 触发），
     使起始 IA 严格从系统给定的 base_im（叠加合约加点后，如 25%）干净起步。
  2) Tab 2 - 组合推演模式（Portfolio Scenario Mode）：
     基于真实或演示组合，在固定组合总市值下按步长调仓，联动重算 L2 集中度、L3 单券 IA 与 L4 组合 IA。
"""
import argparse
import io
import json
import math
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import concentration, margin, webapp

SCENARIO_PAGE = ROOT / "src" / "static" / "scenario.html"

# 架空标的类别定义
ARCHETYPES = {
    "normal": {
        "name": "普通非双创微盘股（主板300/500/1000）",
        "sec_code": "600000.SH",
        "is_sse50": 0,
        "is_csi300": 1,
        "is_cc_micro": 0,
        "desc": "个股10%起征；个股集中度同步带动行业集中度，不触发双创微盘，受r2/r5约束",
    },
    "dual_non300": {
        "name": "双创且非沪深300股（科创/创业板微盘）",
        "sec_code": "688001.SH",
        "is_sse50": 0,
        "is_csi300": 0,
        "is_cc_micro": 1,
        "desc": "个股5%起征；个股集中度同步带动行业集中度与双创微盘集中度，受r2/r5/r7约束",
    },
    "main_micro": {
        "name": "主板微盘股（非300/500/1000主板股）",
        "sec_code": "600999.SH",
        "is_sse50": 0,
        "is_csi300": 0,
        "is_cc_micro": 1,
        "desc": "个股10%起征；个股集中度同步带动行业集中度与双创微盘集中度，受r2/r5/r7约束",
    },
    "sse50": {
        "name": "上证50成分股（豁免r2个股超限）",
        "sec_code": "600519.SH",
        "is_sse50": 1,
        "is_csi300": 1,
        "is_cc_micro": 0,
        "desc": "个股10%起征；豁免r2个股集中度超限，但同步带动行业集中度（受行业加点与r5约束）",
    },
}

# 支持扫描的分析因子元数据（默认按百分比展示交互）
FACTOR_META = {
    "stock_weight": {
        "label": "个股集中度（占组合总市值 %）",
        "unit": "percent",
        "default_start": 0.10,
        "default_step": 0.10,
        "default_steps": 9,
        "min_val": 0.001,
        "max_val": 1.00,
    },
    "industry_weight": {
        "label": "申万一级行业集中度（占组合总市值 %）",
        "unit": "percent",
        "default_start": 0.10,
        "default_step": 0.10,
        "default_steps": 9,
        "min_val": 0.001,
        "max_val": 1.00,
    },
    "cc_micro_weight": {
        "label": "双创/微盘板块集中度（占组合总市值 %）",
        "unit": "percent",
        "default_start": 0.10,
        "default_step": 0.10,
        "default_steps": 9,
        "min_val": 0.0,
        "max_val": 1.00,
    },
    "vol_90d": {
        "label": "90日年化波动率（%）",
        "unit": "percent",
        "default_start": 0.20,
        "default_step": 0.10,
        "default_steps": 13,
        "min_val": 0.0,
        "max_val": 5.00,
    },
    "liquidity_ratio": {
        "label": "单票市值占20日均成交额比例（%）",
        "unit": "percent",
        "default_start": 0.20,
        "default_step": 0.20,
        "default_steps": 9,
        "min_val": 0.01,
        "max_val": 10.00,
    },
    "total_share_ratio": {
        "label": "持股占总股本比例（%）",
        "unit": "percent",
        "default_start": 0.002,
        "default_step": 0.002,
        "default_steps": 14,
        "min_val": 0.0001,
        "max_val": 1.00,
    },
    "float_share_ratio": {
        "label": "持股占流通股本比例（%）",
        "unit": "percent",
        "default_start": 0.005,
        "default_step": 0.005,
        "default_steps": 11,
        "min_val": 0.0001,
        "max_val": 1.00,
    },
}


def generate_step_values(
    start: float,
    step: float,
    steps: Optional[int] = None,
    end: Optional[float] = None,
    min_val: float = 0.0,
    max_val: Optional[float] = None,
) -> List[float]:
    """
    根据起始值、步长（正数递增或负数递减）与步数（或终止值）生成扫描序列。
    包含第 0 步（起始值本身）至第 N 步，共 N + 1 个采样点。
    """
    start = float(start)
    step = float(step)
    if not np.isfinite(start) or not np.isfinite(step) or abs(step) < 1e-9:
        raise ValueError("起始值必须有效，且每次步长不能为 0")

    if steps is not None and int(steps) >= 1:
        n_steps = min(int(steps), 200)
    elif end is not None and np.isfinite(float(end)):
        raw_n = (float(end) - start) / step
        if raw_n < 0:
            raise ValueError("步长方向与起止值方向不一致（递增请用正步长，递减请用负步长）")
        n_steps = min(max(int(round(raw_n)), 1), 200)
    else:
        n_steps = 9

    values = []
    for k in range(n_steps + 1):
        val = round(start + k * step, 8)
        if val < min_val - 1e-9:
            val = min_val
        if max_val is not None and val > max_val + 1e-9:
            val = max_val
        if not values or abs(val - values[-1]) > 1e-9:
            values.append(val)

    if not values:
        raise ValueError("生成的步进序列为空，请检查起始值与步长范围")
    return values


def _build_safe_baseline_row(archetype_key: str, p: Dict) -> Dict:
    """
    构造一台『所有指标均严格处于安全区内（< threshold）』的架空基准单券行。
    在该基准下，所有调整项 = 0，所有限制项 = 0，初始 IA 严格等于 base_im_after_add_on（如 25%）。
    """
    arch = ARCHETYPES.get(archetype_key, ARCHETYPES["normal"])
    total_mv = 100_000_000.0
    # 取一个极小的安全权重，确保默认低于任何集中度起征点与限制线
    safe_w = min(
        p["stock_weight_threshold_dual"] * 0.5,
        p["stock_weight_threshold_other"] * 0.5,
        p["position_weight_limit"] * 0.5,
        0.005,
    )
    safe_w = max(safe_w, 1e-4)
    mv = safe_w * total_mv
    price = 100.0
    abs_shares = mv / price

    # 设定充裕的 20 日均成交额与总/流通股本，确保默认绝不触发 r1 / r3 / r4
    safe_avg_amt = (
        mv / (p["liquidity_threshold"] * 0.1)
        if p["liquidity_threshold"] > 0
        else 1e12
    )
    safe_total_shares = (
        abs_shares / (p["total_share_limit"] * 0.1)
        if p["total_share_limit"] > 0
        else 1e12
    )
    safe_float_shares = (
        abs_shares / (p["float_share_limit"] * 0.1)
        if p["float_share_limit"] > 0
        else 1e12
    )
    safe_vol = p["vol_90d_threshold"] * 0.5

    return {
        "sec_code": arch["sec_code"],
        "sec_name": arch["name"],
        "ind_sw_l1_name": "架空行业",
        "price": price,
        "abs_shares": abs_shares,
        "market_value": mv,
        "total_mv": total_mv,
        "weight_in_portfolio": safe_w,
        "industry_mv": mv,
        "industry_weight": safe_w,
        "weight_in_industry": 1.0,
        "cc_micro_mv": mv if arch["is_cc_micro"] == 1 else 0.0,
        "cc_micro_weight": safe_w if arch["is_cc_micro"] == 1 else 0.0,
        "weight_in_cc_micro": 1.0 if arch["is_cc_micro"] == 1 else 0.0,
        "total_shares": safe_total_shares,
        "total_float_shares": safe_float_shares,
        "avg_amt_20d": safe_avg_amt,
        "vol_90d": safe_vol,
        "is_sse50": arch["is_sse50"],
        "is_csi300": arch["is_csi300"],
        "is_cc_micro": arch["is_cc_micro"],
        "is_suspended": 0,
        "is_risk_warning": 0,
    }


def _apply_theoretical_factor_at_step(
    base_row: Dict, factor: str, x_val: float, p: Dict
) -> Dict:
    """
    在架空基准模式下，将目标分析因子设为 x_val，并同步更新其物理上必然包含的联动项，
    而保持所有非必然相关的维度在安全阈值内（0 触发）。
    """
    row = dict(base_row)
    total_mv = row["total_mv"]
    price = row["price"]

    if factor == "stock_weight":
        # 个股集中度 = x_val：
        # 1) 必然带动其所在行业的集中度至少为 x_val（假设无其他同行业股票额外垫高，weight_in_industry = 1.0）
        # 2) 若该架空股票属于双创/微盘（is_cc_micro == 1），必然同步带动双创/微盘集中度为 x_val
        # 3) 波动率、流动性、股本保持在安全阈值内（不触发）
        w = max(float(x_val), 1e-6)
        mv = w * total_mv
        abs_shares = mv / price
        row["market_value"] = mv
        row["abs_shares"] = abs_shares
        row["weight_in_portfolio"] = w
        row["industry_mv"] = mv
        row["industry_weight"] = w
        row["weight_in_industry"] = 1.0
        if row["is_cc_micro"] == 1:
            row["cc_micro_mv"] = mv
            row["cc_micro_weight"] = w
            row["weight_in_cc_micro"] = 1.0
        else:
            row["cc_micro_mv"] = 0.0
            row["cc_micro_weight"] = 0.0
            row["weight_in_cc_micro"] = 0.0
        # 保持成交额和股本充裕，不额外触发 r1/r3/r4
        row["avg_amt_20d"] = (
            mv / (p["liquidity_threshold"] * 0.1)
            if p["liquidity_threshold"] > 0
            else 1e12
        )
        row["total_shares"] = (
            abs_shares / (p["total_share_limit"] * 0.1)
            if p["total_share_limit"] > 0
            else 1e12
        )
        row["total_float_shares"] = (
            abs_shares / (p["float_share_limit"] * 0.1)
            if p["float_share_limit"] > 0
            else 1e12
        )

    elif factor == "industry_weight":
        # 行业集中度 = x_val（假设由同行业内多只分散个股构成，单票个股集中度保持在安全阈值内）
        # 若该行业标的属于双创/微盘，同步带动双创微盘集中度；否则不触发双创微盘
        w_ind = max(float(x_val), 1e-6)
        ind_mv = w_ind * total_mv
        row["industry_mv"] = ind_mv
        row["industry_weight"] = w_ind
        row["weight_in_industry"] = row["market_value"] / ind_mv
        if row["is_cc_micro"] == 1:
            row["cc_micro_mv"] = ind_mv
            row["cc_micro_weight"] = w_ind
            row["weight_in_cc_micro"] = row["market_value"] / ind_mv

    elif factor == "cc_micro_weight":
        # 双创/微盘集中度 = x_val（假设由板块内不同行业的多只分散个股构成，单票与单行业均在安全阈值内）
        w_cc = max(float(x_val), 0.0)
        cc_mv = w_cc * total_mv
        row["is_cc_micro"] = 1
        row["cc_micro_mv"] = cc_mv
        row["cc_micro_weight"] = w_cc
        row["weight_in_cc_micro"] = (
            row["market_value"] / cc_mv if cc_mv > 0 else 0.0
        )

    elif factor == "vol_90d":
        # 90日波动率 = x_val，其余集中度与容量指标均在安全阈值内
        row["vol_90d"] = max(float(x_val), 0.0)

    elif factor == "liquidity_ratio":
        # 单票持仓市值 / 20日均成交额 = x_val
        ratio = max(float(x_val), 1e-6)
        # 确保单票市值远小于 liquidity_cap（20亿），从而纯粹考察成交额比例阈值
        row["avg_amt_20d"] = row["market_value"] / ratio

    elif factor == "total_share_ratio":
        # 持股数 / 总股本 = x_val（同时由于流通股本 <= 总股本，若不大于总股本则保持流通股本安全或按同等股本检验）
        ratio = max(float(x_val), 1e-6)
        row["total_shares"] = row["abs_shares"] / ratio

    elif factor == "float_share_ratio":
        # 持股数 / 流通股本 = x_val（总股本假设充裕，专门考察流通股本限制项 r4）
        ratio = max(float(x_val), 1e-6)
        row["total_float_shares"] = row["abs_shares"] / ratio
        row["total_shares"] = max(row["total_shares"], row["total_float_shares"] * 10.0)

    else:
        raise ValueError(f"不支持的架空分析变量: {factor}")

    return row


def _detect_threshold_markers(factor: str, archetype_key: str, p: Dict) -> List[Dict]:
    """根据当前分析变量与标的类别，返回需要在图表上标注的阈值拐点竖线"""
    arch = ARCHETYPES.get(archetype_key, ARCHETYPES["normal"])
    markers = []
    if factor == "stock_weight":
        stock_th = (
            p["stock_weight_threshold_dual"]
            if arch["sec_code"].startswith(("300", "301", "688"))
            and arch["is_csi300"] == 0
            else p["stock_weight_threshold_other"]
        )
        markers.append(
            {"value": stock_th, "label": f"个股加点起征 ({stock_th*100:.1f}%)", "color": "#2457d6"}
        )
        markers.append(
            {
                "value": p["industry_weight_threshold"],
                "label": f"行业加点起征 ({p['industry_weight_threshold']*100:.1f}%)",
                "color": "#0e9384",
            }
        )
        if arch["is_cc_micro"] == 1:
            markers.append(
                {
                    "value": p["cc_micro_weight_threshold"],
                    "label": f"双创微盘加点 ({p['cc_micro_weight_threshold']*100:.1f}%)",
                    "color": "#7a5af8",
                }
            )
        if arch["is_sse50"] == 0:
            markers.append(
                {
                    "value": p["position_weight_limit"],
                    "label": f"r2个股超限 ({p['position_weight_limit']*100:.1f}%)",
                    "color": "#d92d20",
                }
            )
        markers.append(
            {
                "value": p["industry_weight_limit"],
                "label": f"r5行业超限 ({p['industry_weight_limit']*100:.1f}%)",
                "color": "#b54708",
            }
        )
        if arch["is_cc_micro"] == 1:
            markers.append(
                {
                    "value": p["cc_micro_weight_limit"],
                    "label": f"r7双创超限 ({p['cc_micro_weight_limit']*100:.1f}%)",
                    "color": "#c11574",
                }
            )
    elif factor == "industry_weight":
        markers.append(
            {
                "value": p["industry_weight_threshold"],
                "label": f"行业加点起征 ({p['industry_weight_threshold']*100:.1f}%)",
                "color": "#2457d6",
            }
        )
        markers.append(
            {
                "value": p["industry_weight_limit"],
                "label": f"r5行业超限 ({p['industry_weight_limit']*100:.1f}%)",
                "color": "#d92d20",
            }
        )
    elif factor == "cc_micro_weight":
        markers.append(
            {
                "value": p["cc_micro_weight_threshold"],
                "label": f"双创微盘加点 ({p['cc_micro_weight_threshold']*100:.1f}%)",
                "color": "#2457d6",
            }
        )
        markers.append(
            {
                "value": p["cc_micro_weight_limit"],
                "label": f"r7双创超限 ({p['cc_micro_weight_limit']*100:.1f}%)",
                "color": "#d92d20",
            }
        )
    elif factor == "vol_90d":
        markers.append(
            {
                "value": p["vol_90d_threshold"],
                "label": f"波动率加点起征 ({p['vol_90d_threshold']*100:.1f}%)",
                "color": "#2457d6",
            }
        )
    elif factor == "liquidity_ratio":
        markers.append(
            {
                "value": p["liquidity_threshold"],
                "label": f"r1流动性超限 ({p['liquidity_threshold']*100:.1f}%)",
                "color": "#d92d20",
            }
        )
    elif factor == "total_share_ratio":
        markers.append(
            {
                "value": p["total_share_limit"],
                "label": f"r3总股本超限 ({p['total_share_limit']*100:.2f}%)",
                "color": "#d92d20",
            }
        )
    elif factor == "float_share_ratio":
        markers.append(
            {
                "value": p["float_share_limit"],
                "label": f"r4流通股本超限 ({p['float_share_limit']*100:.2f}%)",
                "color": "#d92d20",
            }
        )
    return markers


def run_theoretical_scenario(payload: Dict) -> Dict:
    """
    执行 Tab 1：架空基准单因素情景分析。
    默认关闭单股数量 < 20 的 upper_im 强行覆盖（除非用户显式勾选 apply_upper_im=True），
    从而保证在未触碰阈值前，起始 IA 严格等于系统给定的 base_im（如 25%）。
    同时支持 compare_all_archetypes=True 同图对比 4 类架空标的。
    """
    factor = payload.get("factor") or "stock_weight"
    if factor not in FACTOR_META:
        raise ValueError(f"不支持的分析因子: {factor}")
    meta = FACTOR_META[factor]

    raw_params = dict(margin.standard_margin_params())
    raw_params.update(payload.get("params") or {})
    apply_upper_im = bool(payload.get("apply_upper_im", False))
    calc_params = dict(raw_params)
    if not apply_upper_im:
        # 设为 1 使单行架空标的不被 "< 20 只强行取 upper_im (80%)" 掩盖真实规则 IA 曲线
        calc_params["upper_position_limit"] = 1
    else:
        calc_params["upper_position_limit"] = max(
            int(raw_params.get("upper_position_limit", 20)), 2
        )

    p = margin._params_with_defaults(calc_params)

    step_values = generate_step_values(
        start=payload.get("start", meta["default_start"]),
        step=payload.get("step", meta["default_step"]),
        steps=payload.get("steps"),
        end=payload.get("end"),
        min_val=meta["min_val"],
        max_val=meta["max_val"],
    )

    archetype_key = payload.get("archetype") or "normal"
    if archetype_key not in ARCHETYPES:
        archetype_key = "normal"

    compare_all = bool(payload.get("compare_all_archetypes", False))

    def _eval_series_for_arch(arch_k: str) -> List[Dict]:
        base_row = _build_safe_baseline_row(arch_k, p)
        records = []
        prev_ia = None
        for idx, x_val in enumerate(step_values):
            step_row = _apply_theoretical_factor_at_step(base_row, factor, x_val, p)
            df_in = pd.DataFrame([step_row])
            res = margin.calculate_stock_margin(df_in, calc_params).iloc[0]

            mv = float(res["market_value"])
            rmv = float(res["restr_mv"])
            restr_ratio = rmv / mv if mv > 0 else 0.0
            ia_val = float(res["adjusted_ia"])
            rule_ia_val = float(res["rule_margin_requirement"]) / mv if mv > 0 else 0.0
            upper_ia_val = float(p["upper_im"])
            delta_ia = (ia_val - prev_ia) if prev_ia is not None else 0.0
            prev_ia = ia_val

            records.append(
                {
                    "step_index": idx,
                    "factor_value": x_val,
                    "stock_weight": float(res["weight_in_portfolio"]),
                    "industry_weight": float(res["industry_weight"]),
                    "cc_micro_weight": float(res["cc_micro_weight"]),
                    "vol_90d": float(res["vol_90d"]),
                    "base_im_after_add_on": float(res["base_im_after_add_on"]),
                    "stock_adj": float(res["stock_adj"]),
                    "industry_adj": float(res["industry_adj"]),
                    "vol_adj": float(res["vol_adj"]),
                    "star_adj": float(res["star_adj"]),
                    "adjustment_sum": float(res["adjustment_sum"]),
                    "maintenance_im": float(res["maintenance_im"]),
                    "r1_ratio": float(res["restriction1"]) / mv if mv > 0 else 0.0,
                    "r2_ratio": float(res["restriction2"]) / mv if mv > 0 else 0.0,
                    "r3_ratio": float(res["restriction3"]) / mv if mv > 0 else 0.0,
                    "r4_ratio": float(res["restriction4"]) / mv if mv > 0 else 0.0,
                    "r5_ratio": float(res["restriction5"]) / mv if mv > 0 else 0.0,
                    "r7_ratio": float(res["restriction7"]) / mv if mv > 0 else 0.0,
                    "restr_ratio": restr_ratio,
                    "active_restriction": str(res["active_restriction"]),
                    "rule_ia": rule_ia_val,
                    "upper_ia": upper_ia_val,
                    "final_ia": ia_val,
                    "delta_ia": delta_ia,
                }
            )
        return records

    primary_rows = _eval_series_for_arch(archetype_key)
    comparison_series = {}
    if compare_all:
        for ak, ainfo in ARCHETYPES.items():
            s_rows = primary_rows if ak == archetype_key else _eval_series_for_arch(ak)
            comparison_series[ak] = {
                "name": ainfo["name"],
                "values": [r["final_ia"] for r in s_rows],
            }

    starting_base_ia = (
        p["base_im"] * (1.0 + p["contract_add_on"])
        if p["contract_add_on_type"] == "relative"
        else p["base_im"] + p["contract_add_on"]
    )

    return {
        "mode": "theoretical",
        "factor": factor,
        "factor_label": meta["label"],
        "archetype": archetype_key,
        "archetype_name": ARCHETYPES[archetype_key]["name"],
        "starting_base_ia": starting_base_ia,
        "step_count": len(primary_rows),
        "rows": primary_rows,
        "comparison_series": comparison_series,
        "threshold_markers": _detect_threshold_markers(factor, archetype_key, p),
    }


def run_portfolio_scenario(payload: Dict) -> Dict:
    """
    执行 Tab 2：基于组合的情景分析（固定组合总市值调仓模式）。
    当调整某只目标股票的个股集中度 w 时：
      - 目标股票市值 = w * TotalMV，其余股票按原相对权重分配 (1 - w) * TotalMV；
      - 完整执行 L2 (calc_concentration) -> L3 (calculate_stock_margin) -> L4 (calculate_portfolio_ia)，
        使该股票所属的申万一级行业集中度、双创微盘集中度及组合内同行业/同板块其他股票的保证金自然联动。
    """
    positions = payload.get("positions") or []
    if not positions:
        raise ValueError("组合不能为空")

    factor = payload.get("factor") or "stock_weight"
    if factor not in FACTOR_META:
        raise ValueError(f"不支持的分析因子: {factor}")
    meta = FACTOR_META[factor]

    portfolio = pd.DataFrame(positions)
    portfolio = concentration.consolidate_positions(portfolio, merge_duplicates=True)
    security = webapp.get_security_data(portfolio["sec_code"].astype(str).tolist())
    l1 = security.drop(columns=["sec_name", "close", "as_of_date"], errors="ignore")
    base_merged = portfolio.merge(l1, on="sec_code", how="left", validate="one_to_one")

    target_code = concentration.normalize_sec_code(
        payload.get("target_sec_code") or str(base_merged["sec_code"].iloc[0])
    )
    if target_code not in base_merged["sec_code"].values:
        target_code = str(base_merged["sec_code"].iloc[0])

    exempt_capacity = bool(payload.get("exempt_capacity_limits", True))
    apply_upper_im = bool(payload.get("apply_upper_im", False))

    raw_params = dict(margin.standard_margin_params())
    raw_params.update(payload.get("params") or {})
    calc_params = dict(raw_params)
    if not apply_upper_im:
        calc_params["upper_position_limit"] = 1
    p = margin._params_with_defaults(calc_params)

    step_values = generate_step_values(
        start=payload.get("start", meta["default_start"]),
        step=payload.get("step", meta["default_step"]),
        steps=payload.get("steps"),
        end=payload.get("end"),
        min_val=meta["min_val"],
        max_val=meta["max_val"],
    )

    total_mv_0 = float(base_merged["market_value"].sum())
    t_mask = base_merged["sec_code"] == target_code
    t_row_0 = base_merged.loc[t_mask].iloc[0]
    target_name = str(t_row_0["sec_name"] or target_code)
    target_ind = str(t_row_0["ind_sw_l1_name"])

    rows = []
    prev_port_ia = None
    prev_stock_ia = None

    for idx, x_val in enumerate(step_values):
        df_step = base_merged.copy()
        df_step["market_value"] = df_step["market_value"].astype(float)
        df_step["abs_shares"] = df_step["abs_shares"].astype(float)

        if factor == "stock_weight":
            w = min(max(float(x_val), 1e-6), 1.0)
            mv_target = w * total_mv_0
            rest_mv_0 = float(df_step.loc[~t_mask, "market_value"].sum())
            df_step.loc[t_mask, "market_value"] = mv_target
            if rest_mv_0 > 0 and (~t_mask).any():
                if w >= 1.0 - 1e-7:
                    # 100% 集中于目标股票时，给其余股票保留极微小的正数市值避免 L3 零市值校验报错
                    df_step.loc[~t_mask, "market_value"] = 1e-3
                else:
                    scale = (total_mv_0 - mv_target) / rest_mv_0
                    df_step.loc[~t_mask, "market_value"] *= scale
            df_step["abs_shares"] = df_step["market_value"] / df_step["price"]

        elif factor == "industry_weight":
            w_ind = min(max(float(x_val), 1e-6), 1.0)
            ind_mask = df_step["ind_sw_l1_name"] == target_ind
            ind_mv_0 = float(df_step.loc[ind_mask, "market_value"].sum())
            rest_mv_0 = float(df_step.loc[~ind_mask, "market_value"].sum())
            if ind_mv_0 > 0:
                df_step.loc[ind_mask, "market_value"] *= (w_ind * total_mv_0) / ind_mv_0
            if rest_mv_0 > 0 and (~ind_mask).any():
                if w_ind >= 1.0 - 1e-7:
                    df_step.loc[~ind_mask, "market_value"] = 1e-3
                else:
                    df_step.loc[~ind_mask, "market_value"] *= (
                        (1.0 - w_ind) * total_mv_0
                    ) / rest_mv_0
            df_step["abs_shares"] = df_step["market_value"] / df_step["price"]

        elif factor == "cc_micro_weight":
            w_cc = min(max(float(x_val), 1e-6), 1.0)
            cc_mask = df_step["is_cc_micro"] == 1
            if not cc_mask.any():
                # 若组合原本无双创微盘股，将目标股票视为双创微盘股进行推演
                df_step.loc[t_mask, "is_cc_micro"] = 1
                cc_mask = df_step["is_cc_micro"] == 1
            cc_mv_0 = float(df_step.loc[cc_mask, "market_value"].sum())
            rest_mv_0 = float(df_step.loc[~cc_mask, "market_value"].sum())
            if cc_mv_0 > 0:
                df_step.loc[cc_mask, "market_value"] *= (w_cc * total_mv_0) / cc_mv_0
            if rest_mv_0 > 0 and (~cc_mask).any():
                if w_cc >= 1.0 - 1e-7:
                    df_step.loc[~cc_mask, "market_value"] = 1e-3
                else:
                    df_step.loc[~cc_mask, "market_value"] *= (
                        (1.0 - w_cc) * total_mv_0
                    ) / rest_mv_0
            df_step["abs_shares"] = df_step["market_value"] / df_step["price"]

        elif factor == "vol_90d":
            df_step.loc[t_mask, "vol_90d"] = max(float(x_val), 0.0)

        elif factor == "liquidity_ratio":
            ratio = max(float(x_val), 1e-6)
            df_step.loc[t_mask, "avg_amt_20d"] = (
                df_step.loc[t_mask, "market_value"] / ratio
            )

        elif factor == "total_share_ratio":
            ratio = max(float(x_val), 1e-6)
            df_step.loc[t_mask, "total_shares"] = (
                df_step.loc[t_mask, "abs_shares"] / ratio
            )

        elif factor == "float_share_ratio":
            ratio = max(float(x_val), 1e-6)
            df_step.loc[t_mask, "total_float_shares"] = (
                df_step.loc[t_mask, "abs_shares"] / ratio
            )

        if exempt_capacity and factor not in {
            "liquidity_ratio",
            "total_share_ratio",
            "float_share_ratio",
        }:
            df_step["avg_amt_20d"] = 1e14
            df_step["total_shares"] = 1e14
            df_step["total_float_shares"] = 1e14

        l2_df, _, _ = concentration.calc_concentration(df_step)
        l3_df = margin.calculate_stock_margin(l2_df, calc_params)
        l4_sum = margin.calculate_portfolio_ia(l3_df)

        t_res = l3_df.loc[l3_df["sec_code"] == target_code].iloc[0]
        mv_i = float(t_res["market_value"])
        rmv_i = float(t_res["restr_mv"])
        stock_ia = float(t_res["adjusted_ia"])
        port_ia = float(l4_sum["portfolio_ia"])

        delta_port_ia = (
            (port_ia - prev_port_ia) if prev_port_ia is not None else 0.0
        )
        delta_stock_ia = (
            (stock_ia - prev_stock_ia) if prev_stock_ia is not None else 0.0
        )
        prev_port_ia = port_ia
        prev_stock_ia = stock_ia

        rows.append(
            {
                "step_index": idx,
                "factor_value": x_val,
                "stock_weight": float(t_res["weight_in_portfolio"]),
                "industry_weight": float(t_res["industry_weight"]),
                "cc_micro_weight": float(t_res["cc_micro_weight"]),
                "vol_90d": float(t_res["vol_90d"]),
                "stock_adj": float(t_res["stock_adj"]),
                "industry_adj": float(t_res["industry_adj"]),
                "vol_adj": float(t_res["vol_adj"]),
                "star_adj": float(t_res["star_adj"]),
                "adjustment_sum": float(t_res["adjustment_sum"]),
                "maintenance_im": float(t_res["maintenance_im"]),
                "restr_ratio": rmv_i / mv_i if mv_i > 0 else 0.0,
                "active_restriction": str(t_res["active_restriction"]),
                "target_stock_ia": stock_ia,
                "delta_stock_ia": delta_stock_ia,
                "portfolio_ia": port_ia,
                "delta_portfolio_ia": delta_port_ia,
                "final_ia": port_ia,
                "delta_ia": delta_port_ia,
            }
        )

    arch_guess = (
        "sse50"
        if int(t_row_0.get("is_sse50", 0)) == 1
        else "dual_non300"
        if target_code.startswith(("300", "301", "688"))
        and int(t_row_0.get("is_csi300", 0)) == 0
        else "main_micro"
        if int(t_row_0.get("is_cc_micro", 0)) == 1
        else "normal"
    )

    return {
        "mode": "portfolio",
        "factor": factor,
        "factor_label": meta["label"],
        "target_sec_code": target_code,
        "target_sec_name": target_name,
        "target_industry": target_ind,
        "step_count": len(rows),
        "rows": rows,
        "threshold_markers": _detect_threshold_markers(factor, arch_guess, p),
    }


def build_scenario_export(result: Dict, params: Dict) -> bytes:
    """将情景分析的步进序列结果与参数配置导出为格式化 Excel"""
    rows = result.get("rows") or []
    mode = result.get("mode", "theoretical")
    df = pd.DataFrame(rows)

    if mode == "theoretical":
        col_map = {
            "step_index": "步号",
            "factor_value": f"分析变量：{result.get('factor_label', '变量值')}",
            "stock_weight": "个股集中度",
            "industry_weight": "联动行业集中度",
            "cc_micro_weight": "联动双创微盘集中度",
            "vol_90d": "90日波动率",
            "base_im_after_add_on": "起始基础保证金率",
            "stock_adj": "个股调整项",
            "industry_adj": "行业调整项",
            "vol_adj": "波动率调整项",
            "star_adj": "双创微盘调整项",
            "adjustment_sum": "调整项合计",
            "maintenance_im": "维持保证金率",
            "r1_ratio": "r1流动性超限率",
            "r2_ratio": "r2个股超限率",
            "r3_ratio": "r3总股本超限率",
            "r4_ratio": "r4流通股本超限率",
            "r5_ratio": "r5行业超限率",
            "r7_ratio": "r7双创超限率",
            "restr_ratio": "最大超限占比(restr_mv/mv)",
            "active_restriction": "主控限制项",
            "final_ia": "最终IA",
            "delta_ia": "较上一步ΔIA",
        }
    else:
        col_map = {
            "step_index": "步号",
            "factor_value": f"分析变量：{result.get('factor_label', '变量值')}",
            "stock_weight": "目标个股集中度",
            "industry_weight": "目标所属行业集中度",
            "cc_micro_weight": "组合双创微盘集中度",
            "vol_90d": "目标90日波动率",
            "stock_adj": "目标个股调整",
            "industry_adj": "目标行业调整",
            "vol_adj": "目标波动率调整",
            "star_adj": "目标双创调整",
            "maintenance_im": "目标维持保证金率",
            "restr_ratio": "目标超限占比",
            "active_restriction": "目标主控限制项",
            "target_stock_ia": "目标单券IA",
            "delta_stock_ia": "单券ΔIA",
            "portfolio_ia": "组合整体IA",
            "delta_portfolio_ia": "组合ΔIA",
        }

    detail = df[[c for c in col_map if c in df.columns]].rename(columns=col_map)
    meta_items = [
        ("分析模式", "架空基准模式" if mode == "theoretical" else "组合推演模式"),
        ("分析变量", result.get("factor_label", "")),
        (
            "分析对象",
            result.get("archetype_name")
            if mode == "theoretical"
            else f"{result.get('target_sec_code')} ({result.get('target_sec_name')})",
        ),
        ("总采样步数", result.get("step_count", len(detail))),
    ]
    for k, v in (params or {}).items():
        meta_items.append((f"参数：{k}", v))
    meta_df = pd.DataFrame(meta_items, columns=["配置项", "数值"])

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        detail.to_excel(writer, sheet_name="情景分析步进明细", index=False)
        meta_df.to_excel(writer, sheet_name="分析参数配置", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for column in ws.columns:
                letter = column[0].column_letter
                width = min(
                    max(
                        len(str(c.value)) if c.value is not None else 0
                        for c in column
                    )
                    + 4,
                    42,
                )
                ws.column_dimensions[letter].width = max(width, 13)
        ws = writer.sheets["情景分析步进明细"]
        ws.auto_filter.ref = ws.dimensions
        skip_pct = {"步号", "主控限制项", "目标主控限制项"}
        for cell in ws[1]:
            if cell.value and cell.value not in skip_pct:
                for row_cells in ws.iter_cols(
                    min_col=cell.column, max_col=cell.column, min_row=2
                ):
                    for item in row_cells:
                        item.number_format = "0.00%"
    return output.getvalue()


class ScenarioHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, body, content_type: str):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, value):
        self._send(
            status,
            json.dumps(webapp._json_ready(value), ensure_ascii=False),
            "application/json; charset=utf-8",
        )

    def do_GET(self):
        if self.path in {"/", "/index.html", "/scenario"}:
            self._send(200, SCENARIO_PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if self.path == "/api/init":
            portfolio, _ = webapp.load_demo()
            self._json(
                200,
                {
                    "standard_params": margin.standard_margin_params(),
                    "demo_params": margin.default_margin_params(),
                    "archetypes": ARCHETYPES,
                    "factors": FACTOR_META,
                    "positions": portfolio.to_dict(orient="records"),
                },
            )
            return
        self._json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path not in {
            "/api/scenario/theoretical",
            "/api/scenario/portfolio",
            "/api/scenario/upload",
            "/api/scenario/export",
        }:
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 15_000_000:
                raise ValueError("请求体为空或过大")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/api/scenario/upload":
                positions = webapp.parse_portfolio_file(
                    payload.get("filename") or "",
                    payload.get("content") or "",
                )
                self._json(200, {"positions": positions})
                return
            if self.path == "/api/scenario/export":
                content = build_scenario_export(
                    payload.get("result") or {},
                    payload.get("params") or {},
                )
                self._send(
                    200,
                    content,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                return
            if self.path == "/api/scenario/theoretical":
                self._json(200, run_theoretical_scenario(payload))
                return
            if self.path == "/api/scenario/portfolio":
                self._json(200, run_portfolio_scenario(payload))
                return
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description="IA 独立情景与敏感性分析服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ScenarioHandler)
    print(f"IA Scenario Analyzer: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
