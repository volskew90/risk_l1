"""L3 单券保证金与 L4 组合 IA 计算。"""
from typing import Dict

import numpy as np
import pandas as pd


def default_margin_params() -> Dict:
    return {
        "base_im": 0.25,
        "contract_add_on_type": "absolute",
        "contract_add_on": 0.0,
        "upper_im": 0.80,
        "upper_compare": "max",
        "upper_position_limit": 20,
        "stock_weight_threshold_dual": 0.05,
        "stock_weight_threshold_other": 0.10,
        "stock_adj_ratio": 0.7,
        "industry_weight_threshold": 0.20,
        "industry_adj_ratio": 0.4,
        "vol_90d_threshold": 0.60,
        "vol_adj_ratio": 0.7,
        "cc_micro_weight_threshold": 0.30,
        "cc_micro_adj_ratio": 0.4,
        "liquidity_threshold": 0.15,
        "liquidity_cap": 2000000000.0,
        "position_weight_limit": 0.01,
        "total_share_limit": 0.01,
        "float_share_limit": 0.02,
        "industry_weight_limit": 0.40,
        "cc_micro_weight_limit": 0.50,
    }


def _params_with_defaults(params: Dict) -> Dict:
    merged = {**default_margin_params(), **(params or {})}
    for key, value in merged.items():
        if key in {"contract_add_on_type", "upper_compare"}:
            continue
        try:
            merged[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"参数 {key} 必须是数值") from exc
        if not np.isfinite(merged[key]) or merged[key] < 0:
            raise ValueError(f"参数 {key} 必须是非负有限数值")
    if merged["contract_add_on_type"] not in {"absolute", "relative"}:
        raise ValueError("contract_add_on_type 仅支持 absolute / relative")
    if merged["upper_compare"] not in {"max", "min"}:
        raise ValueError("upper_compare 仅支持 max / min")
    if merged["upper_position_limit"] < 1:
        raise ValueError("upper_position_limit 必须大于等于 1")
    return merged


def calculate_stock_margin(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """根据第二层输入逐券计算调整项、限制项、最终保证金和调整后 IA"""
    p = _params_with_defaults(params)
    required = [
        "sec_code", "price", "abs_shares", "market_value", "total_mv",
        "weight_in_portfolio", "industry_mv", "industry_weight",
        "weight_in_industry", "cc_micro_mv", "cc_micro_weight",
        "weight_in_cc_micro", "total_shares", "total_float_shares",
        "avg_amt_20d", "vol_90d", "is_sse50", "is_csi300", "is_cc_micro",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"第三层输入缺少字段: {missing}")
    if df.empty or (df["market_value"] <= 0).any():
        raise ValueError("第三层输入必须为非空组合，且单券市值大于 0")
    numeric = [c for c in required if c not in {"sec_code"}]
    if df[numeric].isna().any().any():
        bad = df.columns[df[numeric].isna().any()].tolist()
        raise ValueError(f"第三层输入存在空值: {bad}")

    out = df.copy()
    code = out["sec_code"].astype(str)
    is_dual_non300 = (code.str.startswith(("300", "301", "688"))
                      & (out["is_csi300"] == 0))
    stock_threshold = np.where(
        is_dual_non300,
        p["stock_weight_threshold_dual"],
        p["stock_weight_threshold_other"],
    )
    out["stock_adj"] = (
        np.maximum(out["weight_in_portfolio"] - stock_threshold, 0)
        * p["stock_adj_ratio"]
    )
    out["industry_adj"] = (
        np.maximum(out["industry_weight"] - p["industry_weight_threshold"], 0)
        * p["industry_adj_ratio"]
    ).round(4)
    out["vol_adj"] = (
        np.maximum(out["vol_90d"] - p["vol_90d_threshold"], 0)
        * p["vol_adj_ratio"]
    ).round(4)
    out["star_adj"] = np.where(
        out["is_cc_micro"] == 1,
        np.clip(
            (out["cc_micro_weight"] - p["cc_micro_weight_threshold"])
            * p["cc_micro_adj_ratio"],
            0,
            1,
        ),
        0.0,
    )
    out["adjustment_sum"] = out[
        ["stock_adj", "industry_adj", "vol_adj", "star_adj"]
    ].sum(axis=1)

    if p["contract_add_on_type"] == "relative":
        out["base_im_after_add_on"] = p["base_im"] * (1 + p["contract_add_on"])
    else:
        out["base_im_after_add_on"] = p["base_im"] + p["contract_add_on"]
    out["maintenance_im"] = (
        out["base_im_after_add_on"] + out["adjustment_sum"]
    ).clip(upper=1.0)

    out["restriction1"] = (
        out["market_value"]
        - np.minimum(
            out["avg_amt_20d"] * p["liquidity_threshold"],
            p["liquidity_cap"],
        )
    ).clip(lower=0)
    out["restriction2"] = np.where(
        out["is_sse50"] == 0,
        (out["market_value"] - p["position_weight_limit"] * out["total_mv"]).clip(lower=0),
        0.0,
    )
    out["restriction3"] = (
        (out["abs_shares"] - p["total_share_limit"] * out["total_shares"]).clip(lower=0)
        * out["price"]
    )
    out["restriction4"] = (
        (out["abs_shares"] - p["float_share_limit"] * out["total_float_shares"]).clip(lower=0)
        * out["price"]
    )
    out["restriction5"] = (
        (out["industry_mv"] - p["industry_weight_limit"] * out["total_mv"]).clip(lower=0)
        * out["weight_in_industry"]
    )
    out["restriction7"] = np.where(
        out["is_cc_micro"] == 1,
        (
            (out["cc_micro_mv"] - p["cc_micro_weight_limit"] * out["total_mv"]).clip(lower=0)
            * out["weight_in_cc_micro"]
        ).round(2),
        0.0,
    )
    restriction_cols = [
        "restriction1", "restriction2", "restriction3",
        "restriction4", "restriction5", "restriction7",
    ]
    out["restr_mv"] = out[restriction_cols].max(axis=1)
    out["rule_margin_requirement"] = (
        (out["market_value"] - out["restr_mv"]) * out["maintenance_im"]
        + out["restr_mv"]
    ).round(2)
    out["upper_margin_requirement"] = (out["market_value"] * p["upper_im"]).round(2)
    if len(out) < int(p["upper_position_limit"]):
        if p["upper_compare"] == "max":
            out["margin_requirement"] = out[
                ["rule_margin_requirement", "upper_margin_requirement"]
            ].max(axis=1)
        else:
            out["margin_requirement"] = out[
                ["rule_margin_requirement", "upper_margin_requirement"]
            ].min(axis=1)
    else:
        out["margin_requirement"] = out["rule_margin_requirement"]
    out["adjusted_ia"] = out["margin_requirement"] / out["market_value"]
    return out


def calculate_portfolio_ia(df: pd.DataFrame) -> Dict:
    """L4：同时返回总保证金/总市值和个券 IA 市值加权平均"""
    if df.empty:
        raise ValueError("无法计算空组合的 IA")
    total_mv = float(df["market_value"].sum())
    total_margin = float(df["margin_requirement"].sum())
    if total_mv <= 0:
        raise ValueError("组合总市值必须大于 0")
    portfolio_ia = total_margin / total_mv
    weighted_ia = float((df["adjusted_ia"] * df["market_value"]).sum() / total_mv)
    return {
        "n_positions": int(len(df)),
        "total_market_value": total_mv,
        "total_margin_requirement": round(total_margin, 2),
        "portfolio_ia": portfolio_ia,
        "portfolio_ia_weighted": weighted_ia,
        "ia_method_difference": weighted_ia - portfolio_ia,
    }
