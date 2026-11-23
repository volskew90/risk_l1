"""L3 单券保证金与 L4 组合 IA 计算及归因剖析模块（对齐《IA_Margin_Rule.html》新规）。"""
from typing import Dict

import numpy as np
import pandas as pd


def default_margin_params() -> Dict:
    """
    默认保证金参数（全面对齐《IA_Margin_Rule.html》第十二节参数速查表）：
    - base_im: 25%
    - stock_adj: 双创非沪深300阈值 5% / 其他 10%，比例 0.7
    - industry_adj: 阈值 20%，比例 0.4
    - vol_adj: 90日波动率阈值 60%，比例 0.4
    - star_adj: 双创/微盘非1800阈值 30%，比例 0.4
    - r1 流动性超限: 20日均成交额 × 100%，上限 20 亿元
    - r2 个股集中度超限: 15%（上证50豁免）
    - r3 总股本超限: 1%
    - r4 流通股本超限: 2%
    - r5 行业集中度超限: 40%
    - r7 微盘+双创超限: 50%
    """
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
        "vol_adj_ratio": 0.4,
        "cc_micro_weight_threshold": 0.30,
        "cc_micro_adj_ratio": 0.4,
        "liquidity_threshold": 1.0,
        "liquidity_cap": 2000000000.0,
        "position_weight_limit": 0.15,
        "total_share_limit": 0.01,
        "float_share_limit": 0.02,
        "industry_weight_limit": 0.40,
        "cc_micro_weight_limit": 0.50,
    }


def standard_margin_params() -> Dict:
    """业务规范标准参数（与 default_margin_params 一致，对齐《IA_Margin_Rule.html》）"""
    return default_margin_params()


def legacy_demo_margin_params() -> Dict:
    """旧版演示参数预设（r1=15%, r2=1%, vol_adj_ratio=0.7）"""
    params = default_margin_params()
    params.update(
        {
            "vol_adj_ratio": 0.7,
            "liquidity_threshold": 0.15,
            "position_weight_limit": 0.01,
        }
    )
    return params


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
    if merged["upper_compare"] not in {"max", "min", "rule2"}:
        raise ValueError("upper_compare 仅支持 max / min / rule2")
    if merged["upper_position_limit"] < 1:
        raise ValueError("upper_position_limit 必须大于等于 1")
    return merged


_RESTRICTION_LABELS = [
    ("restriction1", "r1:流动性"),
    ("restriction2", "r2:个股集中度"),
    ("restriction3", "r3:总股本"),
    ("restriction4", "r4:流通股本"),
    ("restriction5", "r5:行业超限"),
    ("restriction7", "r7:双创微盘"),
]


def _core_margin_calc(out: pd.DataFrame, p: Dict) -> pd.DataFrame:
    """核心单券保证金向量化计算（不含边际敏感度微扰）"""
    orig_index = out.index
    out = out.reset_index(drop=True)
    code = out["sec_code"].astype(str)
    is_dual_non300 = code.str.startswith(("300", "301", "688")) & (
        out["is_csi300"] == 0
    )
    stock_threshold = np.where(
        is_dual_non300,
        p["stock_weight_threshold_dual"],
        p["stock_weight_threshold_other"],
    )

    # 识别特殊 100% 兜底标的（第八节 & TC15）：
    # 停牌股、风险警示股(ST/*ST)、不在 A 股标的表中的未知标的、或 90 日波动率缺失标的
    is_suspended = (
        out["is_suspended"].fillna(0).astype(int)
        if "is_suspended" in out.columns
        else pd.Series(0, index=out.index)
    )
    is_risk_warning = (
        out["is_risk_warning"].fillna(0).astype(int)
        if "is_risk_warning" in out.columns
        else pd.Series(0, index=out.index)
    )
    is_unknown_sec = (
        out["is_unknown_security"].fillna(0).astype(int)
        if "is_unknown_security" in out.columns
        else pd.Series(0, index=out.index)
    )
    if "ind_sw_l1_name" in out.columns:
        is_unknown_sec = is_unknown_sec | (
            out["ind_sw_l1_name"].fillna("未知").astype(str) == "未知"
        ).astype(int)

    vol_missing = out["vol_90d"].isna()
    is_special_100 = (
        (is_suspended == 1)
        | (is_risk_warning == 1)
        | (is_unknown_sec == 1)
        | vol_missing
    )

    reasons = []
    for pos in range(len(out)):
        r_list = []
        if int(is_suspended.iloc[pos]) == 1:
            r_list.append("停牌股")
        if int(is_risk_warning.iloc[pos]) == 1:
            r_list.append("风险警示")
        if int(is_unknown_sec.iloc[pos]) == 1:
            r_list.append("不在A股标的表中")
        if bool(vol_missing.iloc[pos]) and int(is_unknown_sec.iloc[pos]) == 0:
            r_list.append("波动率缺失")
        reasons.append("/".join(r_list) if r_list else "正常")
    out["is_special_100"] = is_special_100.astype(int)
    out["special_reason"] = reasons

    vol_clean = out["vol_90d"].fillna(0.0).astype(float)

    # 第四节：4 项调整项计算
    out["stock_adj"] = (
        np.maximum(out["weight_in_portfolio"] - stock_threshold, 0.0)
        * p["stock_adj_ratio"]
    ).round(6)
    out["industry_adj"] = (
        np.maximum(out["industry_weight"] - p["industry_weight_threshold"], 0.0)
        * p["industry_adj_ratio"]
    ).round(4)
    out["vol_adj"] = (
        np.maximum(vol_clean - p["vol_90d_threshold"], 0.0) * p["vol_adj_ratio"]
    ).round(4)
    # star_adj 仅适用于双创或微盘股且非 1800 成分股（is_cc_micro == 1），非此类标的恒为 0
    out["star_adj"] = np.where(
        out["is_cc_micro"] == 1,
        np.maximum(
            (out["cc_micro_weight"] - p["cc_micro_weight_threshold"])
            * p["cc_micro_adj_ratio"],
            0.0,
        ),
        0.0,
    ).round(6)
    out["adjustment_sum"] = out[
        ["stock_adj", "industry_adj", "vol_adj", "star_adj"]
    ].sum(axis=1)

    # 第六节：基础保证金率 base_im（支持标的级 stock_base_im：25% / 30% / 100%）与框架合约加点
    if "stock_base_im" in out.columns and out["stock_base_im"].notna().any():
        raw_base = out["stock_base_im"].fillna(p["base_im"]).astype(float)
    else:
        raw_base = pd.Series(float(p["base_im"]), index=out.index)
    raw_base = np.where(is_special_100, 1.0, raw_base)

    if p["contract_add_on_type"] == "relative":
        out["base_im_after_add_on"] = raw_base * (1.0 + p["contract_add_on"])
    else:
        out["base_im_after_add_on"] = raw_base + p["contract_add_on"]

    # 正常层保证金率 = MIN(adj_base_im + Σ调整项, 100%)
    out["maintenance_im"] = np.where(
        is_special_100,
        1.0,
        (out["base_im_after_add_on"] + out["adjustment_sum"]).clip(upper=1.0),
    )

    avg_amt_clean = out["avg_amt_20d"].fillna(0.0).astype(float)
    total_shares_clean = out["total_shares"].fillna(0.0).astype(float)
    float_shares_clean = out["total_float_shares"].fillna(0.0).astype(float)

    # 第五节：限制项（超限部分 100% 计收，合并取 MAX）
    out["restriction1"] = (
        out["market_value"]
        - np.minimum(
            avg_amt_clean * p["liquidity_threshold"],
            p["liquidity_cap"],
        )
    ).clip(lower=0.0)
    out["restriction2"] = np.where(
        out["is_sse50"] == 0,
        (out["market_value"] - p["position_weight_limit"] * out["total_mv"]).clip(
            lower=0.0
        ),
        0.0,
    )
    out["restriction3"] = np.where(
        total_shares_clean > 0,
        (out["abs_shares"] - p["total_share_limit"] * total_shares_clean).clip(
            lower=0.0
        )
        * out["price"],
        0.0,
    )
    out["restriction4"] = np.where(
        float_shares_clean > 0,
        (out["abs_shares"] - p["float_share_limit"] * float_shares_clean).clip(
            lower=0.0
        )
        * out["price"],
        0.0,
    )
    out["restriction5"] = (
        (out["industry_mv"] - p["industry_weight_limit"] * out["total_mv"]).clip(
            lower=0.0
        )
        * out["weight_in_industry"]
    )
    out["restriction7"] = np.where(
        out["is_cc_micro"] == 1,
        (
            (out["cc_micro_mv"] - p["cc_micro_weight_limit"] * out["total_mv"]).clip(
                lower=0.0
            )
            * out["weight_in_cc_micro"]
        ).round(2),
        0.0,
    )
    restriction_cols = [c for c, _ in _RESTRICTION_LABELS]
    out["restr_mv"] = out[restriction_cols].max(axis=1)

    # 识别主控限制项（触发最大超限金额的限制项名称）
    active_labels = []
    for pos in range(len(out)):
        rmv = float(out["restr_mv"].iloc[pos])
        if rmv <= 1e-6:
            active_labels.append("无")
        else:
            matched = [
                label
                for col, label in _RESTRICTION_LABELS
                if abs(float(out[col].iloc[pos]) - rmv) <= 1e-2
            ]
            active_labels.append("/".join(matched) if matched else "超限")
    out["active_restriction"] = active_labels

    out["normal_mv"] = (out["market_value"] - out["restr_mv"]).clip(lower=0.0)
    out["rule_margin_requirement"] = np.where(
        is_special_100,
        out["market_value"].round(2),
        (out["normal_mv"] * out["maintenance_im"] + out["restr_mv"]).round(2),
    )
    out["upper_margin_requirement"] = (out["market_value"] * p["upper_im"]).round(2)

    n_pos = len(out)
    limit_n = int(p["upper_position_limit"])
    if p["upper_compare"] == "rule2":
        # 券商B规则2：标的数量 < 阈值时直接取 upper_im (B)；标的数量 >= 阈值时取 MIN(A, B)
        if n_pos < limit_n:
            compared_margin = out["upper_margin_requirement"]
        else:
            compared_margin = out[
                ["rule_margin_requirement", "upper_margin_requirement"]
            ].min(axis=1)
    elif n_pos < limit_n:
        if p["upper_compare"] == "max":
            compared_margin = out[
                ["rule_margin_requirement", "upper_margin_requirement"]
            ].max(axis=1)
        else:
            compared_margin = out[
                ["rule_margin_requirement", "upper_margin_requirement"]
            ].min(axis=1)
    else:
        compared_margin = out["rule_margin_requirement"]

    # 特殊 100% 兜底标的（停牌/ST/未知标的/波动率缺失）始终按 100% 计收，不被 upper_im 取小优惠降级
    out["margin_requirement"] = np.where(
        is_special_100,
        out["market_value"].round(2),
        compared_margin,
    )

    out["adjusted_ia"] = out["margin_requirement"] / out["market_value"]
    out.index = orig_index
    return out


def _recompute_l2_weights(df: pd.DataFrame) -> pd.DataFrame:
    """当单券市值微扰后，重新同步组合总市值、行业集中度和双创微盘集中度"""
    sub = df.copy()
    total_mv = float(sub["market_value"].sum())
    sub["total_mv"] = total_mv
    sub["weight_in_portfolio"] = sub["market_value"] / total_mv

    if "ind_sw_l1_name" in sub.columns:
        ind_mv = sub.groupby("ind_sw_l1_name")["market_value"].transform("sum")
        sub["industry_mv"] = ind_mv
        sub["industry_weight"] = ind_mv / total_mv
        sub["weight_in_industry"] = np.where(
            ind_mv > 0, sub["market_value"] / ind_mv, 0.0
        )

    cc_mv = float(sub.loc[sub["is_cc_micro"] == 1, "market_value"].sum())
    sub["cc_micro_mv"] = cc_mv
    sub["cc_micro_weight"] = cc_mv / total_mv
    sub["weight_in_cc_micro"] = np.where(
        (sub["is_cc_micro"] == 1) & (cc_mv > 0),
        sub["market_value"] / cc_mv,
        0.0,
    )
    return sub


def calculate_stock_margin(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """根据第二层输入逐券计算调整项、限制项、最终保证金、调整后 IA 及全组合边际 IA"""
    p = _params_with_defaults(params)
    required = [
        "sec_code",
        "price",
        "abs_shares",
        "market_value",
        "total_mv",
        "weight_in_portfolio",
        "industry_mv",
        "industry_weight",
        "weight_in_industry",
        "cc_micro_mv",
        "cc_micro_weight",
        "weight_in_cc_micro",
        "total_shares",
        "total_float_shares",
        "avg_amt_20d",
        "vol_90d",
        "is_sse50",
        "is_csi300",
        "is_cc_micro",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"第三层输入缺少字段: {missing}")
    if df.empty or (df["market_value"] <= 0).any():
        raise ValueError("第三层输入必须为非空组合，且单券市值大于 0")

    # 核心结构列不允许为空；vol_90d / 股本 / 成交额允许为空（自动走 100% 特殊标的兜底）
    strict_numeric = [
        c
        for c in required
        if c
        not in {
            "sec_code",
            "vol_90d",
            "total_shares",
            "total_float_shares",
            "avg_amt_20d",
        }
    ]
    if df[strict_numeric].isna().any().any():
        bad = df.columns[df[strict_numeric].isna().any()].tolist()
        raise ValueError(f"第三层输入存在空值: {bad}")

    out = _core_margin_calc(df.copy(), p)

    # 计算全组合边际 IA（Marginal IA）
    sum_mv = float(out["market_value"].sum())
    reported_total_mv = float(out["total_mv"].iloc[0])
    is_full_portfolio = (
        abs(sum_mv - reported_total_mv) <= max(1.0, sum_mv * 1e-4)
        and "ind_sw_l1_name" in out.columns
        and len(out) >= 1
    )

    if is_full_portfolio:
        base_total_margin = float(out["margin_requirement"].sum())
        marginal_list = []
        base_df = df.reset_index(drop=True).copy()
        base_df["market_value"] = base_df["market_value"].astype(float)
        base_df["abs_shares"] = base_df["abs_shares"].astype(float)
        base_df["price"] = base_df["price"].astype(float)
        for pos in range(len(base_df)):
            mv_i = float(base_df["market_value"].iloc[pos])
            delta_mv = max(mv_i * 0.01, 100.0)
            perturbed = base_df.copy()
            perturbed.loc[pos, "market_value"] = mv_i + delta_mv
            perturbed.loc[pos, "abs_shares"] = (
                perturbed.loc[pos, "market_value"] / perturbed.loc[pos, "price"]
            )
            perturbed = _recompute_l2_weights(perturbed)
            p_res = _core_margin_calc(perturbed, p)
            new_total_margin = float(p_res["margin_requirement"].sum())
            marginal_list.append(
                max(0.0, min((new_total_margin - base_total_margin) / delta_mv, 2.0))
            )
        out["marginal_ia"] = pd.Series(marginal_list, index=out.index).round(6)
    else:
        out["marginal_ia"] = out["adjusted_ia"]

    return out


def calculate_portfolio_ia(df: pd.DataFrame) -> Dict:
    """L4：同时返回总保证金/总市值、个券 IA 市值加权平均及组合 IA 构成分解"""
    if df.empty:
        raise ValueError("无法计算空组合的 IA")
    total_mv = float(df["market_value"].sum())
    total_margin = float(df["margin_requirement"].sum())
    if total_mv <= 0:
        raise ValueError("组合总市值必须大于 0")
    portfolio_ia = total_margin / total_mv
    weighted_ia = float((df["adjusted_ia"] * df["market_value"]).sum() / total_mv)

    restr_mv_sum = (
        float(df["restr_mv"].sum()) if "restr_mv" in df.columns else 0.0
    )
    normal_mv_sum = max(total_mv - restr_mv_sum, 0.0)

    if {
        "normal_mv",
        "base_im_after_add_on",
        "stock_adj",
        "industry_adj",
        "vol_adj",
        "star_adj",
        "rule_margin_requirement",
    }.issubset(df.columns):
        normal_mv = df["normal_mv"]
        base_contrib = float((normal_mv * df["base_im_after_add_on"]).sum() / total_mv)
        stock_contrib = float((normal_mv * df["stock_adj"]).sum() / total_mv)
        ind_contrib = float((normal_mv * df["industry_adj"]).sum() / total_mv)
        vol_contrib = float((normal_mv * df["vol_adj"]).sum() / total_mv)
        star_contrib = float((normal_mv * df["star_adj"]).sum() / total_mv)
        restr_contrib = float(restr_mv_sum / total_mv)
        rule_total = float(df["rule_margin_requirement"].sum())
        upper_contrib = float((total_margin - rule_total) / total_mv)
    else:
        base_contrib = portfolio_ia
        stock_contrib = 0.0
        ind_contrib = 0.0
        vol_contrib = 0.0
        star_contrib = 0.0
        restr_contrib = 0.0
        upper_contrib = 0.0

    n_restricted = (
        int((df["restr_mv"] > 0).sum()) if "restr_mv" in df.columns else 0
    )
    n_special = (
        int((df["is_special_100"] == 1).sum())
        if "is_special_100" in df.columns
        else 0
    )

    return {
        "n_positions": int(len(df)),
        "total_market_value": round(total_mv, 2),
        "total_margin_requirement": round(total_margin, 2),
        "portfolio_ia": portfolio_ia,
        "portfolio_ia_weighted": weighted_ia,
        "ia_method_difference": weighted_ia - portfolio_ia,
        "normal_market_value": round(normal_mv_sum, 2),
        "restricted_market_value": round(restr_mv_sum, 2),
        "restricted_mv_ratio": restr_mv_sum / total_mv,
        "n_restricted_stocks": n_restricted,
        "n_special_stocks": n_special,
        "ia_breakdown": {
            "base_im_contrib": base_contrib,
            "stock_adj_contrib": stock_contrib,
            "industry_adj_contrib": ind_contrib,
            "vol_adj_contrib": vol_contrib,
            "star_adj_contrib": star_contrib,
            "restriction_contrib": restr_contrib,
            "upper_im_contrib": upper_contrib,
        },
    }
