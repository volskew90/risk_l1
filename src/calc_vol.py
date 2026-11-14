"""
波动率计算模块
支持单序列与全市场批量多窗口年化波动率计算（向量化加速）。
"""
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_clean_price_array(prices: Sequence) -> Tuple[np.ndarray, bool]:
    """
    将输入价格序列转换为浮点数组，过滤非正数与非法值并保留原序列位置（置为 NaN），
    避免跨停牌缺口被压缩成单日收益。
    """
    n_total = len(prices)
    if n_total == 0:
        return np.empty(0, dtype=float), False

    try:
        arr = np.asarray(prices, dtype=float)
    except (TypeError, ValueError):
        arr = pd.to_numeric(pd.Series(list(prices), dtype="object"), errors="coerce").to_numpy(dtype=float)

    invalid_mask = ~np.isfinite(arr) | (arr <= 0)
    has_gap = bool(invalid_mask.any())
    if has_gap:
        arr = arr.copy()
        arr[invalid_mask] = np.nan
    return arr, has_gap


def calc_volatility(
    prices: Sequence[float],
    windows: List[int] = [20, 40, 90],
    annual_factor: int = 252,
    min_obs_ratio: float = 0.6,
) -> Tuple[dict, dict, dict]:
    """
    计算多窗口年化波动率

    参数:
        prices: 前复权收盘价序列，按交易日升序（最近价格在末尾）
        windows: 波动率窗口列表
        annual_factor: 年化因子（默认252）
        min_obs_ratio: 最小有效样本比例

    返回:
        vols: {window: 波动率值或None}
        ns: {window: 有效样本数}
        flags: {window: 质量标记列表}
    """
    vols = {}
    ns = {}
    flags = {}

    arr, has_gap = _to_clean_price_array(prices)
    price_flags = ["PRICE_GAP"] if has_gap else []

    if np.count_nonzero(~np.isnan(arr)) < 2:
        for w in windows:
            vols[w] = None
            ns[w] = 0
            flags[w] = ["INSUFFICIENT_SAMPLE"]
        return vols, ns, flags

    # 在完整序列上计算对数收益率；含 NaN 的位置收益率天然为 NaN，跨缺口收益不参与波动率
    log_returns = np.log(arr[1:] / arr[:-1])
    sqrt_ann = float(np.sqrt(annual_factor))

    for w in windows:
        # 取最近 w 个收益率
        r_window = log_returns if len(log_returns) < w else log_returns[-w:]

        # 过滤 NaN
        r_clean = r_window[~np.isnan(r_window)]
        n = int(len(r_clean))
        ns[w] = n

        min_obs = max(int(w * min_obs_ratio), 2)
        if n < min_obs:
            vols[w] = None
            flags[w] = ["INSUFFICIENT_SAMPLE"]
            if price_flags:
                flags[w].extend(price_flags)
            flags[w] = list(dict.fromkeys(flags[w]))
            continue

        # 样本标准差 (ddof=1)
        std = float(np.std(r_clean, ddof=1))
        vols[w] = std * sqrt_ann
        flags[w] = list(dict.fromkeys(price_flags)) if price_flags else []

    return vols, ns, flags


def calc_vol_max(vols: dict) -> Optional[float]:
    """取非空波动率中的最大值"""
    valid_vols = [v for v in vols.values() if v is not None and not pd.isna(v)]
    return max(valid_vols) if valid_vols else None


def calc_volatility_batch(
    codes: Sequence[str],
    prices_dict: Dict[str, List[float]],
    windows: List[int] = [20, 40, 90],
    annual_factor: int = 252,
    min_obs_ratio: float = 0.6,
) -> pd.DataFrame:
    """
    批量计算股票列表的多窗口波动率与质量标记，供每日流水线高效调用。
    """
    records = []
    for code in codes:
        prices = prices_dict.get(code, [])
        vols, ns, flags = calc_volatility(
            prices,
            windows=windows,
            annual_factor=annual_factor,
            min_obs_ratio=min_obs_ratio,
        )
        vol_max = calc_vol_max(vols)
        all_flags = set()
        for f in flags.values():
            all_flags.update(f)
        quality_flag = ";".join(sorted(all_flags)) if all_flags else "OK"

        row = {"sec_code": code, "vol_max": vol_max, "quality_flag": quality_flag}
        for w in windows:
            row[f"vol_{w}d"] = vols.get(w)
            row[f"n_{w}d"] = ns.get(w)
        records.append(row)
    return pd.DataFrame(records)


def format_vol_pct(val: Optional[float]) -> str:
    """将波动率小数格式化为百分比字符串，如 0.2501 -> '25.01%'"""
    if val is None or pd.isna(val):
        return ""
    return f"{val * 100:.2f}%"


# ---- 单元测试用例（文档 §4.3） ----

def _test_example():
    """文档 §4.3 验收用例"""
    prices = [10.00, 10.20, 10.10, 10.30, 10.25]
    vols, ns, flags = calc_volatility(prices, windows=[4], annual_factor=252)
    vol = vols[4]
    expected = 0.2501  # 25.01%
    assert vol is not None, "vol should not be None"
    assert abs(vol - expected) < 0.0001, f"Expected {expected}, got {vol}"
    print(f"✓ 单元测试通过: vol={vol:.4f} ({vol*100:.2f}%)")


if __name__ == "__main__":
    _test_example()
