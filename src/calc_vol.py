"""
波动率计算模块
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calc_volatility(prices: List[float],
                    windows: List[int] = [20, 40, 90],
                    annual_factor: int = 252,
                    min_obs_ratio: float = 0.6) -> Tuple[dict, dict, dict]:
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
    
    # 过滤非法价格：保留序列位置（置 NaN），避免跨停牌缺口被压成单日收益
    has_gap = False
    arr = np.full(len(prices), np.nan, dtype=float)
    for i, p in enumerate(prices):
        try:
            v = float(p)
        except (TypeError, ValueError):
            v = np.nan
        if v <= 0 or np.isnan(v):
            has_gap = True
        else:
            arr[i] = v
    price_flags = ["PRICE_GAP"] if has_gap else []
    
    if np.count_nonzero(~np.isnan(arr)) < 2:
        for w in windows:
            vols[w] = None
            ns[w] = 0
            flags[w] = ["INSUFFICIENT_SAMPLE"]
        return vols, ns, flags
    
    # 在完整序列上计算对数收益率；含 NaN 的位置收益率天然为 NaN，跨缺口收益不参与波动率
    log_returns = np.log(arr[1:] / arr[:-1])
    
    for w in windows:
        # 取最近 w 个收益率
        if len(log_returns) < w:
            r_window = log_returns
        else:
            r_window = log_returns[-w:]
        
        # 过滤 NaN
        r_clean = r_window[~np.isnan(r_window)]
        n = len(r_clean)
        ns[w] = n
        
        min_obs = int(w * min_obs_ratio)
        if n < min_obs:
            vols[w] = None
            flags[w] = ["INSUFFICIENT_SAMPLE"]
            if price_flags:
                flags[w].extend(price_flags)
            flags[w] = list(dict.fromkeys(flags[w]))  # 去重
            continue
        
        # 样本标准差 (ddof=1)
        std = np.std(r_clean, ddof=1)
        vol = std * np.sqrt(annual_factor)
        vols[w] = vol
        flags[w] = []
        if price_flags:
            flags[w].extend(list(dict.fromkeys(price_flags)))
    
    return vols, ns, flags


def calc_vol_max(vols: dict) -> Optional[float]:
    """取非空波动率中的最大值"""
    valid_vols = [v for v in vols.values() if v is not None]
    return max(valid_vols) if valid_vols else None


def format_vol_pct(val: Optional[float]) -> str:
    """将波动率小数格式化为百分比字符串，如 0.2501 -> '25.01%'"""
    if val is None:
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
