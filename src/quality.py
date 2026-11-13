"""
质量校验模块
校验规则见文档 §7.3
"""
import logging
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from src import db

logger = logging.getLogger(__name__)


class QualityCheckError(Exception):
    """质量校验未通过，阻断发布"""
    pass


def run_checks(df: pd.DataFrame, trade_date: str, db_path: str,
               config: dict) -> Tuple[bool, Dict, str]:
    """
    执行质量校验
    校验分两级：
      - hard: 不通过则阻断发布（数据不可信）
      - soft: 仅告警不阻断（可能是上游波动，需人工关注）
    soft 列表可通过 config["quality_soft_checks"] 配置
    返回: (passed, check_results, message)
    """
    checks = {}
    messages = []
    soft_set = set(config.get("quality_soft_checks", ["sse50_count", "csi300_count"]))
    
    def _record(name: str, passed: bool, msg: Optional[str] = None, **extra):
        severity = "soft" if name in soft_set else "hard"
        checks[name] = {"passed": bool(passed), "severity": severity, **extra}
        if not passed and msg:
            prefix = "[soft] " if severity == "soft" else ""
            messages.append(prefix + msg)
    
    # 1. 行数较前一日变动 > 阈值
    prev_count = _get_prev_count(trade_date, db_path)
    curr_count = len(df)
    if prev_count > 0:
        change_pct = abs(curr_count - prev_count) / prev_count * 100
    else:
        change_pct = 0
    threshold = config.get("row_change_alert_pct", 2.0)
    _record("row_change", change_pct <= threshold,
            f"Row count changed {change_pct:.1f}% (prev={prev_count}, curr={curr_count})",
            threshold=threshold, actual=round(change_pct, 2))
    
    # 2. 关键字段非空率（列缺失直接判失败——列缺失意味着上游接线断了，不能静默跳过）
    key_cols = ["sec_name", "board", "ind_sw_l1_code"]
    for col in key_cols:
        if col not in df.columns:
            _record(f"nonnull_{col}", False, f"{col} column missing from snapshot",
                    threshold=99.0, actual=None)
            continue
        non_null_rate = df[col].notna().mean() * 100
        _record(f"nonnull_{col}", non_null_rate >= 99.0,
                f"{col} non-null rate {non_null_rate:.1f}% < 99%",
                threshold=99.0, actual=round(non_null_rate, 2))
    
    # 3. 价格覆盖率：大批量取数失败（如配额耗尽）时必须阻断，避免残缺数据覆盖历史好数据
    if "vol_20d" in df.columns:
        cov = df["vol_20d"].notna().mean() * 100
        cov_threshold = config.get("min_price_coverage_pct", 95.0)
        _record("price_coverage", cov >= cov_threshold,
                f"vol_20d coverage {cov:.1f}% < {cov_threshold}% (大批量取数失败?)",
                threshold=cov_threshold, actual=round(cov, 2))
    
    # 4. 波动率取值范围（阈值从配置读取）
    vol_min, vol_max = (config.get("vol_range") or [0.0, 5.0])[:2]
    vol_bad_pct_limit = config.get("vol_bad_pct_alert", 0.5)
    vol_cols = ["vol_20d", "vol_40d", "vol_90d"]
    bad_vol_count = 0
    vol_total = 0
    for col in vol_cols:
        if col in df.columns:
            valid = df[col].dropna()
            vol_total += len(valid)
            bad_vol_count += len(valid[(valid < vol_min) | (valid > vol_max)])
    
    if vol_total > 0:
        bad_pct = bad_vol_count / vol_total * 100
        _record("vol_range", bad_pct <= vol_bad_pct_limit,
                f"Bad volatility ratio {bad_pct:.2f}% > {vol_bad_pct_limit}% (range [{vol_min}, {vol_max}])",
                threshold=vol_bad_pct_limit, actual=round(bad_pct, 2))
    
    # 5. 指数成分计数（默认 soft：指数调样或成分获取失败不应阻断全表发布）
    for flag, expected in [("is_sse50", 50), ("is_csi300", 300), ("is_csi500", 500),
                           ("is_csi1000", 1000), ("is_csi2000", 2000)]:
        if flag in df.columns:
            cnt = int(df[flag].sum())
            name = f"{flag[3:]}_count"
            _record(name, cnt == expected,
                    f"{flag[3:].upper()} count = {cnt} (expected {expected})",
                    expected=expected, actual=cnt)
    
    # 7. 主键唯一性
    pk_dup = int(df.duplicated(subset=["trade_date", "sec_code"]).sum())
    _record("pk_unique", pk_dup == 0,
            f"Primary key duplicates: {pk_dup}",
            duplicates=pk_dup)
    
    # 汇总：只有 hard 校验阻断发布
    hard_failed = [n for n, c in checks.items()
                   if not c.get("passed", True) and c.get("severity") == "hard"]
    all_passed = not hard_failed
    msg = "; ".join(messages) if messages else "All checks passed"
    
    return all_passed, checks, msg


def _get_prev_count(trade_date: str, db_path: str) -> int:
    """获取前一交易日快照行数"""
    # 简化：直接查最近一条成功运行的 rows_count
    conn = db.get_conn(db_path)
    try:
        cur = conn.execute(
            """
            SELECT rows_count FROM pipeline_run
            WHERE status = 'SUCCESS' AND trade_date < ?
            ORDER BY run_id DESC LIMIT 1
            """, (trade_date,)
        )
        row = cur.fetchone()
        return row[0] if row and row[0] else 0
    except Exception:
        return 0
    finally:
        conn.close()
