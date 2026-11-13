"""
重点监控逻辑模块
- 事件表维护
- is_key_monitored 计算
- 监控窗口推导
"""
import logging
from datetime import datetime, timezone
from typing import List, Optional
import pandas as pd
import numpy as np

from src import db

logger = logging.getLogger(__name__)


def merge_events(new_events: pd.DataFrame, db_path: str) -> pd.DataFrame:
    """
    合并新事件到事件表，按 sec_code + event_type + disclose_date 去重
    """
    if new_events.empty:
        return pd.DataFrame()
    
    # 添加时间戳
    new_events = new_events.copy()
    if "created_at" not in new_events.columns:
        new_events["created_at"] = datetime.now(timezone.utc).isoformat()
    
    # 读取已有事件进行去重
    conn = db.get_conn(db_path)
    try:
        existing = pd.read_sql_query(
            "SELECT sec_code, event_type, disclose_date FROM monitor_event",
            conn
        )
    except Exception:
        existing = pd.DataFrame(columns=["sec_code", "event_type", "disclose_date"])
    finally:
        conn.close()
    
    if not existing.empty:
        # 左连接去重
        merged = new_events.merge(
            existing, on=["sec_code", "event_type", "disclose_date"], how="left", indicator=True
        )
        to_insert = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    else:
        to_insert = new_events
    
    if not to_insert.empty:
        db.save_events(to_insert, db_path)
        logger.info("Inserted %s new monitor events", len(to_insert))
    
    return to_insert


def compute_key_monitored(trade_date: str, db_path: str) -> pd.DataFrame:
    """
    计算当日 is_key_monitored 标志
    返回: sec_code, is_key_monitored, monitor_start, monitor_end, monitor_source
    """
    conn = db.get_conn(db_path)
    try:
        # 查询在监控窗口内的事件
        df = pd.read_sql_query(
            """
            SELECT sec_code, event_type, monitor_start, monitor_end, source
            FROM monitor_event
            WHERE monitor_start <= ? AND (monitor_end >= ? OR monitor_end IS NULL)
            ORDER BY sec_code, monitor_end DESC
            """,
            conn, params=(trade_date, trade_date)
        )
    finally:
        conn.close()
    
    if df.empty:
        return pd.DataFrame(columns=["sec_code", "is_key_monitored", "monitor_start", "monitor_end", "monitor_source"])
    
    # 对每个股票取最晚的 monitor_end
    result = df.groupby("sec_code").first().reset_index()
    result["is_key_monitored"] = 1
    result = result.rename(columns={"source": "monitor_source"})
    
    return result[["sec_code", "is_key_monitored", "monitor_start", "monitor_end", "monitor_source"]]


_EXCHANGE_SUFFIX = {"SZSE": ".SZ", "SSE": ".SH"}


def fallback_from_prev(trade_date: str, db_path: str,
                       exchanges: Optional[List[str]] = None) -> pd.DataFrame:
    """
    主通道失败时：沿用前一交易日 is_key_monitored 值
    exchanges: 仅沿用指定交易所（如 ["SZSE"]）的股票；None 表示全部
    """
    last_date = db.get_last_success_date(db_path)
    if not last_date:
        logger.warning("No previous success date for fallback")
        return pd.DataFrame()
    
    sql = """
            SELECT sec_code, is_key_monitored, monitor_start, monitor_end, monitor_source
            FROM security_master_daily
            WHERE trade_date = ? AND is_key_monitored = 1
            """
    params: list = [last_date]
    if exchanges:
        suffixes = [_EXCHANGE_SUFFIX.get(e, "") for e in exchanges]
        suffixes = [s for s in suffixes if s]
        if suffixes:
            sql += " AND (" + " OR ".join("sec_code LIKE ?" for _ in suffixes) + ")"
            params.extend(f"%{s}" for s in suffixes)
    
    conn = db.get_conn(db_path)
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    
    if df.empty:
        return pd.DataFrame()
    
    # 更新 monitor_end 为当日（保守处理）
    df["monitor_end"] = trade_date
    df["monitor_source"] = df["monitor_source"].fillna("") + "_STALE"
    logger.info("Fallback from %s: %s stocks", last_date, len(df))
    return df
