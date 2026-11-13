"""
交易所数据获取模块
- 上交所：严重异常波动页面 JSON 接口
- 深交所：竞价交易公开信息 xlsx 下载
"""
import os
import re
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from urllib.parse import urlencode

import requests
import pandas as pd

from src import trade_cal

logger = logging.getLogger(__name__)

# 深交所 xlsx 下载接口
SZSE_XLSX_URL = "https://www.szse.cn/api/report/ShowReport"

# 上交所 JSON 接口（需要 Referer）
SSE_API_BASE = "https://query.sse.com.cn"
SSE_REFERER = "https://www.sse.com.cn/disclosure/diclosure/ycjyxx/main/"

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def _get_with_retry(url: str, desc: str, retry_times: int = 3, retry_backoff: float = 2,
                    **kwargs) -> requests.Response:
    """带指数退避重试的 GET（交易所接口偶发连接重置）"""
    import time
    last_exc = None
    for attempt in range(1, retry_times + 1):
        try:
            resp = requests.get(url, timeout=60, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            if attempt < retry_times:
                wait = retry_backoff ** attempt
                logger.warning("%s failed (attempt %d/%d): %s, retry in %.1fs",
                               desc, attempt, retry_times, e, wait)
                time.sleep(wait)
    raise last_exc


def _trade_date_add(start: str, n: int, calendar: Optional[List[str]] = None) -> str:
    """start 起第 n 个交易日（优先 XSHG 交易日历，无日历库时退化为跳过周末）"""
    return trade_cal.trade_date_add(start, n, calendar)


# ---------------------------------------------------------------------------
# 深交所
# ---------------------------------------------------------------------------

def fetch_szse_monitor(trade_date: str, retry_times: int = 3, retry_backoff: float = 2) -> pd.DataFrame:
    """
    拉取深交所严重异常波动数据
    返回 DataFrame: [sec_code, sec_name, event_type, trigger_detail, abnormal_period_start, abnormal_period_end, disclose_date, source]
    """
    params = {
        "SHOWTYPE": "xlsx",
        "CATALOGID": "1842_xxpl_after",
        "TABKEY": "tab1",
        "txtStart": trade_date,
        "txtEnd": trade_date,
    }
    url = f"{SZSE_XLSX_URL}?{urlencode(params)}"
    
    try:
        resp = _get_with_retry(url, "SZSE monitor", retry_times, retry_backoff, headers=_UA)
    except Exception as e:
        logger.error("SZSE monitor fetch failed: %s", e)
        raise
    
    # 解析响应内容（不落临时文件）
    try:
        import io
        df = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")
    except Exception as e:
        logger.error("SZSE xlsx parse failed: %s", e)
        raise
    
    if df.empty:
        logger.info("SZSE: no monitor data for %s", trade_date)
        return pd.DataFrame()
    
    # 列名映射（根据实际表格结构调整）
    df.columns = [str(c).strip() for c in df.columns]
    
    records = []
    # 优先从专门的代码列提取，避免误匹配日期/金额中的 6 位数字
    code_col = next((c for c in df.columns if "代码" in c), None)
    for _, row in df.iterrows():
        # 根据深交所披露格式解析
        # 严重异常波动条目通常包含"严重异常波动"字样
        content = " ".join(str(v) for v in row.values if pd.notna(v))
        
        if "严重异常波动" not in content:
            continue
        
        code = None
        if code_col and pd.notna(row.get(code_col)):
            m = re.search(r'(\d{6})', str(row[code_col]))
            if m:
                code = m.group(1)
        if not code:
            # 前后不能紧邻其他数字，避免把 20260904 这类日期截出 6 位
            m = re.search(r'(?<!\d)(\d{6})(?!\d)', content)
            if not m:
                continue
            code = m.group(1)
        sec_code = f"{code}.SZ"
        
        # 判定 event_type
        event_type = "SEVERE_ABNORMAL" if "严重异常波动" in content else "NORMAL_ABNORMAL"
        
        records.append({
            "sec_code": sec_code,
            "sec_name": "",  # 表格中可能有
            "event_type": event_type,
            "trigger_detail": content[:200],
            "abnormal_period_start": None,
            "abnormal_period_end": None,
            "disclose_date": trade_date,
            "source": "SZSE_PAGE",
            "raw_ref": url,
        })
    
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 上交所
# ---------------------------------------------------------------------------

def fetch_sse_monitor(trade_date: str, retry_times: int = 3, retry_backoff: float = 2) -> pd.DataFrame:
    """
    拉取上交所严重异常波动数据
    注意：上交所接口需要动态确认 sqlId，此处用通用方式尝试
    """
    # 上交所页面数据接口（从页面 JS 中确认的 sqlId，可能随时间变化）
    sql_id = "COMMON_SSE_SCSJ_CJGK_CJGKYCYBD_SEVER_ABNORM_FLUCTUATION_L"  # 示例，需实际确认
    
    url = f"{SSE_API_BASE}/commonQuery.do"
    params = {
        "sqlId": sql_id,
        "productId": "",
        "startDate": trade_date,
        "endDate": trade_date,
    }
    headers = {**_UA, "Referer": SSE_REFERER}
    
    try:
        resp = _get_with_retry(url, "SSE monitor", retry_times, retry_backoff,
                               params=params, headers=headers)
        data = resp.json()
    except Exception as e:
        logger.error("SSE monitor fetch failed: %s", e)
        raise
    
    result = data.get("result", [])
    if not result:
        logger.info("SSE: no monitor data for %s", trade_date)
        return pd.DataFrame()
    
    records = []
    for item in result:
        code = item.get("SECURITY_CODE", "")
        name = item.get("SECURITY_NAME", "")
        if not code:
            continue
        sec_code = f"{code}.SH"
        
        records.append({
            "sec_code": sec_code,
            "sec_name": name,
            "event_type": "SEVERE_ABNORMAL",
            "trigger_detail": item.get("DETAIL", "")[:200],
            "abnormal_period_start": item.get("ABNORMAL_PERIOD_START"),
            "abnormal_period_end": item.get("ABNORMAL_PERIOD_END"),
            "disclose_date": trade_date,
            "source": "SSE_PAGE",
            "raw_ref": json.dumps(item, ensure_ascii=False)[:500],
        })
    
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def fetch_all_monitor_events(trade_date: str, retry_times: int = 3,
                             retry_backoff: float = 2) -> tuple:
    """
    拉取沪深交易所所有监控事件
    返回: (统一格式 DataFrame, failed_sources)
    - failed_sources 为空表示两个交易所都成功（结果可能为空，属正常）
    - 任一交易所失败都会记录在 failed_sources，调用方必须据此触发 fallback + 告警，
      不得将残缺数据当作完整数据发布
    """
    all_records = []
    failed_sources = []
    
    # 深交所
    try:
        df_sz = fetch_szse_monitor(trade_date, retry_times, retry_backoff)
        if not df_sz.empty:
            all_records.append(df_sz)
            logger.info("SZSE events fetched: %s rows", len(df_sz))
    except Exception as e:
        logger.error("SZSE fetch error: %s", e)
        failed_sources.append("SZSE")
    
    # 上交所
    try:
        df_sh = fetch_sse_monitor(trade_date, retry_times, retry_backoff)
        if not df_sh.empty:
            all_records.append(df_sh)
            logger.info("SSE events fetched: %s rows", len(df_sh))
    except Exception as e:
        logger.error("SSE fetch error: %s", e)
        failed_sources.append("SSE")
    
    if all_records:
        return pd.concat(all_records, ignore_index=True), failed_sources
    return pd.DataFrame(), failed_sources


def derive_monitor_window(events_df: pd.DataFrame,
                          monitor_days: int = 10,
                          trade_calendar: Optional[List[str]] = None) -> pd.DataFrame:
    """
    推导监控窗口：
    - SEVERE_ABNORMAL: monitor_start = disclose_date, monitor_end = disclose_date + monitor_days 交易日
    - OFFICIAL_LIST: 使用官方给出的起止日（需外部传入已设置好）
    """
    if events_df.empty:
        return events_df
    
    def _calc_end(row):
        if row["event_type"] == "SEVERE_ABNORMAL":
            return _trade_date_add(row["disclose_date"], monitor_days, trade_calendar)
        return row.get("monitor_end") or row["disclose_date"]
    
    events_df = events_df.copy()
    events_df["monitor_start"] = events_df["disclose_date"]
    events_df["monitor_end"] = events_df.apply(_calc_end, axis=1)
    return events_df
