"""
A 股交易日历模块
基于 exchange_calendars 的 XSHG 日历（沪深交易日一致），
未安装时退化为周末 + 固定节假日启发式并告警。
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import exchange_calendars as _ec
    _CAL = _ec.get_calendar("XSHG")
except ImportError:
    _CAL = None
    logger.warning("exchange_calendars not installed, falling back to heuristic trade calendar")

_FIXED_HOLIDAYS = {
    "01-01", "05-01", "05-02", "05-03",
    "10-01", "10-02", "10-03", "10-04", "10-05", "10-06", "10-07",
}


def is_trade_date(date_str: str) -> bool:
    """判断是否为 A 股交易日"""
    if _CAL is not None:
        return bool(_CAL.is_session(date_str))
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if dt.weekday() >= 5:
        return False
    return dt.strftime("%m-%d") not in _FIXED_HOLIDAYS


def trade_date_add(start: str, n: int, calendar: Optional[List[str]] = None) -> str:
    """start 起第 n 个交易日（n=0 返回 start 自身之后的对齐逻辑由调用方处理）"""
    if calendar:
        try:
            idx = calendar.index(start)
            if 0 <= idx + n < len(calendar):
                return calendar[idx + n]
        except ValueError:
            pass
    if _CAL is not None:
        sessions = _CAL.sessions_in_range(
            start, (datetime.strptime(start, "%Y-%m-%d") + timedelta(days=n * 2 + 30)).strftime("%Y-%m-%d")
        )
        dates = [s.strftime("%Y-%m-%d") for s in sessions]
        if start in dates:
            idx = dates.index(start)
        else:
            # start 非交易日时从其后第一个交易日起算
            idx = -1
        target = idx + n
        if 0 <= target < len(dates):
            return dates[target]
        return dates[-1]
    # 无日历库：跳过周末的近似
    dt = datetime.strptime(start, "%Y-%m-%d")
    added = 0
    while added < n:
        dt += timedelta(days=1)
        if dt.weekday() < 5:
            added += 1
    return dt.strftime("%Y-%m-%d")


def recent_trade_dates(end_date: str, n: int) -> List[str]:
    """返回截至 end_date 的最近 n 个交易日（含 end_date，升序）"""
    if _CAL is not None:
        start = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=int(n * 1.6) + 20)).strftime("%Y-%m-%d")
        sessions = _CAL.sessions_in_range(start, end_date)
        return [s.strftime("%Y-%m-%d") for s in sessions][-n:]
    # 退化：近似取日历日
    start = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=int(n * 1.5))
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(int(n * 1.5) + 1)]


def trade_day_count(start: str, end: str) -> Optional[int]:
    """(start, end] 之间的交易日数：start 当日不计，end 计入；无日历库返回 None"""
    if _CAL is None:
        return None
    if start > end:
        return 0
    n = len(_CAL.sessions_in_range(start, end))
    # start 本身是交易日时 sessions_in_range 会包含它，减去
    if n > 0 and _CAL.is_session(start):
        n -= 1
    return n
