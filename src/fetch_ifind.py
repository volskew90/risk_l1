"""
iFinD 数据获取模块
支持三种模式：
  1) prod: iFinDPy SDK（需部署方自备账号，账号密码从环境变量读取，仅 Windows 友好）
  2) http: iFinD HTTP API（refresh_token 鉴权，跨平台，Mac 可用）
  3) proto: Kimi ifind 插件通道（单次限制 3 ticker / 3 年，仅适合验证）

HTTP API 协议参考《iFinD HTTP API 用户手册》：
  POST /api/v1/get_access_token      header: refresh_token -> data.access_token
  POST /api/v1/cmd_history_quotation 历史行情（codes 逗号分隔，functionpara 控制复权/填充）
  POST /api/v1/basic_data_service    基础数据（indipara 指标列表）
  POST /api/v1/data_pool             专题报表（reportname + functionpara + outputpara）
  数据接口请求 header: {"Content-Type":"application/json","access_token":...}
  返回统一含 errorcode(0=成功)/errmsg/tables
板块代码：全部A股 001005010，上证50 001005260，沪深300 001005261
"""
import os
import sys
import time
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src import trade_cal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
BOARD_MAP = {
    # SSE
    "600": ("SSE", "主板"),
    "601": ("SSE", "主板"),
    "603": ("SSE", "主板"),
    "605": ("SSE", "主板"),
    "688": ("SSE", "科创板"),
    # SZSE
    "000": ("SZSE", "主板"),
    "001": ("SZSE", "主板"),
    "002": ("SZSE", "主板"),
    "003": ("SZSE", "主板"),
    "300": ("SZSE", "创业板"),
    "301": ("SZSE", "创业板"),
}

# THS_DR 板块成分报表的板块代码
BLOCK_ALL_A = "001005010"        # 全部A股
INDEX_BLOCK_MAP = {
    "000016.SH": "001005260",    # 上证50
    "000300.SH": "001005261",    # 沪深300
}

# 指数代码 -> 问财查询名（板块报表无权限时的备选通道）
INDEX_NAME_MAP = {
    "000016.SH": "上证50",
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "932000.CSI": "中证2000",
}

# 基础数据规格：(指标, 目标列, indiparams 模板)——{date} 替换为查询日
# 协议来自超级命令 THS_BD 实测：行业指标参数顺序为 "级别,日期"（级别 1=一级 2=二级）
# 行业代码为同花顺申万节点码（S 开头，如 S34=食品饮料/S3405=白酒Ⅱ），非 801xxx 指数代码
# 注意：同一指标不同参数必须分开发请求（同批重复指标只返回最后一组）
# 低频字段（每周刷新入 security_static 缓存）
BASIC_DATA_SPECS = [
    ("ths_ipo_date_stock", "list_date", []),
    ("ths_the_sw_industry_stock", "ind_sw_l1_name", ["1", "{date}"]),
    ("ths_the_sw_industry_stock", "ind_sw_l2_name", ["2", "{date}"]),
    ("ths_the_sw_industry_code_stock", "ind_sw_l1_code", ["1", "{date}"]),
    ("ths_the_sw_industry_code_stock", "ind_sw_l2_code", ["2", "{date}"]),
    ("ths_total_shares_stock", "total_shares", ["{date}"]),
    ("ths_total_float_shares_stock", "total_float_shares", ["{date}"]),
]

# 日频截面字段（每日拉取，不进周缓存）
# ths_avg_vol_nd_stock: N 日平均成交量（股）
# 注意：iFinD 无可用的 N 日平均成交额 BD 指标（ths_amt_nd_stock 返回空），
#       avg_amt_20d 由日线成交额（amount）序列在客户端本地平均得出，见 _fetch_avg_amt
DAILY_BASIC_SPECS = [
    ("ths_avg_vol_nd_stock", "avg_vol_20d", ["{date}", "-20"]),
]

# THS_HistoryQuotes 单批最大代码数
HQ_BATCH_SIZE = 100


def _detect_board(code: str) -> tuple:
    """根据代码前缀判定交易所和板块"""
    for prefix, (ex, board) in BOARD_MAP.items():
        if code.startswith(prefix):
            return ex, board
    return None, None


def _to_dataframe(ths, ths_data) -> pd.DataFrame:
    """THSData -> DataFrame，统一错误处理"""
    if ths_data is None:
        raise RuntimeError("iFinD returned None")
    errorcode = getattr(ths_data, "errorcode", None)
    if errorcode is None:
        # 个别版本直接返回 DataFrame
        if isinstance(ths_data, pd.DataFrame):
            return ths_data
        raise RuntimeError(f"Unexpected iFinD return type: {type(ths_data)}")
    if errorcode != 0:
        raise RuntimeError(f"iFinD error {errorcode}: {getattr(ths_data, 'errmsg', '')}")
    try:
        return ths.THS_Trans2DataFrame(ths_data)
    except Exception:
        data = getattr(ths_data, "data", None)
        if isinstance(data, pd.DataFrame):
            return data
        raise


def _with_retry(fn, retry_times: int, retry_backoff: float, desc: str):
    """指数退避重试包装"""
    last_exc = None
    for attempt in range(1, retry_times + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < retry_times:
                wait = retry_backoff ** attempt
                logger.warning("%s failed (attempt %d/%d): %s, retry in %.1fs",
                               desc, attempt, retry_times, e, wait)
                time.sleep(wait)
    raise last_exc


# ---------------------------------------------------------------------------
# 生产环境：iFinDPy
# ---------------------------------------------------------------------------

class IFindProdClient:
    """基于 iFinDPy 的生产客户端"""

    def __init__(self, retry_times: int = 3, retry_backoff: float = 2,
                 hq_batch_size: int = HQ_BATCH_SIZE):
        self.ths = None
        self.retry_times = retry_times
        self.retry_backoff = retry_backoff
        self.hq_batch_size = hq_batch_size
        self._login()

    def _login(self):
        try:
            import iFinDPy
        except ImportError:
            logger.error("iFinDPy not installed")
            raise
        self.ths = iFinDPy
        account = os.environ.get("IFIND_ACCOUNT", "")
        password = os.environ.get("IFIND_PASSWORD", "")
        if not account or not password:
            raise RuntimeError(
                "iFinD credentials missing: set IFIND_ACCOUNT / IFIND_PASSWORD env vars"
            )
        ret = self.ths.THS_iFinDLogin(account, password)
        if ret not in (0, -201):  # -201 = 已登录
            raise RuntimeError(f"iFinD login failed, ret={ret}")
        logger.info("iFinDPy logged in (ret=%s)", ret)

    def get_stock_pool(self) -> pd.DataFrame:
        """拉取沪深 A 股全池（板块成分报表 p03291）"""
        def _fetch():
            return self.ths.THS_DR(
                "p03291",
                f"date={datetime.now().strftime('%Y%m%d')};blockname={BLOCK_ALL_A};iv_type=allcontract",
                "p03291_f001:Y,p03291_f002:Y,p03291_f003:Y,p03291_f004:Y",
            )
        df = _to_dataframe(self.ths, _with_retry(
            _fetch, self.retry_times, self.retry_backoff, "stock pool"
        ))
        if df is None or df.empty:
            raise RuntimeError("Failed to fetch stock pool from iFinD")
        df.columns = [str(c).lower() for c in df.columns]
        df = df.rename(columns={
            "p03291_f002": "sec_code",
            "p03291_f003": "sec_name",
        })
        if "sec_code" not in df.columns:
            raise RuntimeError(f"Unexpected stock pool columns: {list(df.columns)}")
        if "sec_name" not in df.columns:
            df["sec_name"] = ""
        return df[["sec_code", "sec_name"]]

    def get_basic_data(self, codes: List[str], trade_date: str) -> pd.DataFrame:
        """低频基础数据：上市日期、申万行业、股本等（按规格逐指标请求）"""
        return self._fetch_specs(codes, trade_date, BASIC_DATA_SPECS)

    def get_daily_basic(self, codes: List[str], trade_date: str) -> pd.DataFrame:
        """日频截面数据：20 日日均成交量（BD 指标）/ 日均成交额（日线 amount 本地平均）"""
        df = self._fetch_specs(codes, trade_date, DAILY_BASIC_SPECS)
        amt = self._fetch_avg_amt(codes, trade_date)
        if not amt.empty:
            df = df.merge(amt, on="sec_code", how="left")
        return df

    def _fetch_avg_amt(self, codes: List[str], trade_date: str) -> pd.DataFrame:
        """20 日平均成交额（元）：取近 20 个交易日日线成交额，本地平均（停牌日留空不计入）"""
        sessions = trade_cal.recent_trade_dates(trade_date, 20)
        if not sessions:
            return pd.DataFrame(columns=["sec_code", "avg_amt_20d"])
        start_date = sessions[0]
        params = "Interval:D,Fill:Blank"
        amounts: Dict[str, list] = {c: [] for c in codes}
        for i in range(0, len(codes), self.hq_batch_size):
            batch = codes[i:i + self.hq_batch_size]
            def _fetch(b=batch):
                return self.ths.THS_HistoryQuotes(",".join(b), "amount", params,
                                                  start_date, trade_date)
            try:
                df = _to_dataframe(self.ths, _with_retry(
                    _fetch, self.retry_times, self.retry_backoff,
                    f"avg amt batch {i // self.hq_batch_size}"
                ))
            except Exception as e:
                logger.warning("Avg amount batch %d failed: %s",
                               i // self.hq_batch_size, e)
                continue
            if df is None or df.empty:
                continue
            df.columns = [str(c).lower() for c in df.columns]
            code_col = next((c0 for c0 in ("thscode", "ths_code") if c0 in df.columns), None)
            if not code_col or "amount" not in df.columns:
                logger.warning("Avg amount unexpected columns: %s", list(df.columns))
                continue
            for _, r in df.iterrows():
                if r[code_col] in amounts:
                    try:
                        v = float(r["amount"]) if pd.notna(r["amount"]) else None
                    except (TypeError, ValueError):
                        v = None
                    amounts[r[code_col]].append(v)
        rows = []
        for code, vals in amounts.items():
            valid = [v for v in vals if v is not None]
            rows.append({"sec_code": code,
                         "avg_amt_20d": sum(valid) / len(valid) if valid else None})
        return pd.DataFrame(rows)

    def _fetch_specs(self, codes: List[str], trade_date: str, specs: list) -> pd.DataFrame:
        rows: Dict[str, dict] = {c: {"sec_code": c} for c in codes}
        for i in range(0, len(codes), self.hq_batch_size):
            batch = codes[i:i + self.hq_batch_size]
            for indicator, canon, params_tpl in specs:
                params = ",".join(p.replace("{date}", trade_date) for p in params_tpl)
                def _fetch(b=batch, ind=indicator, p=params):
                    return self.ths.THS_BD(",".join(b), ind, p)
                try:
                    df = _to_dataframe(self.ths, _with_retry(
                        _fetch, self.retry_times, self.retry_backoff, f"basic {indicator}"
                    ))
                except Exception as e:
                    logger.warning("Basic data %s batch %d failed: %s",
                                   indicator, i // self.hq_batch_size, e)
                    continue
                if df is None or df.empty:
                    continue
                df.columns = [str(c).lower() for c in df.columns]
                code_col = next((c0 for c0 in ("thscode", "ths_code") if c0 in df.columns), None)
                if not code_col or indicator.lower() not in df.columns:
                    continue
                for _, r in df.iterrows():
                    v = r[indicator.lower()]
                    if r[code_col] in rows and pd.notna(v) and v != "":
                        rows[r[code_col]][canon] = v
        return pd.DataFrame(rows.values())

    def get_daily_bars(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """前复权日收盘价（批量），返回 DataFrame[sec_code, trade_date, close]"""
        rows = []
        params = "Interval:D,CPS:1,baseDate:1900-01-01,Currency:YSHB,fill:Blank"
        for i in range(0, len(codes), self.hq_batch_size):
            batch = codes[i:i + self.hq_batch_size]
            def _fetch(b=batch):
                return self.ths.THS_HistoryQuotes(",".join(b), "close", params, start_date, end_date)
            try:
                df = _to_dataframe(self.ths, _with_retry(
                    _fetch, self.retry_times, self.retry_backoff,
                    f"price batch {i // self.hq_batch_size}"
                ))
            except Exception as e:
                logger.error("Price batch %d failed: %s", i // self.hq_batch_size, e)
                continue
            if df is None or df.empty:
                continue
            df.columns = [str(c).lower() for c in df.columns]
            code_col = next((c0 for c0 in ("thscode", "ths_code") if c0 in df.columns), None)
            time_col = next((c0 for c0 in ("time", "date") if c0 in df.columns), None)
            if not code_col or not time_col or "close" not in df.columns:
                logger.error("Unexpected price columns: %s", list(df.columns))
                continue
            for _, r in df.iterrows():
                try:
                    close = float(r["close"]) if pd.notna(r["close"]) else float("nan")
                except (TypeError, ValueError):
                    close = float("nan")
                rows.append({"sec_code": r[code_col], "trade_date": str(r[time_col]),
                             "close": close})
        return pd.DataFrame(rows, columns=["sec_code", "trade_date", "close"])

    def get_daily_close(self, codes: List[str], end_date: str, lookback: int = 130) -> Dict[str, List[float]]:
        """
        获取前复权日收盘价序列（批量）。
        fill:Blank —— 停牌日留空，由 calc_vol 按缺口处理，避免前向填充虚增价格序列
        """
        dates = trade_cal.recent_trade_dates(end_date, lookback + 5)
        start_date = dates[0] if dates else (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=int(lookback * 1.5))
        ).strftime("%Y-%m-%d")
        bars = self.get_daily_bars(codes, start_date, end_date)
        prices: Dict[str, List[float]] = {c: [] for c in codes}
        if not bars.empty:
            for code, grp in bars.groupby("sec_code"):
                prices[code] = grp.sort_values("trade_date")["close"].tolist()
        return prices

    def get_realtime_quotes(self, codes: List[str]) -> pd.DataFrame:
        """当日实时行情快照：最新价 / 当日累计成交量（股）/ 累计成交额（元）
        THS_RealtimeQuotes 返回的是最新一次快照，volume/amount 为当日累计值，需自行轮询
        注意：接口原始 volume 单位为手，此处已 ×100 换算为股，与 avg_vol_20d 口径一致"""
        rows = []
        for i in range(0, len(codes), self.hq_batch_size):
            batch = codes[i:i + self.hq_batch_size]
            def _fetch(b=batch):
                return self.ths.THS_RealtimeQuotes(",".join(b), "latest;volume;amount", "")
            try:
                df = _to_dataframe(self.ths, _with_retry(
                    _fetch, self.retry_times, self.retry_backoff,
                    f"realtime batch {i // self.hq_batch_size}"
                ))
            except Exception as e:
                logger.warning("Realtime quotes batch %d failed: %s",
                               i // self.hq_batch_size, e)
                continue
            if df is None or df.empty:
                continue
            df.columns = [str(c).lower() for c in df.columns]
            code_col = next((c0 for c0 in ("thscode", "ths_code") if c0 in df.columns), None)
            if not code_col:
                logger.warning("Realtime quotes unexpected columns: %s", list(df.columns))
                continue
            for _, r in df.iterrows():
                def _num(v):
                    try:
                        return float(v) if pd.notna(v) and v != "" else None
                    except (TypeError, ValueError):
                        return None
                rows.append({
                    "sec_code": r[code_col],
                    "latest": _num(r.get("latest")),
                    "volume": _num(r.get("volume")) * 100 if _num(r.get("volume")) is not None else None,
                    "amount": _num(r.get("amount")),
                })
        return pd.DataFrame(rows, columns=["sec_code", "latest", "volume", "amount"])

    def get_index_components(self, index_code: str, trade_date: str) -> List[str]:
        """获取指数成分股列表（通过板块成分报表）"""
        block = INDEX_BLOCK_MAP.get(index_code)
        if not block:
            logger.warning("No block mapping for index %s", index_code)
            return []
        date_compact = trade_date.replace("-", "")
        def _fetch():
            return self.ths.THS_DR(
                "p03291",
                f"date={date_compact};blockname={block};iv_type=allcontract",
                "p03291_f001:Y,p03291_f002:Y,p03291_f003:Y,p03291_f004:Y",
            )
        try:
            df = _to_dataframe(self.ths, _with_retry(
                _fetch, self.retry_times, self.retry_backoff, f"index components {index_code}"
            ))
        except Exception as e:
            logger.warning("Index component fetch failed for %s: %s", index_code, e)
            return []
        if df is None or df.empty:
            return []
        df.columns = [str(c).lower() for c in df.columns]
        if "p03291_f002" in df.columns:
            return df["p03291_f002"].astype(str).tolist()
        return df.iloc[:, 0].astype(str).tolist()


# ---------------------------------------------------------------------------
# HTTP API 环境：refresh_token 鉴权（跨平台，Mac 可用）
# ---------------------------------------------------------------------------

HTTP_API_BASE = "https://quantapi.51ifind.com/api/v1"


class IFindHttpClient:
    """基于 iFinD HTTP API 的客户端（refresh_token 存环境变量 IFIND_REFRESH_TOKEN）"""

    def __init__(self, retry_times: int = 3, retry_backoff: float = 2,
                 hq_batch_size: int = HQ_BATCH_SIZE, refresh_token: Optional[str] = None):
        import requests
        self._requests = requests
        self.retry_times = retry_times
        self.retry_backoff = retry_backoff
        self.hq_batch_size = hq_batch_size
        # 环境变量优先，其次构造参数（config.yaml 的 ifind_refresh_token）
        self.refresh_token = os.environ.get("IFIND_REFRESH_TOKEN") or refresh_token or ""
        if not self.refresh_token:
            raise RuntimeError(
                "IFIND_REFRESH_TOKEN env var missing; "
                "get it from 超级命令 -> 工具 -> refresh_token 查询/更新"
            )
        self._access_token = None
        self._refresh_access_token()
        logger.info("IFindHttpClient authenticated")

    def _refresh_access_token(self):
        resp = self._requests.post(
            f"{HTTP_API_BASE}/get_access_token",
            headers={"Content-Type": "application/json", "refresh_token": self.refresh_token},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        token = (data.get("data") or {}).get("access_token")
        if not token:
            raise RuntimeError(f"get_access_token failed: {data}")
        self._access_token = token

    def _post(self, endpoint: str, payload: dict, desc: str) -> dict:
        """带重试的 POST；token 过期（错误码非 0 且含 token 字样）时刷新后重试一次"""
        def _call():
            resp = self._requests.post(
                f"{HTTP_API_BASE}/{endpoint}",
                json=payload,
                headers={"Content-Type": "application/json",
                         "access_token": self._access_token},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()

        result = _with_retry(_call, self.retry_times, self.retry_backoff, desc)
        if result.get("errorcode", 0) != 0:
            errmsg = str(result.get("errmsg", ""))
            # -1010 token 失效 / -1302 access_token 无效：刷新后重试一次
            if result.get("errorcode") in (-1010, -1302) or "token" in errmsg.lower():
                logger.warning("access_token may be expired (%s), refreshing", errmsg)
                self._refresh_access_token()
                result = _with_retry(_call, self.retry_times, self.retry_backoff, desc)
            if result.get("errorcode", 0) != 0:
                raise RuntimeError(f"iFinD HTTP error {result.get('errorcode')}: {result.get('errmsg')}")
        return result

    @staticmethod
    def _tables_to_df(tables) -> pd.DataFrame:
        """tables 结构 -> DataFrame。data_pool 返回 [{"table": {列: [...]}}]"""
        if isinstance(tables, list):
            for item in tables:
                if isinstance(item, dict) and isinstance(item.get("table"), dict):
                    return pd.DataFrame(item["table"])
            if tables and isinstance(tables[0], dict):
                return pd.DataFrame(tables)
        if isinstance(tables, dict):
            return pd.DataFrame(tables)
        return pd.DataFrame()

    def _fetch_block(self, block_name: str, date_compact: str) -> pd.DataFrame:
        """板块成分报表 p03291"""
        result = self._post("data_pool", {
            "reportname": "p03291",
            "functionpara": {
                "date": date_compact,
                "blockname": block_name,
                "iv_type": "allcontract",
            },
            "outputpara": "p03291_f001:Y,p03291_f002:Y,p03291_f003:Y,p03291_f004:Y",
        }, f"data_pool {block_name}")
        df = self._tables_to_df(result.get("tables"))
        if not df.empty:
            df.columns = [str(c).lower() for c in df.columns]
            df = df.rename(columns={"p03291_f002": "sec_code", "p03291_f003": "sec_name"})
        return df

    def get_stock_pool(self) -> pd.DataFrame:
        """拉取沪深 A 股全池"""
        df = self._fetch_block(BLOCK_ALL_A, datetime.now().strftime("%Y%m%d"))
        if df.empty or "sec_code" not in df.columns:
            raise RuntimeError(f"Failed to fetch stock pool, columns: {list(df.columns)}")
        if "sec_name" not in df.columns:
            df["sec_name"] = ""
        return df[["sec_code", "sec_name"]]

    def get_index_components(self, index_code: str, trade_date: str) -> List[str]:
        """获取指数成分股列表：优先板块成分报表，无权限/无数据时回退问财智能选股"""
        block = INDEX_BLOCK_MAP.get(index_code)
        if block:
            try:
                df = self._fetch_block(block, trade_date.replace("-", ""))
                if not df.empty and "sec_code" in df.columns:
                    return df["sec_code"].astype(str).tolist()
            except Exception as e:
                logger.warning("Index block fetch failed for %s (%s), falling back to wencai",
                               index_code, e)
        name = INDEX_NAME_MAP.get(index_code)
        if not name:
            logger.warning("No wencai mapping for index %s", index_code)
            return []
        try:
            result = self._post("smart_stock_picking", {
                "searchstring": f"{name}成分股", "searchtype": "stock",
            }, f"wencai {name}")
            tables = result.get("tables") or []
            if tables and isinstance(tables[0], dict) and isinstance(tables[0].get("table"), dict):
                codes = tables[0]["table"].get("股票代码") or []
                logger.info("Wencai %s components: %d", name, len(codes))
                return [str(x) for x in codes]
        except Exception as e:
            logger.warning("Wencai index fetch failed for %s: %s", index_code, e)
        return []

    def _fetch_specs(self, codes: List[str], trade_date: str, specs: list) -> pd.DataFrame:
        """按规格逐指标批量拉取基础数据，输出规范列名"""
        rows: Dict[str, dict] = {c: {"sec_code": c} for c in codes}
        for i in range(0, len(codes), self.hq_batch_size):
            batch = codes[i:i + self.hq_batch_size]
            for indicator, canon, params_tpl in specs:
                params = [p.replace("{date}", trade_date) for p in params_tpl]
                try:
                    result = self._post("basic_data_service", {
                        "codes": ",".join(batch),
                        "indipara": [{"indicator": indicator, "indiparams": params}],
                    }, f"basic {indicator} batch {i // self.hq_batch_size}")
                except Exception as e:
                    logger.warning("Basic data %s batch %d failed: %s",
                                   indicator, i // self.hq_batch_size, e)
                    continue
                for item in result.get("tables") or []:
                    code = item.get("thscode")
                    vals = (item.get("table") or {}).get(indicator)
                    if code in rows and isinstance(vals, list) and vals:
                        v = vals[0]
                        if v not in ("", None):
                            rows[code][canon] = v
        return pd.DataFrame(rows.values())

    def get_basic_data(self, codes: List[str], trade_date: str) -> pd.DataFrame:
        """低频基础数据：上市日期、申万行业、股本等（周缓存）"""
        return self._fetch_specs(codes, trade_date, BASIC_DATA_SPECS)

    def get_daily_basic(self, codes: List[str], trade_date: str) -> pd.DataFrame:
        """日频截面数据：20 日日均成交量（BD 指标）/ 日均成交额（日线 amount 本地平均）"""
        df = self._fetch_specs(codes, trade_date, DAILY_BASIC_SPECS)
        amt = self._fetch_avg_amt(codes, trade_date)
        if not amt.empty:
            df = df.merge(amt, on="sec_code", how="left")
        return df

    def _fetch_avg_amt(self, codes: List[str], trade_date: str) -> pd.DataFrame:
        """20 日平均成交额（元）：取近 20 个交易日日线成交额，本地平均（停牌日留空不计入）"""
        sessions = trade_cal.recent_trade_dates(trade_date, 20)
        if not sessions:
            return pd.DataFrame(columns=["sec_code", "avg_amt_20d"])
        start_date = sessions[0]
        amounts: Dict[str, list] = {c: [] for c in codes}
        for i in range(0, len(codes), self.hq_batch_size):
            batch = codes[i:i + self.hq_batch_size]
            try:
                result = self._post("cmd_history_quotation", {
                    "codes": ",".join(batch),
                    "indicators": "amount",
                    "functionpara": {"Interval": "D", "Fill": "Blank"},
                    "startdate": start_date,
                    "enddate": trade_date,
                }, f"avg amt batch {i // self.hq_batch_size}")
            except Exception as e:
                logger.warning("Avg amount batch %d failed: %s",
                               i // self.hq_batch_size, e)
                continue
            for item in result.get("tables") or []:
                code = item.get("thscode")
                vals = (item.get("table") or {}).get("amount") or []
                if code in amounts:
                    for v in vals:
                        try:
                            amounts[code].append(float(v) if v not in ("", None) else None)
                        except (TypeError, ValueError):
                            amounts[code].append(None)
        rows = []
        for code, vals in amounts.items():
            valid = [v for v in vals if v is not None]
            rows.append({"sec_code": code,
                         "avg_amt_20d": sum(valid) / len(valid) if valid else None})
        return pd.DataFrame(rows)

    def get_realtime_quotes(self, codes: List[str]) -> pd.DataFrame:
        """当日实时行情快照：最新价 / 当日累计成交量（股）/ 累计成交额（元）
        real_time_quotation 返回最新一次快照，volume/amount 为当日累计值，需自行轮询
        注意：接口原始 volume 单位为手，此处已 ×100 换算为股，与 avg_vol_20d 口径一致"""
        rows = []
        for i in range(0, len(codes), self.hq_batch_size):
            batch = codes[i:i + self.hq_batch_size]
            try:
                result = self._post("real_time_quotation", {
                    "codes": ",".join(batch),
                    "indicators": "latest,volume,amount",
                }, f"realtime batch {i // self.hq_batch_size}")
            except Exception as e:
                logger.warning("Realtime quotes batch %d failed: %s",
                               i // self.hq_batch_size, e)
                continue
            for item in result.get("tables") or []:
                code = item.get("thscode")
                table = item.get("table") or {}

                def _first(key):
                    vals = table.get(key)
                    if isinstance(vals, list) and vals:
                        v = vals[0]
                        try:
                            return float(v) if v not in ("", None) else None
                        except (TypeError, ValueError):
                            return None
                    return None

                if code:
                    vol = _first("volume")
                    rows.append({"sec_code": code, "latest": _first("latest"),
                                 "volume": vol * 100 if vol is not None else None,
                                 "amount": _first("amount")})
        return pd.DataFrame(rows, columns=["sec_code", "latest", "volume", "amount"])

    def get_daily_bars(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        前复权日收盘价（批量），返回 DataFrame[sec_code, trade_date, close]
        CPS=2 前复权（分红再投），Fill=Blank 停牌留空
        """
        rows = []
        for i in range(0, len(codes), self.hq_batch_size):
            batch = codes[i:i + self.hq_batch_size]
            try:
                result = self._post("cmd_history_quotation", {
                    "codes": ",".join(batch),
                    "indicators": "close",
                    "functionpara": {
                        "Interval": "D",
                        "CPS": "2",
                        "Fill": "Blank",
                        "Currency": "YSHB",
                    },
                    "startdate": start_date,
                    "enddate": end_date,
                }, f"price batch {i // self.hq_batch_size}")
            except Exception as e:
                logger.error("Price batch %d failed: %s", i // self.hq_batch_size, e)
                continue
            for item in result.get("tables") or []:
                code = item.get("thscode")
                times = item.get("time") or (item.get("table") or {}).get("time") or []
                closes = (item.get("table") or {}).get("close") or []
                for t, v in zip(times, closes):
                    try:
                        close = float(v) if v not in ("", None) else float("nan")
                    except (TypeError, ValueError):
                        close = float("nan")
                    rows.append({"sec_code": code, "trade_date": str(t), "close": close})
        return pd.DataFrame(rows, columns=["sec_code", "trade_date", "close"])

    def get_daily_close(self, codes: List[str], end_date: str, lookback: int = 130) -> Dict[str, List[float]]:
        """兼容接口：取 lookback 窗口的前复权收盘价序列"""
        dates = trade_cal.recent_trade_dates(end_date, lookback + 5)
        start_date = dates[0] if dates else (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=int(lookback * 1.5))
        ).strftime("%Y-%m-%d")
        bars = self.get_daily_bars(codes, start_date, end_date)
        prices: Dict[str, List[float]] = {c: [] for c in codes}
        if not bars.empty:
            for code, grp in bars.groupby("sec_code"):
                prices[code] = grp.sort_values("trade_date")["close"].tolist()
        return prices


# ---------------------------------------------------------------------------
# 原型环境：Kimi ifind 插件通道
# ---------------------------------------------------------------------------

class IFindProtoClient:
    """基于 Kimi ifind 插件的原型客户端（限制较多，仅适合验证）"""

    def __init__(self, retry_times: int = 3, retry_backoff: float = 2, **kwargs):
        self.script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "ifind_tool.py")
        if not os.path.exists(self.script_path):
            # 尝试找已存在的脚本
            logger.warning("ifind_tool.py not found at %s", self.script_path)
        logger.info("IFindProtoClient initialized")

    def _call_script(self, cmd: str) -> Any:
        """调用 ifind_tool.py 脚本"""
        import subprocess
        full_cmd = f"python3 {self.script_path} {cmd}"
        try:
            result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.error("ifind script error: %s", result.stderr)
                return None
            # 尝试解析 JSON
            import json
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return result.stdout
        except Exception as e:
            logger.error("ifind script call failed: %s", e)
            return None

    def get_stock_pool(self) -> pd.DataFrame:
        """使用 ifind 插件获取股票池"""
        # 尝试获取沪深 A 股列表（通过指数成分近似）
        # 由于原型限制，这里使用硬编码方式或简化逻辑
        logger.warning("Prototype mode: stock pool should be provided externally or via manual CSV")
        return pd.DataFrame(columns=['sec_code', 'sec_name'])

    def get_daily_close(self, codes: List[str], end_date: str, lookback: int = 130) -> Dict[str, List[float]]:
        """分批获取价格（每次最多 3 只）"""
        prices = {}
        batch_size = 3
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            for code in batch:
                # 使用 ifind_get_price
                dates = trade_cal.recent_trade_dates(end_date, lookback + 5)
                start_date = dates[0] if dates else (
                    datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=int(lookback * 1.5))
                ).strftime("%Y-%m-%d")
                cmd = f"ifind_get_price --ticker {code} --start {start_date} --end {end_date}"
                result = self._call_script(cmd)
                if result and isinstance(result, dict):
                    prices[code] = result.get("prices", [])
                else:
                    prices[code] = []
            time.sleep(0.5)  # 避免过快请求
        return prices


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

class IFindClient:
    """iFinD 客户端统一入口，自动探测可用模式"""

    def __init__(self, mode: str = "auto", retry_times: int = 3,
                 retry_backoff: float = 2, hq_batch_size: int = HQ_BATCH_SIZE,
                 refresh_token: Optional[str] = None):
        """
        mode: auto / prod / http / proto
        refresh_token: HTTP 模式的 refresh_token（也可走环境变量 IFIND_REFRESH_TOKEN）
        """
        self.mode = mode
        self._client = None
        self._client_kwargs = {
            "retry_times": retry_times,
            "retry_backoff": retry_backoff,
            "hq_batch_size": hq_batch_size,
        }
        self._http_kwargs = {**self._client_kwargs, "refresh_token": refresh_token}
        self._init_client()

    def _init_client(self):
        if self.mode == "auto":
            # 优先 iFinDPy 生产环境，其次 HTTP API，最后 Kimi 插件原型
            try:
                self._client = IFindProdClient(**self._client_kwargs)
                self.mode = "prod"
                logger.info("Auto-selected prod mode (iFinDPy)")
            except Exception:
                try:
                    self._client = IFindHttpClient(**self._http_kwargs)
                    self.mode = "http"
                    logger.info("Auto-selected http mode (iFinD HTTP API)")
                except Exception:
                    self._client = IFindProtoClient(**self._client_kwargs)
                    self.mode = "proto"
                    logger.info("Auto-selected proto mode (Kimi plugin)")
        elif self.mode == "prod":
            self._client = IFindProdClient(**self._client_kwargs)
        elif self.mode == "http":
            self._client = IFindHttpClient(**self._http_kwargs)
        elif self.mode == "proto":
            self._client = IFindProtoClient(**self._client_kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_stock_pool(self) -> pd.DataFrame:
        return self._client.get_stock_pool()

    def get_daily_close(self, codes: List[str], end_date: str, lookback: int = 130) -> Dict[str, List[float]]:
        return self._client.get_daily_close(codes, end_date, lookback)

    def get_daily_bars(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        if hasattr(self._client, 'get_daily_bars'):
            return self._client.get_daily_bars(codes, start_date, end_date)
        logger.warning("get_daily_bars not available in %s mode", self.mode)
        return pd.DataFrame(columns=["sec_code", "trade_date", "close"])

    def get_index_components(self, index_code: str, trade_date: str) -> List[str]:
        if hasattr(self._client, 'get_index_components'):
            return self._client.get_index_components(index_code, trade_date)
        logger.warning("get_index_components not available in %s mode", self.mode)
        return []

    def get_basic_data(self, codes: List[str], trade_date: str) -> pd.DataFrame:
        if hasattr(self._client, 'get_basic_data'):
            return self._client.get_basic_data(codes, trade_date)
        logger.warning("get_basic_data not available in %s mode", self.mode)
        return pd.DataFrame()

    def get_daily_basic(self, codes: List[str], trade_date: str) -> pd.DataFrame:
        if hasattr(self._client, 'get_daily_basic'):
            return self._client.get_daily_basic(codes, trade_date)
        logger.warning("get_daily_basic not available in %s mode", self.mode)
        return pd.DataFrame()

    def get_realtime_quotes(self, codes: List[str]) -> pd.DataFrame:
        if hasattr(self._client, 'get_realtime_quotes'):
            return self._client.get_realtime_quotes(codes)
        logger.warning("get_realtime_quotes not available in %s mode", self.mode)
        return pd.DataFrame(columns=["sec_code", "latest", "volume", "amount"])
