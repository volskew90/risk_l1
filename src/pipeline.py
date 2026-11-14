#!/usr/bin/env python3
"""
主流程编排模块
每日更新流水线（文档 §7.2）
"""
import os
import sys
import yaml
import logging
import argparse
from datetime import datetime, timezone, timedelta
from typing import Optional
import pandas as pd
import numpy as np

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import db, calc_vol, fetch_ifind, fetch_exchange, monitor, quality, export, alert, trade_cal

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str):
    """配置日志"""
    os.makedirs(log_dir, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"pipeline_{today}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def is_trade_date(date_str: str) -> bool:
    """交易日校验：优先使用 XSHG 交易日历（trade_cal），库缺失时退化为启发式"""
    return trade_cal.is_trade_date(date_str)


def run_pipeline(trade_date: Optional[str] = None, config_path: str = "config.yaml",
                 ifind_mode: str = "auto"):
    """
    执行每日流水线
    
    参数:
        trade_date: 指定交易日，默认今天
        config_path: 配置文件路径
        ifind_mode: iFinD 模式 auto/prod/proto
    """
    # 加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    setup_logging(config.get("log_dir", "./logs"))
    
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info("=" * 60)
    logger.info("Pipeline start for trade_date=%s", trade_date)
    
    # 0. 交易日历校验
    if not is_trade_date(trade_date):
        logger.info("%s is not a trade date, exit", trade_date)
        return 0
    
    db_path = config["db_path"]
    db.init_db(db_path)
    run_id = db.log_run_start(trade_date, db_path)
    
    try:
        # 初始化 iFinD 客户端
        ifind_client = fetch_ifind.IFindClient(
            mode=ifind_mode,
            retry_times=int(config.get("retry_times", 3)),
            retry_backoff=float(config.get("retry_backoff", 2)),
            refresh_token=config.get("ifind_refresh_token") or None,
        )
        
        # 1. 拉取全 A 股票池（传入 trade_date 防止历史回溯时间穿越）
        logger.info("Step 1: Fetch stock pool")
        try:
            pool_df = ifind_client.get_stock_pool(trade_date=trade_date)
            if pool_df.empty:
                logger.warning("Empty stock pool from iFinD, using fallback")
                # 从数据库获取上一日股票池作为 fallback
                pool_df = _fallback_pool(trade_date, db_path)
        except Exception as e:
            logger.error("Stock pool fetch failed: %s", e)
            pool_df = _fallback_pool(trade_date, db_path)
        
        if pool_df.empty:
            raise RuntimeError("Cannot obtain stock pool")
        
        # 解析代码
        pool_df["exchange"] = pool_df["sec_code"].apply(_get_exchange)
        pool_df["board"] = pool_df["sec_code"].apply(_get_board)
        
        # 2. 基础数据（行业/上市日期/股本为低频字段：每周指定交易日刷新入静态表，
        #    其余交易日读本地缓存；缓存缺失的股票单独增量补拉）
        logger.info("Step 2: Basic data (weekly cache)")
        codes = pool_df["sec_code"].tolist()
        try:
            basic_df = _get_basic_data(ifind_client, codes, trade_date, db_path, config)
        except Exception as e:
            logger.warning("Basic data fetch failed: %s", e)
            basic_df = pd.DataFrame()
        
        # 2b. 日频截面数据（20 日日均成交量，每日拉取）
        try:
            daily_basic_df = ifind_client.get_daily_basic(codes, trade_date)
        except Exception as e:
            logger.warning("Daily basic fetch failed: %s", e)
            daily_basic_df = pd.DataFrame()
        
        # 3. 拉取行情数据（增量：本地缓存已有的日期不重复拉，除权日自动重建该股缓存）
        logger.info("Step 3: Fetch price data (incremental)")
        lookback = int(config.get("price_lookback_days", 130))
        try:
            prices_dict = _get_prices_incremental(ifind_client, codes, trade_date, lookback, db_path)
        except Exception as e:
            logger.error("Price fetch failed: %s", e)
            raise
        
        # 4. 计算波动率（批量向量化计算）
        logger.info("Step 4: Calculate volatility")
        windows = config.get("windows", [20, 40, 90])
        annual_factor = config.get("annual_factor", 252)
        min_obs_ratio = config.get("min_obs_ratio", 0.6)
        vol_df = calc_vol.calc_volatility_batch(
            codes,
            prices_dict,
            windows=windows,
            annual_factor=annual_factor,
            min_obs_ratio=min_obs_ratio,
        )
        
        # 5. 拉取指数成分（通用循环，问财兜底），并打规模分层标签
        logger.info("Step 5: Fetch index components")
        index_sets = {}
        for key, icode in config.get("index_codes", {}).items():
            try:
                codes_i = ifind_client.get_index_components(icode, trade_date)
                index_sets[key] = set(codes_i)
                logger.info("Index %s (%s): %d components", key, icode, len(codes_i))
            except Exception as e:
                logger.warning("Index component fetch failed for %s: %s", key, e)
                index_sets[key] = set()
        
        for key, members in index_sets.items():
            pool_df[f"is_{key}"] = pool_df["sec_code"].isin(members).astype(int)
        pool_df["size_label"] = pool_df.apply(_size_label, axis=1)

        # 双创微盘：不属于沪深300 / 中证500 / 中证1000
        # （上证50 成分同时带 is_csi300=1，无需单独排除）
        pool_df["is_cc_micro"] = pool_df.apply(_is_cc_micro, axis=1).astype(int)
        
        # 6. 重点监控数据
        logger.info("Step 6: Fetch monitor events")
        monitor_events = pd.DataFrame()
        failed_sources = []
        try:
            monitor_events, failed_sources = fetch_exchange.fetch_all_monitor_events(
                trade_date,
                retry_times=int(config.get("retry_times", 3)),
                retry_backoff=float(config.get("retry_backoff", 2)),
            )
            if not monitor_events.empty:
                monitor_events = fetch_exchange.derive_monitor_window(
                    monitor_events,
                    config.get("monitor_window_trading_days", 10)
                )
                monitor.merge_events(monitor_events, db_path)
        except Exception as e:
            logger.error("Monitor events fetch failed: %s", e)
            failed_sources = ["SZSE", "SSE"]
        
        monitor_fallback = bool(failed_sources)
        
        # 计算 is_key_monitored
        if not monitor_fallback:
            monitor_df = monitor.compute_key_monitored(trade_date, db_path)
        else:
            # 部分或全部交易所失败：先按事件表正常计算，再对失败交易所沿用前一日标记
            logger.warning("Monitor fetch failed for %s, applying stale fallback", failed_sources)
            monitor_df = monitor.compute_key_monitored(trade_date, db_path)
            stale_df = monitor.fallback_from_prev(trade_date, db_path, exchanges=failed_sources)
            if not stale_df.empty:
                if monitor_df.empty:
                    monitor_df = stale_df
                else:
                    # 新鲜数据优先，stale 只补充缺失的 sec_code
                    stale_df = stale_df[~stale_df["sec_code"].isin(monitor_df["sec_code"])]
                    monitor_df = pd.concat([monitor_df, stale_df], ignore_index=True)
            alert.send_alert(
                config.get("alert_wecom_webhook", ""),
                f"IA L1 监控数据降级 [{trade_date}]",
                f"交易所监控数据拉取失败: {', '.join(failed_sources)}\n"
                f"已沿用前一交易日重点监控标记（STALE_MONITOR），请人工核对。",
                email=config.get("alert_email") or None
            )
        
        # 7. 组装快照
        logger.info("Step 7: Assemble snapshot")
        snapshot = pool_df.copy()
        snapshot["trade_date"] = trade_date
        
        # 合并基础数据（上市日期、行业、股本等）
        if not basic_df.empty:
            basic_cols = ["sec_code", "list_date", "ind_sw_l1_code", "ind_sw_l1_name",
                          "ind_sw_l2_code", "ind_sw_l2_name", "special_indicator",
                          "total_shares", "total_float_shares"]
            keep = [c for c in basic_cols if c in basic_df.columns]
            if len(keep) > 1:
                snapshot = snapshot.merge(
                    basic_df[keep].drop_duplicates(subset=["sec_code"]),
                    on="sec_code", how="left"
                )
        else:
            logger.warning("No basic data available, industry/list_date fields will be empty")
        # 保证规格列齐全（缺数据时为 None，由质控的非空率检查兜底告警）
        for col in ["list_date", "ind_sw_l1_code", "ind_sw_l1_name",
                    "ind_sw_l2_code", "ind_sw_l2_name", "total_shares", "total_float_shares"]:
            if col not in snapshot.columns:
                snapshot[col] = None
        
        # 新股标签：上市 20 个交易日内（含上市当日）
        snapshot["is_new_stock"] = snapshot["list_date"].apply(
            lambda d: _is_new_stock(d, trade_date))
        
        # 合并日频截面数据（20 日日均成交量 / 日均成交额）
        if not daily_basic_df.empty and "sec_code" in daily_basic_df.columns:
            keep = [c for c in ["sec_code", "avg_vol_20d", "avg_amt_20d"] if c in daily_basic_df.columns]
            if len(keep) > 1:
                snapshot = snapshot.merge(
                    daily_basic_df[keep].drop_duplicates(subset=["sec_code"]),
                    on="sec_code", how="left"
                )
        if "avg_vol_20d" not in snapshot.columns:
            snapshot["avg_vol_20d"] = None
        if "avg_amt_20d" not in snapshot.columns:
            snapshot["avg_amt_20d"] = None
        
        # 合并波动率数据
        snapshot = snapshot.merge(vol_df, on="sec_code", how="left")
        
        # 合并监控数据
        if not monitor_df.empty:
            snapshot = snapshot.merge(monitor_df, on="sec_code", how="left")
            snapshot["is_key_monitored"] = snapshot["is_key_monitored"].fillna(0).astype(int)
        else:
            snapshot["is_key_monitored"] = 0
            snapshot["monitor_start"] = None
            snapshot["monitor_end"] = None
            snapshot["monitor_source"] = None
        
        # 处理停牌/风险警示
        def _no_price_or_suspended(code):
            p = prices_dict.get(code)
            if not p:
                return "NO_DATA"
            last = p[-1]
            # 最新收盘价为空（fill:Blank 下停牌日留空）视为停牌
            try:
                if last != last or last is None:  # NaN 检查
                    return "SUSPENDED"
            except TypeError:
                return "SUSPENDED"
            return "OK"

        price_status = snapshot["sec_code"].apply(_no_price_or_suspended)
        snapshot["is_suspended"] = (price_status == "SUSPENDED").astype(int)
        # 完全取不到价格不必然是停牌（也可能是取数失败），打质量标记供下游甄别
        no_price = price_status == "NO_DATA"
        snapshot.loc[no_price, "is_suspended"] = 1
        snapshot.loc[no_price, "quality_flag"] = snapshot.loc[no_price, "quality_flag"].apply(
            lambda x: _append_flag(x, "NO_PRICE_DATA")
        )
        # 风险警示：优先用 iFinD 特别处理标识，缺失时退回名称前缀启发式
        if "special_indicator" in snapshot.columns and snapshot["special_indicator"].notna().any():
            snapshot["is_risk_warning"] = snapshot["special_indicator"].apply(
                lambda x: 1 if x and any(k in str(x) for k in ("ST", "退")) else 0
            )
        else:
            snapshot["is_risk_warning"] = snapshot["sec_name"].apply(
                lambda x: 1 if x and (str(x).startswith("ST") or str(x).startswith("*ST") or str(x).startswith("退")) else 0
            )
        
        # 更新 quality_flag（仅对监控数据失败的交易所叠加 STALE_MONITOR 标记）
        if monitor_fallback:
            suffixes = tuple(monitor._EXCHANGE_SUFFIX[s] for s in failed_sources
                             if s in monitor._EXCHANGE_SUFFIX)
            mask = snapshot["sec_code"].str.endswith(suffixes) if suffixes else True
            snapshot.loc[mask, "quality_flag"] = snapshot.loc[mask, "quality_flag"].apply(
                lambda x: _append_flag(x, "STALE_MONITOR")
            )
        
        snapshot["update_time"] = datetime.now(timezone.utc).isoformat()
        
        # 8. 质量校验
        logger.info("Step 8: Quality checks")
        passed, checks, msg = quality.run_checks(snapshot, trade_date, db_path, config)
        
        if not passed:
            logger.error("Quality check BLOCKED: %s", msg)
            db.log_run_end(run_id, "BLOCKED", "QUALITY_CHECK", len(snapshot), msg, db_path, checks)
            alert.send_alert(
                config.get("alert_wecom_webhook", ""),
                f"IA L1 流水线阻断 [{trade_date}]",
                f"质量校验未通过:\n{msg}\n\n检查详情:\n{checks}",
                email=config.get("alert_email") or None
            )
            return 1
        
        logger.info("Quality check passed: %s", msg)
        soft_msgs = [c for c in checks.values() if not c.get("passed", True) and c.get("severity") == "soft"]
        if soft_msgs:
            logger.warning("Soft quality warnings: %s", msg)
            alert.send_alert(
                config.get("alert_wecom_webhook", ""),
                f"IA L1 质控警告 [{trade_date}]",
                f"以下软校验未通过（未阻断发布）:\n{msg}",
                email=config.get("alert_email") or None
            )
        
        # 9. 写入数据库 & 导出
        logger.info("Step 9: Save and export")
        db.save_snapshot(snapshot, db_path)
        
        csv_path, xlsx_path = export.export_snapshot(
            snapshot, trade_date, config.get("output_dir", "./output")
        )
        
        # 记录成功
        db.log_run_end(run_id, "SUCCESS", "EXPORT", len(snapshot), msg, db_path, checks)
        logger.info("Pipeline completed successfully for %s", trade_date)
        logger.info("Output: %s, %s", csv_path, xlsx_path)
        
        return 0
        
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        db.log_run_end(run_id, "FAILED", "ERROR", None, str(e), db_path)
        alert.send_alert(
            config.get("alert_wecom_webhook", ""),
            f"IA L1 流水线失败 [{trade_date}]",
            f"错误: {str(e)}\n\n当日数据未发布，前一日数据保留。",
            email=config.get("alert_email") or None
        )
        return 1


def _get_basic_data(ifind_client, codes: list, trade_date: str,
                    db_path: str, config: dict) -> pd.DataFrame:
    """
    低频基础数据获取策略：
    - 每周指定交易日（basic_data_refresh_weekday，0=周一）全量刷新入 security_static
    - 缓存为空、超过 7 天未刷新或缺少流通股本时，也会触发全量刷新
    - 其余交易日读本地缓存；缓存中没有的股票（如周中 IPO）单独增量补拉
    """
    refresh_weekday = int(config.get("basic_data_refresh_weekday", 0))
    is_refresh_day = datetime.strptime(trade_date, "%Y-%m-%d").weekday() == refresh_weekday

    static_df = db.load_static(db_path)
    stale = True
    if not static_df.empty and "updated_at" in static_df.columns:
        ts_series = pd.to_datetime(static_df["updated_at"], errors="coerce", utc=True).dropna()
        if not ts_series.empty:
            last = ts_series.iloc[ts_series.argmax()]
            stale = (pd.Timestamp.now(tz="UTC") - last).days >= 7

    total_float_missing = (static_df.empty or "total_float_shares" not in static_df.columns
                           or static_df["total_float_shares"].isna().all())
    if is_refresh_day or static_df.empty or stale or total_float_missing:
        logger.info("Basic data full refresh (refresh_day=%s, cache_empty=%s, stale=%s, total_float_missing=%s)",
                    is_refresh_day, static_df.empty, stale, total_float_missing)
        fresh = ifind_client.get_basic_data(codes, trade_date)
        if not fresh.empty:
            db.save_static(fresh, db_path)
            return fresh
        logger.warning("Basic data refresh returned empty, fall back to cache")

    # 增量补拉缓存里没有的股票
    known = set(static_df["sec_code"]) if not static_df.empty else set()
    missing = [c for c in codes if c not in known]
    if missing:
        logger.info("Incremental basic data fetch for %d new codes", len(missing))
        try:
            inc = ifind_client.get_basic_data(missing, trade_date)
            if not inc.empty:
                db.save_static(inc, db_path)
                static_df = pd.concat([static_df, inc], ignore_index=True)
        except Exception as e:
            logger.warning("Incremental basic data fetch failed: %s", e)

    return static_df


def _get_prices_incremental(ifind_client, codes: list, trade_date: str,
                            lookback: int, db_path: str) -> dict:
    """
    增量价格获取（配合 price_history 缓存）：
    - 无缓存的股票：拉完整 lookback 窗口（首次建仓/新股的一次性成本）
    - 有缓存的股票：从缓存最后一日起拉到当日；重叠日价格用于检测除权除息
      （前复权基差变化 → 该股缓存全窗口重建）
    - 当日已缓存的股票零 API 调用（同日重跑免费）
    返回 {sec_code: [close, ...]}（升序，含 NaN 缺口）供波动率计算
    """
    bounds = db.get_price_bounds(db_path)
    sessions = trade_cal.recent_trade_dates(trade_date, lookback + 5)
    full_start = sessions[0] if sessions else trade_date

    # 按拉取起始日分组
    groups: dict = {}
    for code in codes:
        last = bounds.get(code)
        if last and last >= trade_date:
            continue
        start = last if last else full_start
        groups.setdefault(start, []).append(code)

    if groups:
        logger.info("Price fetch: %d groups, %d codes need update",
                    len(groups), sum(len(v) for v in groups.values()))
    else:
        logger.info("Price cache already up to date, zero API calls")

    rebuild_codes = []
    for start, group_codes in sorted(groups.items()):
        bars = ifind_client.get_daily_bars(group_codes, start, trade_date)
        if bars.empty:
            continue
        # 除权检测：重叠日（start，即缓存最后一天）缓存价 vs 新取价
        if start != full_start:
            cached = _load_prices_at(group_codes, start, db_path)
            fresh = bars[bars["trade_date"] == start].set_index("sec_code")["close"]
            for code in group_codes:
                old_v, new_v = cached.get(code), fresh.get(code)
                if old_v is None or new_v is None or pd.isna(old_v) or pd.isna(new_v):
                    continue
                if abs(new_v - old_v) / max(abs(old_v), 1e-9) > 1e-4:
                    rebuild_codes.append(code)
        # 剔除待重建股票的增量行（稍后全窗口重建）
        bars = bars[~bars["sec_code"].isin(rebuild_codes)]
        db.save_prices(bars, db_path)

    if rebuild_codes:
        logger.info("Corporate action detected for %d codes, rebuilding full window",
                    len(rebuild_codes))
        db.delete_prices(rebuild_codes, full_start, trade_date, db_path)
        bars = ifind_client.get_daily_bars(rebuild_codes, full_start, trade_date)
        db.save_prices(bars, db_path)

    return db.load_price_series(codes, trade_date, lookback, db_path)


def _load_prices_at(codes: list, date: str, db_path: str) -> dict:
    """读取指定日期的缓存收盘价: {sec_code: close}"""
    conn = db.get_conn(db_path)
    try:
        out = {}
        for i in range(0, len(codes), 500):
            batch = codes[i:i + 500]
            placeholders = ",".join("?" * len(batch))
            cur = conn.execute(
                f"SELECT sec_code, close FROM price_history WHERE trade_date = ? AND sec_code IN ({placeholders})",
                (date, *batch)
            )
            out.update(dict(cur.fetchall()))
        return out
    finally:
        conn.close()


def _fallback_pool(trade_date: str, db_path: str) -> pd.DataFrame:
    """从数据库获取上一日股票池作为 fallback"""
    last_date = db.get_last_success_date(db_path)
    if not last_date:
        return pd.DataFrame()
    df = db.load_snapshot(last_date, db_path)
    if df.empty:
        return pd.DataFrame()
    # 只保留代码和名称
    return df[["sec_code", "sec_name"]].copy()


# 规模分层优先级：上证50 > 沪深300 > 中证500 > 中证1000 > 中证2000 > 微盘股
# （上证50 是沪深300 的子集，必须排最前）
SIZE_TIER_RULES = [
    ("is_sse50", "上证50"),
    ("is_csi300", "沪深300"),
    ("is_csi500", "中证500"),
    ("is_csi1000", "中证1000"),
    ("is_csi2000", "中证2000"),
]


def _size_label(row) -> str:
    """规模分层标签：按优先级判定，不属于任何指数 = 微盘股"""
    for flag, label in SIZE_TIER_RULES:
        if row.get(flag, 0):
            return label
    return "微盘股"


def _is_cc_micro(row) -> int:
    """双创微盘：不属于沪深300 / 中证500 / 中证1000
    （上证50 成分同时带 is_csi300=1，故只需检查这三个标记）"""
    return 1 if not (row.get("is_csi300", 0) or row.get("is_csi500", 0)
                     or row.get("is_csi1000", 0)) else 0


def _is_new_stock(list_date, trade_date: str, window: int = 20) -> int:
    """新股标签：上市日起 window 个交易日内（含上市当日）为 1"""
    if list_date is None or (isinstance(list_date, float) and pd.isna(list_date)):
        return 0
    s = str(list_date).strip()
    if not s:
        return 0
    if len(s) == 8 and s.isdigit():  # 20010827 -> 2001-08-27
        s = f"{s[:4]}-{s[4:6]}-{s[6:]}"
    try:
        n = trade_cal.trade_day_count(s, trade_date)
    except Exception:
        return 0
    if n is None:
        # 无日历库退化：28 个自然日 ≈ 20 个交易日
        days = (datetime.strptime(trade_date, "%Y-%m-%d") - datetime.strptime(s, "%Y-%m-%d")).days
        return 1 if 0 <= days <= 28 else 0
    return 1 if 0 <= n < window else 0


def _get_exchange(sec_code: str) -> str:
    """从证券代码获取交易所"""
    if sec_code.endswith(".SH"):
        return "SSE"
    elif sec_code.endswith(".SZ"):
        return "SZSE"
    return ""


def _get_board(sec_code: str) -> str:
    """从证券代码获取板块"""
    code = sec_code.split(".")[0]
    _, board = fetch_ifind.BOARD_MAP.get(code[:3], (None, None))
    return board or ""


def _append_flag(flags: str, new_flag: str) -> str:
    """向 quality_flag 追加标记"""
    if not flags or flags == "OK":
        return new_flag
    parts = flags.split(";")
    if new_flag not in parts:
        parts.append(new_flag)
    return ";".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IA Monitor L1 Pipeline")
    parser.add_argument("--date", help="Trade date (YYYY-MM-DD), default today")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--ifind-mode", default="auto", choices=["auto", "prod", "http", "proto"],
                        help="iFinD client mode")
    args = parser.parse_args()
    
    exit_code = run_pipeline(
        trade_date=args.date,
        config_path=args.config,
        ifind_mode=args.ifind_mode
    )
    sys.exit(exit_code)
