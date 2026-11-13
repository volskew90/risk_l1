"""
数据库模块：SQLite 读写、表初始化
"""
import sqlite3
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

DB_SCHEMA = """
-- 主表：每日风控基础快照
CREATE TABLE IF NOT EXISTS security_master_daily (
    trade_date          TEXT NOT NULL,
    sec_code            TEXT NOT NULL,
    sec_name            TEXT,
    exchange            TEXT,
    board               TEXT,
    list_date           TEXT,
    ind_sw_l1_code      TEXT,
    ind_sw_l1_name      TEXT,
    ind_sw_l2_code      TEXT,
    ind_sw_l2_name      TEXT,
    vol_20d             REAL,
    vol_40d             REAL,
    vol_90d            REAL,
    vol_max             REAL,
    n_20d               INTEGER,
    n_40d               INTEGER,
    n_90d              INTEGER,
    is_sse50            INTEGER DEFAULT 0,
    is_csi300           INTEGER DEFAULT 0,
    is_csi500           INTEGER DEFAULT 0,
    is_csi1000          INTEGER DEFAULT 0,
    is_csi2000          INTEGER DEFAULT 0,
    is_cc_micro         INTEGER DEFAULT 0,
    size_label          TEXT,
    is_new_stock        INTEGER DEFAULT 0,
    total_shares        REAL,
    total_float_shares   REAL,
    avg_vol_20d         REAL,
    avg_amt_20d         REAL,
    is_suspended        INTEGER DEFAULT 0,
    is_risk_warning     INTEGER DEFAULT 0,
    is_key_monitored    INTEGER DEFAULT 0,
    monitor_start       TEXT,
    monitor_end         TEXT,
    monitor_source      TEXT,
    quality_flag        TEXT DEFAULT 'OK',
    update_time         TEXT,
    PRIMARY KEY (trade_date, sec_code)
);

CREATE INDEX IF NOT EXISTS idx_smd_date ON security_master_daily(trade_date);
CREATE INDEX IF NOT EXISTS idx_smd_code ON security_master_daily(sec_code);

-- 事件表：重点监控事件
CREATE TABLE IF NOT EXISTS monitor_event (
    event_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    sec_code            TEXT NOT NULL,
    event_type          TEXT NOT NULL,
    trigger_detail      TEXT,
    abnormal_period_start TEXT,
    abnormal_period_end   TEXT,
    disclose_date       TEXT NOT NULL,
    monitor_start       TEXT,
    monitor_end         TEXT,
    source              TEXT,
    raw_ref             TEXT,
    created_at          TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_event_unique 
ON monitor_event(sec_code, event_type, disclose_date);

CREATE INDEX IF NOT EXISTS idx_event_date ON monitor_event(disclose_date);
CREATE INDEX IF NOT EXISTS idx_event_monitor ON monitor_event(monitor_start, monitor_end);

-- 静态信息表：上市日期、行业分类、股本等低频字段（按周刷新，非每日拉取）
CREATE TABLE IF NOT EXISTS security_static (
    sec_code            TEXT PRIMARY KEY,
    list_date           TEXT,
    ind_sw_l1_code      TEXT,
    ind_sw_l1_name      TEXT,
    ind_sw_l2_code      TEXT,
    ind_sw_l2_name      TEXT,
    total_shares        REAL,
    total_float_shares   REAL,
    updated_at          TEXT
);

-- 价格缓存表：日收盘价（前复权），每日增量追加，波动率从本表读序列
-- 联合主键自带唯一索引，无需再建 sec_code+trade_date 索引
CREATE TABLE IF NOT EXISTS price_history (
    sec_code            TEXT NOT NULL,
    trade_date          TEXT NOT NULL,
    close               REAL,
    PRIMARY KEY (sec_code, trade_date)
);

-- 流水线运行日志
CREATE TABLE IF NOT EXISTS pipeline_run (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date          TEXT NOT NULL,
    start_time          TEXT,
    end_time            TEXT,
    status              TEXT,           -- RUNNING / SUCCESS / FAILED / BLOCKED
    step                TEXT,           -- 当前/最后执行步骤
    rows_count          INTEGER,
    msg                 TEXT,
    quality_checks      TEXT            -- JSON
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """初始化数据库，建表建索引，并对已有库做增量列迁移"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(DB_SCHEMA)
    _ensure_columns(conn)
    conn.commit()
    logger.info("Database initialized: %s", db_path)
    return conn


# 后加的列：已有库通过 ALTER TABLE 补齐
_SNAPSHOT_LATE_COLUMNS = {
    "vol_90d": "REAL",
    "n_90d": "INTEGER",
    "is_csi500": "INTEGER DEFAULT 0",
    "is_csi1000": "INTEGER DEFAULT 0",
    "is_csi2000": "INTEGER DEFAULT 0",
    "is_cc_micro": "INTEGER DEFAULT 0",
    "size_label": "TEXT",
    "total_shares": "REAL",
    "total_float_shares": "REAL",
    "avg_vol_20d": "REAL",
    "avg_amt_20d": "REAL",
    "is_new_stock": "INTEGER DEFAULT 0",
}

_STATIC_LATE_COLUMNS = {
    "total_shares": "REAL",
    "total_float_shares": "REAL",
}


def _ensure_columns(conn: sqlite3.Connection) -> None:
    for table, cols in (("security_master_daily", _SNAPSHOT_LATE_COLUMNS),
                        ("security_static", _STATIC_LATE_COLUMNS)):
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for col, ddl in cols.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
                logger.info("Migrated: added column %s to %s", col, table)


def get_conn(db_path: str) -> sqlite3.Connection:
    """获取数据库连接"""
    return sqlite3.connect(db_path)


def save_snapshot(df: pd.DataFrame, db_path: str) -> None:
    """将当日快照写入 security_master_daily，主键冲突则覆盖"""
    if df.empty:
        logger.warning("Empty snapshot, skip writing to DB")
        return
    conn = get_conn(db_path)
    try:
        # 先删除当日数据，保证幂等
        trade_date = df["trade_date"].iloc[0]
        conn.execute("DELETE FROM security_master_daily WHERE trade_date = ?", (trade_date,))
        # 写入新数据
        df.to_sql("security_master_daily", conn, if_exists="append", index=False)
        conn.commit()
        logger.info("Snapshot saved: %s rows for %s", len(df), trade_date)
    finally:
        conn.close()


def load_snapshot(trade_date: str, db_path: str) -> pd.DataFrame:
    """读取指定交易日快照"""
    conn = get_conn(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM security_master_daily WHERE trade_date = ? ORDER BY sec_code",
            conn, params=(trade_date,)
        )
        return df
    finally:
        conn.close()


def save_events(df: pd.DataFrame, db_path: str) -> None:
    """写入监控事件。去重由调用方（monitor.merge_events）负责，此处为纯追加"""
    if df.empty:
        return
    conn = get_conn(db_path)
    try:
        # SQLite UPSERT (INSERT OR REPLACE)
        df.to_sql("monitor_event", conn, if_exists="append", index=False)
        conn.commit()
        logger.info("Events saved: %s rows", len(df))
    finally:
        conn.close()


def load_events(trade_date: str, db_path: str) -> pd.DataFrame:
    """读取在 trade_date 仍处于监控窗口内的事件"""
    conn = get_conn(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT * FROM monitor_event
            WHERE monitor_start <= ? AND (monitor_end >= ? OR monitor_end IS NULL)
            """,
            conn, params=(trade_date, trade_date)
        )
        return df
    finally:
        conn.close()


def log_run_start(trade_date: str, db_path: str) -> int:
    """记录流水线开始，返回 run_id；同时清理此前残留的 RUNNING 僵尸记录"""
    conn = get_conn(db_path)
    try:
        conn.execute(
            """
            UPDATE pipeline_run
            SET status = 'FAILED', end_time = ?, msg = COALESCE(msg, '') || ' [zombie: run interrupted]'
            WHERE status = 'RUNNING'
            """,
            (datetime.now(timezone.utc).isoformat(),)
        )
        cur = conn.execute(
            """
            INSERT INTO pipeline_run (trade_date, start_time, status, step)
            VALUES (?, ?, 'RUNNING', 'START')
            """,
            (trade_date, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def log_run_end(run_id: int, status: str, step: str, rows_count: Optional[int],
                msg: str, db_path: str, quality_checks: Optional[Dict] = None) -> None:
    """记录流水线结束状态"""
    import json
    conn = get_conn(db_path)
    try:
        conn.execute(
            """
            UPDATE pipeline_run
            SET end_time = ?, status = ?, step = ?, rows_count = ?, msg = ?, quality_checks = ?
            WHERE run_id = ?
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                status, step, rows_count, msg,
                json.dumps(quality_checks, ensure_ascii=False) if quality_checks else None,
                run_id
            )
        )
        conn.commit()
    finally:
        conn.close()


def get_last_success_date(db_path: str) -> Optional[str]:
    """获取最近一次成功运行的交易日"""
    conn = get_conn(db_path)
    try:
        cur = conn.execute(
            "SELECT trade_date FROM pipeline_run WHERE status = 'SUCCESS' ORDER BY run_id DESC LIMIT 1"
        )
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def get_snapshot_count(trade_date: str, db_path: str) -> int:
    """获取指定交易日快照行数"""
    conn = get_conn(db_path)
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM security_master_daily WHERE trade_date = ?", (trade_date,)
        )
        return cur.fetchone()[0]
    finally:
        conn.close()


STATIC_COLS = ["sec_code", "list_date", "ind_sw_l1_code", "ind_sw_l1_name",
               "ind_sw_l2_code", "ind_sw_l2_name", "total_shares", "total_float_shares"]


def save_static(df: pd.DataFrame, db_path: str) -> None:
    """写入/更新静态信息表（sec_code 主键覆盖）"""
    if df.empty:
        return
    keep = [c for c in STATIC_COLS if c in df.columns]
    if "sec_code" not in keep or len(keep) == 1:
        return
    df = df[keep].copy()
    df["updated_at"] = datetime.now(timezone.utc).isoformat()
    conn = get_conn(db_path)
    try:
        df.to_sql("security_static", conn, if_exists="append", index=False,
                  method=_upsert_static)
        conn.commit()
        logger.info("Static info saved: %s rows", len(df))
    finally:
        conn.close()


def _upsert_static(table, conn, keys, data_iter):
    """pandas to_sql 的 UPSERT 适配：只更新本次提供的列，不清空其他字段"""
    rows = list(data_iter)
    if not rows:
        return
    col_list = list(keys)
    cols = ", ".join(f'"{k}"' for k in col_list)
    placeholders = ", ".join("?" * len(col_list))
    updates = ", ".join(f'"{k}"=excluded."{k}"' for k in col_list if k != "sec_code")
    sql = (f'INSERT INTO "{table.name}" ({cols}) VALUES ({placeholders}) '
           f'ON CONFLICT(sec_code) DO UPDATE SET {updates}')
    conn.executemany(sql, rows)


def load_static(db_path: str) -> pd.DataFrame:
    """读取静态信息表"""
    conn = get_conn(db_path)
    try:
        try:
            return pd.read_sql_query("SELECT * FROM security_static", conn)
        except Exception:
            return pd.DataFrame(columns=STATIC_COLS + ["updated_at"])
    finally:
        conn.close()


# ---- 价格缓存 ----

def save_prices(df: pd.DataFrame, db_path: str) -> None:
    """价格落缓存（sec_code + trade_date 主键去重覆盖）。df 列: sec_code, trade_date, close"""
    if df.empty:
        return
    df = df[["sec_code", "trade_date", "close"]].copy()
    conn = get_conn(db_path)
    try:
        cols = "sec_code, trade_date, close"
        sql = f"INSERT OR REPLACE INTO price_history ({cols}) VALUES (?, ?, ?)"
        conn.executemany(sql, df.itertuples(index=False, name=None))
        conn.commit()
        logger.info("Price cache saved: %s rows", len(df))
    finally:
        conn.close()


def get_price_bounds(db_path: str) -> dict:
    """每只股票缓存的最大日期: {sec_code: max_trade_date}"""
    conn = get_conn(db_path)
    try:
        cur = conn.execute(
            "SELECT sec_code, MAX(trade_date) FROM price_history GROUP BY sec_code"
        )
        return {r[0]: r[1] for r in cur.fetchall()}
    finally:
        conn.close()


def delete_prices(codes: List[str], start_date: str, end_date: str, db_path: str) -> None:
    """删除指定股票在日期区间内的缓存（用于除权后全窗口重建）"""
    if not codes:
        return
    conn = get_conn(db_path)
    try:
        for i in range(0, len(codes), 500):
            batch = codes[i:i + 500]
            placeholders = ",".join("?" * len(batch))
            conn.execute(
                f"DELETE FROM price_history WHERE trade_date >= ? AND trade_date <= ? "
                f"AND sec_code IN ({placeholders})",
                (start_date, end_date, *batch)
            )
        conn.commit()
        logger.info("Price cache deleted for %d codes [%s, %s]", len(codes), start_date, end_date)
    finally:
        conn.close()


def load_price_series(codes: List[str], end_date: str, lookback: int,
                      db_path: str) -> Dict[str, List[float]]:
    """从缓存读价格序列（按日期升序，最近 lookback+5 个交易日窗口内）"""
    from src import trade_cal
    dates = trade_cal.recent_trade_dates(end_date, lookback + 5)
    if not dates:
        return {c: [] for c in codes}
    start_date = dates[0]
    result: Dict[str, List[float]] = {c: [] for c in codes}
    conn = get_conn(db_path)
    try:
        # 分批避免 SQLite 变量数上限
        for i in range(0, len(codes), 500):
            batch = codes[i:i + 500]
            placeholders = ",".join("?" * len(batch))
            df = pd.read_sql_query(
                f"""
                SELECT sec_code, trade_date, close FROM price_history
                WHERE sec_code IN ({placeholders})
                  AND trade_date >= ? AND trade_date <= ?
                ORDER BY sec_code, trade_date
                """,
                conn, params=(*batch, start_date, end_date)
            )
            for code, grp in df.groupby("sec_code"):
                result[code] = grp["close"].tolist()
    finally:
        conn.close()
    return result
