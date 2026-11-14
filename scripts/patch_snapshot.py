#!/usr/bin/env python3
"""
补丁工具：为已有快照补充股本/日均成交字段，不重新拉取历史价格
用法: ./.venv/bin/python scripts/patch_snapshot.py --date 2026-09-08
"""
import os
import sys
import argparse
import yaml
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import db, export, fetch_ifind

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PATCH_COLS = ["total_shares", "total_float_shares", "avg_vol_20d"]


def main():
    parser = argparse.ArgumentParser(description="Patch snapshot with share/volume fields")
    parser.add_argument("--date", required=True, help="Trade date (YYYY-MM-DD)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    db_path = config["db_path"]
    db.init_db(db_path)

    snapshot = db.load_snapshot(args.date, db_path)
    if snapshot.empty:
        logger.error("No snapshot for %s", args.date)
        return 1
    codes = snapshot["sec_code"].tolist()
    logger.info("Patching %d codes for %s", len(codes), args.date)

    client = fetch_ifind.IFindClient(
        mode="http",
        retry_times=int(config.get("retry_times", 3)),
        retry_backoff=float(config.get("retry_backoff", 2)),
        refresh_token=config.get("ifind_refresh_token") or None,
    )

    # 股本（低频规格）+ 日均成交（日频规格），均不涉及历史价格
    static_df = client.get_basic_data(codes, args.date)
    daily_df = client.get_daily_basic(codes, args.date)

    patch = static_df.merge(daily_df, on="sec_code", how="outer") \
        if not static_df.empty and not daily_df.empty else (static_df if not static_df.empty else daily_df)
    if patch.empty:
        logger.error("No patch data fetched")
        return 1

    # 批量更新快照表（使用 executemany 替代逐行 UPDATE）
    conn = db.get_conn(db_path)
    updated = 0
    try:
        avail_cols = [col for col in PATCH_COLS if col in patch.columns]
        if avail_cols:
            set_clause = ", ".join(f"{col} = COALESCE(?, {col})" for col in avail_cols)
            sql = (
                f"UPDATE security_master_daily SET {set_clause} "
                f"WHERE trade_date = ? AND sec_code = ?"
            )
            params_list = []
            for row in patch[["sec_code"] + avail_cols].itertuples(index=False):
                code = row[0]
                vals = [
                    float(v) if (v is not None and pd.notna(v)) else None
                    for v in row[1:]
                ]
                if any(v is not None for v in vals):
                    params_list.append((*vals, args.date, code))
            if params_list:
                conn.executemany(sql, params_list)
                updated = len(params_list)
        conn.commit()
    finally:
        conn.close()
    logger.info("Patched %d rows", updated)

    # 股本同步进静态缓存
    share_cols = [c for c in ["sec_code", "total_shares", "total_float_shares"] if c in patch.columns]
    if len(share_cols) > 1:
        db.save_static(patch[share_cols].dropna(subset=["total_shares"]), db_path)

    # 重新导出（从库读全量快照，价格波动率原样保留）
    snapshot = db.load_snapshot(args.date, db_path)
    csv_path, xlsx_path = export.export_snapshot(
        snapshot, args.date, config.get("output_dir", "./output"))
    logger.info("Re-exported: %s, %s", csv_path, xlsx_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
