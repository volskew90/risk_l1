#!/usr/bin/env python3
"""
L2 组合风险输入模块
计算单票市值、组合/行业/双创微盘集中度和股本占比。
本层不使用阈值；限制项判断与超限金额计算属于 L3。
"""
import os
import sys
import argparse
import logging
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src import db

logger = logging.getLogger(__name__)

PORTFOLIO_COLS = ["sec_code", "sec_name", "shares", "price"]
UNKNOWN_INDUSTRY = "未知"


def load_portfolio(path: str) -> pd.DataFrame:
    """读取单边组合；空头数量为负，市值和持股数量取绝对值"""
    if path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, dtype={"sec_code": str})
    else:
        df = pd.read_csv(path, dtype={"sec_code": str})

    missing = [c for c in ["sec_code", "shares", "price"] if c not in df.columns]
    if missing:
        raise ValueError(f"组合文件缺少必需列: {missing}")
    if "sec_name" not in df.columns:
        df["sec_name"] = ""
    if df["sec_code"].duplicated().any():
        dup = df.loc[df["sec_code"].duplicated(), "sec_code"].tolist()
        raise ValueError(f"组合文件存在重复证券代码: {dup}")

    df = df.copy()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    bad = df[(df["shares"] == 0) | (df["price"] <= 0) |
             df["shares"].isna() | df["price"].isna()]
    if not bad.empty:
        raise ValueError(f"组合文件存在非法价格/仓位: {bad['sec_code'].tolist()}")
    if (df["shares"] > 0).any() and (df["shares"] < 0).any():
        raise ValueError("第二层仅支持纯多头或纯空头单边组合")

    df["abs_shares"] = df["shares"].abs()
    df["market_value"] = df["abs_shares"] * df["price"]
    return df[PORTFOLIO_COLS + ["abs_shares", "market_value"]]


def _latest_snapshot_date(db_path: str) -> Optional[str]:
    """L1 快照库中最新交易日"""
    conn = db.get_conn(db_path)
    try:
        row = conn.execute("SELECT MAX(trade_date) FROM security_master_daily").fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def attach_industry(portfolio: pd.DataFrame, db_path: str,
                    trade_date: Optional[str] = None) -> pd.DataFrame:
    """关联 L1 快照的行业、股本、成交额和指数标记"""
    if trade_date is None:
        trade_date = _latest_snapshot_date(db_path)
        if trade_date is None:
            raise RuntimeError(f"L1 数据库无快照数据: {db_path}")
    logger.info("Using L1 snapshot date: %s", trade_date)

    snapshot = db.load_snapshot(trade_date, db_path)
    l1_cols = [
        "sec_code", "ind_sw_l1_code", "ind_sw_l1_name", "is_cc_micro",
        "total_shares", "total_float_shares", "avg_amt_20d", "vol_90d", "is_sse50",
        "is_csi300", "is_csi500", "is_csi1000",
    ]
    l1 = snapshot[[c for c in l1_cols if c in snapshot.columns]].drop_duplicates("sec_code")

    df = portfolio.merge(l1, on="sec_code", how="left")
    df["ind_sw_l1_name"] = df["ind_sw_l1_name"].fillna(UNKNOWN_INDUSTRY)
    for col in ["total_shares", "total_float_shares", "avg_amt_20d"]:
        if col not in df.columns:
            df[col] = pd.NA
    for col in ["is_cc_micro", "is_sse50", "is_csi300", "is_csi500", "is_csi1000"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)
    n_unknown = (df["ind_sw_l1_name"] == UNKNOWN_INDUSTRY).sum()
    if n_unknown:
        logger.warning("%d stocks not found in L1 snapshot, industry = %s",
                       n_unknown, UNKNOWN_INDUSTRY)
    df["snapshot_date"] = trade_date
    return df


def calc_concentration(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """计算第二层集中度和股本占比；阈值判断与超限金额属于第三层"""
    total_mv = df["market_value"].sum()
    if total_mv <= 0:
        raise ValueError("组合总市值为 0，无法计算集中度")

    df = df.copy()
    if "abs_shares" not in df.columns:
        df["abs_shares"] = df["shares"].abs()
    df["total_mv"] = total_mv
    df["weight_in_portfolio"] = df["market_value"] / total_mv

    industry = (df.groupby("ind_sw_l1_name", as_index=False)
                  .agg(market_value=("market_value", "sum"),
                       n_stocks=("sec_code", "count")))
    industry["weight_in_portfolio"] = industry["market_value"] / total_mv
    industry = industry.sort_values("weight_in_portfolio", ascending=False,
                                    ignore_index=True)

    ind_mv = industry.set_index("ind_sw_l1_name")["market_value"]
    df["industry_mv"] = df["ind_sw_l1_name"].map(ind_mv)
    df["industry_weight"] = df["industry_mv"] / total_mv
    df["weight_in_industry"] = df["market_value"] / df["industry_mv"]

    cc_micro_mv = df.loc[df.get("is_cc_micro", pd.Series(0, index=df.index)) == 1,
                         "market_value"].sum()
    df["cc_micro_mv"] = cc_micro_mv
    df["cc_micro_weight"] = cc_micro_mv / total_mv
    df["weight_in_cc_micro"] = 0.0
    cc_mask = (df["is_cc_micro"] == 1) & (cc_micro_mv > 0)
    df.loc[cc_mask, "weight_in_cc_micro"] = (
        df.loc[cc_mask, "market_value"] / cc_micro_mv)

    df["total_share_ratio"] = (
        df["abs_shares"] / df["total_shares"].where(df["total_shares"] > 0))
    df["float_share_ratio"] = (
        df["abs_shares"] / df["total_float_shares"].where(df["total_float_shares"] > 0))
    df = df.sort_values(["ind_sw_l1_name", "weight_in_industry"],
                        ascending=[True, False], ignore_index=True)

    top_ind = industry.iloc[0]
    top_stock = df.loc[df["weight_in_portfolio"].idxmax()]
    summary = {
        "total_market_value": total_mv,
        "n_positions": len(df),
        "n_industries": len(industry),
        "industry_hhi": float((industry["weight_in_portfolio"] ** 2).sum()),
        "top_industry": top_ind["ind_sw_l1_name"],
        "top_industry_weight": float(top_ind["weight_in_portfolio"]),
        "top_stock": top_stock["sec_name"] or top_stock["sec_code"],
        "top_stock_weight": float(top_stock["weight_in_portfolio"]),
        "cc_micro_weight": float(cc_micro_mv / total_mv),
        "cc_micro_market_value": float(cc_micro_mv),
    }
    return df, industry, summary


def print_report(df: pd.DataFrame, industry: pd.DataFrame, summary: dict) -> None:
    """打印可读报告"""
    print("=" * 60)
    print("组合风险输入报告")
    print(f"快照日期: {df['snapshot_date'].iloc[0]}    "
          f"总市值: {summary['total_market_value']:,.0f} 元    "
          f"持仓数: {summary['n_positions']}")
    print("-" * 60)
    print(f"{'行业':<10}{'市值(万)':>12}{'占组合':>10}{'股票数':>8}")
    for _, r in industry.iterrows():
        print(f"{r['ind_sw_l1_name']:<12}{r['market_value'] / 1e4:>12,.1f}"
              f"{r['weight_in_portfolio']:>10.1%}{r['n_stocks']:>8}")
    print("-" * 60)
    print(f"{'代码':<12}{'名称':<10}{'行业':<10}{'占组合':>9}{'占行业':>9}")
    for _, r in df.iterrows():
        print(f"{r['sec_code']:<12}{r['sec_name']:<12}{r['ind_sw_l1_name']:<12}"
              f"{r['weight_in_portfolio']:>9.1%}{r['weight_in_industry']:>9.1%}")
    print("-" * 60)
    print(f"行业 HHI: {summary['industry_hhi']:.4f}    "
          f"第一大行业: {summary['top_industry']} ({summary['top_industry_weight']:.1%})    "
          f"第一大重仓: {summary['top_stock']} ({summary['top_stock_weight']:.1%})")
    print(f"双创微盘占比: {summary['cc_micro_weight']:.1%} "
          f"(市值 {summary['cc_micro_market_value'] / 1e4:,.1f} 万)")
    print("=" * 60)


def export_report(df: pd.DataFrame, industry: pd.DataFrame, summary: dict,
                  output_dir: str, trade_date: str) -> str:
    """导出 xlsx 报告（行业汇总 / 个股明细 / 组合摘要三个 sheet）"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"组合集中度_{trade_date}.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        industry.to_excel(writer, sheet_name="行业集中度", index=False)
        df.to_excel(writer, sheet_name="个股明细", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="组合摘要", index=False)
    logger.info("Report exported: %s", path)
    return path


def main():
    parser = argparse.ArgumentParser(description="L2 组合风险输入")
    parser.add_argument("--portfolio", default="data/sample_portfolio.csv",
                        help="组合文件路径（csv/xlsx，列: sec_code, sec_name, shares, price）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径（取 db_path）")
    parser.add_argument("--db", help="L1 数据库路径（覆盖 config）")
    parser.add_argument("--date", help="L1 快照交易日，默认最新")
    parser.add_argument("--out", help="导出目录（不指定则只打印报告）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    db_path = args.db
    if db_path is None:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            db_path = yaml.safe_load(f).get("db_path", "./data/risk_l1.sqlite")

    portfolio = load_portfolio(args.portfolio)
    df = attach_industry(portfolio, db_path, args.date)
    df, industry, summary = calc_concentration(df)
    print_report(df, industry, summary)

    if args.out:
        path = export_report(df, industry, summary, args.out,
                             df["snapshot_date"].iloc[0])
        print(f"报告已导出: {path}")


if __name__ == "__main__":
    main()
