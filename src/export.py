"""
导出模块：CSV / XLSX
"""
import os
import logging
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def export_snapshot(df: pd.DataFrame, trade_date: str, output_dir: str) -> tuple:
    """
    导出当日快照：
    - CSV 保留数值型波动率（供下游程序读取计算）
    - XLSX 波动率格式化为百分比字符串（供人工阅读）
    返回: (csv_path, xlsx_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV: UTF-8-SIG 编码（Excel 双击打开不乱码）
    # 数值列控制精度：波动率 6 位小数，股本/均量取整，避免全精度浮点撑大文件
    csv_df = df.copy()
    for col in ["vol_20d", "vol_40d", "vol_90d", "vol_max"]:
        if col in csv_df.columns:
            csv_df[col] = pd.to_numeric(csv_df[col], errors="coerce").round(6)
    for col in ["total_shares", "total_float_shares", "avg_vol_20d"]:
        if col in csv_df.columns:
            # 完整精度 + 普通十进制（不取整、不用科学计数法，供下游比率计算）
            csv_df[col] = pd.to_numeric(csv_df[col], errors="coerce").map(
                lambda v: "" if pd.isna(v) else np.format_float_positional(v, trim='-'))
    csv_path = os.path.join(output_dir, f"风控基础表_{trade_date}.csv")
    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("CSV exported: %s", csv_path)
    
    # XLSX: 波动率格式化百分比、冻结首行、加筛选
    export_df = df.copy()
    vol_cols = ["vol_20d", "vol_40d", "vol_90d", "vol_max"]
    for col in vol_cols:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(_fmt_vol)
    xlsx_path = os.path.join(output_dir, f"风控基础表_{trade_date}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="风控基础表")
        ws = writer.sheets["风控基础表"]
        
        # 冻结首行
        ws.freeze_panes = "A2"
        
        # 添加筛选（openpyxl 支持）
        from openpyxl.worksheet.table import Table, TableStyleInfo
        tab = Table(displayName="RiskTable", ref=ws.dimensions)
        style = TableStyleInfo(
            name="TableStyleMedium9",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        tab.tableStyleInfo = style
        try:
            ws.add_table(tab)
        except Exception:
            pass  # 若已存在则忽略
    
    logger.info("XLSX exported: %s", xlsx_path)
    return csv_path, xlsx_path


def _fmt_vol(val: Optional[float]) -> str:
    """波动率格式化为百分比字符串"""
    if pd.isna(val) or val is None:
        return ""
    return f"{val * 100:.2f}%"
