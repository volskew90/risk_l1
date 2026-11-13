"""
L1 各模块单元测试：交易日历 / 监控逻辑 / 质量校验 / 波动率缺口处理
可用 pytest 运行，也可直接作为脚本运行
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src import calc_vol, trade_cal, db, monitor, quality


# ---- 交易日历 ----

def test_trade_cal_holiday():
    """2026-02-17 是春节（周二），必须是非交易日"""
    assert trade_cal.is_trade_date("2026-02-17") is False
    assert trade_cal.is_trade_date("2026-09-04") is True   # 周五
    assert trade_cal.is_trade_date("2026-09-05") is False  # 周六
    print("✓ 交易日历节假日测试通过")


def test_trade_date_add():
    """monitor_end 推导：2026-09-04(五) 起第 2 个交易日 = 2026-09-08(二)"""
    assert trade_cal.trade_date_add("2026-09-04", 2) == "2026-09-08"
    print("✓ 交易日推算测试通过")


# ---- 波动率缺口处理 ----

def test_calc_vol_gap_not_compressed():
    """停牌缺口（None）不应被压缩成单日收益"""
    prices = [10.0, 10.2, None, 10.3, 10.25]
    vols, ns, flags = calc_vol.calc_volatility(prices, windows=[4], annual_factor=252)
    # 5 个价格 -> 4 个收益率位置，中间 None 使两侧收益率为 NaN，仅剩 2 个有效
    assert ns[4] == 2, f"expected 2 valid returns, got {ns[4]}"
    assert "PRICE_GAP" in flags[4]
    print("✓ 停牌缺口测试通过")


def test_calc_vol_bad_values():
    """非法价格（0、负数、字符串）不崩溃"""
    prices = [10.0, 0, "bad", 10.3, 10.25, 10.4]
    vols, ns, flags = calc_vol.calc_volatility(prices, windows=[3], annual_factor=252)
    assert "PRICE_GAP" in flags[3] or vols[3] is None
    print("✓ 非法价格测试通过")


# ---- 监控逻辑（使用临时 sqlite） ----

def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    os.unlink(path)
    db.init_db(path)
    return path


def test_monitor_merge_and_window():
    """事件去重 + is_key_monitored 窗口计算"""
    path = _tmp_db()
    try:
        events = pd.DataFrame([{
            "sec_code": "600000.SH",
            "event_type": "SEVERE_ABNORMAL",
            "trigger_detail": "t",
            "abnormal_period_start": None,
            "abnormal_period_end": None,
            "disclose_date": "2026-09-01",
            "monitor_start": "2026-09-01",
            "monitor_end": "2026-09-15",
            "source": "SSE_PAGE",
            "raw_ref": "",
        }])
        monitor.merge_events(events, path)
        monitor.merge_events(events, path)  # 重复合并不应产生重复行
        conn = db.get_conn(path)
        n = conn.execute("SELECT COUNT(*) FROM monitor_event").fetchone()[0]
        conn.close()
        assert n == 1, f"dedupe failed, {n} rows"

        res = monitor.compute_key_monitored("2026-09-10", path)
        assert len(res) == 1 and res.iloc[0]["is_key_monitored"] == 1
        # 窗口外不应命中
        assert monitor.compute_key_monitored("2026-09-20", path).empty
        print("✓ 监控事件合并/窗口测试通过")
    finally:
        os.unlink(path)


def test_monitor_fallback_by_exchange():
    """fallback 仅沿用失败交易所的股票"""
    path = _tmp_db()
    try:
        snap = pd.DataFrame([
            {"trade_date": "2026-09-03", "sec_code": "600000.SH", "is_key_monitored": 1,
             "monitor_start": "2026-09-01", "monitor_end": "2026-09-15", "monitor_source": "SSE_PAGE"},
            {"trade_date": "2026-09-03", "sec_code": "000001.SZ", "is_key_monitored": 1,
             "monitor_start": "2026-09-01", "monitor_end": "2026-09-15", "monitor_source": "SZSE_PAGE"},
        ])
        db.save_snapshot(snap, path)
        run_id = db.log_run_start("2026-09-03", path)
        db.log_run_end(run_id, "SUCCESS", "EXPORT", 2, "ok", path)

        fb = monitor.fallback_from_prev("2026-09-04", path, exchanges=["SZSE"])
        assert len(fb) == 1 and fb.iloc[0]["sec_code"] == "000001.SZ"
        assert fb.iloc[0]["monitor_source"].endswith("_STALE")
        fb_all = monitor.fallback_from_prev("2026-09-04", path)
        assert len(fb_all) == 2
        print("✓ 监控 fallback 交易所过滤测试通过")
    finally:
        os.unlink(path)


# ---- 质量校验 ----

def _base_snapshot():
    return pd.DataFrame({
        "trade_date": ["2026-09-04"] * 3,
        "sec_code": ["600000.SH", "000001.SZ", "300750.SZ"],
        "sec_name": ["浦发银行", "平安银行", "宁德时代"],
        "board": ["主板", "主板", "创业板"],
        "ind_sw_l1_code": ["801780", "801780", "801730"],
        "vol_20d": [0.2, 0.3, 0.5],
        "vol_40d": [0.2, 0.3, 0.5],
        "vol_90d": [0.2, 0.3, 0.5],
        "is_sse50": [1, 0, 0],
        "is_csi300": [1, 1, 1],
    })


def test_quality_missing_column_fails():
    """关键列缺失必须判失败（不能静默跳过）"""
    path = _tmp_db()
    try:
        df = _base_snapshot().drop(columns=["ind_sw_l1_code"])
        passed, checks, msg = quality.run_checks(df, "2026-09-04", path, {})
        assert not passed
        assert checks["nonnull_ind_sw_l1_code"]["passed"] is False
        print("✓ 缺列质控失败测试通过")
    finally:
        os.unlink(path)


def test_quality_soft_check_not_blocking():
    """成分股数量不符是软校验：告警但不阻断"""
    path = _tmp_db()
    try:
        df = _base_snapshot()  # is_sse50 合计 1 ≠ 50
        passed, checks, msg = quality.run_checks(df, "2026-09-04", path, {})
        assert passed, f"soft check should not block: {msg}"
        assert checks["sse50_count"]["severity"] == "soft"
        assert "[soft]" in msg
        print("✓ 软校验不阻断测试通过")
    finally:
        os.unlink(path)


def test_quality_hard_check_blocks():
    """主键重复是硬校验：阻断"""
    path = _tmp_db()
    try:
        df = pd.concat([_base_snapshot(), _base_snapshot().iloc[:1]], ignore_index=True)
        passed, checks, msg = quality.run_checks(df, "2026-09-04", path, {})
        assert not passed
        assert checks["pk_unique"]["passed"] is False
        print("✓ 硬校验阻断测试通过")
    finally:
        os.unlink(path)


def test_quality_price_coverage_blocks():
    """大批量取数失败（vol_20d 大面积为空）必须阻断，防止覆盖历史好数据"""
    path = _tmp_db()
    try:
        df = _base_snapshot()
        df.loc[df.index[1:], "vol_20d"] = None  # 2/3 为空，覆盖率 33%
        passed, checks, msg = quality.run_checks(df, "2026-09-04", path, {})
        assert not passed
        assert checks["price_coverage"]["severity"] == "hard"
        assert checks["price_coverage"]["passed"] is False
        print("✓ 价格覆盖率阻断测试通过")
    finally:
        os.unlink(path)


def test_basic_data_weekly_cache():
    """低频基础数据：刷新日全量拉取入缓存，其余日读缓存，新股增量补拉"""
    path = _tmp_db()
    try:
        from src import pipeline

        class FakeClient:
            def __init__(self):
                self.calls = []
            def get_basic_data(self, codes, d):
                self.calls.append(list(codes))
                return pd.DataFrame({"sec_code": codes,
                                     "list_date": ["20200101"] * len(codes),
                                     "total_float_shares": [1000000.0] * len(codes)})

        cfg = {"basic_data_refresh_weekday": 0}  # 0=周一

        # 2026-09-07 周一：全量刷新
        c1 = FakeClient()
        df = pipeline._get_basic_data(c1, ["600519.SH", "000001.SZ"], "2026-09-07", path, cfg)
        assert len(df) == 2 and c1.calls == [["600519.SH", "000001.SZ"]]

        # 2026-09-08 周二：读缓存，不调 API
        c2 = FakeClient()
        df = pipeline._get_basic_data(c2, ["600519.SH", "000001.SZ"], "2026-09-08", path, cfg)
        assert len(df) == 2 and not c2.calls

        # 周二出现新股：只对新股增量补拉
        c3 = FakeClient()
        df = pipeline._get_basic_data(c3, ["600519.SH", "000001.SZ", "688999.SH"],
                                      "2026-09-08", path, cfg)
        assert len(df) == 3 and c3.calls == [["688999.SH"]]
        print("✓ 基础数据周刷新缓存测试通过")
    finally:
        os.unlink(path)


def test_basic_data_refreshes_missing_total_float():
    """旧缓存没有流通股本时，即使非刷新日也要全量更新"""
    path = _tmp_db()
    try:
        from src import pipeline

        db.save_static(pd.DataFrame({
            "sec_code": ["688057.SH"],
            "list_date": ["20201111"],
        }), path)

        class FakeClient:
            def __init__(self):
                self.calls = []

            def get_basic_data(self, codes, d):
                self.calls.append(list(codes))
                return pd.DataFrame({
                    "sec_code": codes,
                    "total_float_shares": [276000000.0] * len(codes),
                })

        client = FakeClient()
        df = pipeline._get_basic_data(
            client, ["688057.SH"], "2026-09-08", path,
            {"basic_data_refresh_weekday": 0},
        )
        assert client.calls == [["688057.SH"]]
        assert df.iloc[0]["total_float_shares"] == 276000000.0
    finally:
        os.unlink(path)


def test_prices_incremental():
    """价格增量缓存：首次全量、每日增量、同日重跑零调用、除权全窗口重建"""
    path = _tmp_db()
    try:
        from src import pipeline

        class FakeClient:
            def __init__(self, bars):
                self.bars = bars
                self.calls = []
            def get_daily_bars(self, codes, start, end):
                self.calls.append((list(codes), start, end))
                rows = [r for r in self.bars
                        if r[0] in codes and start <= r[1] <= end]
                return pd.DataFrame(rows, columns=["sec_code", "trade_date", "close"])

        codes = ["600519.SH", "000001.SZ"]
        base = [
            ("600519.SH", "2026-09-02", 100.0), ("600519.SH", "2026-09-03", 101.0),
            ("000001.SZ", "2026-09-02", 10.0), ("000001.SZ", "2026-09-03", 10.5),
        ]

        # 首日：无缓存，全窗口拉取
        c1 = FakeClient(base)
        p = pipeline._get_prices_incremental(c1, codes, "2026-09-03", 5, path)
        assert len(c1.calls) == 1 and c1.calls[0][1] < "2026-09-02"
        assert p["600519.SH"] == [100.0, 101.0]

        # 次日：只拉增量（起始日 = 缓存最后一天）
        day2 = base + [("600519.SH", "2026-09-04", 102.0), ("000001.SZ", "2026-09-04", 10.6)]
        c2 = FakeClient(day2)
        p = pipeline._get_prices_incremental(c2, codes, "2026-09-04", 5, path)
        assert len(c2.calls) == 1 and c2.calls[0][1] == "2026-09-03"
        assert p["600519.SH"][-1] == 102.0

        # 同日重跑：缓存已覆盖，零 API 调用
        c3 = FakeClient(day2)
        pipeline._get_prices_incremental(c3, codes, "2026-09-04", 5, path)
        assert not c3.calls

        # 除权除息：重叠日价格变化 -> 该股全窗口重建，另一只不受影响
        bars_adj = [
            ("600519.SH", "2026-09-02", 102.0), ("600519.SH", "2026-09-03", 103.0),
            ("600519.SH", "2026-09-04", 104.0), ("600519.SH", "2026-09-07", 105.0),
            ("000001.SZ", "2026-09-04", 10.6), ("000001.SZ", "2026-09-07", 10.7),
        ]
        c4 = FakeClient(bars_adj)
        p = pipeline._get_prices_incremental(c4, codes, "2026-09-07", 5, path)
        assert p["600519.SH"] == [102.0, 103.0, 104.0, 105.0], p["600519.SH"]
        assert p["000001.SZ"][-1] == 10.7
        print("✓ 价格增量缓存测试通过")
    finally:
        os.unlink(path)


def test_size_label():
    """规模分层标签优先级：上证50 > 沪深300 > 中证500 > 中证1000 > 中证2000 > 微盘股"""
    from src import pipeline
    assert pipeline._size_label({"is_sse50": 1, "is_csi300": 1}) == "上证50"  # 子集优先
    assert pipeline._size_label({"is_csi300": 1, "is_csi500": 1}) == "沪深300"
    assert pipeline._size_label({"is_csi500": 1, "is_csi1000": 1}) == "中证500"
    assert pipeline._size_label({"is_csi1000": 1, "is_csi2000": 0}) == "中证1000"
    assert pipeline._size_label({"is_csi2000": 1}) == "中证2000"
    assert pipeline._size_label({}) == "微盘股"
    assert pipeline._size_label({"is_csi300": 0, "is_sse50": 0}) == "微盘股"
    print("✓ 规模分层标签测试通过")


def test_is_cc_micro():
    """双创微盘：不属于沪深300 / 中证500 / 中证1000
    （上证50 成分在 L1 中必然同时带 is_csi300=1，故无需单独检查）"""
    from src import pipeline
    assert pipeline._is_cc_micro({"is_csi300": 1, "is_csi500": 1, "is_csi1000": 1}) == 0
    assert pipeline._is_cc_micro({"is_sse50": 1, "is_csi300": 1}) == 0
    assert pipeline._is_cc_micro({"is_csi500": 1}) == 0
    assert pipeline._is_cc_micro({"is_csi1000": 1}) == 0
    assert pipeline._is_cc_micro({"is_csi2000": 1}) == 1        # 中证2000 属于双创微盘
    assert pipeline._is_cc_micro({}) == 1
    assert pipeline._is_cc_micro({"is_csi300": 0, "is_csi500": 0, "is_csi1000": 0}) == 1
    print("✓ 双创微盘标签测试通过")


def test_is_new_stock():
    """新股标签：上市 20 个交易日内（含上市当日）"""
    from src import pipeline
    # 上市当日
    assert pipeline._is_new_stock("2026-09-10", "2026-09-10") == 1
    # 2026-08-19 上市，到 2026-09-10 为第 16 个交易日 -> 新股
    assert pipeline._is_new_stock("20260819", "2026-09-10") == 1
    # 老股
    assert pipeline._is_new_stock("19910403", "2026-09-10") == 0
    # 恰好第 20 个交易日之外：2026-08-12 上市，到 2026-09-10 已超 20 个交易日
    assert pipeline._is_new_stock("2026-08-12", "2026-09-10") == 0
    # 空值不崩
    assert pipeline._is_new_stock(None, "2026-09-10") == 0
    print("✓ 新股标签测试通过")


if __name__ == "__main__":
    test_trade_cal_holiday()
    test_trade_date_add()
    test_calc_vol_gap_not_compressed()
    test_calc_vol_bad_values()
    test_monitor_merge_and_window()
    test_monitor_fallback_by_exchange()
    test_quality_missing_column_fails()
    test_quality_soft_check_not_blocking()
    test_quality_hard_check_blocks()
    test_quality_price_coverage_blocks()
    test_basic_data_weekly_cache()
    test_prices_incremental()
    test_size_label()
    test_is_new_stock()
    print("\n所有新增测试通过 ✓")
