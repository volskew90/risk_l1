"""
波动率计算单元测试（文档 §4.3 验收用例）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import calc_vol


def test_example_43():
    """文档 §4.3 示例：价格序列 10.00, 10.20, 10.10, 10.30, 10.25"""
    prices = [10.00, 10.20, 10.10, 10.30, 10.25]
    vols, ns, flags = calc_vol.calc_volatility(prices, windows=[4], annual_factor=252)
    vol = vols[4]
    expected = 0.2501  # 25.01%
    assert vol is not None, "vol should not be None"
    assert abs(vol - expected) < 0.0001, f"Expected {expected}, got {vol}"
    print(f"✓ §4.3 验收用例通过: vol={vol:.4f} ({vol*100:.2f}%)")


def test_insufficient_sample():
    """样本不足时返回 NULL"""
    prices = [10.0, 10.1]  # 只有1个收益率，20日窗口要求至少12个
    vols, ns, flags = calc_vol.calc_volatility(prices, windows=[20], min_obs_ratio=0.6)
    assert vols[20] is None
    assert "INSUFFICIENT_SAMPLE" in flags[20]
    print("✓ 样本不足测试通过")


def test_vol_max():
    """vol_max 取最大值"""
    vols = {20: 0.2, 40: 0.3, 90: None}
    assert calc_vol.calc_vol_max(vols) == 0.3
    print("✓ vol_max 测试通过")


def test_empty_prices():
    """空价格序列处理"""
    vols, ns, flags = calc_vol.calc_volatility([])
    assert all(v is None for v in vols.values())
    print("✓ 空价格序列测试通过")


if __name__ == "__main__":
    test_example_43()
    test_insufficient_sample()
    test_vol_max()
    test_empty_prices()
    print("\n所有测试通过 ✓")
