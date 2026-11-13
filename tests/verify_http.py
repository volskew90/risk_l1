"""HTTP API 连通性验证：用少量股票验证四个接口，不打印任何敏感信息"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.fetch_ifind import IFindHttpClient

with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

client = IFindHttpClient(
    retry_times=int(config.get("retry_times", 3)),
    retry_backoff=float(config.get("retry_backoff", 2)),
    refresh_token=config.get("ifind_refresh_token") or None,
)
print("[1/4] 认证成功，access_token 已获取（%d 字符）" % len(client._access_token))

CODES = ["600519.SH", "000001.SZ", "300750.SZ"]
TRADE_DATE = "2026-09-04"

# 2. 历史行情
prices = client.get_daily_close(CODES, TRADE_DATE, lookback=130)
for c in CODES:
    p = prices.get(c, [])
    valid = [x for x in p if x == x]  # 非 NaN
    print(f"[2/4] {c}: {len(p)} 个价格点，{len(valid)} 个有效，最新收盘={valid[-1] if valid else 'N/A'}")

# 3. 基础数据
basic = client.get_basic_data(CODES, TRADE_DATE)
print(f"[3/4] basic_data: {len(basic)} 行, 列={list(basic.columns)}")
if not basic.empty:
    print(basic.to_string(index=False))

# 4. 指数成分（只验证上证50）
sse50 = client.get_index_components("000016.SH", TRADE_DATE)
print(f"[4/4] 上证50成分: {len(sse50)} 只, 前3只={sse50[:3]}")
