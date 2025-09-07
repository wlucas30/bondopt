import pandas as pd
from bondopt.bond import Bond

# Setup
as_of = pd.Timestamp("2025-09-07")
maturity = pd.Timestamp("2035-09-07")

bond = Bond(
    asset_type="fixed",
    coupon_rate=0.04,   # 4% annual coupon
    coupon_freq=2,      # semiannual
    maturity_date=maturity,
    market_value=99500.0,
    notional=100000.0
)

# Yield curve: 1y=3.1%, 2y=3.2%, …, 10y=4.0%
curve_dates = [as_of + pd.DateOffset(years=i) for i in range(1, 11)]
curve_rates = [0.031 + 0.001 * (i - 1) for i in range(1, 11)]  # 3.1%, 3.2%, ..., 4.0%
yc = pd.Series(curve_rates, index=curve_dates)

# Run valuation
pv = bond.get_present_values_daily(from_date=as_of, yield_curve=yc)
print(pv)
