"""
bond.py

Contains the Bond class and related functions for cashflow projection and basic bond valuation.

Classes:
    Bond - Represents a fixed-income instrument.

Functions:

Notes:
    - Currently supports fixed and zero-coupon bonds.
    - Cashflow projections use ACT/365 day count convention.
    - Present value calculations assume flat yield or input yield curves.

Example:

"""

from dataclasses import dataclass
import pandas as pd

@dataclass
class Bond:
    asset_type: str                     # "fixed" or "zero"
    coupon_rate: float                  # Annual rate e.g. 0.05
    coupon_freq: int                    # Payments per year
    maturity_date: pd.Timestamp
    market_value: float
    notional: float
