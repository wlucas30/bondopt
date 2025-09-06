import pandas as pd
import pytest
from bondopt.bond import Bond

def test_fixed_bond_with_string_date():
    bond = Bond(
        asset_type="fixed",
        coupon_rate=0.05,
        coupon_freq=2,
        maturity_date="2030-06-30",
        market_value=98000,
        notional=100000
    )
    cf = bond.cashflows(pd.Timestamp("2029-01-01"))
    assert not cf.empty
    assert cf.iloc[-1]["Cashflow"] == 102500  # final payment = coupon + notional

def test_fixed_bond_with_timestamp_date():
    bond = Bond(
        asset_type="fixed",
        coupon_rate=0.05,
        coupon_freq=1,
        maturity_date=pd.Timestamp("2030-06-30"),
        market_value=100000,
        notional=100000
    )
    cf = bond.cashflows(pd.Timestamp("2029-01-01"))
    assert cf.shape[0] == 2  # 2029 coupon + 2030 final coupon+notional
    assert cf.iloc[0]["Cashflow"] == 5000.0
    assert cf.iloc[1]["Cashflow"] == 105000.0

def test_zero_coupon_bond():
    bond = Bond(
        asset_type="zero",
        coupon_rate=None,
        coupon_freq=None,
        maturity_date="2030-06-30",
        market_value=75000,
        notional=100000
    )
    cf = bond.cashflows(pd.Timestamp("2029-01-01"))
    assert cf.shape[0] == 1
    assert cf.iloc[0]["Cashflow"] == 100000

def test_invalid_coupon_freq():
    with pytest.raises(ValueError):
        Bond(
            asset_type="fixed",
            coupon_rate=0.05,
            coupon_freq=13,  # invalid
            maturity_date="2030-06-30",
            market_value=98000,
            notional=100000
        )

def test_past_maturity_date():
    with pytest.raises(ValueError):
        Bond(
            asset_type="fixed",
            coupon_rate=0.05,
            coupon_freq=2,
            maturity_date="2000-01-01",
            market_value=98000,
            notional=100000
        )