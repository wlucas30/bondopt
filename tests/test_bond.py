# SPDX-License-Identifier: MIT

import pandas as pd
import pytest
import dateutil.relativedelta as rd
from bondopt.bond import Bond
from bondopt.portfolio import Portfolio

def test_fixed_bond_with_string_date():
    bond = Bond(
        asset_type="fixed",
        coupon_rate=0.05,
        coupon_freq=2,
        maturity_date="2030-06-30",
        issue_date="2029-01-01",
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
        issue_date="2029-01-01",
        market_value=100000,
        notional=100000
    )
    cf = bond.cashflows(pd.Timestamp("2029-01-01"))
    assert cf.shape[0] == 2  # 2029 coupon + 2030 final coupon + notional
    assert cf.iloc[0]["Cashflow"] == 5000.0
    assert cf.iloc[1]["Cashflow"] == 105000.0

def test_zero_coupon_bond():
    bond = Bond(
        asset_type="zero",
        coupon_rate=None,
        coupon_freq=None,
        maturity_date="2030-06-30",
        issue_date="2029-01-01",
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
            issue_date="2029-01-01",
            market_value=98000,
            notional=100000
        )

def test_zero_coupon_no_yield_curve():
    b = Bond(
        asset_type="zero",
        coupon_rate=None,
        coupon_freq=None,
        maturity_date=pd.Timestamp("2030-01-01"),
        issue_date="2029-01-01",
        market_value=95000,
        notional=100000
    )
    as_of = pd.Timestamp("2025-01-01")
    ev = b.expected_value(as_of=as_of, yield_curve=None)
    assert ev == 100000  # no discounting, just the notional

def test_fixed_coupon_no_yield_curve():
    b = Bond(
        asset_type="fixed",
        coupon_rate=0.05,
        coupon_freq=2,
        maturity_date=pd.Timestamp("2026-01-01"),
        issue_date="2025-01-01",
        market_value=100000,
        notional=100000
    )
    as_of = pd.Timestamp("2025-01-01")
    ev = b.expected_value(as_of=as_of, yield_curve=None)
    # Expected cashflows: two coupons left + notional
    cf_df = b.cashflows(as_of)
    expected_sum = cf_df[cf_df["Date"] >= as_of]["Cashflow"].sum()
    assert ev == expected_sum

def test_zero_coupon_with_yield_curve():
    """Zero-coupon bond with discounting using a simple yield curve."""
    b = Bond(
        asset_type="zero",
        coupon_rate=None,
        coupon_freq=None,
        maturity_date=pd.Timestamp("2030-01-01"),
        issue_date="2025-01-01",
        market_value=95000,
        notional=100000
    )
    as_of = pd.Timestamp("2025-01-01")
    # Simple yield curve: 5% annual flat for all future dates
    future_dates = pd.date_range(start=as_of, end=b.maturity_date, freq="YE")
    yc = pd.Series([0.05] * len(future_dates), index=future_dates)
    ev = b.expected_value(as_of=as_of, yield_curve=yc)

    t = (b.maturity_date - as_of).days / 365
    expected = 100000 / (1 + 0.05) ** t
    assert abs(ev - expected) < 1.0  # small tolerance

def test_fixed_coupon_with_yield_curve():
    b = Bond(
        asset_type="fixed",
        coupon_rate=0.06,
        coupon_freq=2, 
        maturity_date=pd.Timestamp("2026-01-01"),
        issue_date="2025-01-01",
        market_value=100000,
        notional=100000
    )
    as_of = pd.Timestamp("2025-01-01").normalize()
    # Yield curve: 5% flat
    cf_df = b.cashflows(valuation_date=as_of)
    future_dates = cf_df[cf_df["Date"] >= as_of]["Date"].dt.normalize()
    yc = pd.Series([0.05] * len(future_dates), index=future_dates)
    ev = b.expected_value(as_of=as_of, yield_curve=yc)

    # Manual calculation for first coupon
    first_cf = cf_df[cf_df["Date"] >= as_of].iloc[0]
    t = (first_cf["Date"] - as_of).days / 365
    expected_first = first_cf["Cashflow"] / (1 + 0.05) ** t

    assert ev > expected_first  # total value includes remaining cashflows

def test_get_present_values_monthly_zero_coupon():
    # Simple zero-coupon bond
    maturity = pd.Timestamp.today().normalize() + pd.Timedelta(days=10)
    bond = Bond(
        asset_type="zero",
        coupon_rate=None,
        coupon_freq=None,
        maturity_date=maturity,
        issue_date=pd.Timestamp.today().normalize(),
        market_value=990,
        notional=1000,
    )

    # Flat yield curve at 5%
    dates = pd.date_range(start=pd.Timestamp.today().normalize(),
                          periods=11, freq="D")
    yield_curve = pd.Series(0.05, index=[dates[-1]])  # single point

    df = bond.get_present_values_monthly(yield_curve=yield_curve)

    # 1. DataFrame structure
    assert list(df.columns) == ["Date", "Value"]

    # 2. Dates run daily until maturity
    assert df["Date"].iloc[0] == dates[0]
    assert df["Date"].iloc[-1] >= maturity
    assert len(df) == 2

    # 3. Values should all be non-negative
    assert (df["Value"] >= 0).all()

    # 4. Present value on the first day should match market_value (after z-spread)
    assert pytest.approx(df["Value"].iloc[0], rel=1e-2) == bond.market_value

def test_get_present_values_monthly_with_multiple_yield_curves():
    # Simple zero-coupon bond
    maturity = pd.Timestamp.today().normalize() + rd.relativedelta(months=12)
    bond = Bond(
        asset_type="zero",
        coupon_rate=None,
        coupon_freq=None,
        maturity_date=maturity,
        issue_date=pd.Timestamp.today().normalize(),
        market_value=900,
        notional=1000,
    )

    dates = []
    working_date = pd.Timestamp.today().normalize()
    while working_date <= bond.maturity_date:
        dates.append(working_date)
        working_date += rd.relativedelta(months=1)
    yield_curve = pd.Series(0.05, index=[dates[-1]])  # single point

    # Assign multiple yield curves
    yield_curves = {}
    for date in dates:
        yield_curves[date] = yield_curve

    df = bond.get_present_values_monthly(yield_curve_dict=yield_curves)
    
    # 1. DataFrame structure
    assert list(df.columns) == ["Date", "Value"]

    # 2. Dates run daily until maturity
    assert df["Date"].iloc[0] == dates[0]
    assert df["Date"].iloc[-1] >= maturity
    assert len(df) == 13

    # 3. Values should all be non-negative
    assert (df["Value"] >= 0).all()

    # 4. Present value on the first day should match market_value (after z-spread)
    assert pytest.approx(df["Value"].iloc[0], rel=1e-2) == bond.market_value

    # 5. Value should increase up to notional
    assert df["Value"].is_monotonic_increasing

def test_bond_with_default_risk():
    # Simple zero-coupon bond
    maturity = pd.Timestamp.today().normalize() + rd.relativedelta(years=3)
    bond = Bond(
        asset_type="zero",
        coupon_rate=None,
        coupon_freq=None,
        maturity_date=maturity,
        issue_date=pd.Timestamp.today().normalize(),
        market_value=900,
        notional=1000,
        default_risk_curve=pd.Series([0.03, 0.04, 0.05], index=[1,2,3])
    )

    # Integer time
    survival_rate = bond.get_total_survival_rate(after_years=3)
    assert isinstance(survival_rate, float)
    assert abs(survival_rate) - 0.857375 <= 1e-5

    # Float time   
    survival_rate = bond.get_total_survival_rate(after_years=2.5)
    assert isinstance(survival_rate, float)
    assert 0.857375 < survival_rate < 0.9216

def test_one_year_bond_with_default_risk():
    # Simple zero-coupon bond
    maturity = pd.Timestamp.today().normalize() + rd.relativedelta(years=1)
    bond = Bond(
        asset_type="zero",
        coupon_rate=None,
        coupon_freq=None,
        maturity_date=maturity,
        issue_date=pd.Timestamp.today().normalize(),
        market_value=900,
        notional=1000,
        default_risk_curve=pd.Series([0.03], index=[1])
    )

    # Integer time
    survival_rate = bond.get_total_survival_rate(after_years=1)
    assert isinstance(survival_rate, float)
    assert survival_rate == 0.97

    # Float time   
    survival_rate = bond.get_total_survival_rate(after_years=0.5)
    assert isinstance(survival_rate, float)
    assert 0.97 < survival_rate < 1

def test_bond_with_default_risk_projected_notional_values():
    # Simple zero-coupon bond
    maturity = pd.Timestamp.today().normalize() + rd.relativedelta(years=3)
    bond = Bond(
        asset_type="zero",
        coupon_rate=None,
        coupon_freq=None,
        maturity_date=maturity,
        issue_date=pd.Timestamp.today().normalize(),
        market_value=900,
        notional=1000,
        default_risk_curve=pd.Series([0.03, 0.04, 0.05], index=[1,2,3])
    )

    nv = bond.get_projected_notional_values()

    assert len(nv) == 37
    assert nv[bond.issue_date] == bond.notional
    assert nv[bond.issue_date + rd.relativedelta(years=1)] == bond.notional * 0.97


def test_projected_notional_values():
    b = Bond(
        asset_type="zero",
        coupon_rate=None,
        coupon_freq=None,
        maturity_date=pd.Timestamp("2030-01-01"),
        default_risk_curve=pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=[1,2,3,4,5]),
        issue_date=pd.Timestamp("2025-01-01"),
        market_value=95000,
        notional=100000
    )

    as_of = pd.Timestamp("2027-01-01")
    # Simple yield curve: 5% annual flat for all future dates
    future_dates = pd.date_range(start=as_of, end=b.maturity_date, freq="YE")
    yc = pd.Series([0.05] * len(future_dates), index=future_dates)

    ev = b.expected_value(as_of=b.issue_date, yield_curve=yc, use_default_risk=False)
    ev1 = b.expected_value(as_of=as_of, yield_curve=yc, use_default_risk=True)

    pvm = b.get_present_values_monthly(yield_curve=yc)
    pvm1 = b.get_present_values_monthly(yield_curve=yc, use_default_risk=True)

    assert ev != ev1
    assert not pvm.equals(pvm1)

def test_list_bonds_returns_dataframe():
    # Create a Portfolio
    portfolio = Portfolio()

    # Create two sample Bond objects
    bond1 = Bond(
        cusip="123456AA1",
        asset_type="fixed",
        coupon_rate=0.05,
        coupon_freq=2,
        issue_date=pd.Timestamp("2020-01-01"),
        maturity_date=pd.Timestamp("2030-01-01"),
        notional=1000,
        market_value=980,
    )

    bond2 = Bond(
        cusip="789012BB2",
        asset_type="zero",
        coupon_rate=None,
        coupon_freq=None,
        issue_date=pd.Timestamp("2021-06-30"),
        maturity_date=pd.Timestamp("2026-06-30"),
        notional=500,
        market_value=450,
    )

    # Add bonds to portfolio
    portfolio.add_bond(bond1)
    portfolio.add_bond(bond2)

    # Call list_bonds
    df = portfolio.list_bonds()

    # Call cashflows
    cf = portfolio.cashflows("2020-01-01")

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["CUSIP", "Notional", "Maturity Date"]
    assert len(df) == 2
    assert set(df["CUSIP"]) == {"123456AA1", "789012BB2"}
    assert df.loc[df["CUSIP"] == "123456AA1", "Notional"].iloc[0] == 1000
    assert df.loc[df["CUSIP"] == "789012BB2", "Maturity Date"].iloc[0] == pd.Timestamp("2026-06-30")
    assert len(cf) == 21


def test_list_bonds_returns_dataframe_with_aggregate_cashflows():
    # Create a Portfolio
    portfolio = Portfolio()

    # Create two sample Bond objects
    bond1 = Bond(
        cusip="123456AA1",
        asset_type="fixed",
        coupon_rate=0.05,
        coupon_freq=2,
        issue_date=pd.Timestamp("2020-01-01"),
        maturity_date=pd.Timestamp("2030-01-01"),
        notional=1000,
        market_value=980,
    )

    bond2 = Bond(
        cusip="789012BB2",
        asset_type="zero",
        coupon_rate=None,
        coupon_freq=None,
        issue_date=pd.Timestamp("2020-01-31"),
        maturity_date=pd.Timestamp("2030-01-01"),
        notional=500,
        market_value=450,
    )

    # Add bonds to portfolio
    portfolio.add_bond(bond1)
    portfolio.add_bond(bond2)

    # Call cashflows
    cf = portfolio.cashflows("2020-01-01", aggregate_cashflows=True)

    # Assertions
    assert len(cf) == 20

def test_get_present_values_monthly_with_multiple_assets():
    # Create a couple of zero-coupon bonds
    today = pd.Timestamp.today().normalize()
    maturity1 = today + pd.Timedelta(days=365)  # 1 year
    maturity2 = today + pd.Timedelta(days=730)  # 2 years

    bond1 = Bond(
        asset_type="zero",
        coupon_rate=None,
        coupon_freq=None,
        issue_date=today,
        maturity_date=maturity1,
        market_value=950,
        notional=1000,
        cusip="BOND1"
    )
    bond2 = Bond(
        asset_type="zero",
        coupon_rate=None,
        coupon_freq=None,
        issue_date=today,
        maturity_date=maturity2,
        market_value=900,
        notional=1000,
        cusip="BOND2"
    )

    portfolio = Portfolio()
    portfolio.add_bond(bond1)
    portfolio.add_bond(bond2)

    # Flat yield curve (1% annual)
    yield_curve = pd.Series(
        [0.01, 0.01],
        index=[maturity1, maturity2]
    )

    df = portfolio.get_present_values_monthly(from_date=today, yield_curve=yield_curve)

    # Basic checks on structure
    assert isinstance(df, pd.DataFrame)
    assert "BOND1" in df.columns
    assert "BOND2" in df.columns
    assert "Total" in df.columns

    # Dates should be monthly starting from today
    assert df.index.is_monotonic_increasing
    assert df.index[0] == today