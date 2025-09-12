"""
reinvest.py

Contains the ReinvestmentStrategy class and related functions for managing
cash reinvestment in a bond portfolio.

Classes:
    ReinvestmentStrategy - Represents a reinvestment policy that allocates
        available cash into new bonds according to predefined rules.

Notes:
    - Supports allocation by credit rating, percentage weights, and spread
      adjustments relative to a base yield curve.
    - Designed to be flexible: reinvestment rules can be provided as a table
      (e.g. CSV, DataFrame) or defined programmatically.
    - New bonds generated through reinvestment are added back to the portfolio,
      producing future cashflows that can themselves be reinvested.

Example:
    >>> import pandas as pd
    >>> from bondopt.reinvest import ReinvestmentStrategy
    >>> strategy_table = pd.DataFrame({
    ...     "Asset Rating": ["BBB", "B"],
    ...     "Allocation %": [0.8, 0.2],
    ...     "Spread Adjustment (bps)": [150, 400],
    ... })
    >>> strategy = ReinvestmentStrategy(strategy_table)
    >>> new_bonds = strategy.apply(cash_available=1_000_000,
    ...                            base_curve=yield_curve,
    ...                            today=pd.Timestamp.today())
"""

# SPDX-License-Identifier: MIT

from bondopt.bond import Bond
from dataclasses import dataclass
import pandas as pd
from typing import Optional
import numpy as np
from scipy.optimize import brentq
import dateutil.relativedelta as rd

@dataclass
class ReinvestmentStrategy:
    """
    Represents a reinvestment strategy for allocating available cash into new bonds.

    The strategy is defined by a set of allocation rules, provided as a Pandas DataFrame.
    Each rule specifies:
        - Asset Rating (e.g. "BBB", "B")
        - Allocation % (fraction of available cash to allocate)
        - Spread Adjustment (in basis points relative to a base yield curve)

    When applied, the strategy generates new bond positions by splitting the
    available cash according to the allocation percentages and applying the
    corresponding credit spreads. The newly created bonds can then be added
    back into the portfolio and will generate future cashflows.

    Args:
        allocation_table (pd.DataFrame): Table containing the reinvestment rules.
            Must have columns ["Asset Rating", "Allocation %", "Spread Adjustment (bps)"].

    Methods:
        apply(cash_available, base_curve, today):
            Splits available cash into new bond positions according to the
            strategy table and returns a list of new Bond objects.
    """

    table: pd.DataFrame
    def __post_init__(self):
        return
    
    def apply(self, cash_available, today, yield_curve: Optional[pd.Series] = None):
        """
        Allocate available cash into new bond positions according to the
        reinvestment strategy table.

        For each row in the strategy table, a proportion of the available
        cash is allocated to a new bond. The coupon rate is determined as
        the given risk-free rate plus the specified spread adjustment. The
        maturity and coupon frequency are set based on table inputs.

        Args:
            cash_available (float): Amount of cash available for reinvestment.
            today (pd.Timestamp): The reinvestment date used to set bond maturities.
            yield_curve (pd.Series, optional): Yield curve beginning from today.

        Returns:
            list[Bond]: A list of new Bond objects created according to the
            reinvestment allocation rules.
        """

        new_assets = []
        for _, row in self.table.iterrows():
            alloc = row["Allocation %"]
            rating = row["Asset Rating"]
            spread = row["Spread Adjustment (bps)"] / 10000
            coupon_freq = row["Coupon Frequency (annual)"]
            maturity_offset = pd.DateOffset(years=row["Maturity Offset (years)"])
            notional = round(cash_available * alloc, 2)
            par_coupon = self.__solve_for_par_coupon(yield_curve=yield_curve, spread=spread, maturity=(today + maturity_offset), coupon_freq=coupon_freq, today=today)

            if notional > 0:
                bond = Bond(
                    cusip=rating,
                    asset_type="fixed",
                    issue_date=today,
                    coupon_rate=par_coupon,
                    coupon_freq=coupon_freq,
                    maturity_date=today + maturity_offset,
                    notional=notional,
                    market_value=notional,
                    ignore_spread=True # z-spread is accounted for by adding spread in this function
                )
                new_assets.append(bond)
                print("Created 1 new bond.")
                print(bond.summary(verbose=True))
        return new_assets
    
    def __solve_for_par_coupon(self, yield_curve, spread, maturity, coupon_freq, today):
        """
        Solve for the par coupon rate such that the bond issues at par (market value = notional).
        """

        issue_date = today

        # Build coupon schedule
        coupon_dates = []
        working_date = today
        while working_date <= maturity:
            coupon_dates.append(working_date)
            working_date += rd.relativedelta(months=12//coupon_freq)

        # Ensure maturity is always included
        if maturity not in coupon_dates:
            coupon_dates = coupon_dates.append(maturity)

        # Time in years (fractional)
        times = np.array([(d - issue_date).days / 365.0 for d in coupon_dates])

        # Interpolate zero rates
        curve_times = (yield_curve.index - issue_date).days / 365.0
        base_rates = np.interp(times, curve_times, yield_curve.values)
        rates = base_rates + spread  # apply z-spread

        def discount_factor(r, t):
            """Discrete compounding discount factor with coupon_freq compounding."""
            return 1.0 / ((1 + r / coupon_freq) ** (coupon_freq * t))

        dfs = np.array([discount_factor(r, t) for r, t in zip(rates, times)])

        def pv_diff(coupon_rate: float) -> float:
            """PV difference between cashflows and notional."""
            cpn = coupon_rate / coupon_freq
            pv_coupons = np.sum(cpn * dfs)
            pv_principal = 1.0 * dfs[-1]  # assume notional = 1
            return pv_coupons + pv_principal - 1.0

        # Root finding
        final_coupon_rate = brentq(pv_diff, -0.1, 1.0)
        return final_coupon_rate