"""
bond.py

Contains the Bond class and related functions for cashflow projection and basic bond valuation.

Classes:
    Bond - Represents a fixed-income instrument.

Functions:
    Bond.cashflows(valuation_date=None) -> pd.DataFrame
        Generates a schedule of future cashflows from the valuation date until maturity.
        
Notes:
    - Currently supports fixed and zero-coupon bonds.
    - This assumes coupon_freq divides 12 evenly (e.g. 1, 2, 3, 4, 6, 12).

Example:
    >>> from bondopt.bond import Bond
    >>> b = Bond(asset_type="fixed", coupon_rate=0.05, coupon_freq=2,
                 maturity_date="2030-06-30", market_value=98000, notional=100000)
    >>> b.cashflows()

            Date    Cashflow
    0 2029-12-30      2500.0
    1 2030-06-30    102500.0
"""

# SPDX-License-Identifier: MIT


from dataclasses import dataclass
from typing import Optional
import pandas as pd
import dateutil.relativedelta as rd

@dataclass
class Bond:
    """
    Represents a fixed-income instrument.

    Attributes:
        asset_type (str): 
            Type of bond. Supported values are "fixed" (coupon-bearing) 
            and "zero" (zero-coupon bond).
        coupon_rate (float, optional): 
            Annual coupon rate (e.g., 0.05 for 5%). Must be non-negative. 
            Not used for zero-coupon bonds.
        coupon_freq (int, optional): 
            Number of coupon payments per year. Typical values are:
              - 1 = annual
              - 2 = semiannual
              - 4 = quarterly
              - 12 = monthly
            Other divisors of 12 (e.g., 3, 6) are supported. Ignored for zero-coupon bonds.
        maturity_date (pd.Timestamp or str): 
            Date when the bond matures and notional is repaid. Must not 
            be in the past. Use format "YYYY-MM-DD" for string input.
        market_value (float): 
            Current market price of the bond. Must be non-negative.
        notional (float): 
            Face value of the bond (principal). Must be non-negative.

    Methods:
        cashflows(valuation_date=None) -> pd.DataFrame
            Generates a schedule of future cashflows from the valuation date until maturity.
        
    Notes:
        - Currently supports fixed and zero-coupon bonds.
        - This assumes coupon_freq divides 12 evenly (e.g. 1, 2, 3, 4, 6, 12).

    Example:
        >>> from bondopt.bond import Bond
        >>> b = Bond(asset_type="fixed", coupon_rate=0.05, coupon_freq=2,
                    maturity_date="2030-06-30", market_value=98000, notional=100000)
        >>> b.cashflows()

                Date    Cashflow
        0 2029-12-30      2500.0
        1 2030-06-30    102500.0
    
    """
    asset_type: str                     # "fixed" or "zero"
    coupon_rate: Optional[float]        # Annual rate e.g. 0.05
    coupon_freq: Optional[int]          # Payments per year (0 to 12)
    maturity_date: pd.Timestamp | str
    market_value: float
    notional: float

    def __post_init__(self):
        # Perform data normalisation
        self.asset_type = self.asset_type.lower()
        if isinstance(self.maturity_date, str):
            self.maturity_date = pd.Timestamp(self.maturity_date)
        elif not isinstance(self.maturity_date, pd.Timestamp):
            raise TypeError("maturity_date must be a pandas.Timestamp object or a convertible string")

        # Perform data validation
        if self.asset_type not in ["fixed", "zero"]:
            raise ValueError(f"Unsupported asset_type: {self.asset_type}")
        if self.asset_type == "fixed" and self.coupon_rate is None:
            raise ValueError("Missing required argument coupon_rate")
        if self.asset_type == "fixed" and self.coupon_freq is None:
            raise ValueError("Missing required argument coupon_freq")
        if self.asset_type == "zero":
            self.coupon_rate, self.coupon_freq = 0.0, 0
        if self.asset_type == "fixed" and (not (0 <= self.coupon_freq <= 12)):
            raise ValueError("coupon_freq must be between 0 and 12")
        if self.asset_type == "fixed" and (12 % self.coupon_freq != 0):
            raise ValueError("coupon_freq must divide 12 evenly (e.g. 1, 2, 3, 4, 6, 12)")
        if self.asset_type == "fixed" and (self.coupon_rate < 0):
            raise ValueError("Expected non-negative value for coupon_rate")
        if self.market_value < 0:
            raise ValueError("Expected non-negative value for market_value")
        if self.notional < 0:
            raise ValueError("Expected non-negative value for notional")
        if self.maturity_date < pd.Timestamp.today().normalize():
            raise ValueError(f"maturity_date {self.maturity_date.date()} is in the past")

    def cashflows(self, valuation_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Generates bond cashflow schedule."""
        if valuation_date is None:
            valuation_date = pd.Timestamp.today().normalize() # sets valuation_date to 00:00 on the current date

        match self.asset_type:
            case "fixed":
                time_between_payments = 12 // self.coupon_freq # time in months between each payment
                cashflow_dates = [] # stores the dates payments will be received
                cashflow_amounts = [] # stores the amount received in each payment
                working_date = self.maturity_date # temporary variable used for calculations

                while working_date > valuation_date: # iterate until all payments added
                    coupon = self.notional * (self.coupon_rate / self.coupon_freq)
                    amount = coupon + (self.notional * (working_date == self.maturity_date)) # adds notional value if last payment

                    # store the payment
                    cashflow_dates.append(working_date)
                    cashflow_amounts.append(amount)

                    # push the working date back by the previously calculated time between each payment
                    working_date -= rd.relativedelta(months=time_between_payments)
                
                return pd.DataFrame({"Date": cashflow_dates[::-1], "Cashflow": cashflow_amounts[::-1]}) # the payments are calculated in reverse order

            case "zero":
                # the notional value is paid at the maturity date
                return pd.DataFrame({"Date": [self.maturity_date], "Cashflow": [self.notional]})