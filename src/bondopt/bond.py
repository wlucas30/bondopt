"""
bond.py

Contains the Bond class and related functions for cashflow projection and basic bond valuation.

Classes:
    Bond - Represents a fixed-income instrument.
        
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
import numpy as np
import dateutil.relativedelta as rd
from scipy.optimize import brentq
from copy import deepcopy

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
            Date when the bond matures and notional is repaid. Use format "YYYY-MM-DD" for string input.
        issue_date (pd.Timestamp or str):
            Date when the bond is issued and the cashflow schedule begins.
            Use format "YYYY-MM-DD" for string input.
        market_value (float): 
            Current market price of the bond. Must be non-negative.
        notional (float): 
            Face value of the bond (principal). Must be non-negative.

    Methods:
        cashflows(valuation_date=None) -> pd.DataFrame
            Generates a schedule of future cashflows from the valuation date until maturity.
        expected_value(as_of=None, yield_curve=None) -> float
            Calculates the expected future value of a bond at a given date, optionally applying discounting using a yield curve.
        
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
    issue_date: pd.Timestamp | str
    market_value: float
    notional: float

    def __post_init__(self):
        # Perform data normalisation
        self.asset_type = self.asset_type.lower()
        if isinstance(self.maturity_date, str):
            self.maturity_date = pd.Timestamp(self.maturity_date)
        elif not isinstance(self.maturity_date, pd.Timestamp):
            raise TypeError("maturity_date must be a pandas.Timestamp object or a convertible string")
        
        if isinstance(self.issue_date, str):
            self.issue_date = pd.Timestamp(self.issue_date)
        elif not isinstance(self.issue_date, pd.Timestamp):
            raise TypeError("issue_date must be a pandas.Timestamp object or a convertible string")

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
        else:
            self.maturity_date = pd.Timestamp(self.maturity_date).normalize()
        if self.issue_date > self.maturity_date:
            raise ValueError("Issue date must be before maturity date")

    def cashflows(self, valuation_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Generates bond cashflow schedule.
        
        Args:
            valuation_date (pd.Timestamp, optional): The date from which the cashflow schedule should
                begin. Defaults to today if None.
        
        Returns:
            pd.Dataframe: A schedule of future cashflows from the valuation date until maturity.
        """
        if valuation_date is None:
            valuation_date = pd.Timestamp.today().normalize() # sets valuation_date to 00:00 on the current date
        else:
            valuation_date = pd.Timestamp(valuation_date).normalize()

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
                    cashflow_dates.append(working_date.normalize())
                    cashflow_amounts.append(amount)

                    # push the working date back by the previously calculated time between each payment
                    working_date -= rd.relativedelta(months=time_between_payments)
                
                return pd.DataFrame({"Date": cashflow_dates[::-1], "Cashflow": cashflow_amounts[::-1]}) # the payments are calculated in reverse order

            case "zero":
                # the notional value is paid at the maturity date
                return pd.DataFrame({"Date": [self.maturity_date], "Cashflow": [self.notional]})
    
    def _solve_z_spread(self, as_of: pd.Timestamp, yield_curve: pd.Series, spread_bounds: tuple = (-1.0, 1.0), tol: float = 2e-10, max_iter: int = 200):
        """
        (Protected method)
        Solves for the constant zero-volatility spread that makes PV(as_of; curve + spread) == market_value.

        Args:
            - yield_curve: pd.Series indexed by dates with annualised zero rates. This indicates the yield for a
                zero-coupon bond issued on as_of and expiring at the date provided in the index
            - spread_bounds: (low, high) bracket for search in decimals (e.g. -0.5 -> -50%, 1.0 -> 100%)
        Returns:
            spread (float) in decimals (e.g. 0.007 = 0.7%)
        """

        # Define objective: pv(spread) - market_value
        def objective(s):
            pv = self.expected_value(as_of=as_of, yield_curve=yield_curve, zspread=s)
            return pv - float(self.market_value)
        
        low, high = spread_bounds

        # Attempts to find the root of the objective (Solve for z-spread)
        try:
            spread = brentq(objective, low, high, xtol=tol, maxiter=max_iter)
        except:
            raise Exception('\033[31mCritical error: Zero-volatility spread could not be calculated!\033[0m')

        return float(spread)
    
    def expected_value(self, as_of: Optional[pd.Timestamp] = None, yield_curve: Optional[pd.Series] = None, zspread=0) -> float:
        """
        Calculates the expected future value of a bond at a given date, optionally applying discounting using a yield curve.

        Args:
            as_of (pd.Timestamp, optional): Target date for valuation. Defaults to today.
            yield_curve: (pd.Series, optional): Series providing annualised zero rates for given dates.
                Should be indexed by pd.Timestamp. If None, no discounting is applied.
        
        Returns:
            float: Expected market (present) value at the target date.
        """

        # Data validation
        if as_of is None:
            as_of = pd.Timestamp.today().normalize()
        else:
            as_of = pd.Timestamp(as_of).normalize()

        # Generate all future cashflows from issue date
        cashflows_dataframe = self.cashflows(self.issue_date)

        # Initialise future value at 0 to be incremented later on
        future_value = 0.0

        # Iterate through each cashflow payment
        for _, row in cashflows_dataframe.iterrows():
            cashflow_date, cashflow_amount = row["Date"].normalize(), row["Cashflow"]

            # Skip if the date is before the target date (cash already received)
            if cashflow_date < as_of:
                continue
                
            # Discounting is only applied when yield_curve != None
            if yield_curve is not None:
                # Linear interpolation to find rate between given rates in the yield curve
                curve_times = (yield_curve.index - as_of).days / 365.0
                target_time = (cashflow_date - as_of).days / 365.0
                spot_rate = float(np.interp(target_time, curve_times, yield_curve.values)) # spot rate at cashflow date

                # Calculate time in years to the cashflow date
                delta_years = (cashflow_date - as_of).days / 365.0

                # Discount cashflow to as_of
                cashflow_amount /= (1 + spot_rate + zspread) ** delta_years
            
            # Add cashflow to total
            future_value += cashflow_amount
         
        return future_value
    
    def _forward_rate(self, spot_short, spot_long, n_short, n_long):
        """
        (Protected method)
        Calculate the implied forward rate starting in n_short years and ending in n_long years.

        Args:
        spot_short: float
            Spot rate (as a decimal, e.g. 0.03 for 3%) for maturity n_short
        spot_long: float
            Spot rate (as a decimal, e.g. 0.04 for 4%) for maturity n_long
        n_short: float
            Shorter maturity (in years)
        n_long: float
            Longer maturity (in years)

        Returns:
        Forward rate (as a decimal) for the period between n_short and n_long
        """
        if n_short == n_long:
            return 0
        # (1+S_long)^(n_long) = (1+S_short)^(n_short) * (1+f)^(n_long - n_short)
        # (1+S_long)^(n_long) / (1+S_short)^(n_short) = (1+f)^(n_long - n_short)
        # ((1+S_long)^(n_long) / (1+S_short)^(n_short)) ^ (1/(n_long - n_short)) = 1 + f
        # ((1+S_long)^(n_long) / (1+S_short)^(n_short)) ^ (1/(n_long - n_short)) - 1 = f
        ratio = (1 + spot_long) ** n_long / (1 + spot_short) ** n_short
        fwd = ratio ** (1 / (n_long - n_short)) - 1
        return fwd
    
    def get_present_values_monthly(self, yield_curve: Optional[pd.Series] = None, yield_curve_dict: Optional[dict] = None) -> pd.DataFrame:
        """
        Calculates the expected future present value of a bond daily from a given date, optionally applying discounting using a yield curve.
        This method also calculates z-spread to ensure that the present value is equal to market value at t=0.

        Args:
            yield_curve (pd.Series, optional): A series providing annualised zero rates for investments maturing on
                given dates, beginning at issue date. If used, forward rates are calculated for investments beginning on
                future dates. If neither yield_curve_dict nor yield_curve is provided, no discounting is applied.
            yield_curve_dict: (dict, optional): A dictionary indexed by dates with pd.Series as values. The key provides the
                start date for each yield curve provided as a pd.Timestamp. The value provides annualised zero rates for given dates.
                Each series should be indexed by pd.Timestamp. A yield curve must be provided for each month beginning from issue.
                If neither yield_curve_dict nor yield_curve is provided, 
                no discounting is applied.
        
        Returns:
            pd.DataFrame: Expected market (present) value at regular monthly intervals.
        """
        # Generate z-spread if needed
        if yield_curve_dict is None:
            if yield_curve is not None:
                zspread = self._solve_z_spread(as_of=self.issue_date, yield_curve=yield_curve)
        else:
            try:
                initial_yield_curve = yield_curve_dict[self.issue_date]
                zspread = self._solve_z_spread(as_of=self.issue_date, yield_curve=initial_yield_curve)
            except KeyError as e:
                raise ValueError(f"The provided yield_curve_dict does not include sufficient data. Error: {e}")
        
        # Initialise new array to hold dates for checking present value
        dates = [self.issue_date]

        # Iterate until the working date is past the maturity date
        working_date = self.issue_date
        while working_date < self.maturity_date:
            working_date += rd.relativedelta(months=1)
            working_date = pd.Timestamp(working_date).normalize()
            dates.append(working_date)
        
        # Generate present value at every date
        present_values = []
        for date in dates:
            # Calculate forward rates if necessary
            temp_yield_curve = None
            if yield_curve_dict is not None:
                # The yield curves from the relevant date should be included
                try:
                    temp_yield_curve = yield_curve_dict[date]
                except KeyError as e:
                    raise ValueError(f"The provided yield_curve_dict does not include sufficient data for {date}. Error: {e}")
            else:
                # We need to calculate forward rates from the valuation date
                temp_yield_curve = pd.Series()

                for maturity_date, rate in yield_curve.items():
                    # Calculate time until maturity from issue
                    n_long = (maturity_date - self.issue_date).days / 365.0

                    # Calculate time from issue until the valuation date
                    n_short = (date - self.issue_date).days / 365.0

                    # Linear interpolation to find spot rate between given rates in the yield curve
                    curve_times = (yield_curve.index - self.issue_date).days / 365.0
                    spot_long = float(np.interp(n_long, curve_times, yield_curve.values)) # spot rate at maturity date
                    spot_short = float(np.interp(n_short, curve_times, yield_curve.values)) # spot rate at valuation date

                    # Calculate forward rate
                    if n_long != n_short:
                        rate = self._forward_rate(spot_short, spot_long, n_short, n_long)
                    else:
                        rate = spot_long

                    # Apply new forward rate to temp_yield_curve
                    temp_yield_curve[maturity_date] = rate

            present_value = self.expected_value(as_of=date, yield_curve=temp_yield_curve, zspread=zspread)
            present_values.append(round(present_value,2))
        
        return pd.DataFrame({"Date": dates, "Value": present_values})