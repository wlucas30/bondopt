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
import uuid

@dataclass
class Bond:
    """
    Represents a fixed-income instrument.

    Attributes:
        cusip (str, optional):
            A bond CUSIP is a unique, nine-character alphanumeric code that serves as a serial number for a specific security.
            If none provided, a random UUID is stored instead.
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
        default_risk_curve (pd.Series, optional):
            Term structure of default risk for this bond.
            Indexed by integer years representing the horizon.
            Values are annualised cumulative probability of default (float) (e.g. 0.0028 = 0.28%).
            Each value is the cumulative annualised probability of default by the end of each horizon.
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
        get_present_values_monthly(yield_curve=None, yield_curve_dict=None, use_default_risk=False) -> pd.DataFrame
            Calculates the expected future present value of a bond monthly.
        get_total_survival_rate(after_years) -> float
            Calculates the expected total survival rate for a bond using the provided default risk curve.
        get_projected_notional_values(interval=rd.relativedelta(months=1)) -> pd.Series
            Generates a pd.Series object representing the expected total notional value after regular time intervals between
            issue and maturity, according to the provided default risk curve.
        summary(verbose=False) -> pd.DataFrame
            Returns a summary of the bond's stored attributes and derived details.
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
    default_risk_curve: Optional[pd.Series] = None
    cusip: Optional[str] = None

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
        if self.default_risk_curve is not None:
            if (len(self.default_risk_curve) < (self.maturity_date - self.issue_date).days // 365):
                raise ValueError("Not enough default risk data provided")
        
        # Store a unique identifier for each bond, if one is not provided
        if self.cusip is None:
            self.cusip = str(uuid.uuid4())

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
    
    def expected_value(self, as_of: Optional[pd.Timestamp] = None, yield_curve: Optional[pd.Series] = None, zspread=0, use_default_risk: Optional[bool]=False) -> float:
        """
        Calculates the expected future value of a bond at a given date, optionally applying discounting using a yield curve.
        Warning: This function does not apply z-spread automatically! Spread must be provided!

        Args:
            as_of (pd.Timestamp, optional): Target date for valuation. Defaults to today.
            yield_curve (pd.Series, optional): Series providing annualised zero rates for given dates.
                Should be indexed by pd.Timestamp. If None, no discounting is applied.
            use_default_risk (bool, optional): Indicates whether expected values should be multiplied by expected survival rate
        
        Returns:
            float: Expected market (present) value at the target date.
        """

        # Data validation
        if as_of is None:
            as_of = pd.Timestamp.today().normalize()
        else:
            as_of = pd.Timestamp(as_of).normalize()
        if use_default_risk and self.default_risk_curve is None:
            raise Exception("No default risk curve provided!")

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

        if use_default_risk:
            after_years = round((as_of - self.issue_date).days / 365.0,2)
            future_value *= self.get_total_survival_rate(after_years)

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
    
    def get_present_values_monthly(self, yield_curve: Optional[pd.Series] = None, yield_curve_dict: Optional[dict] = None, use_default_risk: Optional[bool] = False) -> pd.DataFrame:
        """
        Calculates the expected future present value of a bond monthly, optionally applying discounting using a yield curve.
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
            use_default_risk (bool, optional): Indicates whether expected values should be multiplied by expected survival rate
        
        Returns:
            pd.DataFrame: Expected market (present) value at regular monthly intervals.
        """
        # Generate z-spread if needed
        if yield_curve_dict is None:
            if yield_curve is not None:
                zspread = self._solve_z_spread(as_of=self.issue_date, yield_curve=yield_curve)
            else:
                zspread = 0.0
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
                temp_yield_curve = pd.Series(dtype=float)

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

            present_value = self.expected_value(as_of=date, yield_curve=temp_yield_curve, zspread=zspread, use_default_risk=use_default_risk)
            present_values.append(round(present_value,2))
        
        return pd.DataFrame({"Date": dates, "Value": present_values})
    
    def get_total_survival_rate(self, after_years: float) -> float:  
        """
        Calculates the expected total survival rate for a bond using the provided default risk curve

        Args:
            after_years (float): The number of years after issue for the calculation
        
        Returns:
            Expected survival rate after the given number of years (float) e.g. 0.95 = 95%
        """
        if self.default_risk_curve is None:
            return 1.0
        
        # Get discrete survival rate values
        default_rates = deepcopy(self.default_risk_curve)
        default_rates.sort_index()
        times = default_rates.index.to_numpy(dtype=float)

        # If after_years is within the known range
        if times[0] < after_years < times[-1]:
            # Find index of last time <= after_years
            index_high = np.searchsorted(times, after_years, side='right')
            index_low = index_high-1

            t_low = float(times[index_low])
            D_low = float(default_rates.iloc[index_low])
            t_high = float(times[index_high])
            D_high = float(default_rates.iloc[index_high])

            delta_years = after_years - t_low

            # We have the cumulative rate, and need to convert to marginal
            # The survival rate after after_years will be (Cumulative rate up to n-1) ^ n-1 * (Marginal rate for year n)^delta_years

            # Calculate marginal rate for last time period
            marginal_survival_rate = ((1-D_high) ** t_high) / ((1-D_low) ** t_low)

            return ((1-D_low) ** t_low) * (marginal_survival_rate ** delta_years)

        # If after_years is before the first known time
        elif after_years <= times[0]:
            default_rate_annualised = float(default_rates.iloc[0])
            return (1-default_rate_annualised) ** after_years

        # If after_years is equal to the last known time
        elif after_years == times[-1]:
            return (1 - float(default_rates.iloc[-1])) ** times[-1]

        # If after_years is after the last known time
        elif after_years > times[-1]:
            raise ValueError(f"The value of {after_years} for after_years is outside the range specified by the default risk curve provided.")
                
    def get_projected_notional_values(self, interval: Optional[rd.relativedelta]=rd.relativedelta(months=1)) -> pd.Series:
        """
        Generates a pd.Series object representing the expected total notional value after regular time intervals between
        issue and maturity, according to the provided default risk curve.
        Warning: Will raise an exception if no default risk curve is provided when creating Bond object.

        Args:
            interval (dateutil.relativedelta.relativedelta, optional): The time interval between values to be generated.
                Defaults to 1 month if none provided.
        
        Returns:
            pd.Series object representing the expected total notional value after regular time intervals.
        """

        if self.default_risk_curve is None:
            raise Exception("No default risk curve provided!")

        # Generate a list of dates for which projected notional value should be calculated
        dates = []

        working_date = self.issue_date
        while working_date < self.maturity_date:
            if working_date not in dates:
                dates.append(working_date)

            working_date += interval
        
        if self.maturity_date not in dates:
            dates.append(self.maturity_date)
        
        # Calculated projected notional value for each required date
        projected_notional_values = []
        for date in dates:
            after_years = round((date - self.issue_date).days / 365,2)
            projected_notional_values.append(self.notional * self.get_total_survival_rate(after_years))

        return pd.Series(projected_notional_values, index=dates)
    
    def summary(self, verbose: Optional[bool] = False) -> pd.DataFrame:
        """
        Returns a summary of the bond's stored attributes and derived details.

        Args:
            verbose (str, Optional): Indicates whether a full summary should be provided.

        Returns:
            pd.DataFrame: Tabular summary of the bond's key attributes.
        """

        if verbose:
            data = {
                "CUSIP": self.cusip,
                "Asset Type": self.asset_type,
                "Coupon Rate": f"{self.coupon_rate:.4f}" if self.coupon_rate is not None else None,
                "Coupon Freq": self.coupon_freq,
                "Issue Date": self.issue_date.strftime("%Y-%m-%d"),
                "Maturity Date": self.maturity_date.strftime("%Y-%m-%d"),
                "Notional": self.notional,
                "Market Value": self.market_value,
                "Has Default Risk Curve": self.default_risk_curve is not None,
            }
            return pd.DataFrame(list(data.items()), columns=["CUSIP", "Notional", "Maturity Date"])
        else:
            data = {"CUSIP": self.cusip, "Notional": self.notional, "Maturity Date": self.maturity_date}
            return pd.DataFrame([data], columns=["CUSIP", "Notional", "Maturity Date"])