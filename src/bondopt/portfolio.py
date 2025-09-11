"""
portfolio.py

Contains the Portfolio class and related functions.

Classes:
    Portfolio - Represents a collection of Bond objects.
        
Notes:
    - Bonds must be added to the Portfolio object after creation using bondopt.Bond

Example:
    >>> from bondopt.bond import Portfolio
    >>> p = Portfolio()
    >>> p.add_bond(Bond(...))
    >>> p.cashflows()

            Date    Cashflow
    0 2029-12-30      2500.0
    1 2030-06-30    102500.0
"""

# SPDX-License-Identifier: MIT

from bondopt.bond import Bond
from dataclasses import dataclass
from typing import Optional
import dateutil.relativedelta as rd
import pandas as pd
import numpy as np

@dataclass
class Portfolio:
    """
    Represents a collection of Bond objects.

    Methods:
        add_bond(bond) -> None
            Adds a provided Bond to the Portfolio.
        remove_bond() -> None
            Removes all instances of bonds from the Portfolio with CUSIP matching the provided value.
        list_bonds() -> pd.DataFrame
            Lists all Bond stored in the Portfolio.
        cashflows(valuation_date=None) -> pd.DataFrame
            Generates a schedule of future cashflows for all assets in the Portfolio.
        expected_value() -> 
        get_present_values_monthly(from_date=None, yield_curve=None, yield_curve_dict=None, use_default_risk=False) -> pd.DataFrame:
            Calculates the expected future present value of assets monthly from a given date, or today if none provided.
    """

    def __post_init__(self):
        # Initialise array to store the assets which constitute the portfolio
        self.__assets = []

    def add_bond(self, bond) -> None:
        """
        Adds a provided Bond to the Portfolio.

        Args:
            bond (bondopt.bond.Bond): The Bond to be added to the Portfolio.
        """
        if isinstance(bond, Bond):
            self.__assets.append(bond)
        else:
            raise TypeError("The bond provided is formatted incorrectly. It must be a bondopt.bond.Bond object")
    
    def remove_bond(self, cusip) -> None:
        """
        Removes all instances of bonds from the Portfolio with CUSIP matching the provided value.

        Args:
            cusip (str): The CUSIP value of the bond(s) to be removed.
        """
        found = False
        for bond in self.__assets:
            if bond.cusip == cusip:
                self.__assets.remove(bond)
                found = True
        
        if not found:
            raise ValueError("No Bond with the provided CUSIP value was found.")
        
    def list_bonds(self) -> pd.DataFrame:
        """
        Lists all Bond stored in the Portfolio.
        
        Returns:
            pd.DataFrame: Table displaying information about all bonds held.
        """

        cols = ["CUSIP", "Notional", "Maturity Date"] # Data to be displayed for each bond
        bonds = pd.DataFrame(columns=cols)

        for bond in self.__assets:
            summary = bond.summary(verbose=False)
            bonds = pd.concat([bonds, summary], ignore_index=True)
        
        return bonds
    
    def cashflows(self, valuation_date: Optional[pd.Timestamp] = None, aggregate_cashflows: bool = False) -> pd.DataFrame:
        """
        Generates a schedule of future cashflows for all assets in the Portfolio.

        Args:
            valuation_date (pd.Timestamp, optional)
                The date from which the cashflow schedule should
                begin. Defaults to today if None.
            aggregate_cashflows (bool, optional)
                If True, groups by Date and sums across bonds
        
        Returns:
            pd.DataFrame: Table displaying information about all future cashflows.
        """

        if isinstance(valuation_date, str):
            valuation_date = pd.Timestamp(valuation_date).normalize()

        if valuation_date is None:
            valuation_date = pd.Timestamp.today().normalize()

        cashflows_dataframe = pd.DataFrame(columns=["CUSIP", "Date", "Cashflow"])

        for asset in self.__assets:
            cf = asset.cashflows(valuation_date=valuation_date)

            for _, cashflow in cf.iterrows():
                data = pd.DataFrame([{"CUSIP": asset.cusip, "Date": cashflow["Date"], "Cashflow": cashflow["Cashflow"]}])
                cashflows_dataframe = pd.concat([data, cashflows_dataframe], ignore_index=True)
        
        cashflows_dataframe = cashflows_dataframe.sort_values("Date").reset_index(drop=True)

        if aggregate_cashflows:
            # Group by Date, sum cashflows, and join CUSIPs
            cashflows_dataframe = (
                cashflows_dataframe
                .groupby("Date", as_index=False)
                .agg({
                    "Cashflow": "sum", "CUSIP": lambda x : ", ".join(sorted(set(x)))
                })
                .sort_values("Date")
                .reset_index(drop=True)
            )

            # Reorder columns
            cashflows_dataframe = cashflows_dataframe[["CUSIP", "Date", "Cashflow"]]

        return cashflows_dataframe
    
    def get_present_values_monthly(self, from_date: pd.Timestamp = None, yield_curve: Optional[pd.Series] = None, yield_curve_dict: Optional[dict] = None, use_default_risk: Optional[bool] = False) -> pd.DataFrame:
        """
        Calculates the expected future present value of assets monthly from a given date, optionally applying discounting using a yield curve.
        This method also calculates z-spread to ensure that the present value is equal to market value at t=0.
        If no from_date is provided, it will be valued from today.

        Args:
            from_date (pd.Timestamp, optional): The date from which valuation should begin
            yield_curve (pd.Series, optional): A series providing annualised zero rates for investments maturing on
                given dates, beginning at issue date. If used, forward rates are calculated for investments beginning on
                future dates. If neither yield_curve_dict nor yield_curve is provided, no discounting is applied.
            yield_curve_dict: (dict, optional): A dictionary indexed by dates with pd.Series as values. The key provides the
                start date for each yield curve provided as a pd.Timestamp. The value provides annualised zero rates for given dates.
                Each series should be indexed by pd.Timestamp. A yield curve must be provided monthly from valuation.
                If neither yield_curve_dict nor yield_curve is provided, no discounting is applied.
            use_default_risk (bool, optional): Indicates whether expected values should be multiplied by expected survival rate
        
        Returns:
            pd.DataFrame: Expected market (present) value at regular monthly intervals.
        """

        # Handle empty portfolio
        if not self.__assets:
            return pd.DataFrame(columns=["Date", "Value"])

        # Normalise given from_date
        if from_date is None:
            from_date = pd.Timestamp.today().normalize()
        else:
            from_date = pd.Timestamp(from_date).normalize()

        # Initialise empty list to store results
        values_monthly = []

        # Iterate monthly from today until all assets have matured
        working_date = from_date

        # Create cache for forward rates
        forward_rates = {}

        while working_date <= max(self.list_bonds()["Maturity Date"]):
            total_period_value = 0 # will be incremented

            row = {"Date": working_date}

            # Iterate through each asset held
            for asset in self.__assets:
                if asset.issue_date > working_date or asset.maturity_date < working_date:
                    continue # skip inactive assets

                # Generate z-spread if needed
                if yield_curve_dict is None:
                    if yield_curve is not None:
                        zspread = asset._solve_z_spread(as_of=asset.issue_date, yield_curve=yield_curve)
                    else:
                        zspread = 0.0
                else:
                    try:
                        initial_yield_curve = yield_curve_dict[asset.issue_date]
                        zspread = asset._solve_z_spread(as_of=asset.issue_date, yield_curve=initial_yield_curve)
                    except KeyError as e:
                        raise ValueError(f"The provided yield_curve_dict does not include sufficient data. Error: {e}")

                # Calculate forward rates if necessary
                temp_yield_curve = None
                if yield_curve_dict is not None:
                    # The yield curves from the relevant date should be included
                    try:
                        temp_yield_curve = yield_curve_dict[working_date]
                    except KeyError as e:
                        raise ValueError(f"The provided yield_curve_dict does not include sufficient data for {working_date}. Error: {e}")
                elif yield_curve is not None:
                    # We need to calculate forward rates from the valuation date
                    temp_yield_curve = pd.Series(dtype=float)

                    for maturity_date, rate in yield_curve.items():
                        if (working_date, maturity_date) in forward_rates:
                            rate = forward_rates[(working_date, maturity_date)]
                        else:
                            # Calculate time until maturity from issue
                            n_long = (maturity_date - asset.issue_date).days / 365.0

                            # Calculate time from issue until the valuation date
                            n_short = (working_date - asset.issue_date).days / 365.0

                            # Linear interpolation to find spot rate between given rates in the yield curve
                            curve_times = (yield_curve.index - asset.issue_date).days / 365.0
                            spot_long = float(np.interp(n_long, curve_times, yield_curve.values)) # spot rate at maturity date
                            spot_short = float(np.interp(n_short, curve_times, yield_curve.values)) # spot rate at valuation date

                            # Calculate forward rate
                            if n_long != n_short:
                                rate = asset._forward_rate(spot_short, spot_long, n_short, n_long)
                                forward_rates[(working_date, maturity_date)] = rate
                            else:
                                rate = spot_long
                                forward_rates[(working_date, maturity_date)] = rate

                        # Apply new forward rate to temp_yield_curve
                        temp_yield_curve[maturity_date] = rate

                # Calculate present value using calculated yield curves
                present_value = asset.expected_value(as_of=working_date, yield_curve=temp_yield_curve, zspread=zspread, use_default_risk=use_default_risk)
                
                # Add value to running total for this period
                row[asset.cusip] = present_value
            
            # Add monthly running total to the dict
            values_monthly.append(row)

            working_date += rd.relativedelta(months=1)

        # Convert to DataFrame
        df = pd.DataFrame(values_monthly).set_index("Date")

        # Ensure all CUSIPs exist as columns
        all_cusips = [a.cusip for a in self.__assets]
        df = df.reindex(columns=all_cusips, fill_value=0.0)

        # Calculate cash values for each date in the dataframe
        cash = []
        cf = self.cashflows()
        for date in df.index:
            # Get all cashflows before this date
            cf_before = sum(cf[cf["Date"] < date]["Cashflow"]) # (strict inequality because cash should be added in the next month)
            cash.append(cf_before)

        df = df.assign(Cash = cash)

        # Add total column
        df["Total"] = df.sum(axis=1)

        # Fill missing values
        df = df.fillna(0.0).round(2)

        return df