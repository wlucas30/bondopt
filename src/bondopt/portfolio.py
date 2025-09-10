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
import pandas as pd

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
        get_present_values_monthly() ->
    """

    def __post_init__(self):
        # Initialise array to store the assets which constitute the portfolio
        self.assets = []

    def add_bond(self, bond) -> None:
        """
        Adds a provided Bond to the Portfolio.

        Args:
            bond (bondopt.bond.Bond): The Bond to be added to the Portfolio.
        """
        if isinstance(bond, Bond):
            self.assets.append(bond)
        else:
            raise TypeError("The bond provided is formatted incorrectly. It must be a bondopt.bond.Bond object")
    
    def remove_bond(self, cusip) -> None:
        """
        Removes all instances of bonds from the Portfolio with CUSIP matching the provided value.

        Args:
            cusip (str): The CUSIP value of the bond(s) to be removed.
        """
        found = False
        for bond in self.assets:
            if bond.cusip == cusip:
                self.assets.remove(bond)
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

        for bond in self.assets:
            summary = bond.summary(verbose=False)
            bonds = pd.concat([bonds, summary], ignore_index=True)
        
        return bonds
    
    def cashflows(self, valuation_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Generates a schedule of future cashflows for all assets in the Portfolio.

        Args:
            valuation_date (pd.Timestamp, optional)
                The date from which the cashflow schedule should
                begin. Defaults to today if None.
        
        Returns:
            pd.DataFrame: Table displaying information about all future cashflows.
        """

        if isinstance(valuation_date, str):
            valuation_date = pd.Timestamp(valuation_date).normalize()

        if valuation_date is None:
            valuation_date = pd.Timestamp.today().normalize()

        cashflows_dataframe = pd.DataFrame(columns=["CUSIP", "Date", "Cashflow"])

        for asset in self.assets:
            cf = asset.cashflows(valuation_date=valuation_date)

            for _, cashflow in cf.iterrows():
                data = pd.DataFrame([{"CUSIP": asset.cusip, "Date": cashflow["Date"], "Cashflow": cashflow["Cashflow"]}])
                cashflows_dataframe = pd.concat([data, cashflows_dataframe], ignore_index=True)
        
        return cashflows_dataframe.sort_values("Date").reset_index(drop=True)