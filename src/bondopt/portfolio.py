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
        cashflows(valuation_date=None) -> pd.DataFrame
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
        
