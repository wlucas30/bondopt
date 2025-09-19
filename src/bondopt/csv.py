"""
csv.py

Contains the CSVHandler class and related functions.

Classes:
    CSVHandler - Contains methods for passing data to the library in CSV files.
"""

# SPDX-License-Identifier: MIT

from bondopt.bond import Bond
from bondopt.portfolio import Portfolio
from bondopt.reinvest import ReinvestmentStrategy
from dataclasses import dataclass
from typing import Optional
import dateutil.relativedelta as rd
import pandas as pd
import numpy as np

@dataclass
class CSVHandler:
    """
    Class which contains methods to handle CSV file input.

    Methods:
        encode(path_to_input) -> any
            Generates the required object given a CSV input filepath.
    """

    def encode(self, path_to_input: str) -> any:
        """
        Generates the required object given a CSV input filepath.

        Args:
            path_to_input(str): The filepath for the CSV input.

        Returns:
            The required object for further use.
        """
        # Read the CSV file into a pandas DataFrame
        try:
            df = pd.read_csv(path_to_input)
        except Exception as e:
            raise Exception(f"An error occurred reading the CSV input: {e}")
        
        # Check which columns are included so we can determine which object to create
        col_list = df.columns.tolist()

        # Define the required columns for each object (excludes optional arguments)
        col_reqs = {
            "portfolio": ["Asset Type", "Issue Date", "Maturity Date", "Coupon Rate", "Coupon Frequency", "Market Value", "Notional"],
            "bond": ["Asset Type", "Issue Date", "Maturity Date", "Coupon Rate", "Coupon Frequency", "Market Value", "Notional"],
            "yield_curve_dict": ["From Date"],
            "yield_curve": [], # consists of just dates, no required columns
            "reinvestment_strategy": ["Asset Rating", "Allocation %", "Spread Adjustment (bps)"]
        }

        # Case 1: Portfolio
        if all(col in col_list for col in col_reqs["portfolio"]) and df.shape[0] > 1:
            # Create new Portfolio object
            pf = Portfolio()

            # Iterate through each entry
            for _, asset in df.iterrows():
                # Case 1a: Bond
                if not asset[col_reqs["bond"]].isnull().any(): # this ensures every required column is not null
                    # Create new Bond object
                    try:
                        b = Bond(
                            asset_type = asset["Asset Type"],
                            issue_date = asset["Issue Date"],
                            maturity_date = asset["Maturity Date"],
                            coupon_rate = asset["Coupon Rate"],
                            coupon_freq = asset["Coupon Frequency"],
                            market_value = asset["Market Value"],
                            notional = asset["Notional"],
                            cusip = asset.get("CUSIP", None),
                            default_risk_curve = asset.get("Default Risk Curve", None)
                        )
                    except Exception as e:
                        raise Exception(f"Error: bad data provided! {e}")

                    # Add Bond to Portfolio
                    pf.add_bond(b)
            
            # Return Portfolio object
            return pf
        
        # Case 2: Yield curve dict
        elif all(col in col_list for col in col_reqs["yield_curve_dict"]):
            # Create empty yield curve dict
            ycd = {}

            # Iterate through each provided yield curve
            for _, row in df.iterrows():
                dates = col_list[col_list != "From Date"]
                try:
                    dates_normalised = [pd.Timestamp(date).normalize() for date in dates]
                except Exception as e:
                    raise Exception(f"Error: bad date format provided! {e}")

                rates = [float(row[date]) for date in dates_normalised]
                yc = pd.Series(rates, index=dates_normalised).dropna()
                ycd[row["From Date"]] = yc
            
            return ycd
                
        # Case 3: Individual yield curve
        elif all(self.__is_timestamp(col) for col in col_list):
            _, row = next(df.iterrows())
            dates = col_list
            dates_normalised = [pd.Timestamp(date).normalize() for date in dates]
            rates = [float(row[date]) for date in dates_normalised]
            yc = pd.Series(rates, index=dates_normalised).dropna()
            return yc

        # Case 4: Reinvestment strategy
        elif all(col in col_list for col in col_reqs["reinvestment_strategy"]):
            # Create new ReinvestmentStrategy object
            try:
                rs = ReinvestmentStrategy(table=df)
            except Exception as e:
                raise Exception(f"Error: Failed to create ReinvestmentStrategy with the required data. {e}")

            return rs
            
        # Case 5: Invalid input
        else:
            raise ValueError("The provided CSV file is formatted incorrectly!")
        
    def __is_timestamp(self, s):
        """
        (Private method)
        Checks if a string can be serialised into a pandas Timestamp
        """
        try:
            pd.Timestamp(s)
            return True
        except:
            return False