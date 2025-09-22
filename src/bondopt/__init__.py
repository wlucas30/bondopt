# SPDX-License-Identifier: MIT

import pandas as pd

# Create store for preset default rate tables
default_table = {}

def import_default_rates(path_to_input: str, preset_name: str) -> None:
    """
    Encodes a provided CSV file into a table of default rates to be stored under bondopt.default_table.
    CSV should be in this format:

    Asset Rating    Default Risk Curve
    AAA             0.0003, 0.0004, 0.0005, 0.0006, 0.0007
    AA              0.0004, 0.0004, 0.0006, 0.0007, 0.0009
    A               0.0009, 0.0010, 0.0012, 0.0013, 0.0015
    BBB             0.0028, 0.0031, 0.0034, 0.0037, 0.0039
    
    Args:
        path_to_input(str): The filepath for the CSV input.
        preset_name(str): The name for the default rate table to be stored under.

    Returns:
        None
    
    Example:
        >>> import bondopt
        >>> bondopt.import_default_rates("default_rates.csv", "Bond")
    """

    # Get CSV handler
    from bondopt.csv import CSVHandler
    handler = CSVHandler()

    # Attempt to encode input
    table = handler.encode(path_to_input)

    # Check type
    if isinstance(table, pd.DataFrame) and list(table.columns.values) == ["Asset Rating", "Default Risk Curve"]:
        default_table[preset_name] = table
    else:
        print(table)
        raise Exception("Error occurred while processing CSV input!")

def dfc_format(dfc):
        """
        Formats default risk curve from CSV input
        """
        if dfc is None or not isinstance(dfc, str):
            return None
        else:
            # Split string into list of floats
            values = [float(x.strip()) for x in dfc.split(",")]

            # Generate index for default risk curve
            idx = range(1, len(values)+1)

            return pd.Series(values, index=idx)