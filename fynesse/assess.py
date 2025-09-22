from typing import Any, Union
import pandas as pd
import logging

from .config import *
from . import access

# Set up logging
logger = logging.getLogger(__name__)

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded.
How are missing values encoded, how are outliers encoded? What do columns represent,
makes rure they are correctly labeled. How is the data indexed. Crete visualisation
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct
and correctly timezoned."""


def data() -> Union[pd.DataFrame, Any]:
    """
    Load the data from access and ensure missing values are correctly encoded as well as
    indices correct, column names informative, date and times correctly formatted.
    Return a structured data structure such as a data frame.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR DATA ASSESSMENT CODE:
       - Load data using the access module
       - Check for missing values and handle them appropriately
       - Validate data types and formats
       - Clean and prepare data for analysis

    2. ADD ERROR HANDLING:
       - Handle cases where access.data() returns None
       - Check for data quality issues
       - Validate data structure and content

    3. ADD BASIC LOGGING:
       - Log data quality issues found
       - Log cleaning operations performed
       - Log final data summary

    4. EXAMPLE IMPLEMENTATION:
       df = access.data()
       if df is None:
           print("Error: No data available from access module")
           return None

       print(f"Assessing data quality for {len(df)} rows...")
       # Your data assessment code here
       return df
    """
    logger.info("Starting data assessment")

    # Load data from access module
    df = access.data()

    # Check if data was loaded successfully
    if df is None:
        logger.error("No data available from access module")
        print("Error: Could not load data from access module")
        return None

    logger.info(f"Assessing data quality for {len(df)} rows, {len(df.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your data assessment code here

        # Example: Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts.to_dict()}")
            print(f"Missing values found: {missing_counts.sum()} total")

        # Example: Check data types
        logger.info(f"Data types: {df.dtypes.to_dict()}")

        # Example: Basic data cleaning (students should customize this)
        # Remove completely empty rows
        df_cleaned = df.dropna(how="all")
        if len(df_cleaned) < len(df):
            logger.info(f"Removed {len(df) - len(df_cleaned)} completely empty rows")

        logger.info(f"Data assessment completed. Final shape: {df_cleaned.shape}")
        return df_cleaned

    except Exception as e:
        logger.error(f"Error during data assessment: {e}")
        print(f"Error assessing data: {e}")
        return None


def query(data: Union[pd.DataFrame, Any]) -> str:
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data: Union[pd.DataFrame, Any]) -> None:
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
import pandas as pd

def compute_gender_gaps(df, female_col, male_col, prefix):
    """
    Compute gender gap and parity index for a given male/female column pair.
    
    Parameters
    ----------
    df : DataFrame
    female_col : str
        Column name for female values
    male_col : str
        Column name for male values
    prefix : str
        Prefix to use for new columns (e.g. 'primary_enrol')
    """
    df[female_col] = pd.to_numeric(df[female_col], errors="coerce")
    df[male_col] = pd.to_numeric(df[male_col], errors="coerce")
    
    df[f"gap_{prefix}"] = df[female_col] - df[male_col]
    df[f"gpi_{prefix}"] = df[female_col] / df[male_col]
    
    return df

import pandas as pd

def compute_gender_gaps(master_filled: pd.DataFrame, gap_pairs: dict):
    """
    Compute gender gaps (female - male) for a given set of column pairs.

    Parameters
    ----------
    master_filled : pd.DataFrame
        The merged dataset with gender-disaggregated columns.
    gap_pairs : dict
        Dictionary mapping gap_name -> (female_col, male_col).

    Returns
    -------
    pd.DataFrame
        DataFrame with new gap columns added.
    """
    for gap_name, (female_col, male_col) in gap_pairs.items():
        if female_col in master_filled.columns and male_col in master_filled.columns:
            master_filled[gap_name] = (
                pd.to_numeric(master_filled[female_col], errors="coerce") -
                pd.to_numeric(master_filled[male_col], errors="coerce")
            )
        else:
            print(f"Skipped {gap_name}: missing columns")
    return master_filled


import pandas as pd

def compute_gender_gaps(master_filled: pd.DataFrame, gap_pairs: dict):
    """
    Compute gender gaps (female - male) for a given set of column pairs.

    Parameters
    ----------
    master_filled : pd.DataFrame
        The merged dataset with gender-disaggregated columns.
    gap_pairs : dict
        Dictionary mapping gap_name -> (female_col, male_col).

    Returns
    -------
    pd.DataFrame
        DataFrame with new gap columns added.
    """
    for gap_name, (female_col, male_col) in gap_pairs.items():
        if female_col in master_filled.columns and male_col in master_filled.columns:
            master_filled[gap_name] = (
                pd.to_numeric(master_filled[female_col], errors="coerce") -
                pd.to_numeric(master_filled[male_col], errors="coerce")
            )
        else:
            print(f"Skipped {gap_name}: missing columns")
    return master_filled

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def cluster_gender_gaps(master_filled, gap_prefix="gap_", n_clusters=3, plot=True):
    """
    Cluster countries based on gender gap features using KMeans.
    """
    gap_cols = [col for col in master_filled.columns if gap_prefix in col]
    X = master_filled[gap_cols].dropna()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Assign clusters back
    master_filled.loc[X.index, "cluster"] = clusters

    # Optional visualization (first 2 features)
    if plot and len(gap_cols) >= 2:
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap="viridis")
        plt.xlabel(gap_cols[0])
        plt.ylabel(gap_cols[1])
        plt.title("Clustering of Countries by Gender Gaps")
        plt.show()

    return master_filled, kmeans

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def regression_model(master_filled, target="gap_primary_enrolment", exclude_features=None):
    """
    Train and evaluate a linear regression model to predict a gender gap.
    """
    if exclude_features is None:
        exclude_features = [target, "year", "cluster"]

    numeric_cols = master_filled.select_dtypes(include=np.number).columns.tolist()
    features = [col for col in numeric_cols if col not in exclude_features]

    df_model = master_filled.dropna(subset=features + [target])
    if df_model.shape[0] == 0:
        print("No data available after dropping rows with missing values.")
        return None, None

    X = df_model[features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "R2": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Coefficients": dict(zip(features, model.coef_))
    }

    return model, metrics

