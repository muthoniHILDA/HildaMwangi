"""
Access module for the fynesse framework.

This module handles data access functionality including:
- Data loading from various sources (web, local files, databases)
- Legal compliance (intellectual property, privacy rights)
- Ethical considerations for data usage
- Error handling for access issues

Legal and ethical considerations are paramount in data access.
Ensure compliance with e.g. .GDPR, intellectual property laws, and ethical guidelines.

Best Practice on Implementation
===============================

1. BASIC ERROR HANDLING:
   - Use try/except blocks to catch common errors
   - Provide helpful error messages for debugging
   - Log important events for troubleshooting

2. WHERE TO ADD ERROR HANDLING:
   - File not found errors when loading data
   - Network errors when downloading from web
   - Permission errors when accessing files
   - Data format errors when parsing files

3. SIMPLE LOGGING:
   - Use print() statements for basic logging
   - Log when operations start and complete
   - Log errors with context information
   - Log data summary information

4. EXAMPLE PATTERNS:
   
   Basic error handling:
   try:
       df = pd.read_csv('data.csv')
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
   
   With logging:
   print("Loading data from data.csv...")
   try:
       df = pd.read_csv('data.csv')
       print(f"Successfully loaded {len(df)} rows of data")
       return df
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
"""

from typing import Any, Union
import pandas as pd
import logging
import osmnx as ox
import matplotlib.pyplot as plt

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def data() -> Union[pd.DataFrame, None]:
    """
    Read the data from the web or local file, returning structured format such as a data frame.

    IMPLEMENTATION GUIDE
    ====================

    1. REPLACE THIS FUNCTION WITH YOUR ACTUAL DATA LOADING CODE:
       - Load data from your specific sources
       - Handle common errors (file not found, network issues)
       - Validate that data loaded correctly
       - Return the data in a useful format

    2. ADD ERROR HANDLING:
       - Use try/except blocks for file operations
       - Check if data is empty or corrupted
       - Provide helpful error messages

    3. ADD BASIC LOGGING:
       - Log when you start loading data
       - Log success with data summary
       - Log errors with context

    4. EXAMPLE IMPLEMENTATION:
       try:
           print("Loading data from data.csv...")
           df = pd.read_csv('data.csv')
           print(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
           return df
       except FileNotFoundError:
           print("Error: data.csv file not found")
           return None
       except Exception as e:
           print(f"Error loading data: {e}")
           return None

    Returns:
        DataFrame or other structured data format
    """
    logger.info("Starting data access operation")

    try:
        # IMPLEMENTATION: Replace this with your actual data loading code
        # Example: Load data from a CSV file
        logger.info("Loading data from data.csv")
        df = pd.read_csv("data.csv")

        # Basic validation
        if df.empty:
            logger.warning("Loaded data is empty")
            return None

        logger.info(
            f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    except FileNotFoundError:
        logger.error("Data file not found: data.csv")
        print("Error: Could not find data.csv file. Please check the file path.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        print(f"Error loading data: {e}")
        return None
def plot_city_map(place_name, latitude, longitude, box_size_km=2, tags=None):
    """
    Visualizes the street network, buildings, and points of interest for a given location.

    Parameters
    ----------
    place_name : str
        The name of the place to visualize.
    latitude : float
        Latitude of the center point.
    longitude : float
        Longitude of the center point.
    box_size_km : float
        Size of the bounding box in kilometers.
    tags : dict
        A dictionary of OpenStreetMap tags to include as points of interest.
    """
    # Construct bbox from lat/lon and box_size
    lat_degree_size = box_size_km / 111.0
    lon_degree_size = box_size_km / 111.0 

    north = latitude + lat_degree_size / 2
    south = latitude - lat_degree_size / 2
    west = longitude - lon_degree_size / 2
    east = longitude + lon_degree_size / 2
    bbox = (west, south, east, north)

    try:
        # Get graph from location
        graph = ox.graph_from_bbox(bbox, network_type='drive') # Specify network_type
        # City area
        area = ox.geocode_to_gdf(place_name)
        # Street network
        nodes, edges = ox.graph_to_gdfs(graph)
        # Buildings
        buildings = ox.features_from_bbox(bbox, tags={"building": True})
        # POIs
        if tags is None:
            # Use default tags if none are provided
            tags = {
                "amenity": True,
                "buildings": True,
                "historic": True,
                "leisure": True,
                "shop": True,
                "tourism": True,
                "religion": True,
                "memorial": True
            }
        pois = ox.features_from_bbox(bbox, tags=tags)

        fig, ax = plt.subplots(figsize=(8,8))
        area.plot(ax=ax, color="tan", alpha=0.5)
        buildings.plot(ax=ax, facecolor="gray", edgecolor="gray")
        edges.plot(ax=ax, linewidth=1, edgecolor="black", alpha=0.3)
        nodes.plot(ax=ax, color="black", markersize=1, alpha=0.3)
        if not pois.empty:
            pois.plot(ax=ax, color="green", markersize=5, alpha=1)
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.set_title(place_name, fontsize=14)
        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting the map: {e}")
        print(f"Could not plot map for {place_name} at ({latitude}, {longitude}) with box size {box_size_km} km.")

import pandas as pd
import numpy as np

def load_and_clean_data(primary_path, secondary_path, tertiary_path,Primary_attendance_path, education_General_path, illiterate_path, school_age_path):
    # Load datasets
    df_primary = pd.read_csv(primary_path)
    df_secondary = pd.read_csv(secondary_path)
    df_tertiary = pd.read_csv(tertiary_path)
    df_Primary_attendance=pd.read_csv(Primary_attendance_path)
    df_education_General=pd.read_csv(education_General_path)
    df_illiterate=pd.read_csv(illiterate_path)
    df_school_age=pd.read_csv(school_age_path)

    # Merge datasets
    master = pd.concat([df_primary, df_secondary, df_tertiary], axis=1)

    # Replace junk values
    master = master.replace("#N/B", np.nan)

    # Interpolate missing values within each country
    master = master.groupby("country").apply(lambda g: g.interpolate()).reset_index(drop=True)

    return master

import pandas as pd
import numpy as np

def load_and_clean_data(file_paths: dict):
    """
    Load and clean multiple education-related tables, then merge into one master dataset.

    Parameters
    ----------
    file_paths : dict
        Dictionary with descriptive names as keys and CSV file paths as values.
        Example:
        {
            "primary": "Primary.csv",
            "secondary": "Secondary.csv",
            "tertiary": "Tertiary.csv",
            "literacy": "Literacy.csv",
            "life_expectancy": "SchoolLifeExpectancy.csv",
            "survival": "SurvivalRates.csv",
            "out_school": "OutOfSchool.csv",
            "age_population": "SchoolAgePopulation.csv",
            "expenditure": "Expenditure.csv"
        }

    Returns
    -------
    master : pd.DataFrame
        A merged dataframe with all cleaned indicators.
    """

    dataframes = []

    for name, path in file_paths.items():
        try:
            df = pd.read_csv(path)

            # Basic cleaning
            df = df.replace(["#N/B", "NA", "N/A", ".."], np.nan)

            # Ensure consistent column names
            df = df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"))

            # Keep only country, year, and indicators
            if "country" in df.columns and "year" in df.columns:
                df = df.set_index(["country", "year"])
            else:
                raise ValueError(f"{name} table must contain 'country' and 'year' columns")

            # Prefix columns by table name to avoid clashes
            df = df.add_prefix(f"{name}_")

            dataframes.append(df)

        except Exception as e:
            print(f"Error loading {name}: {e}")

    # Merge all tables on country + year
    master = pd.concat(dataframes, axis=1, join="outer").reset_index()

    # Interpolate missing numeric values by country
    master = master.groupby("country").apply(lambda g: g.interpolate()).reset_index(drop=True)

    return master

import pandas as pd

def load_datasets(base_path="."):
    df_primary = pd.read_csv(f"{base_path}/Primary Education.csv")
    df_secondary = pd.read_csv(f"{base_path}/Secondary Education.csv")
    df_tertiary = pd.read_csv(f"{base_path}/Tertiary Education.csv")
    df_Primary_attendance = pd.read_csv(f"{base_path}/Primary Education Attendance.csv")
    df_education_General = pd.read_csv(f"{base_path}/Education in General.csv")
    df_illiterate = pd.read_csv(f"{base_path}/Illiterate Population.csv")
    df_school_age = pd.read_csv(f"{base_path}/School Age Population.csv")

    datasets = {
        "Primary": df_primary,
        "Secondary": df_secondary,
        "Tertiary": df_tertiary,
        "Primary Attendance": df_Primary_attendance,
        "Education General": df_education_General,
        "Illiterate": df_illiterate,
        "School Age": df_school_age
    }
    return datasets


def inspect_datasets(datasets):
    """Print basic info and missingness for each dataset."""
    for name, df in datasets.items():
        print(f"\n{name} dataset")
        print(df.shape)
        print(df.columns)
        print(df.info())
        print(df.isna().mean().sort_values(ascending=False).head(10))

import pandas as pd

def standardize_table(df, name):
    if "Country" in df.columns:
        df = df.rename(columns={"Country": "country"})
    if "Year" in df.columns:
        df = df.rename(columns={"Year": "year"})
        df["year"] = df["year"].astype(int)
    return df


def load_and_standardize(file_paths: dict):
    """
    Load multiple CSVs and standardize country/year columns.
    """
    datasets = {}
    for name, path in file_paths.items():
        df = pd.read_csv(path)
        df = standardize_table(df, name)
        datasets[name] = df
    return datasets





