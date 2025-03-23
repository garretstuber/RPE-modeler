import pandas as pd
import numpy as np
import io
from typing import Union, Tuple, Optional
import streamlit as st

def load_session_data(file):
    """Load session data from a CSV file"""
    try:
        if isinstance(file, str):  # If path is provided
            session_df = pd.read_csv(file)
        else:  # If file object is provided (from st.file_uploader)
            session_df = pd.read_csv(file)
        
        # Validate the loaded data
        validation_result = validate_session_data(session_df)
        if not validation_result["is_valid"]:
            st.error(f"Session data validation failed: {validation_result['error_message']}")
            return None
            
        return session_df
    except Exception as e:
        st.error(f"Error loading session data: {str(e)}")
        return None

def validate_session_data(df):
    """
    Validates session data format and provides helpful error messages.
    
    Args:
        df: pandas DataFrame containing session data
        
    Returns:
        dict: {
            "is_valid": bool,
            "error_message": str (if is_valid is False),
            "warnings": list of warning strings
        }
    """
    result = {
        "is_valid": True,
        "error_message": "",
        "warnings": []
    }
    
    # Check required columns
    required_columns = ["trial_number", "trial_type", "cs_onset", "reward"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        result["is_valid"] = False
        result["error_message"] = f"Missing required columns: {', '.join(missing_columns)}"
        return result
    
    # Check for expected trial types
    trial_types = df["trial_type"].unique()
    if "CS+" not in trial_types:
        result["is_valid"] = False
        result["error_message"] = "No 'CS+' trials found in trial_type column. Check capitalization and format."
        return result
        
    if "CS-" not in trial_types:
        result["is_valid"] = False
        result["error_message"] = "No 'CS-' trials found in trial_type column. Check capitalization and format."
        return result
    
    # Check reward consistency
    cs_plus_mask = df["trial_type"] == "CS+"
    cs_minus_mask = df["trial_type"] == "CS-"
    
    # Check if CS+ trials have reward values set to 1
    if not all(df.loc[cs_plus_mask, "reward"] == 1):
        result["warnings"].append("Some CS+ trials don't have reward value set to 1.")
    
    # Check if CS- trials have reward values set to 0
    if not all(df.loc[cs_minus_mask, "reward"] == 0):
        result["warnings"].append("Some CS- trials don't have reward value set to 0.")
    
    # Check for reward_time in rewarded trials
    if "reward_time" in df.columns:
        missing_reward_times = cs_plus_mask & df["reward_time"].isna()
        if any(missing_reward_times):
            result["warnings"].append(f"{sum(missing_reward_times)} CS+ trials are missing reward_time values.")
    else:
        result["warnings"].append("Column 'reward_time' is missing. Assuming rewards at CS onset + 3.0s.")
    
    # Check for trial numbering
    if not np.array_equal(df["trial_number"].values, np.arange(len(df))):
        result["warnings"].append("Trial numbers are not sequential starting from 0. This might cause issues with data alignment.")
    
    # Print debug info
    cs_plus_count = sum(cs_plus_mask)
    cs_minus_count = sum(cs_minus_mask)
    print(f"Debug - Trial counts: CS+ = {cs_plus_count}, CS- = {cs_minus_count}")
    print(f"Debug - session_df trial types: {list(trial_types)}")
    print(f"Debug - session_df shape: {df.shape}")
    
    return result

def load_dopamine(data_file, time_file=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load dopamine photometry data from NPY file
    
    Args:
        data_file: Uploaded NPY file containing dopamine data (trials × timepoints)
        time_file: Optional NPY file containing time vector
        
    Returns:
        Tuple of (dopamine_data, time_vector)
    """
    try:
        dopamine_data = np.load(data_file)
        
        # Load time vector if provided
        time_vector = None
        if time_file is not None:
            time_vector = np.load(time_file)
        else:
            # Create default time vector if not provided
            # Assuming 20Hz sampling rate centered around CS onset (0)
            # with -2s to +5s recording window
            time_vector = np.linspace(-2, 5, dopamine_data.shape[1])
            
        return dopamine_data, time_vector
    except Exception as e:
        st.error(f"Error loading dopamine data: {str(e)}")
        return None, None

def load_lick_data(file) -> Union[pd.DataFrame, np.ndarray, None]:
    """
    Load lick data from either:
    - CSV file with trial_number and lick_time columns
    - NPY file with binned lick raster (trials × timepoints)
    
    Args:
        file: Uploaded file containing lick data
        
    Returns:
        DataFrame with lick times or numpy array with lick raster
    """
    try:
        # Determine file type from name
        filename = file.name.lower()
        
        if filename.endswith('.csv'):
            # Load CSV lick times
            df = pd.read_csv(file)
            required_cols = ['trial_number', 'lick_time']
            
            # Check for required columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns in lick data: {', '.join(missing_cols)}")
                return None
                
            return df
            
        elif filename.endswith('.npy'):
            # Load NPY lick raster
            lick_raster = np.load(file)
            return lick_raster
            
        else:
            st.error("Unsupported lick data file format. Please upload .csv or .npy file.")
            return None
            
    except Exception as e:
        st.error(f"Error loading lick data: {str(e)}")
        return None

def generate_time_vector(data_shape: tuple, fs: float = 20.0, 
                         window: tuple = (-2.0, 5.0)) -> np.ndarray:
    """
    Generate a time vector for data visualization
    
    Args:
        data_shape: Shape of the data array (trials, timepoints)
        fs: Sampling frequency in Hz
        window: Time window (start_time, end_time) in seconds
        
    Returns:
        Numpy array with time points
    """
    n_timepoints = data_shape[1]
    return np.linspace(window[0], window[1], n_timepoints) 