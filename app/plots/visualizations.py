import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import Optional, Tuple, Dict, List, Union
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d

# Define a Pacific Northwest forest and mountain color palette
PNW_COLORS = {
    'forest_dark': '#1e4d2b',  # Dark forest green
    'forest_medium': '#2e6d3e', # Medium forest green
    'forest_light': '#5ba172',  # Light forest green
    'mountain_dark': '#4a5859', # Dark slate
    'mountain_medium': '#7a8c8d', # Medium slate
    'mountain_light': '#a9b8b9', # Light slate
    'water_dark': '#0d4b6f',    # Deep water blue
    'water_medium': '#2980b9',  # Medium water blue
    'water_light': '#a8d5f7',   # Light water blue
    'sunset_orange': '#d35400', # Sunset orange
    'sunset_red': '#c0392b',    # Sunset red
    'snow': '#f4f6f7'           # Snow white
}

# Custom colormaps based on PNW theme
def get_pnw_rpe_cmap():
    """Create a custom colormap for RPE data"""
    return mpl.colors.LinearSegmentedColormap.from_list(
        'pnw_rpe', 
        [PNW_COLORS['water_dark'], PNW_COLORS['snow'], PNW_COLORS['forest_dark']]
    )

def get_pnw_dopamine_cmap():
    """Create a custom colormap for dopamine data"""
    return mpl.colors.LinearSegmentedColormap.from_list(
        'pnw_dopamine', 
        [PNW_COLORS['water_light'], PNW_COLORS['water_dark']]
    )

def get_pnw_value_cmap():
    """Create a custom colormap for value data"""
    return mpl.colors.LinearSegmentedColormap.from_list(
        'pnw_value', 
        [PNW_COLORS['forest_light'], PNW_COLORS['forest_dark']]
    )

def get_pnw_lick_cmap():
    """Create a custom colormap for lick data"""
    return mpl.colors.LinearSegmentedColormap.from_list(
        'pnw_lick', 
        [PNW_COLORS['snow'], PNW_COLORS['forest_dark']]
    )

def plot_rpe_heatmap(rpe_data: np.ndarray, 
                    time_points: np.ndarray,
                    session_df: pd.DataFrame,
                    sort_by_value: bool = False,
                    cs_values: Optional[np.ndarray] = None,
                    cmap: str = 'RdBu_r',
                    vmin: Optional[float] = None,
                    vmax: Optional[float] = None,
                    title: str = "RPE Signals",
                    smooth_data: bool = True,
                    smoothing_sigma: float = 1.0,
                    figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
    """
    Plot heatmap of reward prediction error signals with CS+ and CS- trials side by side
    
    Args:
        rpe_data: RPE signal data (trials x timepoints)
        time_points: Time vector
        session_df: Session data
        sort_by_value: Whether to sort trials by CS value
        cs_values: CS value for each trial (if sort_by_value is True)
        cmap: Colormap
        vmin, vmax: Color scale limits
        title: Plot title
        smooth_data: Whether to apply smoothing to the data for better visualization
        smoothing_sigma: Sigma parameter for Gaussian smoothing
        figsize: Figure size as (width, height) tuple
        
    Returns:
        Figure object
    """
    # Default color scale if not provided
    if vmin is None and vmax is None:
        # For RPE data (typically has both positive and negative values)
        if "RPE" in title or "rpe" in title.lower():
            # Symmetric colormap for RPE
            vmax = np.max(np.abs(rpe_data))
            vmin = -vmax
        else:
            # For value function (typically all positive) or other data
            vmin = np.min(rpe_data)
            vmax = np.max(rpe_data)
    
    # Debug: Check trial types
    print(f"Debug - session_df trial types: {session_df['trial_type'].unique()}")
    print(f"Debug - session_df shape: {session_df.shape}")
    
    # Split data by trial type while preserving trial order
    cs_plus_mask = session_df['trial_type'] == 'CS+'
    cs_minus_mask = ~cs_plus_mask
    
    # Debug: Count number of trials of each type
    cs_plus_count = cs_plus_mask.sum()
    cs_minus_count = cs_minus_mask.sum()
    print(f"Debug - Trial counts: CS+ = {cs_plus_count}, CS- = {cs_minus_count}")
    
    # Get indices for each trial type
    cs_plus_indices = np.where(cs_plus_mask)[0]
    cs_minus_indices = np.where(cs_minus_mask)[0]
    
    # Debug: Check indices
    print(f"Debug - CS+ indices count: {len(cs_plus_indices)}")
    print(f"Debug - CS- indices count: {len(cs_minus_indices)}")
    
    # Extract data for each trial type (preserving original trial order)
    cs_plus_data = rpe_data[cs_plus_indices]
    cs_minus_data = rpe_data[cs_minus_indices]
    
    # Apply smoothing to the data if requested
    if smooth_data:
        # Apply smoothing along the time dimension (axis 1)
        for i in range(cs_plus_data.shape[0]):
            cs_plus_data[i, :] = gaussian_filter1d(cs_plus_data[i, :], sigma=smoothing_sigma)
        for i in range(cs_minus_data.shape[0]):
            cs_minus_data[i, :] = gaussian_filter1d(cs_minus_data[i, :], sigma=smoothing_sigma)
    
    # Debug: Check data shapes
    print(f"Debug - CS+ data shape: {cs_plus_data.shape}")
    print(f"Debug - CS- data shape: {cs_minus_data.shape}")
    
    # Create figure with side-by-side layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=False)
    
    # Select the appropriate colormap
    if cmap == 'RdBu_r':
        # Use the custom PNW RPE colormap as default
        plot_cmap = get_pnw_rpe_cmap()
    elif cmap == 'viridis':
        # Use the custom PNW value colormap for value functions
        plot_cmap = get_pnw_value_cmap()
    else:
        # Use the specified matplotlib colormap
        plot_cmap = cmap
    
    # Use seaborn style
    sns.set_style("white")
    
    # Create DataFrames for seaborn heatmaps
    if len(cs_plus_data) > 0:
        cs_plus_df = pd.DataFrame(
            cs_plus_data,
            index=range(len(cs_plus_data)),
            columns=time_points
        )
        
        # Plot CS+ heatmap
        sns.heatmap(cs_plus_df, ax=ax1, cmap=plot_cmap, vmin=vmin, vmax=vmax,
                   cbar_kws={'label': 'Value' if 'Value' in title else 'RPE', 'shrink': 0.8})
    else:
        print("Debug - No CS+ data to plot")
        ax1.text(0.5, 0.5, "No CS+ trials", ha='center', va='center', fontsize=18, transform=ax1.transAxes)
        ax1.set_xlim(0, len(time_points))
        ax1.set_ylim(0, 10)
    
    if len(cs_minus_data) > 0:
        cs_minus_df = pd.DataFrame(
            cs_minus_data,
            index=range(len(cs_minus_data)),
            columns=time_points
        )
        
        # Plot CS- heatmap
        sns.heatmap(cs_minus_df, ax=ax2, cmap=plot_cmap, vmin=vmin, vmax=vmax,
                   cbar_kws={'label': 'Value' if 'Value' in title else 'RPE', 'shrink': 0.8})
    else:
        print("Debug - No CS- data to plot")
        ax2.text(0.5, 0.5, "No CS- trials", ha='center', va='center', fontsize=18, transform=ax2.transAxes)
        ax2.set_xlim(0, len(time_points))
        ax2.set_ylim(0, 10)
    
    # Add custom x-tick marks at half-second intervals
    half_sec_ticks = np.arange(np.floor(time_points[0] * 2) / 2, 
                             np.ceil(time_points[-1] * 2) / 2 + 0.5, 
                             0.5)
    
    # Find x positions for the time ticks
    tick_positions = []
    for tick in half_sec_ticks:
        closest_idx = np.argmin(np.abs(time_points - tick))
        tick_positions.append(closest_idx)
    
    # Configure CS+ plot
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([f"{t:.1f}" for t in half_sec_ticks], rotation=45)
    ax1.axvline(x=np.argmin(np.abs(time_points - 0)), color=PNW_COLORS['mountain_dark'], 
               linestyle='--', linewidth=2, label='CS Onset')
    ax1.axvline(x=np.argmin(np.abs(time_points - 3)), color=PNW_COLORS['sunset_red'], 
               linestyle='--', linewidth=2, label='Reward')
    ax1.set_xlabel('Time from CS Onset (s)', fontsize=16)
    ax1.set_ylabel('Trial', fontsize=16)
    ax1.set_title(f"CS+ Trials", fontsize=20, fontweight='bold')
    ax1.invert_yaxis()  # Adjust y-axis to show first trial at top, last at bottom
    
    # Configure CS- plot
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f"{t:.1f}" for t in half_sec_ticks], rotation=45)
    ax2.axvline(x=np.argmin(np.abs(time_points - 0)), color=PNW_COLORS['mountain_dark'], 
               linestyle='--', linewidth=2, label='CS Onset')
    ax2.set_xlabel('Time from CS Onset (s)', fontsize=16)
    ax2.set_ylabel('Trial', fontsize=16)
    ax2.set_title(f"CS- Trials", fontsize=20, fontweight='bold')
    ax2.invert_yaxis()  # Adjust y-axis to show first trial at top, last at bottom
    
    # Set y-ticks at 10-trial increments for both axes
    for ax, data in zip([ax1, ax2], [cs_plus_data, cs_minus_data]):
        if len(data) > 0:
            n_trials = len(data)
            y_ticks = np.arange(0, n_trials, 10)
            if n_trials - 1 not in y_ticks:
                y_ticks = np.append(y_ticks, n_trials - 1)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(int(y)) for y in y_ticks])
        
        # Increase font size for tick labels
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Create legend with larger font
        ax.legend(loc='upper right', fontsize=14)
    
    # Add overall title
    fig.suptitle(title, fontsize=24, fontweight='bold', y=0.98)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def plot_photometry_heatmap(photometry_data: np.ndarray, 
                           time_points: np.ndarray,
                           session_df: pd.DataFrame,
                           sort_trials: bool = False,
                           zscore: bool = True,
                           cmap: str = 'viridis',
                           title: str = "Dopamine Photometry") -> plt.Figure:
    """
    Plot heatmap of dopamine photometry signals with CS+ and CS- trials side by side
    
    Args:
        photometry_data: Photometry data (trials x timepoints)
        time_points: Time vector
        session_df: Session data
        sort_trials: Whether to sort trials by trial type (CS+ first)
        zscore: Whether to z-score the data
        cmap: Colormap
        title: Plot title
        
    Returns:
        Figure object
    """
    # Z-score data if requested
    if zscore:
        # Z-score per trial
        mean = np.mean(photometry_data[:, time_points < 0], axis=1, keepdims=True)
        std = np.std(photometry_data[:, time_points < 0], axis=1, keepdims=True)
        std[std == 0] = 1  # Avoid division by zero
        plot_data = (photometry_data - mean) / std
        plot_title = title + " (Z-scored)"
    else:
        plot_data = photometry_data
        plot_title = title
    
    # Split data by trial type while preserving trial order
    cs_plus_mask = session_df['trial_type'] == 'CS+'
    cs_minus_mask = ~cs_plus_mask
    
    # Get indices for each trial type
    cs_plus_indices = np.where(cs_plus_mask)[0]
    cs_minus_indices = np.where(cs_minus_mask)[0]
    
    # Extract data for each trial type (preserving original trial order)
    cs_plus_data = plot_data[cs_plus_indices]
    cs_minus_data = plot_data[cs_minus_indices]
    
    # Create figure with side-by-side layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), sharey=False)
    
    # Use a custom PNW colormap
    if cmap == 'viridis':
        # Use the simplified dopamine colormap by default
        plot_cmap = get_pnw_dopamine_cmap()
    else:
        # Use the specified matplotlib colormap
        plot_cmap = cmap
    
    # Use seaborn style
    sns.set_style("white")
    
    # Set color scale limits
    vmax = np.max(np.abs(plot_data))*0.8 if zscore else np.max(plot_data)
    vmin = -vmax if zscore else np.min(plot_data)
    
    # Plot CS+ heatmap
    if len(cs_plus_data) > 0:
        sns.heatmap(cs_plus_data, ax=ax1, cmap=plot_cmap, vmin=vmin, vmax=vmax,
                   cbar_kws={'label': 'dF/F' if not zscore else 'Z-score', 'shrink': 0.8})
    else:
        ax1.text(0.5, 0.5, "No CS+ trials", ha='center', va='center', fontsize=18)
        ax1.set_xlim(0, len(time_points))
        ax1.set_ylim(0, 10)
    
    # Plot CS- heatmap
    if len(cs_minus_data) > 0:
        sns.heatmap(cs_minus_data, ax=ax2, cmap=plot_cmap, vmin=vmin, vmax=vmax,
                   cbar_kws={'label': 'dF/F' if not zscore else 'Z-score', 'shrink': 0.8})
    else:
        ax2.text(0.5, 0.5, "No CS- trials", ha='center', va='center', fontsize=18)
        ax2.set_xlim(0, len(time_points))
        ax2.set_ylim(0, 10)
    
    # Add custom x-tick marks at half-second intervals
    half_sec_ticks = np.arange(np.floor(time_points[0] * 2) / 2, 
                             np.ceil(time_points[-1] * 2) / 2 + 0.5, 
                             0.5)
    
    # Find x positions for the time ticks
    tick_positions = []
    for tick in half_sec_ticks:
        closest_idx = np.argmin(np.abs(time_points - tick))
        tick_positions.append(closest_idx)
    
    # Configure CS+ plot
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([f"{t:.1f}" for t in half_sec_ticks], rotation=45)
    ax1.axvline(x=np.argmin(np.abs(time_points - 0)), color=PNW_COLORS['mountain_dark'], 
               linestyle='--', linewidth=2, label='CS Onset')
    ax1.axvline(x=np.argmin(np.abs(time_points - 3)), color=PNW_COLORS['sunset_red'], 
               linestyle='--', linewidth=2, label='Reward')
    ax1.set_xlabel('Time from CS Onset (s)', fontsize=16)
    ax1.set_ylabel('Trial', fontsize=16)
    ax1.set_title(f"CS+ Trials", fontsize=20, fontweight='bold')
    ax1.invert_yaxis()  # Adjust y-axis to show first trial at top, last at bottom
    
    # Configure CS- plot
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f"{t:.1f}" for t in half_sec_ticks], rotation=45)
    ax2.axvline(x=np.argmin(np.abs(time_points - 0)), color=PNW_COLORS['mountain_dark'], 
               linestyle='--', linewidth=2, label='CS Onset')
    ax2.set_xlabel('Time from CS Onset (s)', fontsize=16)
    ax2.set_ylabel('Trial', fontsize=16)
    ax2.set_title(f"CS- Trials", fontsize=20, fontweight='bold')
    ax2.invert_yaxis()  # Adjust y-axis to show first trial at top, last at bottom
    
    # Set y-ticks at 10-trial increments for both plots
    for ax, data in zip([ax1, ax2], [cs_plus_data, cs_minus_data]):
        if len(data) > 0:
            n_trials = len(data)
            y_ticks = np.arange(0, n_trials, 10)
            if n_trials - 1 not in y_ticks:
                y_ticks = np.append(y_ticks, n_trials - 1)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(int(y)) for y in y_ticks])
        
        # Increase font size for tick labels
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Create legend with larger font
        ax.legend(loc='upper right', fontsize=14)
    
    # Add overall title
    fig.suptitle(plot_title, fontsize=24, fontweight='bold', y=0.98)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def plot_lick_raster(lick_data: Union[pd.DataFrame, np.ndarray],
                    time_points: Optional[np.ndarray] = None,
                    session_df: Optional[pd.DataFrame] = None,
                    bin_size: float = 0.1,
                    window: Tuple[float, float] = (-2, 5),
                    sort_trials: bool = False) -> plt.Figure:
    """
    Plot lick raster aligned to CS onset with CS+ and CS- trials side by side
    
    Args:
        lick_data: Either DataFrame with trial_number and lick_time columns, 
                or 2D binned lick array (trials x timepoints)
        time_points: Time vector (required if lick_data is 2D array)
        session_df: Session data (optional, used for separating trial types)
        bin_size: Bin size in seconds (for DataFrame input)
        window: Time window for plotting
        sort_trials: Whether to sort trials by trial type
        
    Returns:
        Figure object
    """
    # Create figure with side-by-side layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), sharey=False)
    
    # Use seaborn style
    sns.set_style("white")
    
    # Use custom lick color
    lick_color = PNW_COLORS['forest_dark']
    
    if session_df is None:
        # No session data means we can't separate by trial type
        ax1.text(0.5, 0.5, "No session data available\nCannot separate by trial type",
                ha='center', va='center', fontsize=18, transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "No session data available\nCannot separate by trial type",
                ha='center', va='center', fontsize=18, transform=ax2.transAxes)
        return fig
    
    if isinstance(lick_data, pd.DataFrame):
        # We have discrete lick times
        # Create bins for raster plot
        bins = np.arange(window[0], window[1] + bin_size, bin_size)
        n_bins = len(bins) - 1
        bin_centers = bins[:-1] + bin_size/2
        
        # Create empty lick raster
        n_trials = len(session_df)
        lick_raster = np.zeros((n_trials, n_bins), dtype=bool)
        
        # Fill in lick raster
        for _, row in lick_data.iterrows():
            trial = int(row['trial_number'])
            time = row['lick_time']
            if window[0] <= time <= window[1] and trial < n_trials:
                bin_idx = np.digitize(time, bins) - 1
                if 0 <= bin_idx < n_bins:
                    lick_raster[trial, bin_idx] = True
        
        raster_data = lick_raster
        time_vector = bin_centers
    else:
        # We already have binned lick raster
        raster_data = lick_data
        if time_points is None:
            # Create default time vector
            time_vector = np.linspace(window[0], window[1], raster_data.shape[1])
        else:
            time_vector = time_points
            # Filter time window if needed
            in_window = (time_vector >= window[0]) & (time_vector <= window[1])
            time_vector = time_vector[in_window]
            raster_data = raster_data[:, in_window]
    
    # Split data by trial type while preserving trial order
    cs_plus_mask = session_df['trial_type'] == 'CS+'
    cs_minus_mask = ~cs_plus_mask
    
    # Get indices for each trial type
    cs_plus_indices = np.where(cs_plus_mask)[0]
    cs_minus_indices = np.where(cs_minus_mask)[0]
    
    # Extract lick data for each trial type (preserving original trial order)
    cs_plus_licks = raster_data[cs_plus_indices]
    cs_minus_licks = raster_data[cs_minus_indices]
    
    # Plot lick rasters for each trial type
    # Plot CS+ licks
    if len(cs_plus_indices) > 0:
        for i, trial_idx in enumerate(range(len(cs_plus_indices))):
            lick_times = time_vector[cs_plus_licks[trial_idx]]
            ax1.eventplot(lick_times, lineoffsets=i, linelengths=0.8, 
                         linewidths=1.5, color=lick_color)
    else:
        ax1.text(0.5, 0.5, "No CS+ trials", ha='center', va='center', fontsize=18,
                transform=ax1.transAxes)
    
    # Plot CS- licks
    if len(cs_minus_indices) > 0:
        for i, trial_idx in enumerate(range(len(cs_minus_indices))):
            lick_times = time_vector[cs_minus_licks[trial_idx]]
            ax2.eventplot(lick_times, lineoffsets=i, linelengths=0.8, 
                         linewidths=1.5, color=lick_color)
    else:
        ax2.text(0.5, 0.5, "No CS- trials", ha='center', va='center', fontsize=18,
                transform=ax2.transAxes)
    
    # Create half-second tick marks
    half_sec_ticks = np.arange(np.floor(window[0] * 2) / 2, 
                            np.ceil(window[1] * 2) / 2 + 0.5, 
                            0.5)
    
    # Configure CS+ plot
    ax1.axvline(x=0, color=PNW_COLORS['mountain_dark'], linestyle='--', linewidth=2, label='CS Onset')
    ax1.axvline(x=3, color=PNW_COLORS['sunset_red'], linestyle='--', linewidth=2, label='Reward')
    ax1.set_xlabel('Time from CS Onset (s)', fontsize=16)
    ax1.set_ylabel('Trial', fontsize=16)
    ax1.set_title(f"CS+ Trials", fontsize=20, fontweight='bold')
    ax1.set_xlim(window)
    ax1.set_xticks(half_sec_ticks)
    ax1.set_xticklabels([f"{t:.1f}" for t in half_sec_ticks], rotation=45)
    
    # Configure CS- plot
    ax2.axvline(x=0, color=PNW_COLORS['mountain_dark'], linestyle='--', linewidth=2, label='CS Onset')
    ax2.set_xlabel('Time from CS Onset (s)', fontsize=16)
    ax2.set_ylabel('Trial', fontsize=16)
    ax2.set_title(f"CS- Trials", fontsize=20, fontweight='bold')
    ax2.set_xlim(window)
    ax2.set_xticks(half_sec_ticks)
    ax2.set_xticklabels([f"{t:.1f}" for t in half_sec_ticks], rotation=45)
    
    # Set y-limits and y-ticks for both plots
    for ax, data in zip([ax1, ax2], [cs_plus_licks, cs_minus_licks]):
        n_trials_plot = len(data)
        if n_trials_plot > 0:
            ax.set_ylim(-1, n_trials_plot)
            
            # Set y-ticks at 10-trial increments
            y_ticks = np.arange(0, n_trials_plot, 10)
            if n_trials_plot - 1 not in y_ticks:
                y_ticks = np.append(y_ticks, n_trials_plot - 1)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(int(y)) for y in y_ticks])
        
        # Increase font size for tick labels
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Create legend with larger font
        ax.legend(loc='upper right', fontsize=14)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add overall title
    fig.suptitle("Lick Raster", fontsize=24, fontweight='bold', y=0.98)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def plot_average_traces(data_dict: Dict, 
                       time_points: np.ndarray,
                       session_df: pd.DataFrame,
                       window: Tuple[float, float] = (-2, 5),
                       baseline_window: Tuple[float, float] = (-2, 0),
                       zscore: bool = True) -> plt.Figure:
    """
    Plot average traces for CS+ and CS- trials
    
    Args:
        data_dict: Dictionary with keys for each data type to plot
                   (e.g., 'rpe', 'value', 'dopamine')
        time_points: Time vector
        session_df: Session data
        window: Time window for plotting
        baseline_window: Window for baseline normalization
        zscore: Whether to z-score data
        
    Returns:
        Figure object
    """
    # Get trial types
    cs_plus_mask = session_df['trial_type'] == 'CS+'
    cs_minus_mask = ~cs_plus_mask
    
    # Filter time window
    in_window = (time_points >= window[0]) & (time_points <= window[1])
    plot_time = time_points[in_window]
    
    # Get number of data types to plot
    n_plots = len(data_dict)
    
    # Use seaborn style
    sns.set_style("white")
    
    # Create figure with larger dimensions
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 6*n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    # Create half-second tick marks
    half_sec_ticks = np.arange(np.floor(window[0] * 2) / 2, 
                             np.ceil(window[1] * 2) / 2 + 0.5, 
                             0.5)
    
    # For each data type
    for i, (data_name, data) in enumerate(data_dict.items()):
        ax = axes[i]
        
        # Filter data to window
        plot_data = data[:, in_window]
        
        # Get baseline window
        baseline_mask = (plot_time >= baseline_window[0]) & (plot_time <= baseline_window[1])
        
        if zscore:
            # Z-score data based on baseline
            baseline_mean = np.mean(plot_data[:, baseline_mask], axis=1, keepdims=True)
            baseline_std = np.std(plot_data[:, baseline_mask], axis=1, keepdims=True)
            baseline_std[baseline_std == 0] = 1  # Avoid division by zero
            plot_data = (plot_data - baseline_mean) / baseline_std
        
        # Calculate mean and SEM for CS+ trials
        cs_plus_mean = np.mean(plot_data[cs_plus_mask], axis=0)
        cs_plus_sem = np.std(plot_data[cs_plus_mask], axis=0) / np.sqrt(np.sum(cs_plus_mask))
        
        # Calculate mean and SEM for CS- trials
        cs_minus_mean = np.mean(plot_data[cs_minus_mask], axis=0)
        cs_minus_sem = np.std(plot_data[cs_minus_mask], axis=0) / np.sqrt(np.sum(cs_minus_mask))
        
        # Create DataFrames for seaborn plotting
        df_cs_plus = pd.DataFrame({
            'Time': plot_time,
            'Value': cs_plus_mean,
            'SEM': cs_plus_sem,
            'Trial Type': 'CS+'
        })
        
        df_cs_minus = pd.DataFrame({
            'Time': plot_time,
            'Value': cs_minus_mean,
            'SEM': cs_minus_sem,
            'Trial Type': 'CS-'
        })
        
        # Combine the data
        plot_df = pd.concat([df_cs_plus, df_cs_minus])
        
        # Plot CS+ mean ± SEM with enhanced styling and PNW theme
        sns.lineplot(data=plot_df, x='Time', y='Value', hue='Trial Type', 
                    palette={'CS+': PNW_COLORS['sunset_orange'], 'CS-': PNW_COLORS['water_medium']}, 
                    ax=ax, linewidth=3)
        
        # Add SEM as shaded area
        for trial_type, color in zip(['CS+', 'CS-'], 
                                    [PNW_COLORS['sunset_orange'], PNW_COLORS['water_medium']]):
            subset = plot_df[plot_df['Trial Type'] == trial_type]
            ax.fill_between(subset['Time'], 
                          subset['Value'] - subset['SEM'],
                          subset['Value'] + subset['SEM'],
                          color=color, alpha=0.3)
        
        # Add event markers
        ax.axvline(x=0, color=PNW_COLORS['mountain_dark'], linestyle='--', linewidth=2, label='CS Onset')
        ax.axvline(x=3, color=PNW_COLORS['sunset_red'], linestyle='--', linewidth=2, label='Reward (CS+ only)')
        
        # Add zero line
        ax.axhline(y=0, color=PNW_COLORS['mountain_medium'], linestyle='-', alpha=0.3, linewidth=1.5)
        
        # Improve labels with larger fonts
        if zscore:
            ax.set_ylabel(f"{data_name.capitalize()} (z-score)", fontsize=16)
        else:
            ax.set_ylabel(f"{data_name.capitalize()}", fontsize=16)
        
        # Add title with larger font
        ax.set_title(f"Average {data_name.capitalize()} Trace", fontsize=20, fontweight='bold')
        
        # Set x-ticks at half-second intervals
        ax.set_xticks(half_sec_ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in half_sec_ticks], rotation=45)
        
        # Increase legend font size
        legend = ax.legend(fontsize=14, frameon=True, facecolor='white', framealpha=0.9)
        
        # Customize formatting for all axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=14)
        ax.grid(True, linestyle='--', alpha=0.4)
    
    # Set common x label only on bottom plot with larger font
    axes[-1].set_xlabel('Time from CS Onset (s)', fontsize=16)
    
    # Add overall title if more than one plot
    if n_plots > 1:
        fig.suptitle("Average Response Traces", fontsize=24, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    
    return fig

def convert_lick_times_to_raster(lick_times_df: pd.DataFrame,
                               n_trials: int,
                               time_vector: np.ndarray) -> np.ndarray:
    """
    Convert lick times DataFrame to lick raster matrix
    
    Args:
        lick_times_df: DataFrame with trial_number and lick_time columns
        n_trials: Number of trials
        time_vector: Time vector for binning
        
    Returns:
        Boolean array of shape (n_trials, len(time_vector))
    """
    # Create empty lick raster
    lick_raster = np.zeros((n_trials, len(time_vector)), dtype=bool)
    
    # Fill in lick raster
    for _, row in lick_times_df.iterrows():
        trial = int(row['trial_number'])
        time = row['lick_time']
        
        # Find closest time bin
        time_idx = np.argmin(np.abs(time_vector - time))
        
        # Make sure trial index is valid
        if trial < n_trials:
            lick_raster[trial, time_idx] = True
    
    return lick_raster 