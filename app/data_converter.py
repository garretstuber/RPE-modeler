import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

st.set_page_config(
    page_title="TD Learning Data Converter",
    layout="wide",
)

st.title("🔄 TD Learning Data Converter")
st.markdown("""
This tool converts your behavioral and neural data into the format required by the TD Learning Visualization app.
""")

def create_output_dir():
    """Create output directory for converted files"""
    output_dir = Path("converted_data")
    output_dir.mkdir(exist_ok=True)
    return output_dir

# Create output directory
output_dir = create_output_dir()

# File uploader for the main pickle file
st.subheader("1. Upload Data File")
uploaded_file = st.file_uploader("Upload your pickle (.pkl) file containing behavioral data", type=['pkl'])

# Initialize session state for storing data
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.session_data = None
    st.session_state.cs_onsets = None
    st.session_state.reward_times = None
    st.session_state.dopamine_data = None
    st.session_state.time_vector = None
    st.session_state.lick_times = None

if uploaded_file is not None:
    try:
        # Load the pickle file if not already loaded
        if st.session_state.data is None:
            st.session_state.data = pickle.load(uploaded_file)
            st.success("Data loaded successfully!")
            
        # Display basic info about the data
        st.write(f"Data type: `{type(st.session_state.data)}`")
        
        if isinstance(st.session_state.data, pd.DataFrame):
            df = st.session_state.data
            st.write(f"DataFrame shape: {df.shape}")
            st.write("Columns:", list(df.columns))
            
            # Display a sample of the data
            with st.expander("Preview Data"):
                st.dataframe(df.head())
                
                # Show detailed column information
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Contains NaN': df.isna().any().values,
                    'Sample Values': [str(df[col].dropna().head(3).tolist()) if not df[col].empty else "" for col in df.columns]
                })
                st.write("Column Details:")
                st.dataframe(col_info)
            
            # Step 2: Data Analysis and Mapping
            st.subheader("2. Analyze and Map Data")
            
            # Add filtering options for animal and day
            if 'animal' in df.columns and 'day' in df.columns:
                with st.expander("Filter Data"):
                    # Get unique animals and days
                    animals = sorted(df['animal'].unique())
                    days = sorted(df['day'].unique())
                    
                    # Create filter dropdowns
                    selected_animal = st.selectbox("Select Animal", options=["All"] + list(animals))
                    selected_day = st.selectbox("Select Day", options=["All"] + list(days))
                    
                    # Apply filters
                    filtered = False
                    if selected_animal != "All":
                        df = df[df['animal'] == selected_animal]
                        filtered = True
                    
                    if selected_day != "All":
                        df = df[df['day'] == selected_day]
                        filtered = True
                    
                    if filtered:
                        st.success(f"Data filtered: {len(df)} events remaining")
                        st.write("Filtered data preview:")
                        st.dataframe(df.head())
            
            # Analyze the event structure
            if 'event' in df.columns:
                events = df['event'].unique()
                st.write(f"Found {len(events)} unique event types:")
                st.write(", ".join([f"`{e}`" for e in events]))
                
                # Allow user to map events to TD Learning concepts
                st.write("Please map your event types to TD Learning concepts:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    trial_start_event = st.selectbox(
                        "Trial start event",
                        options=events,
                        index=events.tolist().index("trial_start") if "trial_start" in events else 0
                    )
                    
                    odor_start_event = st.selectbox(
                        "Odor/cue onset event",
                        options=events,
                        index=events.tolist().index("odor_start") if "odor_start" in events else 0
                    )
                    
                    reward_event = st.selectbox(
                        "Reward delivery event",
                        options=events,
                        index=events.tolist().index("sol_open") if "sol_open" in events else 0
                    )
                
                with col2:
                    lick_event = st.selectbox(
                        "Lick event (optional)",
                        options=list(events) + ["N/A - No lick data"],
                        index=events.tolist().index("lick") if "lick" in events else 0
                    )
                    
                    # Check if 'timestamp' is in milliseconds or seconds
                    timestamp_unit = st.radio(
                        "Timestamp unit",
                        options=["seconds", "milliseconds"],
                        index=0
                    )
                    
                    # Time window around CS onset
                    time_window = st.slider(
                        "Time window around CS onset (seconds)",
                        min_value=1.0,
                        max_value=10.0,
                        value=(2.0, 5.0),
                        step=0.5
                    )
                
                # Add trial_identity mapping section if column exists
                if 'trial_identity' in df.columns:
                    st.write("### Trial Identity Mapping")
                    st.write("Your data contains a trial_identity column that can be used to determine CS+ and CS- trials.")
                    
                    # Get unique values and show detailed information
                    trial_id_values = df['trial_identity'].unique()
                    
                    # Show table with trial identity counts
                    trial_id_counts = df['trial_identity'].value_counts().reset_index()
                    trial_id_counts.columns = ['Trial Identity Value', 'Count']
                    st.write("Trial identity distribution:")
                    st.dataframe(trial_id_counts)
                    
                    # Create trial identity mapping
                    st.write("#### Select which trial identity values represent CS+ and CS- trials:")
                    
                    # Allow binary 0/1 selection if there are exactly 2 values
                    if len(trial_id_values) == 2:
                        # Sort the values to ensure consistent ordering
                        sorted_values = sorted([str(v) for v in trial_id_values])
                        
                        # Determine which is CS+ and which is CS-
                        cs_mapping = st.radio(
                            "Trial Identity Mapping",
                            options=[
                                f"{sorted_values[0]} = CS+, {sorted_values[1]} = CS-",
                                f"{sorted_values[1]} = CS+, {sorted_values[0]} = CS-",
                            ],
                            index=0
                        )
                        
                        # Extract mapping
                        if cs_mapping.startswith(f"{sorted_values[0]} = CS+"):
                            cs_plus_identity = sorted_values[0]
                            cs_minus_identity = sorted_values[1]
                        else:
                            cs_plus_identity = sorted_values[1]
                            cs_minus_identity = sorted_values[0]
                            
                        st.success(f"CS+ trials identified by trial_identity = {cs_plus_identity}")
                        st.success(f"CS- trials identified by trial_identity = {cs_minus_identity}")
                    else:
                        # If more than 2 values, provide dropdown selection
                        sorted_values = sorted([str(v) for v in trial_id_values])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            cs_plus_identity = st.selectbox(
                                "Which trial_identity value represents CS+ (rewarded) trials?",
                                options=sorted_values,
                                index=0
                            )
                        with col2:
                            filtered_values = [v for v in sorted_values if v != cs_plus_identity]
                            cs_minus_identity = st.selectbox(
                                "Which trial_identity value represents CS- (unrewarded) trials?",
                                options=filtered_values,
                                index=0 if filtered_values else None
                            )
                    
                    # Show the number of trials expected for each type
                    cs_plus_count = len(df[df['trial_identity'].astype(str) == cs_plus_identity])
                    cs_minus_count = len(df[df['trial_identity'].astype(str) == cs_minus_identity])
                    
                    st.write(f"Found {cs_plus_count} events with trial_identity = {cs_plus_identity} (CS+)")
                    st.write(f"Found {cs_minus_count} events with trial_identity = {cs_minus_identity} (CS-)")
                
                # Button to analyze the data based on user selections
                if st.button("Analyze Data"):
                    with st.spinner("Analyzing data structure..."):
                        # Record original DataFrame size
                        st.write(f"Starting with {len(df)} total events")
                        
                        # Debug column information
                        st.write("Columns available:", list(df.columns))
                        
                        # Add special handling for trial_identity column
                        if 'trial_identity' in df.columns:
                            trial_id_values = df['trial_identity'].value_counts()
                            st.write("Trial identity value counts:")
                            st.write(trial_id_values)
                        
                        # Convert timestamps if needed
                        conversion_factor = 1.0 if timestamp_unit == "seconds" else 0.001
                        
                        # Check if we have lick data
                        has_lick_data = lick_event != "N/A - No lick data"
                        
                        # Convert timestamps
                        if 'timestamp' in df.columns:
                            # Add timestamp_sec column to the main dataframe first
                            df['timestamp_sec'] = df['timestamp'] * conversion_factor
                            
                            # Create a copy of the DataFrame with string-converted trial_identity for consistent comparisons
                            if 'trial_identity' in df.columns:
                                df['trial_identity_str'] = df['trial_identity'].astype(str)
                            
                            # Extract trial events
                            trial_start_data = df[df['event'] == trial_start_event].copy()
                            odor_start_data = df[df['event'] == odor_start_event].copy()
                            reward_data = df[df['event'] == reward_event].copy()
                            
                            # Show event counts
                            st.write(f"Found {len(trial_start_data)} {trial_start_event} events")
                            st.write(f"Found {len(odor_start_data)} {odor_start_event} events")
                            st.write(f"Found {len(reward_data)} {reward_event} events")
                            
                            # If using trial_identity, filter odor events by trial type
                            using_trial_identity = 'trial_identity' in df.columns and 'cs_plus_identity' in locals()
                            if using_trial_identity:
                                st.write("### Filtering odor events by trial_identity")
                                
                                # Filter odor events by trial identity
                                cs_plus_odors = odor_start_data[odor_start_data['trial_identity_str'] == cs_plus_identity].copy()
                                cs_minus_odors = odor_start_data[odor_start_data['trial_identity_str'] == cs_minus_identity].copy()
                                
                                st.write(f"Found {len(cs_plus_odors)} CS+ odor events")
                                st.write(f"Found {len(cs_minus_odors)} CS- odor events")
                            
                            if has_lick_data:
                                lick_data = df[df['event'] == lick_event].copy()
                                st.write(f"Found {len(lick_data)} {lick_event} events")
                                
                            # Session Construction Strategy
                            st.write("### Session Construction")
                            construction_method = st.radio(
                                "How should trials be constructed?",
                                options=[
                                    "Using trial_start events with trial_identity",
                                    "Using only odor events with trial_identity",
                                    "Simple - use all odor events sequentially"
                                ],
                                index=0 if using_trial_identity else 2
                            )
                            
                            # Add option to limit the number of trials
                            max_trials_col1, max_trials_col2 = st.columns(2)
                            
                            with max_trials_col1:
                                limit_trials = st.checkbox("Limit number of trials", value=True)
                            
                            with max_trials_col2:
                                if limit_trials:
                                    max_cs_plus = st.number_input("Max CS+ trials", min_value=1, max_value=1000, value=50)
                                    max_cs_minus = st.number_input("Max CS- trials", min_value=1, max_value=1000, value=50)
                                else:
                                    max_cs_plus = None
                                    max_cs_minus = None
                        
                        # Now construct the session data
                        session_rows = []
                        
                        if construction_method == "Using trial_start events with trial_identity" and using_trial_identity:
                            st.write("Constructing trials using trial_start events and trial_identity...")
                            
                            # Create temporary lists to hold the trial data before limiting
                            cs_plus_trials = []
                            cs_minus_trials = []
                            
                            # Process all trials
                            for trial_idx, trial_row in trial_start_data.iterrows():
                                trial_time = trial_row['timestamp_sec']
                                
                                # Use trial_count if available
                                if 'trial_count' in trial_row and not pd.isna(trial_row['trial_count']):
                                    trial_num = int(trial_row['trial_count'])
                                else:
                                    trial_num = trial_idx
                                
                                # Find odor onset after trial start
                                odor_mask = (odor_start_data['timestamp_sec'] > trial_time) & (odor_start_data['timestamp_sec'] <= trial_time + 5)
                                odor_rows = odor_start_data[odor_mask]
                                
                                if len(odor_rows) == 0:
                                    # Skip trials with no odor onset
                                    continue
                                    
                                # Get the first odor onset time
                                odor_row = odor_rows.iloc[0]
                                odor_time = odor_row['timestamp_sec']
                                
                                # Determine CS type by trial_identity
                                if 'trial_identity_str' in odor_row:
                                    trial_identity = odor_row['trial_identity_str']
                                    if trial_identity == cs_plus_identity:
                                        trial_type = 'CS+'
                                        # Look for reward
                                        reward_mask = (reward_data['timestamp_sec'] > odor_time) & (reward_data['timestamp_sec'] <= odor_time + 5)
                                        reward_rows = reward_data[reward_mask]
                                        
                                        if len(reward_rows) > 0:
                                            reward_time = reward_rows.iloc[0]['timestamp_sec']
                                            reward_value = 1
                                        else:
                                            reward_time = None
                                            reward_value = 0
                                            
                                        # Calculate ITI
                                        next_trial_mask = (trial_start_data['timestamp_sec'] > trial_time)
                                        if next_trial_mask.any():
                                            next_trial_time = trial_start_data[next_trial_mask].iloc[0]['timestamp_sec']
                                            iti = next_trial_time - trial_time
                                        else:
                                            iti = 15.0  # Default ITI if this is the last trial
                                        
                                        # Add to CS+ trials list
                                        cs_plus_trials.append({
                                            'trial_number': trial_num,
                                            'trial_type': trial_type,
                                            'cs_onset': odor_time,
                                            'reward_time': reward_time,
                                            'reward': reward_value,
                                            'iti': iti
                                        })
                                        
                                    elif trial_identity == cs_minus_identity:
                                        trial_type = 'CS-'
                                        reward_time = None
                                        reward_value = 0
                                        
                                        # Calculate ITI
                                        next_trial_mask = (trial_start_data['timestamp_sec'] > trial_time)
                                        if next_trial_mask.any():
                                            next_trial_time = trial_start_data[next_trial_mask].iloc[0]['timestamp_sec']
                                            iti = next_trial_time - trial_time
                                        else:
                                            iti = 15.0  # Default ITI if this is the last trial
                                        
                                        # Add to CS- trials list
                                        cs_minus_trials.append({
                                            'trial_number': trial_num,
                                            'trial_type': trial_type,
                                            'cs_onset': odor_time,
                                            'reward_time': reward_time,
                                            'reward': reward_value,
                                            'iti': iti
                                        })
                                    else:
                                        # Skip trials with unknown identity
                                        continue
                                else:
                                    # Skip trials without identity
                                    continue
                            
                            # Limit trials if requested
                            if limit_trials and max_cs_plus is not None:
                                # Sort by CS onset time
                                cs_plus_trials.sort(key=lambda x: x['cs_onset'])
                                cs_minus_trials.sort(key=lambda x: x['cs_onset'])
                                
                                # Limit to max trials
                                cs_plus_trials = cs_plus_trials[:max_cs_plus]
                                cs_minus_trials = cs_minus_trials[:max_cs_minus]
                                
                                st.info(f"Limited to first {len(cs_plus_trials)} CS+ trials and {len(cs_minus_trials)} CS- trials")
                            
                            # Combine and add to session_rows
                            session_rows.extend(cs_plus_trials)
                            session_rows.extend(cs_minus_trials)
                        
                        elif construction_method == "Using only odor events with trial_identity" and using_trial_identity:
                            st.write("Constructing trials using odor events and trial_identity...")
                            
                            # Limit the number of odor events if specified
                            if limit_trials and max_cs_plus is not None:
                                # Sort odor events by timestamp to ensure chronological order
                                cs_plus_odors = cs_plus_odors.sort_values('timestamp_sec')
                                cs_minus_odors = cs_minus_odors.sort_values('timestamp_sec')
                                
                                # Limit to the specified number of trials
                                cs_plus_odors = cs_plus_odors.head(max_cs_plus)
                                cs_minus_odors = cs_minus_odors.head(max_cs_minus)
                                
                                st.info(f"Limited to first {len(cs_plus_odors)} CS+ trials and {len(cs_minus_odors)} CS- trials")
                            
                            # Process CS+ trials
                            for idx, odor_row in cs_plus_odors.iterrows():
                                odor_time = odor_row['timestamp_sec']
                                
                                # Use trial_count if available
                                if 'trial_count' in odor_row and not pd.isna(odor_row['trial_count']):
                                    trial_num = int(odor_row['trial_count'])
                                else:
                                    trial_num = idx
                                
                                # Look for reward
                                reward_mask = (reward_data['timestamp_sec'] > odor_time) & (reward_data['timestamp_sec'] <= odor_time + 5)
                                reward_rows = reward_data[reward_mask]
                                
                                if len(reward_rows) > 0:
                                    reward_time = reward_rows.iloc[0]['timestamp_sec']
                                    reward_value = 1
                                else:
                                    reward_time = None
                                    reward_value = 0
                                
                                # Calculate ITI
                                next_odor_mask = (odor_start_data['timestamp_sec'] > odor_time)
                                if next_odor_mask.any():
                                    next_odor_time = odor_start_data[next_odor_mask].iloc[0]['timestamp_sec']
                                    iti = next_odor_time - odor_time
                                else:
                                    iti = 15.0  # Default ITI if this is the last trial
                                
                                # Add to session data
                                session_rows.append({
                                    'trial_number': trial_num,
                                    'trial_type': 'CS+',
                                    'cs_onset': odor_time,
                                    'reward_time': reward_time,
                                    'reward': reward_value,
                                    'iti': iti
                                })
                            
                            # Process CS- trials
                            for idx, odor_row in cs_minus_odors.iterrows():
                                odor_time = odor_row['timestamp_sec']
                                
                                # Use trial_count if available
                                if 'trial_count' in odor_row and not pd.isna(odor_row['trial_count']):
                                    trial_num = int(odor_row['trial_count'])
                                else:
                                    trial_num = idx
                                
                                # Calculate ITI
                                next_odor_mask = (odor_start_data['timestamp_sec'] > odor_time)
                                if next_odor_mask.any():
                                    next_odor_time = odor_start_data[next_odor_mask].iloc[0]['timestamp_sec']
                                    iti = next_odor_time - odor_time
                                else:
                                    iti = 15.0  # Default ITI if this is the last trial
                                
                                # Add to session data
                                session_rows.append({
                                    'trial_number': trial_num,
                                    'trial_type': 'CS-',
                                    'cs_onset': odor_time,
                                    'reward_time': None,
                                    'reward': 0,
                                    'iti': iti
                                })
                                
                        else:
                            st.write("Using simple sequential construction of trials...")
                            
                            # Create temporary lists to hold trial data by type
                            cs_plus_trials = []
                            cs_minus_trials = []
                            
                            # Process all odor events sequentially
                            for idx, odor_row in odor_start_data.iterrows():
                                odor_time = odor_row['timestamp_sec']
                                
                                # Use trial_count if available
                                if 'trial_count' in odor_row and not pd.isna(odor_row['trial_count']):
                                    trial_num = int(odor_row['trial_count'])
                                else:
                                    trial_num = idx
                                
                                # Determine trial type based on rewards
                                reward_mask = (reward_data['timestamp_sec'] > odor_time) & (reward_data['timestamp_sec'] <= odor_time + 5)
                                reward_rows = reward_data[reward_mask]
                                
                                # Calculate ITI
                                next_odor_mask = (odor_start_data['timestamp_sec'] > odor_time)
                                if next_odor_mask.any():
                                    next_odor_time = odor_start_data[next_odor_mask].iloc[0]['timestamp_sec']
                                    iti = next_odor_time - odor_time
                                else:
                                    iti = 15.0  # Default ITI if this is the last trial
                                
                                if len(reward_rows) > 0:
                                    # This is a CS+ trial
                                    trial_type = 'CS+'
                                    reward_time = reward_rows.iloc[0]['timestamp_sec']
                                    reward_value = 1
                                    
                                    cs_plus_trials.append({
                                        'trial_number': trial_num,
                                        'trial_type': trial_type,
                                        'cs_onset': odor_time,
                                        'reward_time': reward_time,
                                        'reward': reward_value,
                                        'iti': iti
                                    })
                                else:
                                    # This is a CS- trial
                                    trial_type = 'CS-'
                                    reward_time = None
                                    reward_value = 0
                                    
                                    cs_minus_trials.append({
                                        'trial_number': trial_num,
                                        'trial_type': trial_type,
                                        'cs_onset': odor_time,
                                        'reward_time': reward_time,
                                        'reward': reward_value,
                                        'iti': iti
                                    })
                            
                            # Limit the number of trials if requested
                            if limit_trials and max_cs_plus is not None:
                                # Sort by CS onset time
                                cs_plus_trials.sort(key=lambda x: x['cs_onset'])
                                cs_minus_trials.sort(key=lambda x: x['cs_onset'])
                                
                                # Limit to max trials
                                cs_plus_trials = cs_plus_trials[:max_cs_plus]
                                cs_minus_trials = cs_minus_trials[:max_cs_minus]
                                
                                st.info(f"Limited to first {len(cs_plus_trials)} CS+ trials and {len(cs_minus_trials)} CS- trials")
                            
                            # Combine and add to session_rows
                            session_rows.extend(cs_plus_trials)
                            session_rows.extend(cs_minus_trials)
                        
                        # Create session DataFrame and sort by CS onset time
                        session_df = pd.DataFrame(session_rows)
                        session_df = session_df.sort_values('cs_onset').reset_index(drop=True)
                        
                        # Reassign trial numbers sequentially
                        session_df['trial_number'] = np.arange(len(session_df))
                        
                        # Store session data in session state
                        st.session_state.session_data = session_df
                        
                        # Compile lick times if available
                        if has_lick_data:
                            lick_rows = []
                            
                            # For each trial in the session data
                            for trial_idx, trial_row in session_df.iterrows():
                                # Get the odor (CS) onset time for this trial
                                cs_time = trial_row['cs_onset']
                                
                                # Find licks in relevant window around CS onset
                                window_start = cs_time - time_window[0]
                                window_end = cs_time + time_window[1]
                                
                                trial_licks = lick_data[
                                    (lick_data['timestamp_sec'] >= window_start) & 
                                    (lick_data['timestamp_sec'] <= window_end)
                                ]
                                
                                # Add to lick dataframe with times aligned to CS onset
                                for _, lick in trial_licks.iterrows():
                                    lick_time_aligned = lick['timestamp_sec'] - cs_time  # Align to CS onset
                                    lick_rows.append({
                                        'trial_number': trial_idx,
                                        'lick_time': lick_time_aligned
                                    })
                            
                            if lick_rows:
                                st.session_state.lick_times = pd.DataFrame(lick_rows)
                                st.write(f"Extracted {len(lick_rows)} lick events across {len(session_df)} trials.")
                            else:
                                st.warning("No lick events found in the specified time windows.")
                        
                        # Display results
                        st.success(f"Analysis complete! Found {len(session_df)} trials ({sum(session_df['trial_type'] == 'CS+')} CS+ and {sum(session_df['trial_type'] == 'CS-')} CS-)")
                        
                        # Display parsed session data
                        st.write("### Extracted Session Data")
                        st.dataframe(session_df)
                        
                        if has_lick_data and st.session_state.lick_times is not None:
                            st.write(f"### Extracted Lick Events")
                            st.dataframe(st.session_state.lick_times.head(20))
                            
                            # Plot lick histogram
                            st.write("### Lick Distribution")
                            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                            
                            # Separate CS+ and CS- trials
                            cs_plus_trials = session_df[session_df['trial_type'] == 'CS+']['trial_number'].values
                            cs_minus_trials = session_df[session_df['trial_type'] == 'CS-']['trial_number'].values
                            
                            # Get licks for each trial type
                            cs_plus_licks = st.session_state.lick_times[st.session_state.lick_times['trial_number'].isin(cs_plus_trials)]['lick_time'].values
                            cs_minus_licks = st.session_state.lick_times[st.session_state.lick_times['trial_number'].isin(cs_minus_trials)]['lick_time'].values
                            
                            # Plot histograms
                            bins = np.linspace(-time_window[0], time_window[1], 50)
                            ax.hist(cs_plus_licks, bins=bins, alpha=0.5, label='CS+', color='red')
                            ax.hist(cs_minus_licks, bins=bins, alpha=0.5, label='CS-', color='blue')
                            
                            # Add vertical lines
                            ax.axvline(x=0, color='k', linestyle='--', label='CS Onset')
                            if np.any(session_df['reward'] == 1):
                                # Find average reward time relative to CS onset
                                mean_reward_time = np.mean(session_df[session_df['reward'] == 1]['reward_time'].values - 
                                                          session_df[session_df['reward'] == 1]['cs_onset'].values)
                                ax.axvline(x=mean_reward_time, color='g', linestyle='--', label=f'Reward ({mean_reward_time:.2f}s)')
                            
                            ax.set_xlabel('Time from CS Onset (s)')
                            ax.set_ylabel('Lick Count')
                            ax.set_title('Lick Distribution Relative to CS Onset')
                            ax.legend()
                            
                            st.pyplot(fig)
            
            # Step 3: Optional - Neural Data
            st.subheader("3. Neural Data (Optional)")
            
            # Option to upload neural data
            has_neural_data = st.checkbox("I have neural recording data")
            
            if has_neural_data:
                neural_data_type = st.radio(
                    "Neural data format",
                    options=["NumPy array (.npy)", "CSV file", "None/Skip"],
                    index=2
                )
                
                if neural_data_type != "None/Skip":
                    if neural_data_type == "NumPy array (.npy)":
                        neural_file = st.file_uploader("Upload neural data file", type=['npy'])
                        time_file = st.file_uploader("Upload time vector file (optional)", type=['npy'])
                        
                        if neural_file is not None:
                            # Load neural data
                            try:
                                # Save temporarily
                                with open('temp_neural.npy', 'wb') as f:
                                    f.write(neural_file.getbuffer())
                                
                                neural_data = np.load('temp_neural.npy')
                                os.remove('temp_neural.npy')
                                
                                st.write(f"Neural data shape: {neural_data.shape}")
                                
                                # Store
                                st.session_state.dopamine_data = neural_data
                                
                                # Load time vector if available
                                if time_file is not None:
                                    with open('temp_time.npy', 'wb') as f:
                                        f.write(time_file.getbuffer())
                                    
                                    time_vector = np.load('temp_time.npy')
                                    os.remove('temp_time.npy')
                                    
                                    st.write(f"Time vector shape: {time_vector.shape}")
                                    st.session_state.time_vector = time_vector
                            
                            except Exception as e:
                                st.error(f"Error loading neural data: {str(e)}")
                    
                    elif neural_data_type == "CSV file":
                        neural_file = st.file_uploader("Upload neural data CSV", type=['csv'])
                        
                        if neural_file is not None:
                            try:
                                neural_df = pd.read_csv(neural_file)
                                st.write(f"Neural data shape: {neural_df.shape}")
                                
                                # Convert to NumPy array - assuming first column might be time
                                if st.checkbox("First column is time vector"):
                                    time_col = neural_df.columns[0]
                                    data_cols = neural_df.columns[1:]
                                    
                                    st.session_state.time_vector = neural_df[time_col].values
                                    st.session_state.dopamine_data = neural_df[data_cols].values.T  # Transpose to trials × timepoints
                                else:
                                    st.session_state.dopamine_data = neural_df.values.T  # Transpose to trials × timepoints
                                
                                st.write(f"Converted neural data shape: {st.session_state.dopamine_data.shape}")
                                
                            except Exception as e:
                                st.error(f"Error loading neural data CSV: {str(e)}")
            
            # Step 4: Generate Files
            st.subheader("4. Generate Files for TD Learning App")
            
            if st.session_state.session_data is not None:
                if st.button("Generate Files"):
                    with st.spinner("Generating files..."):
                        # Create timestamped output directory
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        dataset_dir = output_dir / f"converted_dataset_{timestamp}"
                        dataset_dir.mkdir(exist_ok=True)
                        
                        # Save session data
                        session_file = dataset_dir / "session_data.csv"
                        st.session_state.session_data.to_csv(session_file, index=False)
                        
                        # Save lick times if available
                        if st.session_state.lick_times is not None:
                            lick_file = dataset_dir / "lick_times.csv"
                            st.session_state.lick_times.to_csv(lick_file, index=False)
                        
                        # Save neural data if available
                        if st.session_state.dopamine_data is not None:
                            dopamine_file = dataset_dir / "dopamine_data.npy"
                            np.save(dopamine_file, st.session_state.dopamine_data)
                            
                            if st.session_state.time_vector is not None:
                                time_file = dataset_dir / "dopamine_time.npy"
                                np.save(time_file, st.session_state.time_vector)
                        
                        # Create a README file
                        with open(dataset_dir / "README.txt", "w") as f:
                            f.write(f"TD Learning Dataset\n")
                            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                            f.write(f"Files included:\n")
                            f.write(f"- session_data.csv: Trial information\n")
                            if st.session_state.lick_times is not None:
                                f.write(f"- lick_times.csv: Lick event times\n")
                            if st.session_state.dopamine_data is not None:
                                f.write(f"- dopamine_data.npy: Neural activity data\n")
                            if st.session_state.time_vector is not None:
                                f.write(f"- dopamine_time.npy: Time vector for neural data\n")
                        
                        st.success(f"Files generated successfully in {dataset_dir}!")
                        
                        # Create a zip file for easy download
                        import shutil
                        zip_path = dataset_dir.with_suffix('.zip')
                        shutil.make_archive(zip_path.with_suffix(''), 'zip', dataset_dir)
                        
                        # Create download button
                        with open(zip_path, 'rb') as f:
                            st.download_button(
                                label="Download Converted Dataset",
                                data=f.read(),
                                file_name=f"td_learning_dataset_{timestamp}.zip",
                                mime="application/zip"
                            )
                        
                        # Instructions for using the files
                        st.info("""
                        ## Next Steps
                        
                        1. Download the converted dataset
                        2. Extract the ZIP file
                        3. Open the main TD Learning app
                        4. Upload the generated files:
                           - session_data.csv (required)
                           - dopamine_data.npy (optional)
                           - dopamine_time.npy (optional)
                           - lick_times.csv (optional)
                        """)
                        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.exception(e)
else:
    st.info("Please upload your data file to begin.")
    
    # Show an example of what the converted files will look like
    with st.expander("Example of converted files"):
        st.markdown("""
        ### session_data.csv
        ```
        trial_number,trial_type,cs_onset,reward_time,reward,iti
        0,CS+,0.0,3.0,1,16.2
        1,CS-,16.2,,0,13.8
        2,CS+,30.0,33.0,1,14.5
        ```
        
        ### lick_times.csv
        ```
        trial_number,lick_time
        0,-0.2
        0,1.3
        1,0.1
        ```
        
        ### dopamine_data.npy
        A NumPy array with shape (n_trials, n_timepoints)
        
        ### dopamine_time.npy
        A 1D NumPy array with shape (n_timepoints,) representing seconds
        """) 