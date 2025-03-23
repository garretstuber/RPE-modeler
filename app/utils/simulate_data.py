import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from typing import Tuple, Dict, List, Optional

def create_pavlovian_session(n_trials: int = 100, 
                            cs_duration: float = 2.0,
                            trace_interval: float = 1.0,
                            reward_delay: float = 3.0,
                            mean_iti: float = 15.0, 
                            iti_std: float = 2.0,
                            reward_probability: float = 1.0) -> pd.DataFrame:
    """
    Create a simulated Pavlovian conditioning session with CS+ and CS- trials
    
    Args:
        n_trials: Number of trials in session
        cs_duration: Duration of CS presentation in seconds
        trace_interval: Time between CS offset and reward onset
        reward_delay: Time from CS onset to reward delivery
        mean_iti: Mean inter-trial interval in seconds
        iti_std: Standard deviation of ITI
        reward_probability: Probability of reward delivery on CS+ trials (0-1)
        
    Returns:
        DataFrame with columns: trial_number, trial_type, cs_onset, reward_time, reward, iti
    """
    # Create alternating CS+/CS- trials
    trial_types = []
    for i in range(n_trials):
        if i % 2 == 0:
            trial_types.append('CS+')
        else:
            trial_types.append('CS-')
    
    # Randomize ITIs from normal distribution
    itis = np.random.normal(mean_iti, iti_std, n_trials)
    itis = np.clip(itis, mean_iti - 2*iti_std, mean_iti + 2*iti_std)  # Clip extreme values
    
    # Calculate CS onset times
    cs_onsets = np.zeros(n_trials)
    for i in range(1, n_trials):
        cs_onsets[i] = cs_onsets[i-1] + itis[i-1]
    
    # Initialize reward times (will be set for CS+ trials only)
    reward_times = np.full(n_trials, np.nan)
    
    # Default to NaN for CS- trials
    rewards = np.zeros(n_trials)
    
    # Apply probabilistic reward delivery for CS+ trials
    for i in range(n_trials):
        if trial_types[i] == 'CS+':
            if np.random.random() < reward_probability:
                # This CS+ trial gets a reward
                reward_times[i] = cs_onsets[i] + reward_delay
                rewards[i] = 1
    
    # Create dataframe
    session_df = pd.DataFrame({
        'trial_number': np.arange(n_trials),
        'trial_type': trial_types,
        'cs_onset': cs_onsets,
        'reward_time': reward_times,
        'reward': rewards,
        'iti': itis
    })
    
    return session_df

def simulate_td_model(session_df: pd.DataFrame, 
                     alpha: float = 0.2, 
                     gamma: float = 0.95,
                     time_resolution: float = 0.05,
                     cs_duration: float = 2.0,
                     reward_delay: float = 3.0,
                     trial_window: Tuple[float, float] = (-2.0, 5.0)) -> Dict:
    """
    Simulate Temporal Difference (TD) learning model based on Sutton & Barto
    
    Args:
        session_df: DataFrame with session info
        alpha: Learning rate (0 to 1)
        gamma: Discount factor (0 to 1)
        time_resolution: Time step size in seconds
        cs_duration: Duration of CS in seconds
        reward_delay: Time from CS onset to reward
        trial_window: Time window around CS onset for simulation
        
    Returns:
        Dictionary with model results:
            - time_points: time vector
            - values: state values for each trial and time point (shape: trials × time points)
            - rpes: reward prediction errors (shape: trials × time points)
            - cs_value: learned value of CS for each trial
    """
    n_trials = len(session_df)
    
    # Create time vector for simulation
    time_points = np.arange(trial_window[0], trial_window[1] + time_resolution, time_resolution)
    n_timepoints = len(time_points)
    
    # Initialize arrays
    values = np.zeros((n_trials, n_timepoints))
    rpes = np.zeros((n_trials, n_timepoints))
    cs_values = np.zeros(n_trials)  # Track the CS value over trials
    
    # Create feature representation (eligibility traces)
    # We'll use a simple stimulus representation where CS is active during presentation
    # plus a trace that extends after CS offset
    def get_features(t):
        features = np.zeros(2)  # [CS, trace]
        
        # CS feature (active during CS)
        if 0 <= t < cs_duration:
            features[0] = 1
        
        # Trace feature (active after CS, decaying)
        if cs_duration <= t < reward_delay:
            # Exponential decay from CS offset to reward
            decay_rate = 3.0  # Controls decay speed
            features[1] = np.exp(-decay_rate * (t - cs_duration))
            
        return features
    
    # Initialize weights (represent value function)
    weights = np.zeros(2)
    
    # Run TD learning algorithm
    for trial_idx in range(n_trials):
        trial_type = session_df.iloc[trial_idx]['trial_type']
        is_rewarded = trial_type == 'CS+'
        
        # For each time step in trial
        for t_idx in range(n_timepoints - 1):
            t = time_points[t_idx]
            next_t = time_points[t_idx + 1]
            
            # Get state features
            features = get_features(t - trial_window[0])
            next_features = get_features(next_t - trial_window[0])
            
            # Current state value
            current_value = np.dot(weights, features)
            
            # Next state value
            next_value = np.dot(weights, next_features)
            
            # Reward (only at reward time for CS+ trials)
            reward = 0
            if is_rewarded and (reward_delay - time_resolution/2 <= t - trial_window[0] < reward_delay + time_resolution/2):
                reward = 1
            
            # Calculate TD error (RPE)
            td_error = reward + gamma * next_value - current_value
            
            # Update weights
            weights += alpha * td_error * features
            
            # Store values and RPEs
            values[trial_idx, t_idx] = current_value
            rpes[trial_idx, t_idx] = td_error
        
        # Store final CS value for this trial
        cs_features = get_features(0)  # Features at CS onset
        cs_values[trial_idx] = np.dot(weights, cs_features)
    
    return {
        'time_points': time_points,
        'values': values,
        'rpes': rpes,
        'cs_value': cs_values
    }

def simulate_dopamine_from_rpe(rpe_data: np.ndarray, 
                             time_points: np.ndarray,
                             noise_level: float = 0.05,
                             kernel_width: float = 0.2,
                             gain: float = 1.2,
                             add_realistic_features: bool = True) -> np.ndarray:
    """
    Simulate biologically realistic dopamine transients based on RPE signals
    
    Args:
        rpe_data: RPE signals from TD model (trials × timepoints)
        time_points: Time vector
        noise_level: Amount of noise to add
        kernel_width: Width of Gaussian kernel in seconds
        gain: Amplification factor for RPE signal
        add_realistic_features: Add biological features like adaptation and baseline drift
        
    Returns:
        Simulated dopamine signals (trials × timepoints)
    """
    n_trials, n_timepoints = rpe_data.shape
    
    # Create Gaussian kernel for convolution
    time_resolution = time_points[1] - time_points[0]
    kernel_size = int(kernel_width / time_resolution)
    kernel = sp_signal.gaussian(kernel_size, std=kernel_size/5)
    kernel = kernel / np.sum(kernel)  # Normalize
    
    # Initialize dopamine array
    dopamine = np.zeros_like(rpe_data)
    
    # Process each trial
    for trial in range(n_trials):
        # Amplify RPE signal to create clearer correlation
        amplified_rpe = rpe_data[trial] * gain
        
        # Convolve RPE with kernel
        da_signal = sp_signal.convolve(amplified_rpe, kernel, mode='same')
        
        if add_realistic_features:
            # Add biologically realistic features
            
            # 1. Baseline drift (slow fluctuations)
            drift_freq = 0.2  # Hz
            drift_amp = 0.1
            drift = drift_amp * np.sin(2 * np.pi * drift_freq * time_points / np.ptp(time_points))
            
            # 2. Signal adaptation (response decreases with repeated stimulation)
            # Find peaks in signal to apply adaptation
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(da_signal, height=0.1, distance=int(0.5/time_resolution))
            
            # Apply adaptation to each peak
            for peak_idx in peaks:
                if peak_idx > 0:
                    # Calculate adaptation window after peak
                    adapt_window = np.arange(peak_idx, min(peak_idx + int(1.0/time_resolution), n_timepoints))
                    if len(adapt_window) > 0:
                        # Exponential decay adaptation
                        adapt_factor = np.exp(-np.arange(len(adapt_window)) * time_resolution * 3)
                        da_signal[adapt_window] *= adapt_factor
            
            # 3. Add small spontaneous transients
            n_transients = np.random.poisson(3)  # Random number of spontaneous events
            for _ in range(n_transients):
                # Random position and amplitude
                pos = np.random.randint(0, n_timepoints)
                amp = np.random.uniform(0.1, 0.3)
                width = int(np.random.uniform(2, 5) / time_resolution)
                
                # Create transient template
                transient = amp * np.exp(-np.arange(width) * time_resolution * 5)
                
                # Add to signal
                end_idx = min(pos + width, n_timepoints)
                da_signal[pos:end_idx] += transient[:end_idx-pos]
            
            # Add drift to signal
            da_signal += drift
        
        # Add noise - smaller noise level for more correlation
        # Use colored noise for more biological realism
        
        # Generate white noise
        white_noise = np.random.normal(0, noise_level, n_timepoints)
        
        # Create pink noise (1/f) using IIR filter
        b, a = sp_signal.butter(2, 0.1, btype='lowpass')
        colored_noise = sp_signal.filtfilt(b, a, white_noise)
        
        # Scale the colored noise to the desired level
        colored_noise = colored_noise * noise_level / np.std(colored_noise)
        
        da_signal += colored_noise
        
        dopamine[trial] = da_signal
    
    return dopamine

def simulate_lick_behavior(session_df: pd.DataFrame,
                          time_points: np.ndarray,
                          value_data: np.ndarray,
                          anticipation_threshold: float = 0.3,
                          reward_response_prob: float = 0.9,
                          spontaneous_rate: float = 0.01,  # Reduced spontaneous licking
                          bout_probability: float = 0.9,  # Higher bout probability
                          bout_length_range: tuple = (4, 10),  # Longer bouts
                          bout_interval_range: tuple = (0.2, 0.4),
                          well_learned: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Simulate realistic licking behavior based on learned values and trial type
    
    Args:
        session_df: Session information dataframe
        time_points: Time vector
        value_data: Value function from TD model (trials × timepoints)
        anticipation_threshold: Value threshold to trigger anticipatory licking
        reward_response_prob: Probability of licking in response to reward
        spontaneous_rate: Rate of spontaneous licking
        bout_probability: Probability of a lick being part of a bout
        bout_length_range: Range of possible lick bout lengths
        bout_interval_range: Range of inter-lick intervals within a bout (seconds)
        well_learned: Simulate behavior for a well-learned task
        
    Returns:
        Tuple of (lick_times_df, lick_raster)
    """
    n_trials = len(session_df)
    time_resolution = time_points[1] - time_points[0]
    n_timepoints = len(time_points)
    
    # Initialize lick raster matrix
    lick_raster = np.zeros((n_trials, n_timepoints), dtype=bool)
    
    # Will store all lick times in a list of (trial, time) tuples
    lick_times = []
    
    # For a well-learned task, use more stereotyped licking patterns
    if well_learned:
        # Simulate licks for each trial
        for trial in range(n_trials):
            trial_type = session_df.iloc[trial]['trial_type']
            is_cs_plus = trial_type == 'CS+'
            
            # 1. Very low baseline licking
            baseline_window = time_points < 0
            # Random spontaneous licks
            if is_cs_plus:
                spont_prob = spontaneous_rate * 0.5  # Even lower for CS+ trials
            else:
                spont_prob = spontaneous_rate
                
            # Generate sparse spontaneous licks
            for t_idx in np.where(baseline_window)[0]:
                if np.random.random() < spont_prob * time_resolution:
                    lick_raster[trial, t_idx] = True
                    lick_times.append((trial, time_points[t_idx]))
            
            # 2. CS+ trials: Burst of licking after cue and at reward
            if is_cs_plus:
                # Initial response to CS+ (0.5-2s after cue onset)
                cue_response_window = (time_points >= 0.5) & (time_points <= 2.0)
                # Start a lick bout in response to cue
                if np.random.random() < 0.9:  # 90% chance of responding to cue
                    # Pick a random time point for first lick in the cue response window
                    cue_resp_indices = np.where(cue_response_window)[0]
                    if len(cue_resp_indices) > 0:
                        first_lick_idx = np.random.choice(cue_resp_indices)
                        # Generate a lick bout starting at this point
                        generate_lick_bout(
                            trial, first_lick_idx, time_points, lick_raster, lick_times,
                            bout_length=np.random.randint(4, 8),
                            interval_range=bout_interval_range
                        )
                
                # Reward response (3.0-4.0s after cue)
                reward_window = (time_points >= 3.0) & (time_points <= 4.0)
                # Start a lick bout in response to reward
                if np.random.random() < 0.95:  # 95% chance of responding to reward
                    # Pick a time point right after reward delivery
                    reward_indices = np.where(reward_window)[0]
                    if len(reward_indices) > 0:
                        # Start the bout shortly after reward
                        reward_start_idx = reward_indices[0] + np.random.randint(1, 5)
                        if reward_start_idx < n_timepoints:
                            # Generate a lick bout starting at this point
                            generate_lick_bout(
                                trial, reward_start_idx, time_points, lick_raster, lick_times,
                                bout_length=np.random.randint(6, 12),  # Longer bout for reward
                                interval_range=bout_interval_range
                            )
            
            # 3. CS- trials: Very occasional random licks
            else:
                # Much lower probability of licking for CS-
                cs_minus_window = time_points >= 0
                for t_idx in np.where(cs_minus_window)[0]:
                    if np.random.random() < spontaneous_rate * 0.3 * time_resolution:
                        lick_raster[trial, t_idx] = True
                        lick_times.append((trial, time_points[t_idx]))
    
    else:
        # Original licking simulation for untrained animals
        for trial in range(n_trials):
            trial_type = session_df.iloc[trial]['trial_type']
            is_cs_plus = trial_type == 'CS+'
            
            # Define time windows for different licking probabilities
            baseline_window = time_points < 0
            anticipation_window = (time_points >= 0) & (time_points < 3)
            reward_window = (time_points >= 3) & (time_points < 4) if is_cs_plus else np.zeros_like(time_points, dtype=bool)
            
            # Base probabilities for each time window
            baseline_prob = spontaneous_rate * time_resolution * np.ones(n_timepoints)
            anticipation_prob = np.zeros(n_timepoints)
            reward_prob = np.zeros(n_timepoints)
            
            # Calculate anticipatory licking probability based on value
            anticipation_prob[anticipation_window] = value_data[trial, anticipation_window] * anticipation_threshold
            
            # Add ramping up to reward for CS+ trials (learned expectation)
            if is_cs_plus:
                # Create ramping function peaking at reward time
                ramp = np.zeros(n_timepoints)
                # Find indices of time points in anticipation window
                ant_indices = np.where(anticipation_window)[0]
                if len(ant_indices) > 0:
                    # Create linear ramp normalized to [0, 1]
                    ramp[ant_indices] = np.linspace(0, 1, len(ant_indices))
                    # Apply the ramp to anticipation probability
                    anticipation_prob[anticipation_window] *= (0.5 + 0.5 * ramp[anticipation_window])
            
            # Add reward consumption licking (after reward delivery)
            reward_prob[reward_window] = reward_response_prob * time_resolution * 2  # Higher probability during reward
            
            # Combine all probabilities
            lick_prob = baseline_prob + anticipation_prob + reward_prob
            
            # Generate initial licks
            initial_licks = np.random.random(n_timepoints) < lick_prob
            potential_lick_indices = np.where(initial_licks)[0]
            
            # Process each potential lick and generate bouts
            processed_indices = set()
            for idx in potential_lick_indices:
                if idx in processed_indices:
                    continue
                    
                # Mark as processed
                processed_indices.add(idx)
                
                # Add the initial lick
                lick_raster[trial, idx] = True
                lick_times.append((trial, time_points[idx]))
                
                # Decide if this starts a licking bout
                if np.random.random() < bout_probability:
                    # Generate a bout of licks
                    bout_length = np.random.randint(bout_length_range[0], bout_length_range[1]+1)
                    
                    # Find indices for additional licks in bout
                    t = time_points[idx]
                    cumulative_time = 0
                    for _ in range(1, bout_length):
                        # Random interval between licks in bout
                        interval = np.random.uniform(bout_interval_range[0], bout_interval_range[1])
                        cumulative_time += interval
                        
                        # Find closest time point
                        next_t = t + cumulative_time
                        if next_t > time_points[-1]:
                            break
                            
                        next_idx = np.argmin(np.abs(time_points - next_t))
                        
                        # Don't add if already processed
                        if next_idx in processed_indices:
                            continue
                        
                        # Add the bout lick
                        lick_raster[trial, next_idx] = True
                        lick_times.append((trial, time_points[next_idx]))
                        processed_indices.add(next_idx)
    
    # Convert lick times to DataFrame
    lick_times_df = pd.DataFrame({
        'trial_number': [lt[0] for lt in lick_times],
        'lick_time': [lt[1] for lt in lick_times]
    })
    
    return lick_times_df, lick_raster

def generate_lick_bout(trial: int, 
                      start_idx: int, 
                      time_points: np.ndarray, 
                      lick_raster: np.ndarray, 
                      lick_times: list,
                      bout_length: int = 7, 
                      interval_range: tuple = (0.2, 0.4)):
    """
    Helper function to generate a bout of licks
    
    Args:
        trial: Trial number
        start_idx: Index in time_points for the first lick
        time_points: Time vector
        lick_raster: Lick raster matrix to update
        lick_times: List of lick times to update
        bout_length: Number of licks in the bout
        interval_range: Range of inter-lick intervals (seconds)
    """
    # Add the first lick
    lick_raster[trial, start_idx] = True
    lick_times.append((trial, time_points[start_idx]))
    
    # Add subsequent licks in the bout
    current_idx = start_idx
    for _ in range(1, bout_length):
        # Random interval between licks
        interval = np.random.uniform(interval_range[0], interval_range[1])
        next_time = time_points[current_idx] + interval
        
        # Find closest time point
        if next_time > time_points[-1]:
            break
            
        next_idx = np.argmin(np.abs(time_points - next_time))
        
        # Avoid duplicate licks at the same time
        if next_idx == current_idx:
            next_idx += 1
            if next_idx >= len(time_points):
                break
        
        # Add the lick
        lick_raster[trial, next_idx] = True
        lick_times.append((trial, time_points[next_idx]))
        current_idx = next_idx

def generate_example_dataset(reward_probability: float = 1.0,
                        alpha: float = 0.2, 
                        gamma: float = 0.95) -> Dict:
    """
    Generate a complete example dataset for demo purposes
    
    Args:
        reward_probability: Probability of reward delivery on CS+ trials (0-1)
        alpha: Learning rate parameter
        gamma: Discount factor parameter
    
    Returns:
        Dictionary with all simulated data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create session data
    session_df = create_pavlovian_session(n_trials=100, reward_probability=reward_probability)
    
    # Debug: Check trial types
    cs_plus_count = (session_df['trial_type'] == 'CS+').sum()
    cs_minus_count = (session_df['trial_type'] == 'CS-').sum()
    print(f"Debug - Trial counts: CS+ = {cs_plus_count}, CS- = {cs_minus_count}")
    
    # Debug: Check specific trial types
    print("First 10 trial types:", session_df['trial_type'].iloc[:10].values)
    print("Is 'CS+' in trial_type unique values?", 'CS+' in session_df['trial_type'].unique())
    
    # Calculate how many rewarded trials we actually got
    rewarded_trials = (session_df['reward'] == 1).sum()
    print(f"Debug - Rewarded trials: {rewarded_trials}/{cs_plus_count} ({rewarded_trials/cs_plus_count:.1%})")
    
    # Run TD model
    td_results = simulate_td_model(
        session_df=session_df,
        alpha=alpha,
        gamma=gamma
    )
    
    # Simulate biologically realistic dopamine signals
    dopamine_data = simulate_dopamine_from_rpe(
        td_results['rpes'], 
        td_results['time_points'],
        noise_level=0.05,
        gain=1.2,
        add_realistic_features=True
    )
    
    # Add additional signal at CS onset and reward time to ensure better correlation
    cs_mask = (td_results['time_points'] >= -0.1) & (td_results['time_points'] <= 0.1)
    reward_mask = (td_results['time_points'] >= 2.9) & (td_results['time_points'] <= 3.1)
    
    for trial in range(len(session_df)):
        trial_type = session_df.iloc[trial]['trial_type']
        
        # Add correlated signal at CS onset
        rpe_peak_cs = np.max(td_results['rpes'][trial, cs_mask])
        dopamine_data[trial, cs_mask] += rpe_peak_cs * 0.8 + 0.1
        
        # Add correlated signal at reward (CS+ only)
        if trial_type == 'CS+':
            rpe_peak_reward = np.max(td_results['rpes'][trial, reward_mask])
            dopamine_data[trial, reward_mask] += rpe_peak_reward * 0.8 + 0.1
    
    # Simulate realistic licking behavior
    lick_times_df, lick_raster = simulate_lick_behavior(
        session_df,
        td_results['time_points'],
        td_results['values'],
        anticipation_threshold=0.4,
        reward_response_prob=0.95,
        spontaneous_rate=0.02,
        bout_probability=0.8,
        bout_length_range=(3, 7),
        bout_interval_range=(0.2, 0.4)
    )
    
    return {
        'session_data': session_df,
        'time_points': td_results['time_points'],
        'values': td_results['values'],
        'rpes': td_results['rpes'],
        'dopamine_data': dopamine_data,
        'lick_times': lick_times_df,
        'lick_raster': lick_raster
    } 