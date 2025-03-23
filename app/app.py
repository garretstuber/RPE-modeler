import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import json

# Add utils directory to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from utils.load_data import load_session_data, load_dopamine, load_lick_data, generate_time_vector, validate_session_data
from utils.simulate_data import (
    simulate_td_model, simulate_dopamine_from_rpe, 
    simulate_lick_behavior, generate_example_dataset,
    create_pavlovian_session
)
from plots.visualizations import (
    plot_rpe_heatmap, plot_photometry_heatmap, 
    plot_lick_raster, plot_average_traces,
    convert_lick_times_to_raster
)

# Set page config
st.set_page_config(
    page_title="TD Learning Visualization",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to create directories if they don't exist
def create_data_dirs():
    """Create data directories if they don't exist"""
    os.makedirs("app/data", exist_ok=True)
    os.makedirs("app/data/example_simulated", exist_ok=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.data_loaded = False
    st.session_state.example_data = None
    st.session_state.last_params = {
        "alpha": 0.2,
        "gamma": 0.95,
        "reward_probability": 1.0
    }
    create_data_dirs()

# Helper function to load or generate example data
def get_example_data(alpha, gamma, reward_probability):
    """Get example data for demo purposes"""
    # Check if parameters have changed since last generation
    current_params = {
        "alpha": alpha,
        "gamma": gamma,
        "reward_probability": reward_probability
    }
    
    # If parameters changed or no data exists, regenerate example data
    if st.session_state.example_data is None or current_params != st.session_state.last_params:
        st.session_state.example_data = generate_example_dataset(
            reward_probability=reward_probability,
            alpha=alpha,
            gamma=gamma
        )
        st.session_state.last_params = current_params.copy()
        
    return st.session_state.example_data

# Title and description
st.markdown("""
# 🧠 Interactive Temporal Difference Learning Visualization
""")

# Add custom CSS to style the header
st.markdown("""
<style>
.main-header {
    background-color: #f0f5ff;
    border-radius: 10px;
    padding: 20px;
    border-left: 5px solid #4b84ff;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Add a brief description in a styled container
st.markdown("""
<div class="main-header">
<p>This interactive tool allows you to <b>simulate</b> and <b>visualize</b> reward prediction error (RPE) signals 
using Temporal Difference (TD) learning, and <b>compare</b> them with real or synthetic data from Pavlovian conditioning experiments.</p>
<p>Adjust model parameters, upload your own data, or use the example dataset to explore how TD learning captures dopamine neuron activity 
and behavioral responses during associative learning.</p>
</div>
""", unsafe_allow_html=True)

# Create sidebar with parameters and file upload
with st.sidebar:
    st.header("TD Model Parameters")
    
    # Add option to load parameters from file
    load_params = st.expander("Load Parameters from File", expanded=False)
    with load_params:
        st.markdown("""
        Upload a JSON file with model parameters. The file should contain a JSON object with keys:
        - `alpha`: Learning rate (0.01-1.0)
        - `gamma`: Discount factor (0.5-1.0)
        - `reward_probability`: Probability of reward delivery on CS+ trials (0-1.0)
        
        Example:
        ```json
        {
            "alpha": 0.2,
            "gamma": 0.95,
            "reward_probability": 1.0
        }
        ```
        """)
        
        # Add template download button
        template_params = {
            "alpha": 0.2,
            "gamma": 0.95,
            "reward_probability": 1.0,
            "description": "Template parameters for TD Learning model"
        }
        
        template_json = json.dumps(template_params, indent=4)
        st.download_button(
            label="Download Template Parameters",
            data=template_json,
            file_name="td_parameters_template.json",
            mime="application/json"
        )
        
        param_file = st.file_uploader("Upload parameter file (JSON)", type=["json", "txt"])
        
        if param_file is not None:
            try:
                import json
                params = json.load(param_file)
                
                # Extract parameters with validation
                if "alpha" in params and 0.01 <= params["alpha"] <= 1.0:
                    alpha_from_file = params["alpha"]
                else:
                    alpha_from_file = None
                    st.warning("Alpha parameter missing or out of range (0.01-1.0).")
                
                if "gamma" in params and 0.5 <= params["gamma"] <= 1.0:
                    gamma_from_file = params["gamma"]
                else:
                    gamma_from_file = None
                    st.warning("Gamma parameter missing or out of range (0.5-1.0).")
                
                if "reward_probability" in params and 0.0 <= params["reward_probability"] <= 1.0:
                    reward_prob_from_file = params["reward_probability"]
                else:
                    reward_prob_from_file = None
                    
                # Success message
                if alpha_from_file is not None or gamma_from_file is not None:
                    st.success("Parameters loaded successfully!")
                    
            except Exception as e:
                st.error(f"Error loading parameters: {str(e)}")
                alpha_from_file = None
                gamma_from_file = None
                reward_prob_from_file = None
        else:
            alpha_from_file = None
            gamma_from_file = None
            reward_prob_from_file = None
    
    # Model parameters with loaded values as defaults if available
    alpha = st.slider("Learning rate (α)", 0.01, 1.0, alpha_from_file if alpha_from_file is not None else 0.2, 0.01,
                    help="Controls how quickly the model updates its predictions based on new information")
    gamma = st.slider("Discount factor (γ)", 0.5, 1.0, gamma_from_file if gamma_from_file is not None else 0.95, 0.01,
                    help="Determines how much future rewards are valued compared to immediate rewards")
    
    # Add reward omission controls
    st.header("Reward Omission Settings")
    reward_probability = st.slider("Reward Probability for CS+ trials", 0.0, 1.0, 
                                 reward_prob_from_file if reward_prob_from_file is not None else 1.0, 0.05,
                                 help="Probability of reward delivery on CS+ trials (1.0 = all CS+ trials rewarded, 0.0 = no rewards)")
    
    if reward_probability < 1.0:
        st.info(f"With {reward_probability:.0%} reward probability, approximately {int(50 * reward_probability)} out of 50 CS+ trials will be rewarded.")
    
    # Input file uploads
    st.header("Data Upload")
    
    # Session data
    session_file = st.file_uploader("Upload session_data.csv", type="csv",
                                   help="CSV file with trial information (required)")
    
    # Dopamine data
    dopamine_file = st.file_uploader("Upload dopamine_data.npy", type="npy",
                                   help="NPY file with dopamine fluorescence data (trials × timepoints)")
    
    # Optional dopamine time vector
    dopamine_time_file = st.file_uploader("Upload dopamine_time.npy (optional)", type="npy",
                                       help="NPY file with time vector for dopamine data")
    
    # Lick data
    lick_file = st.file_uploader("Upload lick data", type=["csv", "npy"],
                              help="CSV with lick times or NPY with lick raster")
    
    # Button to generate example data
    if st.button("Use Example Data"):
        st.session_state.data_loaded = True
        st.session_state.using_example = True
        st.session_state.example_data = None  # Force regeneration
        st.success(f"Example data loaded! Using reward probability: {reward_probability:.0%}")

# Main content
# Check if we have session data (either uploaded or example)
if session_file is not None or (st.session_state.data_loaded and st.session_state.using_example):
    
    # Load data
    if st.session_state.using_example:
        # Use example data
        example_data = get_example_data(alpha, gamma, reward_probability)
        session_df = example_data['session_data']
        time_points = example_data['time_points']
        rpe_data = example_data['rpes']
        value_data = example_data['values']
        dopamine_data = example_data['dopamine_data']
        lick_times_df = example_data['lick_times']
        lick_raster = example_data['lick_raster']
        using_real_dopamine = False
        using_real_licks = False
        
        # Show info about rewarded trials
        rewarded_trials = (session_df['reward'] == 1).sum()
        cs_plus_count = (session_df['trial_type'] == 'CS+').sum()
        st.info(f"Generated {rewarded_trials} rewarded trials out of {cs_plus_count} CS+ trials ({rewarded_trials/cs_plus_count:.0%})")
    else:
        # Load real data
        st.session_state.using_example = False
        
        # Load session data
        session_df = load_session_data(session_file)
        
        if session_df is None:
            st.error("Error loading session data. Please check the file format.")
            st.stop()
        
        # Show data preview and validation warnings
        validation_result = validate_session_data(session_df)
        
        if validation_result["warnings"]:
            with st.expander("⚠️ Data Validation Warnings", expanded=True):
                for warning in validation_result["warnings"]:
                    st.warning(warning)
        
        # Option to preview the data
        with st.expander("📊 Preview Session Data", expanded=False):
            st.dataframe(session_df.head(10), use_container_width=True)
            st.caption("Showing first 10 trials")
            
            # Summary statistics
            st.write("#### Summary Statistics:")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Total trials: {len(session_df)}")
                st.write(f"CS+ trials: {sum(session_df['trial_type'] == 'CS+')}")
                st.write(f"CS- trials: {sum(session_df['trial_type'] == 'CS-')}")
            with col2:
                if 'reward_time' in session_df.columns:
                    reward_times = session_df[session_df['reward_time'].notna()]['reward_time'].values
                    if len(reward_times) > 0:
                        avg_reward_time = np.mean(reward_times - session_df[session_df['reward_time'].notna()]['cs_onset'].values)
                        st.write(f"Average reward delay: {avg_reward_time:.2f}s")
                
                if 'iti' in session_df.columns:
                    st.write(f"Average ITI: {session_df['iti'].mean():.2f}s")
        
        # Run TD model
        st.info(f"Running TD model with parameters: α = {alpha}, γ = {gamma}, Reward Prob = {reward_probability:.0%}")
        
        td_results = simulate_td_model(
            session_df=session_df,
            alpha=alpha,
            gamma=gamma
        )
        
        time_points = td_results['time_points']
        rpe_data = td_results['rpes']
        value_data = td_results['values']
        cs_values = td_results['cs_value']
        
        # Load dopamine data if available
        if dopamine_file is not None:
            with st.spinner("Processing dopamine data..."):
                dopamine_data, dopa_time = load_dopamine(dopamine_file, dopamine_time_file)
                
                # Show preview of dopamine data
                with st.expander("📊 Preview Dopamine Data", expanded=False):
                    st.write(f"Dopamine data shape: {dopamine_data.shape}")
                    st.line_chart(pd.DataFrame(dopamine_data[:5].T).rename(columns={i: f"Trial {i+1}" for i in range(5)}))
                    st.caption("Showing first 5 trials")
                
                if dopa_time is not None and len(dopa_time) != rpe_data.shape[1]:
                    # Need to interpolate to match time points
                    st.info(f"Interpolating dopamine data to match model time points ({len(dopa_time)} → {len(time_points)})")
                    from scipy.interpolate import interp1d
                    dopamine_interp = np.zeros((dopamine_data.shape[0], len(time_points)))
                    
                    for trial in range(dopamine_data.shape[0]):
                        f = interp1d(dopa_time, dopamine_data[trial], 
                                   bounds_error=False, fill_value="extrapolate")
                        dopamine_interp[trial] = f(time_points)
                    
                    dopamine_data = dopamine_interp
                using_real_dopamine = True
                st.success("Dopamine data loaded successfully")
        else:
            # Simulate dopamine from RPE
            dopamine_data = simulate_dopamine_from_rpe(rpe_data, time_points)
            using_real_dopamine = False
        
        # Load lick data if available
        if lick_file is not None:
            with st.spinner("Processing lick data..."):
                lick_data = load_lick_data(lick_file)
                
                if isinstance(lick_data, pd.DataFrame):
                    # Convert lick times to raster format
                    lick_times_df = lick_data
                    
                    # Show preview of lick times data
                    with st.expander("📊 Preview Lick Times Data", expanded=False):
                        st.dataframe(lick_times_df.head(20))
                        st.caption("Showing first 20 lick events")
                        st.write(f"Total licks: {len(lick_times_df)}")
                        st.write(f"Trials with licks: {lick_times_df['trial_number'].nunique()}")
                    
                    lick_raster = convert_lick_times_to_raster(
                        lick_times_df, len(session_df), time_points)
                else:
                    # Already in raster format
                    lick_raster = lick_data
                    # Create dummy lick times DataFrame (not used for plotting)
                    lick_times_df = pd.DataFrame(columns=['trial_number', 'lick_time'])
                    
                    # Show preview of lick raster data
                    with st.expander("📊 Preview Lick Raster Data", expanded=False):
                        st.write(f"Lick raster shape: {lick_raster.shape}")
                        # Show heatmap of first few trials
                        fig, ax = plt.subplots(figsize=(10, 3))
                        ax.imshow(lick_raster[:10], aspect='auto', cmap='binary')
                        ax.set_xlabel('Time bins')
                        ax.set_ylabel('Trial')
                        ax.set_title('Lick Raster Preview (First 10 Trials)')
                        st.pyplot(fig)
                
                using_real_licks = True
                st.success("Lick data loaded successfully")
        else:
            # Simulate lick behavior
            lick_times_df, lick_raster = simulate_lick_behavior(
                session_df, time_points, value_data)
            using_real_licks = False
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Visualizations", "Average Traces", "Analysis", "Export Data"])
    
    with tab1:
        st.header("Trial-by-Trial Visualization")
        
        # Display RPE heatmap
        st.subheader("Reward Prediction Error (RPE) Signals")
        fig = plot_rpe_heatmap(
            rpe_data=rpe_data, 
            time_points=time_points,
            session_df=session_df,
            title="RPE Signals" + (" (α={:.2f}, γ={:.2f})".format(alpha, gamma) 
                                if not st.session_state.using_example else "")
        )
        st.pyplot(fig)
        
        # Display dopamine heatmap
        st.subheader("Dopamine Signals")
        fig = plot_photometry_heatmap(
            photometry_data=dopamine_data,
            time_points=time_points,
            session_df=session_df,
            sort_trials=True,
            zscore=True,
            title="Dopamine Photometry" + (" (Real Data)" if using_real_dopamine else " (Simulated)")
        )
        st.pyplot(fig)
        
        # Display lick raster
        st.subheader("Lick Raster")
        fig = plot_lick_raster(
            lick_data=lick_raster,
            time_points=time_points,
            session_df=session_df,
            sort_trials=True
        )
        st.pyplot(fig)
        if not using_real_licks:
            st.caption("Note: Simulated lick behavior based on TD model values")
        
        # Display value function
        st.subheader("Value Function")
        fig = plot_rpe_heatmap(
            rpe_data=value_data,
            time_points=time_points,
            session_df=session_df,
            cmap='viridis',
            title="State Values"
        )
        st.pyplot(fig)
    
    with tab2:
        st.header("Trial-Averaged Responses")
        
        # RPE and Value traces
        st.subheader("TD Model Traces")
        td_traces = {
            'rpe': rpe_data,
            'value': value_data
        }
        
        fig = plot_average_traces(
            data_dict=td_traces,
            time_points=time_points,
            session_df=session_df,
            zscore=False
        )
        st.pyplot(fig)
        
        # Dopamine and lick traces
        st.subheader("Neural and Behavioral Traces")
        da_lick_traces = {
            'dopamine': dopamine_data
        }
        
        # Convert lick raster to continuous signal for plotting
        lick_rate = np.zeros_like(lick_raster, dtype=float)
        window_size = 5  # Time bins
        for i in range(lick_raster.shape[0]):
            lick_rate[i] = np.convolve(lick_raster[i].astype(float), 
                                     np.ones(window_size)/window_size, 
                                     mode='same')
        
        da_lick_traces['lick_rate'] = lick_rate
        
        fig = plot_average_traces(
            data_dict=da_lick_traces,
            time_points=time_points,
            session_df=session_df,
            zscore=True
        )
        st.pyplot(fig)
    
    with tab3:
        st.header("Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate trial-by-trial correlations between RPE and dopamine
            st.subheader("RPE vs Dopamine Correlation")
            
            # Get CS onsets and reward times
            reward_window = (2.9, 3.1)  # Time window around reward
            cs_window = (-0.1, 0.1)     # Time window around CS onset
            
            # Extract peak responses
            reward_mask = (time_points >= reward_window[0]) & (time_points <= reward_window[1])
            cs_mask = (time_points >= cs_window[0]) & (time_points <= cs_window[1])
            
            # Extract peaks
            rpe_reward_peaks = np.max(rpe_data[:, reward_mask], axis=1)
            da_reward_peaks = np.max(dopamine_data[:, reward_mask], axis=1)
            
            rpe_cs_peaks = np.max(rpe_data[:, cs_mask], axis=1)
            da_cs_peaks = np.max(dopamine_data[:, cs_mask], axis=1)
            
            # Plot correlation scatter
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # Import stats module
            from scipy import stats
            
            # CS response correlation
            ax1.scatter(rpe_cs_peaks, da_cs_peaks, alpha=0.7)
            ax1.set_xlabel("RPE at CS")
            ax1.set_ylabel("Dopamine at CS")
            ax1.set_title("CS Response Correlation")
            
            # Add regression line with error handling
            try:
                # Remove any NaN or inf values
                cs_mask = ~(np.isnan(rpe_cs_peaks) | np.isnan(da_cs_peaks) | 
                         np.isinf(rpe_cs_peaks) | np.isinf(da_cs_peaks))
                
                # Check if we have enough valid data points and variation
                if np.sum(cs_mask) > 2 and np.std(rpe_cs_peaks[cs_mask]) > 0 and np.std(da_cs_peaks[cs_mask]) > 0:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        rpe_cs_peaks[cs_mask], da_cs_peaks[cs_mask])
                    
                    x = np.linspace(np.min(rpe_cs_peaks[cs_mask]), np.max(rpe_cs_peaks[cs_mask]), 100)
                    ax1.plot(x, slope * x + intercept, 'r')
                    ax1.text(0.05, 0.95, f"r = {r_value:.2f}, p = {p_value:.3f}", 
                           transform=ax1.transAxes, fontsize=10,
                           verticalalignment='top')
                else:
                    ax1.text(0.05, 0.95, "Insufficient data variation for regression", 
                           transform=ax1.transAxes, fontsize=10,
                           verticalalignment='top')
            except Exception as e:
                ax1.text(0.05, 0.95, f"Error calculating regression: {str(e)}", 
                       transform=ax1.transAxes, fontsize=8,
                       verticalalignment='top')
            
            # Reward response correlation
            ax2.scatter(rpe_reward_peaks, da_reward_peaks, alpha=0.7)
            ax2.set_xlabel("RPE at Reward")
            ax2.set_ylabel("Dopamine at Reward")
            ax2.set_title("Reward Response Correlation")
            
            # Add regression line with error handling
            try:
                # Remove any NaN or inf values
                reward_mask = ~(np.isnan(rpe_reward_peaks) | np.isnan(da_reward_peaks) | 
                             np.isinf(rpe_reward_peaks) | np.isinf(da_reward_peaks))
                
                # Check if we have enough valid data points and variation
                if np.sum(reward_mask) > 2 and np.std(rpe_reward_peaks[reward_mask]) > 0 and np.std(da_reward_peaks[reward_mask]) > 0:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        rpe_reward_peaks[reward_mask], da_reward_peaks[reward_mask])
                    
                    x = np.linspace(np.min(rpe_reward_peaks[reward_mask]), np.max(rpe_reward_peaks[reward_mask]), 100)
                    ax2.plot(x, slope * x + intercept, 'r')
                    ax2.text(0.05, 0.95, f"r = {r_value:.2f}, p = {p_value:.3f}", 
                           transform=ax2.transAxes, fontsize=10,
                           verticalalignment='top')
                else:
                    ax2.text(0.05, 0.95, "Insufficient data variation for regression", 
                           transform=ax2.transAxes, fontsize=10,
                           verticalalignment='top')
            except Exception as e:
                ax2.text(0.05, 0.95, f"Error calculating regression: {str(e)}", 
                       transform=ax2.transAxes, fontsize=8,
                       verticalalignment='top')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Lick latency analysis
            st.subheader("Lick Latency Analysis")
            
            # Calculate first lick latency for each trial
            first_licks = []
            cs_type = []
            trial_nums = []
            
            for trial in range(len(session_df)):
                trial_lick_mask = lick_times_df['trial_number'] == trial
                if np.sum(trial_lick_mask) > 0:
                    trial_licks = lick_times_df.loc[trial_lick_mask, 'lick_time'].values
                    
                    # Only consider licks after CS onset
                    post_cs_licks = trial_licks[trial_licks >= 0]
                    
                    if len(post_cs_licks) > 0:
                        first_lick = np.min(post_cs_licks)
                        first_licks.append(first_lick)
                        cs_type.append(session_df.iloc[trial]['trial_type'])
                        trial_nums.append(trial)
            
            # Create DataFrame for analysis
            latency_df = pd.DataFrame({
                'trial': trial_nums,
                'first_lick': first_licks,
                'cs_type': cs_type
            })
            
            # Create box plot of latencies by trial type
            if len(first_licks) > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                import seaborn as sns
                # Set seaborn style for better visualization
                sns.set_style("whitegrid")
                
                # Create a more attractive plot using seaborn
                # Use violin plot with box plot inside to show distribution
                sns.violinplot(x='cs_type', y='first_lick', data=latency_df, 
                             inner='box', hue='cs_type', hue_order=latency_df['cs_type'].unique(),
                             palette='Set3', dodge=False,
                             ax=ax, alpha=0.7)
                
                # Add individual data points
                sns.stripplot(x='cs_type', y='first_lick', data=latency_df, 
                           jitter=True, size=5, color='black', alpha=0.5, ax=ax)
                
                # Improve appearance
                ax.set_xlabel('Trial Type', fontsize=12, fontweight='bold')
                ax.set_ylabel('Latency to First Lick (s)', fontsize=12, fontweight='bold')
                ax.set_title('Lick Latency by Trial Type', fontsize=14, fontweight='bold')
                
                # Add reward time marker
                ax.axhline(y=3.0, color='r', linestyle='--', linewidth=2, label='Reward Time')
                
                # Add a horizontal line at mean latency for each group
                trial_types = latency_df['cs_type'].unique()
                for i, trial_type in enumerate(trial_types):
                    mean_latency = latency_df[latency_df['cs_type'] == trial_type]['first_lick'].mean()
                    ax.hlines(y=mean_latency, xmin=i-0.3, xmax=i+0.3, colors='darkred', 
                            linestyles='solid', linewidth=2, label=f'Mean {trial_type}')
                
                # Customize appearance
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.legend(loc='upper right')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # T-test for difference in latencies
                cs_plus_latencies = latency_df[latency_df['cs_type'] == 'CS+']['first_lick'].values
                cs_minus_latencies = latency_df[latency_df['cs_type'] == 'CS-']['first_lick'].values
                
                if len(cs_plus_latencies) > 1 and len(cs_minus_latencies) > 1:
                    try:
                        t_stat, p_val = stats.ttest_ind(cs_plus_latencies, cs_minus_latencies)
                        
                        st.write(f"**T-test Results:**")
                        st.write(f"- CS+ mean latency: {np.mean(cs_plus_latencies):.2f}s")
                        st.write(f"- CS- mean latency: {np.mean(cs_minus_latencies):.2f}s")
                        st.write(f"- T-statistic: {t_stat:.2f}")
                        st.write(f"- p-value: {p_val:.4f}")
                        
                        if p_val < 0.05:
                            st.success("Significant difference in lick latencies between CS+ and CS- trials!")
                        else:
                            st.info("No significant difference in lick latencies between CS+ and CS- trials.")
                    except Exception as e:
                        st.error(f"Error in statistical test: {str(e)}")
                else:
                    st.info("Not enough lick data for statistical comparison.")
            else:
                st.info("No lick data available for latency analysis.")

    with tab4:
        st.header("Export Data")
        st.markdown("""
        Download model outputs and processed data for further analysis in your preferred software.
        All files are saved in common formats (CSV or NumPy) that can be easily loaded in Python, MATLAB, or R.
        """)
        
        # Create columns for the export options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Outputs")
            
            # Export TD model results as CSV
            if st.button("Export TD Model Results"):
                # Create DataFrame with model outputs
                model_df = pd.DataFrame({
                    'time': time_points
                })
                
                # Add trial-averaged data
                cs_plus_mask = np.array([t == "CS+" for t in session_df["trial_type"]])
                cs_minus_mask = np.array([t == "CS-" for t in session_df["trial_type"]])
                
                model_df['rpe_csplus_avg'] = np.mean(rpe_data[cs_plus_mask], axis=0)
                model_df['rpe_csminus_avg'] = np.mean(rpe_data[cs_minus_mask], axis=0)
                model_df['value_csplus_avg'] = np.mean(value_data[cs_plus_mask], axis=0)
                model_df['value_csminus_avg'] = np.mean(value_data[cs_minus_mask], axis=0)
                
                # Convert to CSV
                csv = model_df.to_csv(index=False)
                
                # Create download button
                st.download_button(
                    label="Download TD Model Results (CSV)",
                    data=csv,
                    file_name="td_model_results.csv",
                    mime="text/csv",
                )
            
            # Export full trial-by-trial data
            if st.button("Export Full Trial-by-Trial Data"):
                # Create a BytesIO object to store the NPZ file
                import io
                from scipy.io import savemat
                
                # First save as .npz
                output_file = io.BytesIO()
                np.savez(
                    output_file,
                    time_points=time_points,
                    rpe_data=rpe_data,
                    value_data=value_data,
                    session_info=session_df.to_numpy(),
                    session_columns=session_df.columns.values
                )
                
                # Create download button
                output_file.seek(0)
                st.download_button(
                    label="Download Trial-by-Trial Data (NPZ)",
                    data=output_file,
                    file_name="td_model_trial_data.npz",
                    mime="application/octet-stream"
                )
                
                # Also provide MATLAB format
                mat_output = io.BytesIO()
                
                # Convert DataFrame to dictionary suitable for MATLAB
                session_dict = {col: session_df[col].values for col in session_df.columns}
                
                # Save to MATLAB format
                savemat(
                    mat_output, 
                    {
                        'time_points': time_points,
                        'rpe_data': rpe_data,
                        'value_data': value_data,
                        'session_info': session_dict
                    }
                )
                
                # Create download button for MATLAB format
                mat_output.seek(0)
                st.download_button(
                    label="Download Trial-by-Trial Data (MATLAB)",
                    data=mat_output,
                    file_name="td_model_trial_data.mat",
                    mime="application/octet-stream"
                )
                
        with col2:
            st.subheader("Plots & Figures")
            
            # Export current figures as SVG or PNG
            plot_format = st.radio("Export Format", ["PNG", "SVG", "PDF"], horizontal=True)
            file_extension = plot_format.lower()
            mime_type = f"image/{file_extension}" if file_extension != "pdf" else "application/pdf"
            
            if st.button(f"Generate Exportable Figures"):
                # Create a BytesIO object for each figure
                from io import BytesIO
                import matplotlib.pyplot as plt
                
                # Create 3 main figures
                
                # Figure 1: RPE Heatmap
                fig1 = plot_rpe_heatmap(
                    rpe_data=rpe_data, 
                    time_points=time_points,
                    session_df=session_df,
                    figsize=(10, 8),
                    dpi=300,
                    title="RPE Signals" + (" (α={:.2f}, γ={:.2f})".format(alpha, gamma) 
                                        if not st.session_state.using_example else "")
                )
                
                # Figure 2: Value Function
                fig2 = plot_rpe_heatmap(
                    rpe_data=value_data,
                    time_points=time_points,
                    session_df=session_df,
                    cmap='viridis',
                    figsize=(10, 8),
                    dpi=300,
                    title="State Values"
                )
                
                # Figure 3: Average Traces
                td_traces = {
                    'rpe': rpe_data,
                    'value': value_data
                }
                
                fig3 = plot_average_traces(
                    data_dict=td_traces,
                    time_points=time_points,
                    session_df=session_df,
                    figsize=(10, 6),
                    dpi=300,
                    zscore=False
                )
                
                # Save to BytesIO
                for i, fig in enumerate([fig1, fig2, fig3], 1):
                    buffer = BytesIO()
                    fig.savefig(buffer, format=file_extension, bbox_inches='tight', dpi=300)
                    buffer.seek(0)
                    
                    # Create download button
                    fig_name = ["RPE_Heatmap", "Value_Function", "Average_Traces"][i-1]
                    download_label = f"Download {fig_name}.{file_extension}"
                    
                    st.download_button(
                        label=download_label,
                        data=buffer,
                        file_name=f"{fig_name}.{file_extension}",
                        mime=mime_type
                    )
                    
                    # Show small preview
                    st.image(buffer, caption=fig_name, width=300)
                    
                    # Close the figure to avoid memory leaks
                    plt.close(fig)
            
            # Export all data and figures as a zip file
            if st.button("Export All Results (ZIP)"):
                import zipfile
                from io import BytesIO
                import matplotlib.pyplot as plt
                
                # Create a ZIP file in memory
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add CSV data
                    model_df = pd.DataFrame({'time': time_points})
                    cs_plus_mask = np.array([t == "CS+" for t in session_df["trial_type"]])
                    cs_minus_mask = np.array([t == "CS-" for t in session_df["trial_type"]])
                    
                    model_df['rpe_csplus_avg'] = np.mean(rpe_data[cs_plus_mask], axis=0)
                    model_df['rpe_csminus_avg'] = np.mean(rpe_data[cs_minus_mask], axis=0)
                    model_df['value_csplus_avg'] = np.mean(value_data[cs_plus_mask], axis=0)
                    model_df['value_csminus_avg'] = np.mean(value_data[cs_minus_mask], axis=0)
                    
                    # Save CSV to zip file
                    csv = model_df.to_csv(index=False)
                    zip_file.writestr('td_model_results.csv', csv)
                    
                    # Add NPZ file with trial data
                    npz_buffer = BytesIO()
                    np.savez(
                        npz_buffer,
                        time_points=time_points,
                        rpe_data=rpe_data,
                        value_data=value_data,
                        session_info=session_df.to_numpy(),
                        session_columns=session_df.columns.values
                    )
                    npz_buffer.seek(0)
                    zip_file.writestr('td_model_trial_data.npz', npz_buffer.read())
                    
                    # Add PNG figures
                    figure_names = ["RPE_Heatmap", "Value_Function", "Average_Traces"]
                    for i, plot_func in enumerate([
                        lambda: plot_rpe_heatmap(
                            rpe_data=rpe_data, 
                            time_points=time_points,
                            session_df=session_df,
                            figsize=(10, 8),
                            dpi=300,
                            title="RPE Signals"
                        ),
                        lambda: plot_rpe_heatmap(
                            rpe_data=value_data,
                            time_points=time_points,
                            session_df=session_df,
                            cmap='viridis',
                            figsize=(10, 8),
                            dpi=300,
                            title="State Values"
                        ),
                        lambda: plot_average_traces(
                            data_dict={'rpe': rpe_data, 'value': value_data},
                            time_points=time_points,
                            session_df=session_df,
                            figsize=(10, 6),
                            dpi=300,
                            zscore=False
                        )
                    ]):
                        fig_name = figure_names[i] if i < len(figure_names) else f"Figure_{i}"
                        fig = plot_func()
                        
                        fig_buffer = BytesIO()
                        fig.savefig(fig_buffer, format='png', bbox_inches='tight', dpi=300)
                        fig_buffer.seek(0)
                        zip_file.writestr(f'figures/{fig_name}.png', fig_buffer.read())
                        plt.close(fig)
                    
                    # Add a README file
                    readme_text = f"""
                    # TD Learning Model Results
                    
                    Generated by the TD Learning Visualization Tool
                    
                    ## Parameters
                    
                    - Learning rate (α): {alpha}
                    - Discount factor (γ): {gamma}
                    - Total trials: {len(session_df)}
                    - CS+ trials: {sum(cs_plus_mask)}
                    - CS- trials: {sum(cs_minus_mask)}
                    
                    ## Contents
                    
                    - td_model_results.csv: Trial-averaged model outputs
                    - td_model_trial_data.npz: Complete trial-by-trial data (NumPy format)
                    - figures/: Visualizations in PNG format
                    
                    ## How to load the data
                    
                    ### Python
                    ```python
                    import numpy as np
                    import pandas as pd
                    
                    # Load trial-averaged results
                    avg_data = pd.read_csv('td_model_results.csv')
                    
                    # Load trial-by-trial data
                    with np.load('td_model_trial_data.npz') as data:
                        time_points = data['time_points']
                        rpe_data = data['rpe_data']
                        value_data = data['value_data']
                        session_info = data['session_info']
                        session_columns = data['session_columns']
                    ```
                    
                    ### MATLAB
                    ```matlab
                    % Load trial-averaged results
                    avg_data = readtable('td_model_results.csv');
                    
                    % Load trial-by-trial data
                    trial_data = load('td_model_trial_data.npz');
                    ```
                    """
                    zip_file.writestr('README.md', readme_text)
                
                # Create download button for the ZIP file
                zip_buffer.seek(0)
                st.download_button(
                    label="Download All Results (ZIP)",
                    data=zip_buffer,
                    file_name="td_model_results.zip",
                    mime="application/zip"
                )

else:
    # No data loaded yet, show instructions
    st.info("👈 Upload your data files using the sidebar, or click 'Use Example Data' to see a demo.")
    
    st.subheader("About this app")
    st.markdown("""
    ## Interactive TD Learning Visualization Tool

    This application provides an interactive platform for neuroscientists and researchers to visualize, simulate, and analyze Temporal Difference (TD) learning models in the context of Pavlovian conditioning experiments. It allows you to compare model predictions with real neural recordings and behavioral data.

    ### Theoretical Background
    
    Temporal Difference (TD) learning is a computational approach that models how animals learn to predict future rewards based on environmental cues. In classical conditioning experiments, animals learn to associate neutral stimuli (CS) with rewards (US), and their neural activity patterns reflect this learning process.
    
    Dopamine neurons in the midbrain are known to encode reward prediction errors (RPEs) - the difference between expected and actual rewards. This application allows you to simulate these RPE signals using TD learning algorithms, and compare them with real dopamine recordings.

    #### Standard RPE Model
    
    This tool implements the standard reward prediction error model where:
    """)
    
    # Use LaTeX for the equation
    st.latex(r"\delta_t = R_t - V(t)")
    
    st.markdown("""
    Where:
    - **δ** (RPE) represents the difference between actual and predicted reward
    - **R_t** is the actual reward received at time t
    - **V(t)** is the predicted value of the reward at time t
    
    The RPE signal has these properties:
    - **Positive RPE (δ > 0)**: When actual reward is greater than predicted reward (better than expected)
    - **Negative RPE (δ < 0)**: When actual reward is less than predicted reward (worse than expected)
    - **Zero RPE (δ = 0)**: When actual reward matches predicted reward (exactly as expected)
    
    In TD learning, this error signal is used to update future predictions according to:
    """)
    
    # Use LaTeX for the update rule
    st.latex(r"V(t) \leftarrow V(t) + \alpha \cdot \delta")
    
    st.markdown("""
    Where **α** is the learning rate that determines how quickly predictions are updated based on new information.
    
    ### Key Features
    
    - **TD Model Simulation**: Adjust learning rate (α) and discount factor (γ) to see how predictions change
    - **Interactive Visualizations**: Trial-by-trial heatmaps and average response traces 
    - **Neural Data Comparison**: Upload and analyze real dopamine recordings alongside model predictions
    - **Behavioral Analysis**: Visualize and analyze licking behavior in relation to predictions
    - **Statistical Tools**: Correlation analyses between model RPEs and neural data
    
    ### Research Applications
    
    - Test hypotheses about how dopamine neurons encode prediction errors
    - Explore how different TD model parameters affect predictions
    - Compare model predictions with your experimental data
    - Analyze behavioral learning in relation to neural activity
    """)
    
    st.subheader("Data Format Requirements")
    
    st.markdown("""
    To use your own data with this application, please format your files as follows:
    
    ### 1. Session Data (REQUIRED): `session_data.csv`
    
    This file must contain information about each trial in your experiment:
    
    ```csv
    trial_number,trial_type,cs_onset,reward_time,reward,iti
    0,CS+,0.0,3.0,1,16.2
    1,CS-,16.2,,0,13.8
    2,CS+,30.0,33.0,1,14.5
    ```
    
    **Required Columns**:
    - `trial_number`: Sequential numbering of trials (0-indexed)
    - `trial_type`: Must contain 'CS+' and 'CS-' labels (exact spelling and capitalization)
    - `cs_onset`: Time (in seconds) when the CS stimulus was presented
    - `reward`: Binary value (1 for rewarded trials, 0 for unrewarded)
    - `reward_time`: Time (in seconds) when reward was delivered (can be empty for unrewarded trials)
    - `iti`: Inter-trial interval (in seconds)
    
    ### 2. Dopamine Data (OPTIONAL): `dopamine_data.npy`
    
    Neural recording data (e.g., fiber photometry) aligned to CS onset:
    
    - A NumPy array with shape `(n_trials, n_timepoints)`
    - Each row represents a trial, each column a timepoint
    - Time should be aligned to CS onset (t=0)
    - Recommended timerange: -2s to +5s relative to CS onset
    
    **Time Vector** (OPTIONAL): `dopamine_time.npy`
    
    - A 1D NumPy array with shape `(n_timepoints,)` specifying the time points
    - If not provided, the app will assume regular intervals matching TD model time points
    
    ### 3. Lick Data (OPTIONAL): `lick_times.csv` OR `lick_raster.npy`
    
    Option 1: Timestamped lick events in CSV format:
    ```csv
    trial_number,lick_time
    0,-0.2
    0,1.3
    1,0.1
    ```
    
    - `trial_number`: Trial index matching the session data
    - `lick_time`: Time of lick relative to CS onset (seconds)
    
    Option 2: Pre-binned lick raster as NumPy array:
    - Shape: `(n_trials, n_timepoints)` with binary values (0 or 1)
    - Time bins should match the model time points
    """)
    
    st.subheader("Getting Started")
    st.markdown("""
    1. Use the sidebar to adjust TD model parameters (learning rate α and discount factor γ)
    2. Either upload your own data files or click "Use Example Data" to see a demonstration
    3. Explore the visualizations in the different tabs
    4. For detailed analysis, check the "Analysis" tab for correlations and statistics
    
    Need help? Check out the README.md file in the repository for more information.
    """)

# Run the Streamlit app
if __name__ == "__main__":
    # This won't be used when running with streamlit run
    pass 