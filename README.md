# Interactive Temporal Difference Learning GUI

An interactive GUI built with **Python + Streamlit** to simulate and visualize **reward prediction error (RPE)** signals using **Temporal Difference (TD) learning**, and to compare them with real or synthetic data from Pavlovian conditioning experiments.

## 📋 Features

- Simulate TD learning model with adjustable parameters (α, γ)
- Visualize RPE signals, value function, dopamine activity, and licking behavior
- Upload and analyze real experimental data
- Compare model predictions with neural and behavioral data
- Statistical analysis of model-data correlations
- Data converter for transforming experimental data into the required format

## 🧠 Pavlovian Conditioning Task Model

- 2-second odor cue (CS+ or CS−)
- 1-second trace interval
- Reward (sucrose) delivered at 3s only for **CS+ trials**
- Inter-trial interval (ITI): random, centered around 15s
- **100 trials per session**: 50 CS+, 50 CS−

## 💻 Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
cd app
streamlit run app.py
```

4. To use the data converter utility:

```bash
cd app
streamlit run data_converter.py
```

## 📁 Input File Formats

### 1. session_data.csv (REQUIRED)
```csv
trial_number,trial_type,cs_onset,reward_time,reward,iti
0,CS+,0.0,3.0,1,16.2
1,CS-,16.2,,0,13.8
2,CS+,30.0,33.0,1,14.5
```

### 2. dopamine_data.npy (OPTIONAL)
- Shape: `(n_trials, n_timepoints)`
- Time vector: `dopamine_time.npy` (e.g., [-2.0, ..., +5.0])

### 3. lick_times.csv OR lick_raster.npy (OPTIONAL)
```csv
# lick_times.csv
trial_number,lick_time
0,-0.2
0,1.3
1,0.1
```
OR
- `lick_raster.npy` shape: `(n_trials, n_timepoints)` (binned licks)

## 🔄 Data Converter

The data converter utility helps transform experimental data into the format required by the main app:

1. Upload your data file (Pickle format)
2. Map event types (trial_start, odor_start, reward, lick) to your data
3. Select CS+ and CS- trial identification method
4. Analyze and limit the number of trials as needed
5. Generate and download the converted data files

## 🧰 Project Structure

```
project_root/
├── app.py                            # Main Streamlit app entry point
├── data_converter.py                 # Data conversion utility
├── utils/                            # Utility functions
│   ├── load_data.py                  # Data loading utilities
│   └── simulate_data.py              # TD model and synthetic data
├── plots/                            # Visualization modules
│   └── visualizations.py             # Heatmaps, rasters, traces
├── data/                             # User and demo data
│   └── example_simulated/            # Example synthetic data
└── converted_data/                   # Output directory for converted data
```

## 📊 TD Model Visualizations

The app provides several visualizations:

1. **Heatmaps**: Trial-by-trial visualization of RPE, dopamine signals, and value function
2. **Raster Plots**: Lick behavior aligned to CS onset
3. **Average Traces**: CS+ vs CS- comparison of RPE, value, dopamine, and lick rate
4. **Correlation Analysis**: RPE vs dopamine correlation at CS and reward times
5. **Lick Latency Analysis**: Statistical comparison of first lick timing

## 🔍 How TD Learning Works

The TD model implemented follows the approach from Sutton & Barto:

1. **State Representation**: CS presentation and trace interval
2. **Value Function**: Estimated expected future reward at each time point
3. **RPE Computation**: Difference between predicted and actual reward (`RPE (δ) = Rt - V(t)`)
4. **Learning**: Update value function based on RPE and learning rate (`V(t) ← V(t) + α · δ`)

## 📝 License

This project is open source and available under the MIT License.

## 👥 Contributors

This project was created by the Stuber Lab.

## 📚 References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
- Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. Science, 275(5306), 1593-1599. 