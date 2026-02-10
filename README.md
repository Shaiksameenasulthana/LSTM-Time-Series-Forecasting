# ✈️ LSTM Time Series Forecasting — Airline Passengers

## About

This project demonstrates how to use a **Long Short-Term Memory (LSTM)** recurrent neural network to forecast monthly international airline passenger numbers. It uses the classic *Box-Jenkins Airline Passengers* dataset (1949–1960) and frames the time-series prediction as a supervised regression problem.

## Dataset

| Detail | Value |
|---|---|
| **File** | `airline-passengers.csv` |
| **Records** | 144 monthly observations |
| **Period** | January 1949 – December 1960 |
| **Columns** | `month` (YYYY-MM), `total_passengers` (in thousands) |
| **Source** | Box & Jenkins (1976) — classic benchmark for time-series analysis |

## Project Structure

```
.
├── Lstm_airline-passengers_.ipynb   # Jupyter notebook with full LSTM pipeline
├── airline-passengers.csv           # Dataset
└── README.md                        # This file
```

## Workflow

The notebook follows these steps:

1. **Import Libraries** — NumPy, Pandas, Matplotlib, TensorFlow/Keras, scikit-learn.
2. **Load Data** — Read the CSV; extract the `total_passengers` column as float32.
3. **Normalize** — Scale values to the [0, 1] range using `MinMaxScaler`.
4. **Train/Test Split** — 67% training (~96 samples), 33% testing (~48 samples).
5. **Create Supervised Dataset** — Convert the time series into input/output pairs using a sliding window (`look_back=1`, i.e., predict *t+1* from *t*).
6. **Reshape for LSTM** — Reshape input to `[samples, time_steps, features]`.
7. **Build & Train LSTM Model** — A Sequential model with one LSTM layer (4 units) and one Dense output layer, compiled with Adam optimizer and MSE loss, trained for 100 epochs.
8. **Evaluate** — Invert scaling on predictions, compute RMSE for both train and test sets.
9. **Visualize** — Plot actual vs. predicted values for train and test sets.

## Model Architecture

```
Layer             Output Shape     Params
─────────────────────────────────────────
LSTM (4 units)    (None, 4)        96
Dense (1 unit)    (None, 1)        5
─────────────────────────────────────────
Total params: 101
```

## Requirements

- Python 3.8+
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

Install all dependencies:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## Usage

1. Clone or download this repository.
2. Ensure `airline-passengers.csv` is in the same directory as the notebook.
3. Open and run the notebook:

```bash
jupyter notebook Lstm_airline-passengers_.ipynb
```

## Evaluation Metric

The model is evaluated using **Root Mean Squared Error (RMSE)** on both the training and test sets after inverse-transforming predictions back to the original passenger-count scale.

## License

This project is for educational and demonstration purposes.
