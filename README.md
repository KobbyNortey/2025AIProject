# Energy-Use Forecasting – 2025 AI Project  
AI Final Project – Streamlit demo + Keras regression model_

> **Purpose**  
> Using Machine Learning to predict energy consumption on campus per semster


---
## Deployed Project link
https://aifinalproject10.streamlit.app

## 1. High-level goals
| # | Outcome |
|---|---------|
| 1 | **Data pipeline** that cleans raw meter readings, engineers calendar / weather features, and stores a tidy frame ready for modelling. |
| 2 | **Keras network** (GRU + dense head) that predicts the kWh demand with < 6 % MAPE on the public test split. |
| 3 | **Preprocessing bundle** (`preprocessor.joblib`) keeping train / inference transformations identical. |
| 4 | **Model artefact** (`energy_net.keras`) saved after early-stopping. |
| 5 | **Streamlit UI** (`app.py`) for quick experimentation: CSV upload, single-row form, graphs, and error metrics. |
| 6 | **Dev-container** so you can open the folder in VS Code, hit _Run_, and everything _just works_. |

---

## 2. Folder map
```
2025AIProject
├── .devcontainer/        # VS Code remote-container settings
│   └── devcontainer.json
├── artefacts/            # Reusable objects
│   ├── energy_net.keras
│   └── preprocessor.joblib
├── app.py                # Streamlit entry-point
├── EnergyConsumption.ipynb  # Notebook: EDA → training → export artefacts
├── train_energy_data.csv # Dataset
├── requirements.txt      # Python libs for prod
└── README.md             # ← you are here
```

---

## 3. Quick run (local Python)

1. **Clone and activate a venv**
   ```bash
   git clone https://github.com/<your-org>/2025AIProject.git
   cd 2025AIProject
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Streamlit**
   ```bash
   streamlit run app.py
   ```
   Your browser should open `http://localhost:8501` with two tabs:

    * **Upload CSV** – drop a file shaped like `train_energy_data.csv`; get predictions.
    * **Single row** – manual feature sliders for “what-if” analysis.

---

## 4. Re-training in 4 steps

> _All code lives in the **EnergyConsumption.ipynb** notebook – nothing hidden._

1. **Load data**  
   `pd.read_csv('train_energy_data.csv')`

2. **Preprocess**
    * converts timestamps
    * adds hour-of-day, day-of-week, holiday, °C bin indicators
    * splits 80 / 10 / 10 train-val-test
    * saves `preprocessor.joblib`

3. **Fit model**
   ```python
   model.fit(
       X_train,
       y_train,
       epochs=200,
       callbacks=[EarlyStopping(...), ModelCheckpoint(...)]
   )
   ```

4. **Export weights** – automatically saved to `artefacts/energy_net.keras`.  
   _Restart Streamlit and it will hot-load the new file._



## 7. Evaluation snapshot

| Split | MAE (kWh) | RMSE (kWh) | MAPE (%) |
|-------|-----------|------------|----------|
| Train | 1.79      | 2.45       | 4.8      |
| Val   | 1.92      | 2.67       | 5.2      |
| Test  | 2.01      | 2.74       | 5.9      |

> Validation uses an expanding-window walk-forward scheme to respect temporal ordering.



## 8. Credits

| Name                   |
|------------------------|
| Shawn Owusu-Nortey     | 
| Maame Afua Kome-Mensah |
| Debora Suday           |
| Kingsford Amissah      |

---