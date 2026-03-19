
# 📱 Customer Churn Intervention Simulator

> **Predict** telecom customer churn, **explain** it with SHAP, and **simulate** business interventions — live.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-GPU-green)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 What This Project Does

Most churn predictors stop at **"will this customer churn? Yes/No."**

This project goes further:

1. **Predicts** churn probability per customer (XGBoost, ROC-AUC = 0.82)
2. **Explains** the prediction with SHAP waterfall and force plots
3. **Simulates** business interventions — change contract type, tenure, charges — and sees the risk update **live**
4. **Logs** every simulation to SQLite for team review

---

## 🖥️ Live Demo

Live app coming soon on Streamlit Cloud.

---

## 🏗️ Project Structure
```
churn-intervention-simulator/
├── src/
│ ├── data_loader.py # Load + clean + split Telco CSV
│ ├── preprocessor.py # ColumnTransformer pipeline
│ ├── trainer.py # XGBoost training + artifacts
│ ├── predictor.py # Single + batch inference API
│ ├── explainer.py # SHAP TreeExplainer wrappers
│ ├── evaluator.py # Metrics + evaluation plots
│ └── database.py # SQLite session logging
├── app/
│ ├── main.py # Streamlit entry point
│ ├── components/
│ │ └── sidebar.py # Customer selector
│ └── tabs/
│ ├── tab_simulator.py # What-If Simulator
│ ├── tab_explainability.py # SHAP global + local plots
│ ├── tab_performance.py # Model metrics dashboard
│ └── tab_history.py # Session history + trend chart
├── configs/
│ └── config.yaml # Hyperparams + paths
├── models/ # Trained artifacts (gitignored)
├── data/raw/ # Raw CSV (gitignored)
├── requirements.txt
└── setup.py
```

---

## ⚡ Key Features

| Feature | Detail |
|---|---|
| Live What-If Simulator | Adjust contract, tenure, charges and churn risk updates instantly |
| SHAP Force Plots | Interactive before and after explanation side by side |
| SHAP Beeswarm | Global feature importance across 300 test customers |
| Waterfall Charts | Per-customer detailed feature contribution breakdown |
| Session Logging | SQLite DB logs every simulation with optional notes |
| Risk Trend Chart | Risk reduction over logged sessions visualised with Plotly |
| GPU Training | XGBoost trained on CUDA, inference switched to CPU automatically |

---

## 🚀 Run Locally

### 1. Clone and setup

```bash
git clone https://github.com/Priyanshukv06/churn-intervention-simulator.git
cd churn-intervention-simulator
python -m venv projects
projects\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Download the dataset

Download **Telco Customer Churn** from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it at:

```
data/raw/telco_churn.csv
```

### 3. Train the model

```bash
python -m src.trainer
```

### 4. Launch the app

```bash
streamlit run app/main.py
```


Open http://localhost:8501

📊 Model Results
| Metric    | Score  |
| --------- | ------ |
| ROC-AUC   | 0.8228 |
| PR-AUC    | 0.6272 |
| Recall    | 71.66% |
| Precision | 52.34% |
| F1 Score  | 60.50% |
| Accuracy  | 75.12% |

Class imbalance is handled with scale_pos_weight = 2.763 (negatives / positives ratio).


🧠 How SHAP Explainability Works

The module uses shap.TreeExplainer on the trained XGBoost model.

    Force plots show which features pushed the prediction up or down from the base rate

    Waterfall charts stack individual feature contributions to reach the final prediction

    Beeswarm plot gives a global view — each dot is one customer, color is feature value, x-axis is SHAP impact

    Bar chart shows mean absolute SHAP value per feature across the test set

The What-If simulator recomputes SHAP on every slider change so the explanation updates in real time.


🗃️ Session Logging

Every simulation is saved to a local SQLite database at data/sessions.db. Each session stores:

    Timestamp

    Original vs simulated churn probability and delta

    All intervention parameters (contract, tenure, charges, internet, payment)

    Optional free-text notes

The Session History tab shows stats cards, a filterable table, a Plotly risk trend chart, and a danger zone to delete sessions.

🔧 Tech Stack
| Layer             | Tools                                       |
| ----------------- | ------------------------------------------- |
| ML Model          | XGBoost with GPU training and CPU inference |
| Preprocessing     | scikit-learn ColumnTransformer              |
| Explainability    | SHAP TreeExplainer                          |
| App Framework     | Streamlit                                   |
| Visualisation     | Plotly, Matplotlib                          |
| SHAP in Streamlit | streamlit-shap                              |
| Storage           | SQLite via Python sqlite3                   |
| Language          | Python 3.11                                 |


👤 Author

Priyanshu Verma
GitHub: https://github.com/Priyanshukv06
Location: Indore, Madhya Pradesh, India

📄 License

This project is licensed under the MIT License.