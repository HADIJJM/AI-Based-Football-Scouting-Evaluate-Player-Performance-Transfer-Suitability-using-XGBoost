# AI-Based Football Scouting: Evaluate Player Performance & Transfer Suitability using XGBoost

An advanced, data-driven football scouting pipeline designed to objectively evaluate player performance and predict transfer suitability using high-resolution event data. Developed as part of the Machine Learning course at the Department of Computer Engineering, College of Computer Science and Information Technology, **Imam Abdulrahman Bin Faisal University (IAU)**.

---

# Project Overview

Traditional football scouting heavily relies on subjective evaluations, leading to costly transfer mistakes. While early analytical models bridged some gaps, they heavily biased attacking positions by focusing solely on surface-level metrics like goals and assists, leaving defensive contributions statistically underrepresented.

This project introduces an end-to-end automated scouting framework powered by the **XGBoost** algorithm and a custom **Justice Formula**. By utilizing high-granularity spatial and contextual event data from the StatsBomb Open Data dataset, the model captures both offensive and defensive performance fairly, calculates a normalized performance index, and evaluates players exclusively within their positional peer groups.

## Key Features

- **Justice Formula:** Balances offensive and defensive contributions to generate a normalized performance score.
- **Position-Normalized Evaluation:** Compares players only against others in the same position.
- **Interactive Streamlit Dashboard:** Provides an easy-to-use interface for scouts to evaluate players using AI-powered predictions.
- **Young Talent Detection:** Automatically identifies promising players aged **23 or younger**.
- **Playing Style Classification:** Classifies players based on the quality and frequency of their technical actions.

---

# Justice Formula & Performance Index

To reduce the historical bias toward attacking players, a weighted performance metric was designed:

```text
Justice Formula Score =
(Goals × 3)
+ (Assists × 2)
+ (Interceptions × 1.5)
+ (Clearances × 1)
```

The overall Performance Index is then calculated as:

```text
Performance Index =
((Offensive Score + Defensive Score) / Total Actions) × 100
```

This metric reflects a player's overall contribution relative to their involvement during the match.

---

# Machine Learning Model

The project uses **XGBoost (Extreme Gradient Boosting)** as the primary classifier and compares its performance with a Logistic Regression baseline.

## Model Features

- `total_actions` – Total passes, carries, duels, shots, tackles, and other actions.
- `avg_x` – Average field position (length).
- `avg_y` – Average field position (width).
- `pressure_events` – Actions performed under opponent pressure.
- `offensive_score` – Aggregated attacking contribution.
- `defensive_score` – Aggregated defensive contribution.
- `performance_index` – Normalized efficiency score.
- `duels` – Defensive and physical contests.
- `carries` – Ball progression actions.

---

# Experimental Results

The optimized XGBoost model was trained on more than **422,000 event records** collected from **200 matches** across Europe's Top Five leagues.

| Metric | Score |
|---------|-------|
| Classification Accuracy | **96.50%** |
| ROC-AUC | **0.9943** |
| Elite Precision | **87%** |
| Elite Recall | **95%** |

---

# Streamlit Dashboard

The application (`app.py`) serves as an AI-powered decision support tool for football scouts.

Features include:

- Manual player metric input.
- Predefined scouting profiles.
- Instant prediction of transfer suitability.
- Classification as **Elite Target** or **Standard Prospect**.
- Interactive visualizations and normalized performance charts.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ai-football-scout.git
cd ai-football-scout
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Required libraries include:

- streamlit
- xgboost
- pandas
- scikit-learn
- joblib

Run the application:

```bash
streamlit run app.py
```

---

# Research & Development Team

Developed by Computer Engineering students at **Imam Abdulrahman Bin Faisal University (IAU)**.

- **Hadi Al-Hajji** (Lead)
- **Mahdi Al-Khamis** - @Psycodem
- **Omar Al-Subgh** – 

**Instructor**

Mohammad Aftab

---

# Dataset

This project is built using the **StatsBomb Open Data** event dataset, which provides high-resolution football event data for research and educational purposes.

---

# Technologies Used

- Python
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Matplotlib
- Joblib

---

# Project Structure

```text
ai-football-scout/
│
├── app.py
├── model.pkl
├── requirements.txt
├── data/
├── notebooks/
├── images/
└── README.md
```

---

# License

This project was developed for educational and academic purposes as part of the Machine Learning course at Imam Abdulrahman Bin Faisal University.
