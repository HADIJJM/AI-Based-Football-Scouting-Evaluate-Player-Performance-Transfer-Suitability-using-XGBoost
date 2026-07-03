# AI-Based Football Scouting: Evaluate Player Performance & Transfer Suitability using XGBoost

An AI-powered football scouting system that objectively evaluates player performance and predicts transfer suitability using machine learning and high-resolution football event data.

Developed as part of the **Machine Learning** course at the **Department of Computer Engineering, College of Computer Science and Information Technology, Imam Abdulrahman Bin Faisal University (IAU).**

---

## Project Overview

Traditional football scouting often relies on subjective opinions and basic statistics such as goals and assists, which can undervalue players in defensive positions.

This project introduces a data-driven scouting framework that evaluates players more fairly by considering both offensive and defensive contributions. Using **XGBoost** and detailed event data from **StatsBomb Open Data**, the system provides objective player evaluation and transfer suitability prediction.

---

## Key Features

- Fair player evaluation using a custom **Justice Formula**
- Position-based player comparison
- AI-powered transfer suitability prediction
- Young talent detection (players aged **23 or younger**)
- Interactive **Streamlit** dashboard
- Performance visualization and player analysis

---

## Machine Learning Model

The model was trained using **more than 422,000 football event records** extracted from **200 matches** across **Europe's Top Five leagues**.

### Features Used

- Total Actions
- Average Field Position
- Pressure Events
- Offensive Score
- Defensive Score
- Performance Index
- Duels
- Carries

The primary classifier is **XGBoost**, with **Logistic Regression** used as a baseline for comparison.

---

## Results

| Metric | Score |
|--------|------:|
| **Accuracy** | **96.50%** |
| **ROC-AUC** | **0.9943** |
| **Elite Precision** | **87%** |
| **Elite Recall** | **95%** |

---

## Streamlit Application

The project includes an interactive **Streamlit** dashboard that allows users to:

- Enter player statistics
- Predict transfer suitability
- Classify players as **Elite** or **Standard**
- Visualize player performance
- Explore predefined scouting profiles

Run the application using:

```bash
streamlit run app.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ai-football-scout.git
cd ai-football-scout
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```

---

## Dataset

This project uses the **StatsBomb Open Data** dataset for research and educational purposes.

> **Note:** The dataset is **not included** in this repository due to its size.

Please download the dataset from the official repository:

https://github.com/statsbomb/open-data

After downloading, place the required files inside the `data/` directory before running the application.

---

## Technologies Used

- Python
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Matplotlib
- Joblib

---

## Project Structure

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

## Research & Development Team

Developed by Computer Engineering students at **Imam Abdulrahman Bin Faisal University (IAU).**

- **Hadi Al-Hajji** — Team Lead
- **Mahdi Al-Khamis** - @Psycodem
- **Omar Al-Subgh**

### Instructor

- **Mohammad Aftab**

---

## License

This project was developed for educational and academic purposes as part of the **Machine Learning** course at **Imam Abdulrahman Bin Faisal University (IAU)**.
