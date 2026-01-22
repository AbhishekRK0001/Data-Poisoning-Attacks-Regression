📌 Data Poisoning Attacks on Regression Models
This mini project experimentally analyzes the impact of data poisoning attacks on regression-based machine learning models and evaluates multiple defense mechanisms to improve robustness. The project simulates adversarial manipulation of training data and measures performance degradation using standard regression metrics.


🎯 Objectives
Study vulnerabilities of regression models to data poisoning
Implement label flipping and outlier injection attacks
Evaluate defense techniques such as:
Z-score filtering
IQR filtering
Isolation Forest
RANSAC regression
Compare performance using RMSE and R²


🧠 Project Overview
Machine learning models often assume training data is trustworthy. However, when data comes from untrusted sources, attackers can poison the data to degrade model performance. This project demonstrates how even small poisoning fractions can significantly affect regression models and how robust defenses can mitigate these attacks.


🗂 Project Structure
data-poisoning-regression/
│
├── src/                  # Core logic (attacks, defenses, models)
├── notebooks/            # Step-by-step experiments
├── scripts/              # Automation and analysis
├── results/              # CSVs and plots
├── models/               # Saved models
├── reports/              # Project report
├── requirements.txt
└── README.md


⚙️ Requirements
Python 3.10+
Libraries:
numpy
pandas
scikit-learn
matplotlib
seaborn


Install dependencies:
pip install -r requirements.txt

▶️ How to Run the Project
Step 1: Activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

Step 2: Run baseline and experiments
python scripts/run_grid.py

Step 3: Analyze results
python scripts/analyze_results.py

Results are saved under:
results/
results/figures/


📊 Evaluation Metrics
RMSE (Root Mean Square Error)
R² Score


📈 Key Findings
Regression models are highly sensitive to poisoned data
IQR filtering and RANSAC provide strong robustness
Isolation Forest may remove valid samples if not tuned


🔮 Future Scope
Extend to classification and deep learning models
Implement adaptive defenses
Study real-time and streaming poisoning attacks


👨‍🎓 Academic Use
This project was developed as a Mini Project for academic purposes and follows VTU / APS College guidelines.


📄 License
This project is for educational and research purposes only.
