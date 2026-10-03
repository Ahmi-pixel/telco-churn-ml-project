# Telco Customer Churn Prediction

**CS-13410 Introduction to Machine Learning — Semester Project (Fall 2026)**
The University of Lahore · Department of Computer Science & IT · BSCS

## Group Members
| Name | Roll No. |
|---|---|
| Syed Ahmad Ali | 70147064 |
| Umair Sajjid | 70146750 |
| Jawad Raza | 70146212 |

## Project Overview
Telecom companies lose revenue when customers cancel their subscriptions (**churn**). This project builds a machine learning model that predicts whether a customer is likely to leave, based on their account, service and billing information. The company could then contact high-risk customers with retention offers before they leave.

- **Task:** Binary classification (Churn: Yes / No)
- **Main metric:** ROC-AUC
- **Other metrics:** F1-score and recall for the churn class (accuracy is misleading because only ~26.5% of customers churn)

## Dataset
- **Name:** IBM Telco Customer Churn (also on Kaggle as "Telco Customer Churn")
- **Source:** IBM sample dataset. The customers are fictional and it is released for educational use.
- **Size:** 7,043 customers × 21 columns. Numeric columns include tenure and charges; categorical columns include contract, internet service and payment method.
- **File:** `Telco-Customer-Churn.csv`

## Repository Structure
| File | Description |
|---|---|
| `Part1_Data_Preparation_Pipeline.ipynb` | Part 1 notebook: problem framing, EDA, preprocessing, leakage-free pipeline |
| `Part1_Project_Proposal.docx` | One-page project proposal |
| `Telco-Customer-Churn.csv` | Raw dataset |

## Project Progress
| Part | Topic | Status |
|---|---|---|
| Part 1 | Data understanding, preprocessing and leakage-free pipeline | ✅ Done |
| Part 2 | Naive Bayes vs. decision tree baseline | ⏳ Upcoming |
| Part 3 | Ensembles, regularisation and hyperparameter tuning | ⏳ Upcoming |
| Part 4 | Neural network, clustering, interpretability and fairness | ⏳ Upcoming |
| Part 5 | Final notebook, report and viva | ⏳ Upcoming |

## Part 1 Summary
- Stratified 80/20 train-test split **before** any analysis, so the test set stays unseen.
- EDA on the training data with 6 visualisations. Churn is highest for new customers, month-to-month contracts, fiber-optic internet and electronic-check payments.
- Four engineered features, e.g. charge-to-tenure ratio (the most informative feature found).
- A scikit-learn `Pipeline` + `ColumnTransformer` that handles missing values, outliers, scaling, one-hot encoding and feature selection. It is fitted on training data only, so there is no data leakage.
- Class imbalance is handled with stratified splitting, class weights and imbalance-aware metrics.
- 5-fold cross-validation sanity check: ROC-AUC ≈ 0.85.

## How to Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook Part1_Data_Preparation_Pipeline.ipynb
```
Keep `Telco-Customer-Churn.csv` in the same folder as the notebook. If it is missing, the notebook downloads it automatically.
