# Dataset Documentation: Credit Card Fraud Detection

## Overview
The dataset contains transactions made by credit cards in September 2013 by European cardholders. It covers 284,807 transactions that occurred across two days, where 492 transactions are fraudulent (0.172%).

## Features
- **Time**: Elapsed seconds between the given transaction and the first transaction in the dataset.
- **V1 - V28**: Principal components obtained via PCA due to confidentiality constraints.
- **Amount**: Transaction transaction amount in Euros.
- **Class**: Target indicator (1 for fraudulent transactions, 0 for legitimate transactions).

## Data Cleaning & Splitting
- Raw transactions: 284,807
- Deduplicated transactions: 283,726 (1,081 duplicate transactions dropped)
- Train split: 80% (226,980 transactions; 378 frauds, 226,602 legitimate)
- Test split: 20% (56,746 transactions; 95 frauds, 56,651 legitimate)

## Source
The raw `creditcard.csv` dataset is available on Kaggle:
[Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place `creditcard.csv` directly inside this `data/` directory to enable full exploratory data analysis.
