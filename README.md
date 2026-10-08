# ML Techniques Comparison and Optimization

A practical machine learning study comparing multiple classification models and preprocessing techniques on the **Default of Credit Card Clients** dataset.

The project evaluates different approaches for handling outliers, class imbalance, dimensionality reduction, and feature selection, followed by lightweight hyperparameter tuning for the best-performing models.

## Project Overview

The goal of this project is to investigate how different machine learning techniques affect classification accuracy rather than relying on a single model or preprocessing approach.

The workflow includes:

1. Dataset exploration and preprocessing
2. Comparison of five classification models
3. Outlier detection comparison
4. Class imbalance handling comparison
5. Dimensionality reduction comparison
6. Correlation-based feature selection
7. Hyperparameter tuning
8. Final model comparison

---

## Dataset

**Default of Credit Card Clients Dataset**

* 30,000 samples
* 25 original columns
* 23 predictive features after removing `ID`
* Binary classification target: `default.payment.next.month`
* No missing values
* Train/Test split: 80% / 20%
* Stratified split
* `random_state = 42`

### Target Distribution

| Class | Samples | Percentage |
| ----- | ------: | ---------: |
| 0     |  23,364 |     77.88% |
| 1     |   6,636 |     22.12% |

The dataset is imbalanced, with the non-default class representing the majority.

---

## Baseline Model Comparison

Five classification algorithms were evaluated using their default parameters:

* Logistic Regression
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Random Forest
* XGBoost

Standardization was applied to models that require scaled features.

### Results

| Model               |   Accuracy |
| ------------------- | ---------: |
| **SVM**             | **0.8160** |
| Random Forest       |     0.8137 |
| XGBoost             |     0.8108 |
| Logistic Regression |     0.8077 |
| KNN                 |     0.7928 |

The **SVM** achieved the highest baseline accuracy and was therefore used as the main model for the technique-comparison experiments.

---

## Outlier Detection

Five outlier detection techniques were evaluated using SVM:

* IQR
* Z-Score
* Isolation Forest
* Local Outlier Factor (LOF)
* Elliptic Envelope

### Results

| Method            | SVM Accuracy |
| ----------------- | -----------: |
| IQR               |       0.7788 |
| Z-Score           |       0.8130 |
| Isolation Forest  |       0.8130 |
| **LOF**           |   **0.8163** |
| Elliptic Envelope |       0.8143 |

LOF produced the highest accuracy among the tested outlier detection methods, although its improvement over the baseline was very small.

---

## Class Imbalance

Several approaches were evaluated for handling the imbalanced target:

* Random Over Sampling
* Random Under Sampling
* SMOTE
* ADASYN
* SMOTEENN
* Borderline-SMOTE
* Tomek Links
* SMOTETomek
* Class Weight

### Results

| Method                | SVM Accuracy |
| --------------------- | -----------: |
| Random Over Sampling  |       0.7745 |
| Random Under Sampling |       0.7750 |
| SMOTE                 |       0.7422 |
| ADASYN                |       0.7253 |
| SMOTEENN              |       0.6282 |
| Borderline-SMOTE      |       0.7380 |
| **Tomek Links**       |   **0.8155** |
| SMOTETomek            |       0.7408 |
| Class Weight          |       0.7763 |

Tomek Links performed best among the tested imbalance techniques, although the original imbalanced dataset still produced slightly better accuracy.

---

## Dimensionality Reduction

Two dimensionality reduction techniques were evaluated:

* PCA
* LDA

### PCA

Different numbers of principal components were evaluated using 5-fold cross-validation:

| Components | CV Accuracy |
| ---------: | ----------: |
|          2 |      0.8035 |
|          3 |      0.8032 |
|          5 |      0.8034 |
|         10 |      0.8059 |
|         15 |      0.8165 |
|     **20** |  **0.8198** |

PCA with 20 components achieved the best cross-validation result.

Final test accuracy:

**PCA + SVM: 0.8158**

### LDA

LDA reduced the 23 original features to one discriminant component.

**LDA + SVM Accuracy: 0.8122**

---

## Technique Combination

The best-performing techniques from the previous experiments were combined:

**LOF → Tomek Links → StandardScaler → PCA(20) → SVM**

Final accuracy:

**0.8177**

The combined pipeline improved only slightly over the original SVM baseline of 0.8160.

---

## Correlation-Based Feature Selection

Instead of continuing with more complex preprocessing combinations, correlation-based feature selection was evaluated.

The absolute correlation between each feature and the target was calculated using the training data.

Several correlation thresholds were tested:

| Threshold | Features |   Accuracy |
| --------: | -------: | ---------: |
|      0.01 |       21 |     0.8162 |
|  **0.05** |   **13** | **0.8177** |
|      0.10 |        7 |     0.8172 |
|      0.15 |        7 |     0.8172 |
|      0.20 |        5 |     0.8168 |

The best result was obtained using a correlation threshold of **0.05**, reducing the feature set from 23 to 13 features while achieving an accuracy of **0.8177**.

### Selected Features

```text
PAY_0
PAY_2
PAY_3
PAY_4
PAY_5
PAY_6
LIMIT_BAL
PAY_AMT1
PAY_AMT2
PAY_AMT3
PAY_AMT4
PAY_AMT5
PAY_AMT6
```

This showed that a simpler feature-selection approach could achieve a result comparable to the more complex combined pipeline.

---

## Hyperparameter Tuning

After feature selection, lightweight hyperparameter tuning was performed for both SVM and Random Forest.

### Tuned SVM

The following parameters were tested:

* `C`: 0.1, 1, 10
* `gamma`: `scale`, 0.01, 0.1
* `kernel`: RBF

The best validation configuration was:

```text
C = 1
gamma = scale
kernel = rbf
```

Validation accuracy:

**0.8194**

Final test accuracy:

**0.8177**

---

### Tuned Random Forest

The following parameters were tested:

* `n_estimators`: 100, 200, 300
* `max_depth`: None, 10, 20

The best validation configuration was:

```text
n_estimators = 300
max_depth = 10
```

Validation accuracy:

**0.8183**

Final test accuracy:

**0.8183**

---

## Final Comparison

| Approach                                    | Test Accuracy |
| ------------------------------------------- | ------------: |
| Baseline SVM                                |        0.8160 |
| Correlation Feature Selection + SVM         |        0.8177 |
| Tuned SVM + Selected Features               |        0.8177 |
| **Tuned Random Forest + Selected Features** |    **0.8183** |

### Final Model

The best final test accuracy was achieved by:

**Random Forest + Correlation Feature Selection**

Configuration:

```text
Features: 13
n_estimators: 300
max_depth: 10
Accuracy: 0.8183
```

---

## Key Findings

* SVM was the strongest baseline model among the five tested classifiers.
* Most outlier and class-imbalance techniques did not improve raw accuracy.
* PCA with 20 components performed better than the smaller PCA configurations, but its final test accuracy was still slightly below the baseline SVM.
* Combining multiple preprocessing techniques provided only a small improvement.
* Correlation-based feature selection reduced the feature count from **23 to 13** while maintaining a similar accuracy.
* Lightweight tuning improved the Random Forest result, making it the final best-performing model with **81.83% accuracy**.

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* imbalanced-learn
* Google Colab
* Matplotlib / Seaborn

## Project Structure

```text
ML_Techniques_Comparison_and_Optimization/
│
├── ML_Techniques_Comparison_and_Optimization.ipynb
└── README.md
```

## Conclusion

This project demonstrates a practical comparison of machine learning models and preprocessing techniques for a binary classification problem.

Rather than assuming that more preprocessing always produces better results, the experiments show that simpler approaches can perform just as well or better. The final model achieved **81.83% test accuracy** using only 13 selected features with a tuned Random Forest classifier.
