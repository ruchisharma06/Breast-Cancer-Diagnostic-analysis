# Breast-Cancer-Diagnostic-analysis
This repository contains a comprehensive analysis of the Breast Cancer Wisconsin (Diagnostic) dataset. The dataset is used for binary classification to distinguish between malignant and benign breast tumors based on various features computed from digitized images of fine needle aspirates (FNA) of breast masses.

## Table of Contents
1. [Installation](#installation)
2. [Data Description](#data-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Feature Importance](#feature-importance)
6. [Model Training](#model-training)
7. [Evaluation Metrics](#evaluation-metrics)
8. [TensorBoard Visualization](#tensorboard-visualization)

## Installation

To set up the environment, run:

```bash
pip install ucimlrepo pandas seaborn numpy matplotlib scipy scikit-learn torch tensorboard Pillow
```

## Data Description

The dataset is fetched from the UCI Machine Learning Repository using the `ucimlrepo` package. It contains 569 instances with 30 numerical features and one categorical target column (`Diagnosis`).

- **ID:** Identifier (ignored in analysis)
- **Diagnosis:** Categorical (M = Malignant, B = Benign)
- **Features:** 30 numerical features computed from FNA images

## Data Preprocessing

1. **Remove Outliers:** Outliers are replaced with the median of the respective feature.
2. **Map Target Values:** 'M' is mapped to 0 and 'B' is mapped to 1 for model training.

```python
def remove_outlier(x):
    median = np.median(df[x])
    Q1 = np.percentile(df[x], 25, method='midpoint')
    Q3 = np.percentile(df[x], 75, method='midpoint')
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    df[x] = np.where(df[x] > upper, median, df[x])
    df[x] = np.where(df[x] < lower, median, df[x])

for feature in df.columns[:-1]:  # Excluding 'Diagnosis'
    remove_outlier(feature)

df['Diagnosis'] = df['Diagnosis'].replace({'M': 0, 'B': 1})
```

## Exploratory Data Analysis

- **Boxplot:** Identifies the presence of outliers.
- **Correlation Heatmap:** Shows the correlation between features and the target variable.

```python
plt.figure(figsize=(20, 9))
sns.heatmap(df.corr(), annot=True).set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=20)
```

## Feature Importance

Chi-Square test is used to determine the importance of features.

```python
from scipy.stats import chisquare

feature_columns = df.columns[:-1]
result = pd.DataFrame(columns=["Features", "Chi2Weights"])

for column in feature_columns:
    chi2, _ = chisquare(df[column])
    result = result.append({"Features": column, "Chi2Weights": chi2}, ignore_index=True)

result = result.sort_values(by="Chi2Weights", ascending=False)
print(result)
```

## Model Training

The dataset is split into training and testing sets. A simple neural network is implemented using PyTorch.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['Diagnosis'], test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Evaluation Metrics

Performance is evaluated using confusion matrix, accuracy, precision, recall, and F1-score.

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Predictions and evaluation
y_pred = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
y_pred = (y_pred > 0.5).astype(int)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
```

## TensorBoard Visualization

TensorBoard is used to visualize training metrics.

```python
from torch.utils.tensorboard import SummaryWriter

# Clear previous logs
import shutil
import os

log_dir = 'runs'
shutil.rmtree(log_dir, ignore_errors=True)
os.makedirs(log_dir, exist_ok=True)

writer = SummaryWriter(log_dir)

# Training loop with TensorBoard logging
for epoch in range(num_epochs):
    # Training code...
    writer.add_scalar('Loss/train', loss.item(), epoch)

writer.close()
```

## Repository Structure

```
.
├── data_preprocessing.py
├── eda.py
├── feature_importance.py
├── model_training.py
├── evaluation.py
└── README.md
```

This README provides a high-level overview of the steps involved in the analysis of the Breast Cancer Wisconsin (Diagnostic) dataset. For detailed code and further explanations, refer to the respective Python scripts.
