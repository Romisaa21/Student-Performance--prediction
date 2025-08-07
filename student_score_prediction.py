
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data =pd.read_csv("D:\projects(training)\StudentPerformanceFactors.csv")
data

data.info()

data.isnull().sum()

pd.get_dummies(data.dropna()).sum()

pd.get_dummies(data).sum()

data.dropna()

pd.get_dummies(data).sum()

plt.boxplot(data['Exam_Score'])
plt.show()

plt.hist(data.Exam_Score, bins=30, rwidth=0.9)
plt.xlabel('Exam_Score')
plt.title('Data with outlier values')
plt.show()

data.Exam_Score.describe()

Q1 = data.Exam_Score.quantile(0.25)
Q3 = data.Exam_Score.quantile(0.75)
IQR = Q3 - Q1
IQR

lower_limit_IQR=Q1-1.5*IQR
upper_limit_IQR=Q3+1.5*IQR

lower_limit_IQR

upper_limit_IQR

data=data[(data.Exam_Score<upper_limit_IQR) & (data.Exam_Score>lower_limit_IQR)]
data

plt.hist(data.Exam_Score, bins=20, rwidth=0.9)
plt.xlabel('Exam_Score')
plt.title('Data without outliers using the IQR metrics')
plt.show()

plt.boxplot(data['Hours_Studied'])
plt.show()

Q1 = data.Hours_Studied.quantile(0.25)
Q3 = data.Hours_Studied.quantile(0.75)
IQR = Q3 - Q1
IQR

lower_limit_IQR=Q1-1.5*IQR
upper_limit_IQR=Q3+1.5*IQR

lower_limit_IQR

upper_limit_IQR

data=data[(data.Hours_Studied<upper_limit_IQR) & (data.Hours_Studied>lower_limit_IQR)]
data

plt.hist(data.Hours_Studied, bins=20, rwidth=0.9)
plt.xlabel('Hours_Studied')
plt.title('Data without outliers using the IQR metrics')
plt.show()

import seaborn as sns
corr_matrix = data[['Exam_Score', 'Hours_Studied']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix with Heatmap')
plt.show()

from sklearn.model_selection import train_test_split
x = data[['Hours_Studied']]
y = data['Exam_Score']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x_train)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

y_pred

y_test

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score

print('MAE:', mean_absolute_error(y_test, y_pred))
print('MAPE:', mean_absolute_percentage_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('R² Score:', r2_score(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.scatter(x_test, y_test, color='blue', label='Actual')
plt.scatter(x_test, y_pred, color='red', label='Predicted')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Actual vs Predicted Exam Scores')
plt.legend()
plt.show()

plt.figure(figsize=(18, 9))

plt.scatter(x_test, y_test, color='blue', label='Actual Data', alpha=0.5)

X_test_sorted = np.sort(x_test.values.flatten())
y_pred_sorted = model.predict(X_test_sorted.reshape(-1, 1))
plt.plot(X_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Linear Regression: Hours Studied vs Exam Score')
plt.legend()
plt.grid(True)
plt.show()

"""polynomial regression"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

degree = 3
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg.fit(x_train, y_train)
y_pred_poly = polyreg.predict(x_test)

print('MAE:', mean_absolute_error(y_test, y_pred_poly))
print('MAPE:', mean_absolute_percentage_error(y_test, y_pred_poly))
print('MSE:', mean_squared_error(y_test, y_pred_poly))
print('R² Score:', r2_score(y_test, y_pred_poly))

plt.figure(figsize=(12, 8))
plt.scatter(x_test, y_test, color='blue', label='Actual Data', alpha=0.5)
y_pred_poly_sorted = polyreg.predict(X_test_sorted.reshape(-1, 1))
plt.plot(X_test_sorted, y_pred_poly_sorted, color='green', linewidth=2, label=f'Polynomial Regression (Degree {degree})')

plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Linear vs Polynomial Regression: Hours Studied vs Exam Score')
plt.legend()
plt.grid(True)
plt.show()

new_hours_studied = np.array([[9.25]])
predicted_exam_score_p = polyreg.predict(new_hours_studied)
print(f"Predicted Exam Score using poly_reg for {new_hours_studied[0][0]} hours studied: {predicted_exam_score_p[0]}")

predicted_exam_score_l = model.predict(new_hours_studied)
print(f"Predicted Exam Score using linear_reg for {new_hours_studied[0][0]} hours studied: {predicted_exam_score_l[0]}")

"""# Linear Regression and Polynomial using Multi features"""

features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions', 'Sleep_Hours']
X = data[features]
Y = data['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

degree = 3
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)
y_pred_poly = polyreg.predict(X_test)

print("\nLinear Regression Metrics:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_linear):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred_linear):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_linear):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred_linear):.4f}")

print(f"\nPolynomial Regression (Degree = {degree}) Metrics:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_poly):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred_poly):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_poly):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred_poly):.4f}")