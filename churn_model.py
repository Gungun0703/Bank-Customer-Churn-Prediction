# 📊 Data handling
import pandas as pd
import numpy as np

# 📈 Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# 🤖 Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv("Churn_Modelling.csv")
print(df.head())
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
X = df.drop('Exited', axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

feature_importance = pd.DataFrame({
    'Feature': df.drop('Exited', axis=1).columns,
    'Importance': rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print(feature_importance.head(10))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title("Top Factors Affecting Customer Churn")
plt.show()
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
new_customer = [[
    600,   # CreditScore
    1,     # Gender (1 = Male, 0 = Female)
    40,    # Age
    3,     # Tenure
    60000, # Balance
    2,     # NumOfProducts
    1,     # HasCrCard
    1,     # IsActiveMember
    50000, # EstimatedSalary
    0,     # Geography_Germany
    1      # Geography_Spain
]]
new_customer = scaler.transform(new_customer)
prediction = rf_model.predict(new_customer)
if prediction[0] == 1:
    print("⚠️ Customer will leave the bank (CHURN)")
else:
    print("✅ Customer will stay")
    importance_df = feature_importance.head(10)
print(importance_df)
import joblib

joblib.dump(rf_model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")