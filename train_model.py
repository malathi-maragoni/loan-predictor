import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("loan_data.csv")

# Encode categorical features
categorical_cols = ['Gender','Married','Education','Self_Employed','Property_Area']
for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# Encode target
data['Loan_Status'] = LabelEncoder().fit_transform(data['Loan_Status'])

# Split features and target
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "loan_model.pkl")
print("✅ Model trained and saved as loan_model.pkl")
