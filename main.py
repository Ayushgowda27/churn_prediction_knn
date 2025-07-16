import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load Dataset
print("Loading dataset...")
df = pd.read_csv("data/churn_data.csv")
print("Dataset loaded. Shape:", df.shape)

# Optional: Show basic stats
print("\nBasic Info:")
print(df['Churn'].value_counts())

# Step 2: Data Preprocessing
df.drop(columns=['customerID'], inplace=True, errors='ignore')
df = pd.get_dummies(df, drop_first=True)
df = df.dropna()

# Step 3: Split Data
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train k-NN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Step 6: Predictions and Evaluation
y_pred = knn.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Step 8: Bar Graph (Churn distribution)
plt.figure(figsize=(5, 4))
sns.countplot(x="Churn_Yes", data=df)
plt.title("Churn Count (0 = No, 1 = Yes)")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# Step 9: Scatter Plot (Tenure vs Monthly Charges)
if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df['tenure'], y=df['MonthlyCharges'], hue=df['Churn_Yes'], alpha=0.7)
    plt.title("Tenure vs Monthly Charges by Churn")
    plt.xlabel("Tenure (months)")
    plt.ylabel("Monthly Charges")
    plt.legend(title="Churn")
    plt.tight_layout()
    plt.show()
    plt.close()
else:
    print("Scatter plot skipped: 'tenure' or 'MonthlyCharges' column missing.")
