import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Step 1: Load the clinical data
file_path = "../data/dlbcl_duke_2017/data_clinical_patient.txt"
df = pd.read_csv(file_path, sep="\t", comment="#")

# Step 2: Inspect and clean your target variable
# Print the unique responses to see what values you have
print("Unique therapy responses:", df["INITIAL_TX_RESPONSE"].unique())

# Drop rows where the therapy response is missing
df = df.dropna(subset=["INITIAL_TX_RESPONSE"])

# Create a binary target variable: 1 for "Complete response", 0 for all others
df["Response"] = df["INITIAL_TX_RESPONSE"].str.strip().str.lower().apply(
    lambda x: 1 if x == "complete response" else 0
)

# Step 3: Encode categorical variables using one-hot encoding
# Here we assume "SEX" is a categorical variable; adjust if necessary.
df_encoded = pd.get_dummies(df, columns=["SEX"], drop_first=True)

# Display columns to check encoding
print("Encoded columns:", df_encoded.columns)

# Step 4: Define features and target
# Using AGE_AT_DIAGNOSIS, IPI, and the encoded SEX column if present.
features = ["AGE_AT_DIAGNOSIS", "IPI"]
if "SEX_Female" in df_encoded.columns:
    features.append("SEX_Female")

X = df_encoded[features]
y = df_encoded["Response"]

# Step 5: Handle missing values in the features via imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 8: Evaluate the model on the test set
y_pred = clf.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
