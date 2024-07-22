import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
import sqlite3

# Load data
df = pd.read_csv('kidney_disease.csv', na_values='?')
df.info()

# Rename columns for consistency and readability
df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'aanemia', 'class']

# Handle incorrect entries and missing values
df['diabetes_mellitus'] = df['diabetes_mellitus'].replace({'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'})
df['coronary_artery_disease'] = df['coronary_artery_disease'].replace({'\tno': 'no'})
df['class'] = df['class'].replace({'ckd\t': 'ckd', 'notckd': 'not ckd'}).map({'ckd': 0, 'not ckd': 1})

# Convert numeric columns to appropriate types
num_cols = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'blood_urea', 'blood_glucose_random',
            'serum_creatinine', 'sodium', 'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count',
            'red_blood_cell_count']

# Describe numeric columns to check for issues
df.describe()
df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')
# Convert columns to numeric, coercing errors
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Impute missing values with column mean for numeric columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Identify categorical columns
cat_cols = [col for col in df.columns if df[col].dtype == 'object']

# Impute missing values with mode for categorical columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Check unique values in categorical columns
for col in cat_cols:
    print(f"{col} has {df[col].unique()} values\n")

# Encode categorical variables
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Plot distribution of numeric features
plt.figure(figsize=(20, 15))
plotnumber = 1

for column in num_cols:
    if plotnumber <= 14:
        ax = plt.subplot(3, 5, plotnumber)
        sns.histplot(df[column], kde=True)
        plt.xlabel(column)
        
    plotnumber += 1

plt.tight_layout()
plt.show()

# Plot count of categorical features
plt.figure(figsize=(20, 15))
plotnumber = 1

for column in cat_cols:
    if plotnumber <= 11:
        ax = plt.subplot(3, 4, plotnumber)
        sns.countplot(x=column, data=df, hue=column, legend=False, palette='rocket')
        plt.xlabel(column)
        
    plotnumber += 1

plt.tight_layout()
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(df.corr(), annot=True, linewidths=2, linecolor='lightgrey')
plt.show()

# Split data into features and target variable
X = df.drop('class', axis=1)
y = df['class']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scale data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Save scaler
pickle.dump(sc, open("scaler.pkl", 'wb'))

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

# Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

# Saving decision tree
plt.figure(figsize=(12, 8))
plot_tree(dtc, filled=True, feature_names=X.columns, class_names=['No CKD', 'CKD'])
plt.title("Decision Tree Classifier")

buffer = io.BytesIO()
plt.savefig(buffer, format='png')
plt.close()


buffer.seek(0)
image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

with open("decision_tree_plot.bin", "wb") as f:
    f.write(image_base64.encode('utf-8'))

# Random Forest Classifier
rd_clf = RandomForestClassifier(criterion='entropy', max_depth=10, max_features='sqrt', min_samples_leaf=5,
                                min_samples_split=2, n_estimators=400, random_state=42)
rd_clf.fit(X_train, y_train)
rd_clf_acc = accuracy_score(y_test, rd_clf.predict(X_test))

# Logistic Regression Evaluation
y_pred_lr = lr.predict(X_test)
print("Logistic Regression:")
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
conf_mat_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat_lr, annot=True, cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Decision Tree Classifier Evaluation
y_pred_dtc = dtc.predict(X_test)
print("\nDecision Tree Classifier:")
print("Classification Report:")
print(classification_report(y_test, y_pred_dtc))
print("Accuracy:", accuracy_score(y_test, y_pred_dtc))
conf_mat_dtc = confusion_matrix(y_test, y_pred_dtc)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat_dtc, annot=True, cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix - Decision Tree Classifier")
plt.show()

# Random Forest Classifier Evaluation
y_pred_rd = rd_clf.predict(X_test)
print("\nRandom Forest Classifier:")
print("Classification Report:")
print(classification_report(y_test, y_pred_rd))
print("Accuracy:", accuracy_score(y_test, y_pred_rd))
conf_mat_rd = confusion_matrix(y_test, y_pred_rd)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat_rd, annot=True, cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix - Random Forest Classifier")
plt.show()

# Save the Random Forest model
pickle.dump(rd_clf, open("models_kidney.pkl", 'wb'))

# Save accuracies
ac = [lr_acc, dtc_acc, rd_clf_acc]
pickle.dump(ac, open("accuracy.pkl", 'wb'))

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('kidney_disease.db')
c = conn.cursor()

# Create doctor table
c.execute('''
CREATE TABLE IF NOT EXISTS doctor (
    doctor_mail_id TEXT PRIMARY KEY,
    doctor_name TEXT
)
''')

# Create patient table
c.execute('''
CREATE TABLE IF NOT EXISTS patient (
    patient_mail_id TEXT PRIMARY KEY,
    patient_name TEXT,
    doctor_mail_id TEXT,
    age INTEGER,
    blood_pressure INTEGER,
    specific_gravity REAL,
    albumin INTEGER,
    sugar INTEGER,
    red_blood_cells INTEGER,
    pus_cell INTEGER,
    pus_cell_clumps INTEGER,
    bacteria INTEGER,
    blood_glucose_random INTEGER,
    blood_urea INTEGER,
    serum_creatinine REAL,
    sodium INTEGER,
    potassium REAL,
    haemoglobin REAL,
    packed_cell_volume INTEGER,
    white_blood_cell_count INTEGER,
    red_blood_cell_count REAL,
    hypertension INTEGER,
    diabetes_mellitus INTEGER,
    coronary_artery_disease INTEGER,
    appetite INTEGER,
    peda_edema INTEGER,
    aanemia INTEGER,
    FOREIGN KEY (doctor_mail_id) REFERENCES doctor(doctor_mail_id)
)
''')

# Insert a sample doctor record
# c.execute('insert into doctor values("sat@","sat")')
c.execute('select * from doctor')
c.execute('insert into doctor values("sathwik@gmail.com","sathwik")')
c.execute('insert into doctor values("sunil@gmail.com","sunil")')
data = c.fetchall()
for d in data:
    print(d)

# Commit changes and close the connection
conn.commit()
conn.close()
