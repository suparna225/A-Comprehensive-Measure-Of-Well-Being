import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/Human Development Index - Full.csv")

# ==========================================
# 2. SELECT REQUIRED COLUMNS
# ==========================================

data = df[
    [
        "Country",
        "Life Expectancy at Birth (2021)",
        "Expected Years of Schooling (2021)",
        "Mean Years of Schooling (2021)",
        "Gross National Income Per Capita (2021)",
        "Human Development Index (2021)"
    ]
].copy()

# ==========================================
# 3. HANDLE MISSING VALUES
# ==========================================

numeric_columns = [
    "Life Expectancy at Birth (2021)",
    "Expected Years of Schooling (2021)",
    "Mean Years of Schooling (2021)",
    "Gross National Income Per Capita (2021)",
    "Human Development Index (2021)"
]

for column in numeric_columns:
    data[column] = data[column].fillna(data[column].mean())

# ==========================================
# 4. EXPLORATORY DATA ANALYSIS
# ==========================================

plt.figure(figsize=(8, 5))
sns.histplot(
    data["Human Development Index (2021)"],
    kde=True
)
plt.title("Distribution of Human Development Index")
plt.xlabel("HDI Score")
plt.ylabel("Number of Countries")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=data,
    x="Life Expectancy at Birth (2021)",
    y="Human Development Index (2021)"
)
plt.title("Life Expectancy vs HDI")
plt.xlabel("Life Expectancy")
plt.ylabel("HDI Score")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=data,
    x="Mean Years of Schooling (2021)",
    y="Human Development Index (2021)"
)
plt.title("Mean Years of Schooling vs HDI")
plt.xlabel("Mean Years of Schooling")
plt.ylabel("HDI Score")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=data,
    x="Gross National Income Per Capita (2021)",
    y="Human Development Index (2021)"
)
plt.title("GNI Per Capita vs HDI")
plt.xlabel("GNI Per Capita")
plt.ylabel("HDI Score")
plt.show()

plt.figure(figsize=(10, 7))

correlation = data[numeric_columns].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix of HDI Indicators")
plt.show()

# ==========================================
# 5. DEFINE FEATURES AND TARGET
# ==========================================

X = data[
    [
        "Life Expectancy at Birth (2021)",
        "Expected Years of Schooling (2021)",
        "Mean Years of Schooling (2021)",
        "Gross National Income Per Capita (2021)"
    ]
]

y = data["Human Development Index (2021)"]

# ==========================================
# 6. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# ==========================================
# 7. CREATE LINEAR REGRESSION MODEL
# ==========================================

model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")

# Save the trained model
with open("hdi_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as hdi_model.pkl")

# ==========================================
# 8. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 9. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-------------------------")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-Squared Score:", r2)

# ==========================================
# 10. SHOW ACTUAL VS PREDICTED VALUES
# ==========================================

results = pd.DataFrame({
    "Actual HDI": y_test.values,
    "Predicted HDI": y_pred
})

print("\nACTUAL VS PREDICTED:")
print(results.head(10))