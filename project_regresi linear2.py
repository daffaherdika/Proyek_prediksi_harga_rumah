import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'C:/Users/daffa/Downloads/DATA RUMAH.xlsx'
data = pd.read_excel(file_path)

# Data cleaning: Separate numeric and non-numeric columns manually
numeric_columns = ['HARGA', 'Luas Bangunan', 'Luas Tanah', 'Kamar Tidur', 'Kamar Mandi', 'Garasi']
data = data[numeric_columns]

# Handle missing or invalid data: Drop rows with NaN values
data = data.dropna()

# Define features (X) and target (y)
X = data.drop(columns=["HARGA"])
y = data["HARGA"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=356)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

# Print the coefficients and intercept
print("Intercept:", model.intercept_)
coefficients = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
print("\nRegression Coefficients:\n", coefficients)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Perform 10-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=356)

# Cross-validation for R² Score
r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
mean_r2_score = np.mean(r2_scores)

print("\n10-Fold Cross-Validation R² Score:")
print(f"Mean R² Score: {mean_r2_score:.2f} ({mean_r2_score * 100:.2f}%)")


# Visualisasi regresi untuk data latih dan data uji
plt.figure(figsize=(10, 6))

# Plot data latih
plt.scatter(y_train, y_train_pred, color='orange', label='Data Latih')

# Plot data uji
plt.scatter(y_test, y_pred, color='red', label='Data Uji')

# Plot garis identitas
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], linestyle='--', color='gray', label='Garis Identitas')

# Label sumbu x dan y
plt.xlabel('Nilai Sebenarnya')
plt.ylabel('Nilai Prediksi')

# Judul plot
plt.title('Prediksi Model Regresi Linier')

# Menambahkan legenda
plt.legend()

# Menampilkan plot
plt.show()

# Function to predict price for new input
def predict_new_data():
    print("\nEnter the feature values for the new house:")
    new_data = []
    
    # Replace the list of features with the actual ones used in your dataset
    for feature in X.columns:
        value = float(input(f"Enter value for {feature}: "))
        new_data.append(value)
    
    # Convert the input into a DataFrame
    new_data_df = pd.DataFrame([new_data], columns=X.columns)
    
    # Make prediction
    predicted_price = model.predict(new_data_df)
    print(f"\nPredicted Price for the new house: {predicted_price[0]:,.2f} Rupiah")

# Call the function to allow manual input and prediction
predict_new_data()
