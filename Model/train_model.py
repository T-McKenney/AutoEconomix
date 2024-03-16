import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the dataset
data = pd.read_csv('CarsData.csv')

print('Running the model')

# Preliminary data preprocessing
# Dropping any rows with missing values for simplicity
data.dropna(inplace=True)

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, drop_first=True)

# Defining features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
error = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {error}')

# Visualize actual vs predicted prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictions)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Car Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.show()

# Save the trained model
with open('model/model.pkl', 'wb') as file:
    pickle.dump(model, file)