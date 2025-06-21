#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score


# In[22]:


# Add CSV file
import pandas as pd

df = pd.read_csv("YesBank.csv")

# Check null values
print("\n🔍 Null values in each column:")
print(df.isnull().sum())

# Data information
print("\nℹ️ DataFrame info:")
print(df.info())

# Convert Date column to datetime (Month-Year format)
df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')

# Create a YearMonth column for grouping
df['YearMonth'] = df['Date'].dt.to_period('M')

# Group data monthly and compute mean values
monthly_df = df.groupby('YearMonth').agg({
    'Open': 'mean',
    'High': 'mean',
    'Low': 'mean',
    'Close': 'mean'
}).reset_index()

# Check processed data
print("\n✅ First few rows of the processed monthly data:")
print(monthly_df.head())



# In[3]:


# Define features (inputs) and target (output)
X = monthly_df[['Open', 'High', 'Low']]
y = monthly_df['Close']

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[4]:


# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Train, predict, and evaluate
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'r2_score': r2
    }
    
    print(f"🔹 {name} R2 Score: {r2:.4f}")


# In[5]:


from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression


# In[6]:


# Initialize scaler normalization
scaler = StandardScaler()

# Fit scaler on training data and transform
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[7]:


# Initialize feature selector to pick top 2 features (you can adjust k)
selector = SelectKBest(score_func=f_regression, k=2)

# Fit on training data
X_train_selected = selector.fit_transform(X_train_scaled, y_train)

# Transform test data
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print(f"✅ Selected Features: {list(selected_features)}")


# In[8]:


# Use the same models dictionary

results = {}

for name, model in models.items():
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'r2_score': r2
    }
    
    print(f"🔹 {name} R2 Score after Feature Selection + Normalization: {r2:.4f}")


# In[13]:


y_actual = y_test.values
y_pred_lr = results['Linear Regression']['y_pred']

plt.figure(figsize=(8,5))
plt.plot(y_actual, 'o-', label='Actual', color='black')
plt.plot(y_pred_lr, 'o--', label='Predicted Linear Regression', color='blue')

plt.xlabel('Test Sample Index')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted: Linear Regression (After Feature Selection + Normalization)')
plt.legend()
plt.grid(True)
plt.show()


# In[14]:


y_pred_rf = results['Random Forest']['y_pred']

plt.figure(figsize=(8,5))
plt.plot(y_actual, 'o-', label='Actual', color='red')
plt.plot(y_pred_rf, 'o--', label='Predicted Random Forest', color='green')

plt.xlabel('Test Sample Index')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted: Random Forest (After Feature Selection + Normalization)')
plt.legend()
plt.grid(True)
plt.show()


# In[16]:


y_pred_gb = results['Gradient Boosting']['y_pred']

plt.figure(figsize=(8,5))
plt.plot(y_actual, 'o-', label='Actual', color='black')
plt.plot(y_pred_gb, 'o--', label='Predicted Gradient Boosting', color='gold')

plt.xlabel('Test Sample Index')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted: Gradient Boosting (After Feature Selection + Normalization)')
plt.legend()
plt.grid(True)
plt.show()


# In[17]:


# Collect model names and R² scores
model_names = list(results.keys())
r2_scores = [res['r2_score'] for res in results.values()]

# Plot
plt.figure(figsize=(6,4))
plt.bar(model_names, r2_scores, color=['blue', 'green', 'orange'])
plt.ylabel('R² Score (Accuracy)')
plt.title('Model Accuracy Comparison (After Feature Selection + Normalization)')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[21]:


# Get feature names that were selected
selected_features = X.columns[selector.get_support()]

# Get importance
rf_model = results['Random Forest']['model']
importance_rf = rf_model.feature_importances_

# Plot
plt.figure(figsize=(6,4))
plt.bar(selected_features, importance_rf, color='brown')
plt.ylabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[20]:


gb_model = results['Gradient Boosting']['model']
importance_gb = gb_model.feature_importances_

# Plot
plt.figure(figsize=(6,4))
plt.bar(selected_features, importance_gb, color='purple')
plt.ylabel('Importance')
plt.title('Feature Importance (Gradient Boosting)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[ ]:




