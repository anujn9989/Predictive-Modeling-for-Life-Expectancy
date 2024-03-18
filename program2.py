import pandas as pd
from math import sqrt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('final_data1.csv', thousands=',')
df['GNI per Capita'] = pd.to_numeric(df['GNI per Capita'])
print("------------------------------------Linear Regression-------------------------------------")
x = df.values[:, 3:]      
y = df.values[:, 0]        
print("------------------Training and testing set----------------------")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
print("------------------ Feature scaling ----------------------")
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
regr = linear_model.LinearRegression().fit(X_train, y_train)
print("------------------ Sample cases: ----------------------")
sample = [6, 0.5]       
print("1.")
for column, value in zip(list(df)[3:], sample):
    print(column + ': ' + str(value))
sample_pred = regr.predict([sample])
print('Predicted length of Life Expectancy at Birth:', sample_pred)
print('-----------------------')
sample = [5, 0.5]       
print("2.")
for column, value in zip(list(df)[3:], sample):
    print(column + ': ' + str(value))
sample_pred = regr.predict([sample])
print('Predicted length of Life Expectancy at Birth:', sample_pred)
print('-----------------------')
sample = [6, 1.5]       
print("3.")
for column, value in zip(list(df)[1:], sample):
    print(column + ': ' + str(value))
sample_pred = regr.predict([sample])
print('Predicted length of Life Expectancy at Birth:', sample_pred)
print('-----------------------')

print('Coefficients:')
print(regr.coef_)
print('Intercept:')
print(regr.intercept_)
print("------------------ Predicting the test set results ----------------------")
y_pred = regr.predict(X_test)
df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_pred)

mse = metrics.mean_squared_error(y_test, y_pred)
print('Root mean squared error (RMSE):', sqrt(mse))

print('R-squared score:', metrics.r2_score(y_test, y_pred))
print("\n")
