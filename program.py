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
print("----------------------------Linearity Checking----------------------------------")
sns.set_theme(style="whitegrid")
ax = sns.scatterplot(y="Life Expectancy at Birth", x="Happiness Score", data=df,
               palette="Set1")
plt.title("Life Expectancy at Birth vs Happiness Score", y=1,x=0.5, fontsize = 16)
ax.set( ylabel='Life Expectancy at Birth (years)')
plt.show()
ax = sns.scatterplot(y="Life Expectancy at Birth", x="GDP per Capita", data=df,
               palette="Set1")
plt.title("Life Expectancy at Birth vs GDP per Capita", y=1,x=0.5, fontsize = 16)
ax.set( ylabel='Life Expectancy at Birth (years)')
plt.show()
ax = sns.scatterplot(y="Life Expectancy at Birth", x="GNI per Capita", data=df,
               palette="Set1")
plt.title("Life Expectancy at Birth vs GNI per Capita", y=1,x=0.5, fontsize = 16)
ax.set( ylabel='Life Expectancy at Birth (years)')
plt.show()
ax = sns.scatterplot(y="Life Expectancy at Birth", x="Freedom", data=df,
               palette="Set1")
plt.title("Life Expectancy at Birth vs Freedom", y=1,x=0.5, fontsize = 16)
ax.set( ylabel='Life Expectancy at Birth (years)')
plt.show()

sns.residplot(y="Life Expectancy at Birth", x="Happiness Score", data=df, color='magenta')
plt.title('Residual plot Happiness', size=24)
plt.xlabel('Happiness Score', size=18)
plt.ylabel('Life Expectancy at Birth (years)', size=18);
plt.show()

sns.residplot(y="Life Expectancy at Birth", x="GDP per Capita", data=df, color='magenta')
plt.title('Residual plot GDP per Capita', size=24)
plt.xlabel('GDP per Capita', size=18)
plt.ylabel('Life Expectancy at Birth (years)', size=18);
plt.show()

sns.residplot(y="Life Expectancy at Birth", x="GNI per Capita", data=df, color='magenta')
plt.title('Residual plot GNI per Capita', size=24)
plt.xlabel('GNI per Capita', size=18)
plt.ylabel('Life Expectancy at Birth (years)', size=18);
plt.show()

sns.residplot(y="Life Expectancy at Birth", x="Freedom", data=df, color='magenta')
plt.title('Residual plot Freedom', size=24)
plt.xlabel('Freedom', size=18)
plt.ylabel('Life Expectancy at Birth (years)', size=18);
plt.show()
print("------------------------------------Linear Regression-------------------------------------")
x = df.values[:, 1:3]      
y = df.values[:, 0]        
print("------------------Training and testing set----------------------")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print("------------------ Feature scaling ----------------------")
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
regr = linear_model.LinearRegression().fit(X_train, y_train)
print("------------------ Sample cases: ----------------------")
sample = [6, 1.5]       
print("1.")
for column, value in zip(list(df)[1:3], sample):
    print(column + ': ' + str(value))
sample_pred = regr.predict([sample])
print('Predicted length of Life Expectancy at Birth:', sample_pred)
print('-----------------------')
sample = [6, 2.5]       
print("2.")
for column, value in zip(list(df)[1:3], sample):
    print(column + ': ' + str(value))
sample_pred = regr.predict([sample])
print('Predicted length of Life Expectancy at Birth:', sample_pred)
print('-----------------------')
sample = [7, 1.5]       
print("3.")
for column, value in zip(list(df)[1:3], sample):
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
