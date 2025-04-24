## Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Store data in a structured format (e.g., CSV, DataFrame).
2. Use a Simple Linear Regression model to fit the training data.
3. Use the trained model to predict values for the test set.
4. Evaluate performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). 

## Program:

## NAME : THARUN.V
## REG NO : 212224230290

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df= pd.read_csv('/content/student_scores.csv')

df.head()
```

## Output:
![ml s1](https://github.com/user-attachments/assets/b6db29af-6687-4789-a9a8-74dffd6324d3)


```
df.tail()
```
## Output:
![ml s2](https://github.com/user-attachments/assets/93c39355-0ec7-4e74-a4af-c36169b239ef)


```
df.info()
```
## Output:
![ml s3](https://github.com/user-attachments/assets/50667f44-91d5-4607-9193-a80df047bede)


```
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,-1].values
print(y)
```
## Output:
![ml s4](https://github.com/user-attachments/assets/c76d084e-5b4e-40db-88c2-7638e1caf60d)


```
print(x.shape)
print(y.shape)
```
## Output:
![ml s5](https://github.com/user-attachments/assets/e808ad7a-af39-4057-a193-76886ae60b4a)


```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
```

## Output:
![ml s6](https://github.com/user-attachments/assets/79466a34-97aa-4095-acdc-a81ec067e279)

```
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df
```

## Output:
![ml s7](https://github.com/user-attachments/assets/3073ab79-a9b6-476f-91fd-5779a12ab272)


```
mse = mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae = mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse = np.sqrt(mse)
print('RMSE = ',rmse)
```
## Output:

![ml s8](https://github.com/user-attachments/assets/3ddc35ae-f668-4248-b85e-4b24638cb844)

```
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:
![training set](https://github.com/user-attachments/assets/07b90a13-5fe0-4030-b61d-26b621b9835c)


```
plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,y_pred,color="red")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:
![testing set](https://github.com/user-attachments/assets/7d22c497-476c-46f7-b8bf-7d30b4a4f064)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
