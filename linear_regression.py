import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("C:\\Users\\USER\\OneDrive\\Documents\\C Programming\\ml\\potti\\housing.csv")

#preprocessing 

df.columns
df.isnull().sum()
df = df.dropna(subset=[
    "total_rooms",
    "households",
    "median_house_value"
])

df = df[df["households"] > 0]

X=(df['total_rooms']/df['households']).values.reshape(-1,1)
y=df['median_house_value'].values/100000

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

X_train=X_train.flatten()
X_test=X_test.flatten()
#gradient descent 

w=0.0
b=0.0

learning_rate=0.001
epoch=2000

n=len(X_train)

for i in range(epoch):
    y_pred=w*X_train+b
    dw=(-2/n)*np.sum(X_train*(y_train-y_pred))
    db=(-2/n)*np.sum(y_train-y_pred)
    
    w=w-learning_rate*dw
    b=b-learning_rate*db

print("Gradient Descent Results:\nWeight:",w,"\nBias:",b)
y_pred_tg=w*X_test + b

#normal equation

X_b=np.c_[np.ones(len(X_train)),X_train]

theta=np.linalg.pinv(X_b.T@X_b)@X_b.T@y_train

b=theta[0]
w=theta[1]

print("Normal Equation Results:\nWeight:",w,"\nBias:",b)

y_pred_ne=X_b@theta

y_pred_tn=w*X_test + b

#evaluation

mse_gd=mean_squared_error(y_test,y_pred_tg)
mse_ne=mean_squared_error(y_test,y_pred_tn)

r2_gd=r2_score(y_test,y_pred_tg)
r2_ne=r2_score(y_test,y_pred_tn)

print("Gradient Descent MSE:",mse_gd,"\nR2 Score:",r2_gd)
print("Normal Equation MSE:",mse_ne,"\nR2 Score:",r2_ne)


indices = np.argsort(X_test)

plt.figure(figsize=(10, 6))

plt.scatter(
    X_test,
    y_test,
    color="blue",
    alpha=0.5,
    label="Actual Data"
)

plt.plot(
    X_test[indices],
    y_pred_tg[indices],
    color="red",
    label="Gradient Descent"
)

plt.plot(
    X_test[indices],
    y_pred_tn[indices],
    color="green",
    linestyle="--",
    label="Normal Equation"
)

plt.xlabel("Standardized Average Rooms")
plt.ylabel("Median House Value (100,000 dollars)")
plt.title("Linear Regression - California Housing Dataset")

plt.legend()
plt.grid()
plt.show()

