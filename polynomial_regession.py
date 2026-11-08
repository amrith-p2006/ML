import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import make_pipeline

df=pd.read_csv("C:\\Users\\USER\\OneDrive\\Documents\\C Programming\\ml\\potti\\auto-mpg.csv")

print(df.head())
print(df.columns)

print(df.isnull().sum())

df=df.dropna(subset=["displacement","mpg"])   

X=df["displacement"].values.reshape(-1,1)
y=df["mpg"].values

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)


linear_model=LinearRegression()

linear_model.fit(xtrain,ytrain)

ypred=linear_model.predict(xtest)

degrees=[2,3,4,5]
models={}
results=[]
best_model=None
for d in degrees:
    model=make_pipeline(PolynomialFeatures(degree=d, include_bias=False),StandardScaler(),LinearRegression())
    model.fit(xtrain,ytrain)
    y_pred_poly=model.predict(xtest)
    models[d]=model
    mse=mean_squared_error(ytest,y_pred_poly)
    r2=r2_score(ytest,y_pred_poly)
    results.append((d,mse,r2))
    print("\n\nDegree:",d,"\nMSE:",mse,"\nR2:",r2)

    if best_model is None or r2>best_model[2]:
        best_model=(d,mse,r2)   

print("\n\nBest Model:\nDegree:",best_model[0],"\nMSE:",best_model[1],"\nR2:",best_model[2])

X_curve = np.linspace(
    X.min(), X.max(), 300
).reshape(-1, 1)


plt.figure(figsize=(10,6))
plt.scatter(xtest.flatten(),ytest,color="blue",label="Actual Data",alpha=0.5)
plt.plot(X_curve.flatten(),linear_model.predict(X_curve),color="red",label="Linear Regression")
plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.title("Polynomial Regression")
plt.legend()
plt.show()
for degree in degrees:

    y_curve = models[degree].predict(X_curve)

    plt.plot(
        X_curve.flatten(),
        y_curve,
        label=f"Polynomial Degree {degree}"
    )

plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")

plt.title("Linear vs Polynomial Regression - Auto MPG")

plt.legend()
plt.grid(True)
plt.show()