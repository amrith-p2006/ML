# ============================================================
# MACHINE LEARNING LAB - 15 PROGRAMS
# All datasets are loaded manually from local CSV files using
# pandas read_csv(). Keep the CSV files in the same folder as
# this Python script, or replace the filenames with full paths.
# ============================================================


# ------------------------------------------------------------
# QUESTION 1: Linear Regression - California Housing
# Expected CSV: california_housing.csv
# Required columns: AveBedrms (4th feature in the original
# sklearn dataset, matching data.data[:, 3]) and MedHouseVal.
# ------------------------------------------------------------
import pandas as pd# ------------------------------------------------------------

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("california_housing.csv")
print(data.head())
x =data[["total_rooms"]].values
y = data["median_house_value"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n = len(x_train)
b0 = 0
b1 = 0
lr = 0.01
iter = 1000

for i in range(iter):
    y_pred = b0 + b1 * x_train.flatten()
    error = y_pred - y_train
    db1 = (1/n) * np.sum(error * x_train.flatten())
    db0 = (1/n) * np.sum(error)
    b0 = b0 - lr * db0
    b1 = b1 - lr * db1

y_pred_gd = b0 + b1 * x_test.flatten()

A = np.c_[np.ones((len(x_train), 1)), x_train]
theta = np.linalg.inv(A.T @ A) @ A.T @ y_train
b0 = theta[0]
b1 = theta[1]
y_pred_nor = b0 + b1 * x_test.flatten()

print("Gradient Descent")
print("MSE:", mean_squared_error(y_test, y_pred_gd))
print("R2:", r2_score(y_test, y_pred_gd))

print("Normal Equation")
print("MSE:", mean_squared_error(y_test, y_pred_nor))
print("R2:", r2_score(y_test, y_pred_nor))

plt.scatter(x_test, y_test, color="blue", label="Actual Data", alpha=0.3)
plt.scatter(x_test, y_pred_gd, color="red",
            label="Gradient Descent Predictions", alpha=0.3)
plt.scatter(x_test, y_pred_nor, color="green",
            label="Normal Equation Predictions", alpha=0.3)
plt.xlabel("AveBedrms (Standardized)")
plt.ylabel("Median House Value")
plt.title("Linear Regression on California Housing Dataset")
plt.legend()
plt.show()



# ------------------------------------------------------------
# QUESTION 2: Polynomial Regression - Auto MPG
# Expected CSV: auto-mpg.csv
# Required columns: displacement, mpg
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("auto-mpg.csv")
data = data[["displacement", "mpg"]].dropna()

X = data[["displacement"]]
y = data["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Linear Regression")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

X_curve = np.linspace(X.min().iloc[0], X.max().iloc[0], 100).reshape(-1, 1)
y_curve = model.predict(X_curve)
plt.plot(X_curve, y_curve, label="Linear Regression", color="black")

for degree in [2, 3, 4]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)

    print("\nPolynomial Degree:", degree)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    X_curve_poly = poly.transform(X_curve)
    y_curve = model.predict(X_curve_poly)
    plt.plot(X_curve, y_curve, label="Degree " + str(degree))

plt.scatter(X, y, color="blue", label="Actual Data", alpha=0.5)
plt.xlabel("Engine Displacement")
plt.ylabel("MPG")
plt.title("Linear and Polynomial Regression")
plt.legend()
plt.show()


# ------------------------------------------------------------
# QUESTION 3: Ridge and Lasso Regression - Diabetes
# Expected CSV: diabetes.csv
# Required columns: all feature columns plus target.
# ------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("diabetes.csv")
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear = LinearRegression()
linear.fit(X_train, y_train)
linear_pred = linear.predict(X_test)

ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

lasso = LassoCV(alphas=[0.01, 0.1, 1, 10, 100],
                cv=5, max_iter=10000)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

models = {
    "Linear": linear_pred,
    "Ridge": ridge_pred,
    "Lasso": lasso_pred
}

mse_values = []
r2_values = []

for name, pred in models.items():
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    mse_values.append(mse)
    r2_values.append(r2)
    print("\n", name, "Regression")
    print("MSE:", round(mse, 2))
    print("R2 Score:", round(r2, 4))

print("\nBest Ridge Alpha:", ridge.alpha_)
print("Best Lasso Alpha:", lasso.alpha_)

names = list(models.keys())
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(names, mse_values)
plt.title("MSE Comparison")
plt.ylabel("MSE")

plt.subplot(1, 2, 2)
plt.bar(names, r2_values)
plt.title("R2 Score Comparison")
plt.ylabel("R2 Score")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
for name, pred in models.items():
    plt.scatter(y_test, pred, label=name)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()


# ------------------------------------------------------------
# QUESTION 4: Logistic Regression MLE and MAP - Breast Cancer
# Expected CSV: breast_cancer.csv
# Required columns: feature columns plus target.
# target must use sklearn-style labels: 0 = malignant, 1 = benign.
# ------------------------------------------------------------
imimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, f1_score

data=pd.read_csv("breast-cancer.csv")
print(data.head())
x=data.drop("class",axis=1)
x = pd.get_dummies(x, drop_first=True, dtype=int)
y=(data["class"]=="recurrence-events").astype(int)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

mod1=LogisticRegression(penalty=None)
mod1.fit(x_train,y_train)
pred1=mod1.predict(x_test)

mod2=LogisticRegression(penalty='l1',solver='liblinear',max_iter=100)
mod2.fit(x_train,y_train)
pred2=mod2.predict(x_test)

mod3=LogisticRegression(penalty='l2',max_iter=100)
mod3.fit(x_train,y_train)
pred3=mod3.predict(x_test)

models={"log":pred1,"l11":pred2,"l22":pred3}
accuracy = []
f1 = []
for name,value in models.items():
    a = accuracy_score(y_test, value)
    f = f1_score(y_test, value)
    print(name)
    print(a,f)
    accuracy.append(a)
    f1.append(f)

name=models.keys()
plt.bar(name,accuracy)
plt.title("accuracy")
plt.show()
plt.bar(name,f1)
plt.title("f1_score")
plt.show()
# ------------------------------------------------------------
# QUESTION 5: MLE and MAP - 20 Newsgroups
# Expected CSV: 20_newsgroups_train.csv
# Required columns: text. Each row contains one training document.
# ------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

data = pd.read_csv("20_newsgroups_train.csv")
data = data.dropna(subset=["text"])

vectorizer = CountVectorizer(max_features=1000, stop_words="english")
X = vectorizer.fit_transform(data["text"].astype(str))

counts = np.asarray(X.sum(axis=0)).ravel()
N = counts.sum()
V = len(counts)

mle = counts / N
print("\nMLE probabilities", mle[:10])

priors = [1, 1.5, 2, 5]
map_results = {}

for alpha in priors:
    map_estimate = (counts + alpha - 1) / (N + V * (alpha - 1))
    map_results[alpha] = map_estimate
    print(alpha)
    print(map_estimate[:10])

words = vectorizer.get_feature_names_out()[:10]
plt.figure(figsize=(10, 5))
plt.plot(words, mle[:10], "o-", label="MLE")

for alpha in priors:
    plt.plot(words, map_results[alpha][:10], "-o",
             label=f"MAP alpha={alpha}")

plt.xlabel("Words")
plt.ylabel("Probability")
plt.title("Comparison of MLE and MAP")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# QUESTION 6: Logistic Regression - Pima Indians Diabetes
# Expected CSV: diabetes.csv
# Required columns: Outcome and predictor columns.
# This is the Pima dataset, not the regression Diabetes dataset
# used in Question 3. Keep separate CSV files if both are used.
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("pima_diabetes.csv")
data = data.dropna()

X = data.drop(columns=["Outcome"])
y = data["Outcome"]

xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(xtrain, ytrain)
pred = model.predict(xtest)

print("Without Scaling\n")
print("Accuracy:", accuracy_score(ytest, pred))
print("Classification Report\n", classification_report(ytest, pred))

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

model.fit(xtrain, ytrain)
pred = model.predict(xtest)

print("With Scaling\n")
print("Accuracy:", accuracy_score(ytest, pred))
print("Classification Report\n", classification_report(ytest, pred))


# ------------------------------------------------------------
# QUESTION 7: Multinomial and Bernoulli Naive Bayes
# Expected CSVs: 20_newsgroups_train.csv and 20_newsgroups_test.csv
# Required columns in both: text, target.
# target contains the numeric class/category for each document.
# ------------------------------------------------------------
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv("20_newsgroups_train.csv").dropna(subset=["text", "target"])
test = pd.read_csv("20_newsgroups_test.csv").dropna(subset=["text", "target"])

vectorizer = CountVectorizer(stop_words="english")
xtrain = vectorizer.fit_transform(train["text"].astype(str))
xtest = vectorizer.transform(test["text"].astype(str))

ytrain = train["target"]
ytest = test["target"]

model = MultinomialNB()
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
print("Multinomial Naive Bayes")
print("Accuracy:", accuracy_score(ytest, pred))
print("F1 Score:", f1_score(ytest, pred, average="weighted"))

model = BernoulliNB()
model.fit(xtrain, ytrain)
pred = model.predict(xtest)
print("Bernoulli Naive Bayes")
print("Accuracy:", accuracy_score(ytest, pred))
print("F1 Score:", f1_score(ytest, pred, average="weighted"))


# ------------------------------------------------------------
# QUESTION 8: KNN - Fashion MNIST
# Expected CSVs: fashion-mnist_train.csv, fashion-mnist_test.csv
# Required columns: label and 784 pixel columns.
# ------------------------------------------------------------
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

train = pd.read_csv("fashion-mnist_train.csv")
test = pd.read_csv("fashion-mnist_test.csv")

xtrain = train.drop("label", axis=1)
xtest = test.drop("label", axis=1)
ytrain = train["label"]
ytest = test["label"]

xtrain = xtrain[:10000] / 255.0
ytrain = ytrain[:10000]
xtest = xtest[:2000] / 255.0
ytest = ytest[:2000]

for k in [1, 3, 5, 7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(xtrain, ytrain)
    pred = model.predict(xtest)
    print("\nK =", k)
    print("Accuracy:", accuracy_score(ytest, pred))
    print("Classification Report\n", classification_report(ytest, pred))


# ------------------------------------------------------------
# QUESTION 9: Decision Tree - Online Retail
# Expected CSV: OnlineRetail.csv
# Required columns: CustomerID, InvoiceNo, Quantity, UnitPrice.
# ------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("OnlineRetail.csv", encoding="latin1")
df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

df["CustomerID"] = df["CustomerID"].astype(int)
df["Amount"] = df["Quantity"] * df["UnitPrice"]

customer = df.groupby("CustomerID").agg(
    TotalAmount=("Amount", "sum"),
    Frequency=("InvoiceNo", "nunique"),
    AvgQuantity=("Quantity", "mean"),
    AvgPrice=("Amount", "mean")
)

customer["segment"] = pd.qcut(
    customer["TotalAmount"].rank(method="first"),
    q=3, labels=["low", "medium", "high"]
)

x = customer[["Frequency", "AvgQuantity", "AvgPrice"]]
y = customer["segment"]

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(
    criterion="entropy", max_depth=3, random_state=42
)
model.fit(xtrain, ytrain)
pred = model.predict(xtest)

print("Accuracy Score:", accuracy_score(ytest, pred))
print(classification_report(ytest, pred, zero_division=0))

plt.figure(figsize=(24, 12))
plot_tree(model, feature_names=x.columns,
          class_names=[str(c) for c in model.classes_],
          filled=True, rounded=True)
plt.title("Decision Tree")
plt.tight_layout()
plt.show()

importance = pd.Series(
    model.feature_importances_, index=x.columns
).sort_values(ascending=False)
print("\nFeature Importances\n")
print(importance)


# ------------------------------------------------------------
# QUESTION 10: Logistic Regression vs Decision Tree - Adult Income
# Expected CSV: adult.csv
# Required columns: income plus predictor columns.
# Income values should look like <=50K and >50K (optional final dot
# is removed below).
# ------------------------------------------------------------
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

data = pd.read_csv("adult.csv")
df = data.dropna()

x = df.drop("income", axis=1)
income = df["income"].astype(str).str.strip().str.rstrip(".")
y = (income == ">50K").astype(int)

cat = x.select_dtypes(include=["object", "category"]).columns
num = x.select_dtypes(exclude=["object", "category"]).columns

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", StandardScaler(), num)
])

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=42
)

xtrain = pre.fit_transform(xtrain)
xtest = pre.transform(xtest)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(xtrain, ytrain)
    pred = model.predict(xtest)
    print("\n", name)
    print("Accuracy:", accuracy_score(ytest, pred))
    print("Classification Report:\n", classification_report(ytest, pred))


# ------------------------------------------------------------
# QUESTION 11: Linear SVM - Iris
# Expected CSV: iris.csv
# Required columns: first two feature columns and target.
# target must use sklearn-style labels: 0 = Setosa.
# ------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iris = pd.read_csv("iris.csv")
x = iris.iloc[:, :2].values
y = (iris["target"] == 0).astype(int)

scaler = StandardScaler()
x = scaler.fit_transform(x)

model = SVC(kernel="linear")
model.fit(x, y)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap="coolwarm")

w = model.coef_[0]
b = model.intercept_[0]

X = x[:, 0]
boundary = -(w[0] * X + b) / w[1]
margin1 = -(w[0] * X + b - 1) / w[1]
margin2 = -(w[0] * X + b + 1) / w[1]

plt.plot(X, boundary, "k-")
plt.plot(X, margin1, "k--")
plt.plot(X, margin2, "k--")
plt.xlabel("Sepal Length (scaled)")
plt.ylabel("Sepal Width (scaled)")
plt.title("Linear SVM - Iris Classification")
plt.show()


# ------------------------------------------------------------
# QUESTION 12: SVM Kernels - Fashion MNIST
# Expected CSVs: fashion-mnist_train.csv, fashion-mnist_test.csv
# Required columns: label and 784 pixel columns.
# ------------------------------------------------------------
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

X_train = train_df.drop("label", axis=1) / 255.0
y_train = train_df["label"]
X_test = test_df.drop("label", axis=1) / 255.0
y_test = test_df["label"]

models = {
    "Linear": SVC(kernel="linear"),
    "Polynomial": SVC(kernel="poly", degree=3),
    "RBF": SVC(kernel="rbf")
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))


# ------------------------------------------------------------
# QUESTION 13: MLP - Wine Quality
# Expected CSV: winequality-red.csv
# This dataset is semicolon-separated, so sep=";" is required.
# Required column: quality plus numeric feature columns.
# ------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("winequality-red.csv", sep=";")
X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

architectures = {
    "One hidden layer - 10 neurons": (10,),
    "One hidden layer - 50 neurons": (50,),
    "Two hidden layers": (50, 30),
    "Three hidden layers": (100, 50, 25)
}

for name, layers in architectures.items():
    model = MLPClassifier(
        hidden_layer_sizes=layers,
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, zero_division=0))


# ------------------------------------------------------------
# QUESTION 14: Activation Functions - MNIST
# Expected CSVs: mnist_train.csv, mnist_test.csv
# Required columns: label and 784 pixel columns.
# ------------------------------------------------------------
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")

X_train = train.drop("label", axis=1).values / 255.0
y_train = train["label"].values
X_test = test.drop("label", axis=1).values / 255.0
y_test = test["label"].values

activations = ["logistic", "relu", "tanh"]

for act in activations:
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation=act,
        max_iter=20,
        random_state=42
    )

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    pred = model.predict(X_test)

    print("\nActivation:", act)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Training time:", round(end - start, 2), "seconds")
    print("Iterations:", model.n_iter_)

    plt.plot(model.loss_curve_, label=act)

plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.title("Convergence for Different Activation Functions")
plt.legend()
plt.show()


# ------------------------------------------------------------
# QUESTION 15: MLP Hyperparameter Tuning - Fashion MNIST
# Expected CSV: fashion-mnist_train.csv
# Required columns: label and 784 pixel columns.
# Uses the first 5000 rows, as in the original program.
# ------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("fashion-mnist_train.csv")

x = data.drop("label", axis=1).values[:5000]
y = data["label"].values[:5000]

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

learning_rates = [0.001, 0.01]
batch_sizes = [32, 64]
epochs = [50, 100]

results = {}

for lr in learning_rates:
    for batch in batch_sizes:
        for ep in epochs:
            model = MLPClassifier(
                hidden_layer_sizes=(100,),
                learning_rate_init=lr,
                batch_size=batch,
                max_iter=ep,
                random_state=42
            )

            model.fit(xtrain, ytrain)
            ypred = model.predict(xtest)

            acc = accuracy_score(ytest, ypred)
            key = str(lr) + " " + str(batch) + " " + str(ep)
            results[key] = acc
            print(key, "Accuracy =", round(acc, 4))

plt.figure(figsize=(12, 5))
plt.bar(results.keys(), results.values())
plt.title("Fashion MNIST Hyperparameter Comparison")
plt.xlabel("Hyperparameter Settings")
plt.ylabel("Accuracy")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()





import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
data=pd.read_csv("digits.csv.gz",header=None)
x=data.iloc[:,:-1]
scaler=StandardScaler()
x=scaler.fit_transform(x)

k=[2,5,6,7,8,10]

inrtia=[]
sil=[]

for i in k:
    kmeans=KMeans(n_clusters=i,n_init=10,random_state=42)
    labels=kmeans.fit_predict(x)
    inrtia.append(kmeans.inertia_)
    score=silhouette_score(x,labels)
    sil.append(score)
    print(i)
    print(kmeans.inertia_)
    print(score)

plt.plot(k,inrtia)
plt.show()





import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier
df=pd.read_csv('titanic.csv')
df=df.dropna()

x=df[['Pclass','Sex','Age','Fare']]
y=df['Survived']

x['Sex']=x['Sex'].map({'male': 0,'female':1})


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier(random_state=42)

bagging=BaggingClassifier(estimator=model,n_estimators=10,random_state=42)
bagging.fit(xtrain,ytrain)
ypred=bagging.predict(xtest)

boosting=AdaBoostClassifier(estimator=model,n_estimators=10,random_state=42)
boosting.fit(xtrain,ytrain)
pred=boosting.predict(xtest)

print("Bagging")
print("Accuracy:", accuracy_score(ytest, ypred))
print("Precision:", precision_score(ytest, ypred))
print("Recall:", recall_score(ytest,ypred))
print("F1 Score:", f1_score(ytest, ypred))

print("\nBoosting")
print("Accuracy:", accuracy_score(ytest, pred))
print("Precision:", precision_score(ytest,   pred))
print("Recall:", recall_score(ytest, pred))
print("F1 Score:", f1_score(ytest, pred))




import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('BostonHousing.csv')

X = df[['rm']]
y = df['medv']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

degrees = range(1, 11)
train_errors = []
val_errors = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    train_pred = model.predict(X_train_poly)
    val_pred = model.predict(X_val_poly)

    train_errors.append(mean_squared_error(y_train, train_pred))
    val_errors.append(mean_squared_error(y_val, val_pred))

plt.plot(degrees, train_errors, marker='o', label='Training Error')
plt.plot(degrees, val_errors, marker='o', label='Validation Error')

plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.show()