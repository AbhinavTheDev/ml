import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""#Regressor 001""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##Installing Required Libraries""")
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets, linear_model, model_selection
    return datasets, linear_model, model_selection, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##Loading Dataset""")
    return


@app.cell
def _(datasets):
    df = datasets.load_diabetes(return_X_y=True)
    return (df,)


@app.cell
def _(df):
    X, y = df
    print(X.shape)
    print(X[0])
    return X, y


@app.cell
def _(X, np):
    x = X[:, np.newaxis, 0]
    print(x.shape)
    return (x,)


@app.cell
def _(mo):
    mo.md(r"""##Splitting Data into train and test""")
    return


@app.cell
def _(model_selection, x, y):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y)
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""Call Model and fit data""")
    return


@app.cell
def _(X_train, linear_model, y_train):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""Make Prediction using trained model""")
    return


@app.cell
def _(X_test, model):
    y_pred = model.predict(X_test)
    return (y_pred,)


@app.cell
def _(mo):
    mo.md(r"""Plot the model prediction""")
    return


@app.cell
def _(X_test, plt, y_pred, y_test):
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    return


@app.cell
def _(np, y_pred, y_test):
    from sklearn.metrics import mean_squared_error, r2_score

    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)

    rmse = np.sqrt(mse)
    print("RMSE:", rmse)

    r2 = r2_score(y_test, y_pred)
    print("R2:", r2)
    return


if __name__ == "__main__":
    app.run()
