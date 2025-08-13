import marimo

__generated_with = "0.14.16"
app = marimo.App(
    width="medium",
    app_title="",
    layout_file="layouts/regressor_002.slides.json",
    auto_download=["ipynb"],
)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets, linear_model, model_selection
    return datasets, linear_model, model_selection, np


@app.cell
def _(datasets, mo):
    mo.ui.table(
        data=datasets.load_linnerud(),
        # use pagination when your table has many rows
        pagination=True,
        label="Dataframe")
    return


@app.cell
def _(datasets):
    df = datasets.load_linnerud(return_X_y=True)
    # print(df)
    return (df,)


@app.cell
def _(df):
    X, y = df
    print(X.shape)
    print(X[0])
    print(y.shape)
    print(y[0])
    return X, y


@app.cell
def _(X, np):
    # Using SitUps as Data Variable
    situps = X[:, np.newaxis, 1]
    print(situps.shape)
    print(situps[0])

    return (situps,)


@app.cell
def _(X, np):
    # Using Chins as Data Variable
    chins = X[:, np.newaxis, 0]
    print(chins.shape)
    print(chins[0])
    return (chins,)


@app.cell
def _(X, np):
    # Using Jumps as Data Variable
    jumps = X[:, np.newaxis, 2]
    print(jumps.shape)
    print(jumps[0])
    return (jumps,)


@app.cell
def _(np, y):
    # Using Waistline as Target
    waistlines = y[:, np.newaxis, 1]
    print(waistlines.shape)
    print(waistlines[0])
    return (waistlines,)


@app.cell
def _(np, y):
    # Using weight as target Variable
    weight = y[:, np.newaxis, 0]
    print(weight.shape)
    print(weight[0])
    return (weight,)


@app.cell
def _(np, y):
    # Using pulse as target Variable
    pulse = y[:, np.newaxis, 2]
    print(pulse.shape)
    print(pulse[0])
    return (pulse,)


@app.cell
def _(chins, jumps, model_selection, pulse, situps, waistlines, weight):
    X1_train, X1_test, y1_train, y1_test = model_selection.train_test_split(situps, waistlines)
    X2_train, X2_test, y2_train, y2_test = model_selection.train_test_split(chins, weight)
    X3_train, X3_test, y3_train, y3_test = model_selection.train_test_split(jumps, pulse)
    return (
        X1_test,
        X1_train,
        X2_test,
        X2_train,
        X3_test,
        X3_train,
        y1_test,
        y1_train,
        y2_test,
        y2_train,
        y3_test,
        y3_train,
    )


@app.cell
def _(
    X1_train,
    X2_train,
    X3_train,
    linear_model,
    y1_train,
    y2_train,
    y3_train,
):
    model = linear_model.LinearRegression()
    model1 = model.fit(X1_train, y1_train)
    model2 = model.fit(X2_train, y2_train)
    model3 = model.fit(X3_train, y3_train)
    return model1, model2, model3


@app.cell
def _(X1_test, X2_test, X3_test, model1, model2, model3):
    y1_pred = model1.predict(X1_test)
    y2_pred = model2.predict(X2_test)
    y3_pred = model3.predict(X3_test)
    return y1_pred, y2_pred, y3_pred


@app.cell
def _(
    X1_test,
    X2_test,
    X3_test,
    y1_pred,
    y1_test,
    y2_pred,
    y2_test,
    y3_pred,
    y3_test,
):
    def _():
        import matplotlib.pyplot as plt

        # Create 3 subplots in a row (1 row, 3 columns)
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Plot for Dataset 1
        axs[0].scatter(X1_test, y1_test, color='black')
        axs[0].plot(X1_test, y1_pred, color='red', linewidth=3)
        axs[0].set_title('Situps / Waistline')

        # Plot for Dataset 2
        axs[1].scatter(X2_test, y2_test, color='blue')
        axs[1].plot(X2_test, y2_pred, color='red', linewidth=3)
        axs[1].set_title('Chins / Weight')

        # Plot for Dataset 3
        axs[2].scatter(X3_test, y3_test, color='green')
        axs[2].plot(X3_test, y3_pred, color='red', linewidth=3)
        axs[2].set_title('Jumps / Pulse Rate')

        plt.suptitle('Linear Relation between Exercise and Health', fontsize=14)
        plt.tight_layout()
        return plt.show()


    _()
    return


@app.cell
def _(np, y1_pred, y1_test, y2_pred, y2_test, y3_pred, y3_test):
    from sklearn.metrics import mean_squared_error, r2_score

    mse1 = mean_squared_error(y1_test, y1_pred)
    mse2 = mean_squared_error(y2_test, y2_pred)
    mse3 = mean_squared_error(y3_test, y3_pred)
    print("MSE for Situps / Waistline:", mse1)
    print("MSE for Chins / Weight:", mse2)
    print("MSE for Jumps / Pulse Rate:", mse2)
    print("\n")
    rmse1 = np.sqrt(mse1)
    rmse2 = np.sqrt(mse2)
    rmse3 = np.sqrt(mse3)
    print("RMSE for Situps / Waistline:", rmse1)
    print("RMSE for Chins / Weight:", rmse2)
    print("RMSE for Jumps / Pulse Rate:", rmse3)
    print("\n")
    r2_1 = r2_score(y1_test, y1_pred)
    r2_2 = r2_score(y2_test, y2_pred)
    r2_3 = r2_score(y3_test, y3_pred)
    print("R2 for Situps / Waistline:", r2_1)
    print("R2 for Chins / Weight:", r2_2)
    print("R2 for Jumps / Pulse Rate:", r2_3)
    print("\n")
    return


if __name__ == "__main__":
    app.run()
