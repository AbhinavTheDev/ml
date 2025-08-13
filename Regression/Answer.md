### Q - In your own words, describe how to create a Regression model that would plot the relationship between the waistline and how many situps are accomplished. Do the same for the other datapoints in this dataset.

### A -
In linnerud dataset, we have dataset consisting three data variables (Chins, Situps, Jumps) and three target variables (Weight, Waist, Pulse). For creating a linear regression model, we need to define X and Y variables, for sit-ups and waistline. I defined X as numpy column consisting only sit-ups data and defined Y as numpy column consisting only waist data. 

```
x = X[:, np.newaxis, 1]
Y = y[:, np.newaxis, 1]
```
Then, we can split data into train and test sets using `model_selection.train_test_split()` . Afterwards, we can call model and fit our data for training model. After training, we can predict data. After successful prediction, we can plot results in scatter plot with a linear line of model continuity.