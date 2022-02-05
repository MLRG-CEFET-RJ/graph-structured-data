# %%
import pandas as pd
import tensorflow_decision_forests as tfdf

# %%
def load_data():
    train = pd.read_csv('./train_dataset.csv')
    test = pd.read_csv('./val_dataset.csv')
    return train, test

# %%
train_ds_pd, train_ds_pd = load_data()

# %%
label = "labelling"

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)

# %%
# Configure the model.
model_7 = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)

# Optional.
model_7.compile(metrics=["mse"])

# Train the model.
model_7.fit(x=train_ds)


