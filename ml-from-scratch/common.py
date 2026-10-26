import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pandas
df = pd.read_csv("salary_data.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# Clean and prepare the data
df = df.drop_duplicates()

# Separate features and target
X = df.drop(columns=["price"])
y = df["price"]

# Convert the data into NumPy arrays when needed
x = X.to_numpy()
y = y.to_numpy()

# dataset split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# matplotlib
x = df['YearsExperience']
y = df['Salary']

fig, axes = plt.subplots(1, 2, figsize=(15, 4))

axes[0].set_title("Salary vs Experience")
axes[0].set_xlabel("Experience")
axes[0].set_ylabel("Salary")
axes[0].scatter(x, y)

loss = [1, 2, 3, 4]
axes[1].plot(loss)

plt.tight_layout()
plt.show()