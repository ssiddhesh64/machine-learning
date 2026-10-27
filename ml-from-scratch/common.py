import numpy as np
import pandas as pd
import torch
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

# visualize histograms
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
  if isinstance(layer, Tanh):
    t = layer.out
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends);
plt.title('activation distribution')


# visualize histograms
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i,p in enumerate(parameters):
  t = p.grad
  if p.ndim == 2:
    print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.title('weights gradient distribution');