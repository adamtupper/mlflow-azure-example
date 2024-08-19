# %%
"""Exploratory Data Analysis (EDA) for the RetinaMNIST dataset."""

import matplotlib.pyplot as plt
import medmnist
import pandas as pd
import seaborn as sns

# %%
# Load the dataset
DATASET_DIR = "~/Downloads"

train_dataset = medmnist.RetinaMNIST(
    split="train", download=True, size=224, root="../.."
)
val_dataset = medmnist.RetinaMNIST(split="val", download=True, size=224, root="../..")
test_dataset = medmnist.RetinaMNIST(split="test", download=True, size=224, root="../..")

# %%
# Plot the class distribution of the training, validation, and test sets

# Create a DataFrame with the class labels and the split
examples = []
for dataset in (train_dataset, val_dataset, test_dataset):
    for example in dataset:
        examples.append({"split": dataset.split, "label": example[1][0]})
df = pd.DataFrame(examples)

# Plot the class distribution for each split
graph = sns.catplot(data=df, x="split", hue="label", kind="count")


# Add the count of each class to the plot
ax = graph.facet_axis(0, 0)  # or ax = g.axes.flat[0]
for c in ax.containers:
    labels = [f"{int(v.get_height())}" for v in c]
    ax.bar_label(c, labels=labels, label_type="edge")

# Set the title and labels
plt.title("Class Distribution for each Split of the RetinaMNIST Dataset")
plt.xlabel("Split")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# %%
# Plot five examples from each of the five classes from the training set
fig, axes = plt.subplots(5, 5, figsize=(5, 5))

examples_per_class = {i: [] for i in range(5)}
for example in train_dataset:
    label = example[1][0]
    if len(examples_per_class[label]) < 5:
        examples_per_class[label].append(example)

for i, examples in examples_per_class.items():
    for j, example in enumerate(examples):
        axes[i, j].imshow(example[0], cmap="gray")
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

        if j == 0:
            axes[i, j].set_ylabel(f"Level {i}")

plt.suptitle("Levels of Diabetic Retinopathy Severity")
plt.tight_layout()
plt.show()

# %%
