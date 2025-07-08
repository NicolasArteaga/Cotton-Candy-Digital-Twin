import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import time

# Example data
data = pd.DataFrame({
    'cook_time': [105, 110, 100, 108, 106, 102],
    'cooldown_time': [60, 80, 45, 75, 65, 55],
    'wait_time': [5, 4, 6, 3, 5, 7],
    'iteration': [1, 5, 10, 15, 3, 12],
    'Quality': ['Good', 'Bad', 'Bad', 'Bad', 'Good', 'Bad']
})

# Convert labels to binary
data['Quality'] = data['Quality'].map({'Good': 1, 'Bad': 0})

# Features and labels
X = data[['cook_time', 'cooldown_time', 'wait_time', 'iteration', ]]
y = data['Quality']

print("Training started...")
start = time.time()

# Train decision tree
model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X, y)

end = time.time()
print(f"Training finished in {end - start:.4f} seconds")

# Plot the tree
plt.figure(figsize=(8,5))
tree.plot_tree(model, 
               feature_names=['cook_time', 'cooldown_time', 'wait_time', 'iteration'], 
               class_names=['Bad', 'Good'], 
               filled=True)
plt.title("Cotton Candy Quality Decision Tree")
plt.show()