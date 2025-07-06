import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import time

# Example data
data = pd.DataFrame({
    'Humidity_in': [45, 55, 48, 60, 49, 51],
    'Temp_out': [32, 31, 28, 29, 33, 27],
    'Quality': ['Good', 'Bad', 'Bad', 'Bad', 'Good', 'Bad']
})

# Convert labels to binary
data['Quality'] = data['Quality'].map({'Good': 1, 'Bad': 0})

# Features and labels
X = data[['Humidity_in', 'Temp_out']]
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
               feature_names=['Humidity_in', 'Temp_out'], 
               class_names=['Bad', 'Good'], 
               filled=True)
plt.title("Cotton Candy Quality Decision Tree")
plt.show()