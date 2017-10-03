from sklearn import tree

# 1 - smooth, 0 - bympy
# 140 - grams
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

# 0 - apple, 1 - orange
labels = [0, 0, 1, 1]

# Decision Tree
classifier = tree.DecisionTreeClassifier()
classifier.fit(features, labels)
print(classifier.predict([[160, 0]]))