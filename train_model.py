import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

x = []
y = []

labels = ["HELLO", "YES" , "NO", "THANKS"]

for label in labels:
    data = np.load(f"{label}.npy")
    x.extend(data)
    y.extend([label] * len(data))

model = RandomForestClassifier()
model.fit(x, y)

pickle.dump(model, open("model.pkl", "wb"))