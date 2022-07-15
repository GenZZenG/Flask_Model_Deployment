import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("model_data.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = LogisticRegression(class_weight='balanced')
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
