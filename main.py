import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

fake_db = pd.read_csv("Fake.csv")
true_db = pd.read_csv("True.csv")

# true = 0 or 1

# 23481 rows

#     0,    1,       2,    3,    4,
# title, text, subject, date, true,

fake_db["true"] = 0
true_db["true"] = 1

final_db = pd.concat([fake_db, true_db],  ignore_index=True)

final_db["feature"] = final_db["title"] + final_db["text"] + final_db["subject"] + final_db["date"]

X = final_db["feature"]  # input
y = final_db["true"]     # output

vectoriser = CountVectorizer()

input_vectors = vectoriser.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(input_vectors, y, test_size=0.2)

trainer = BernoulliNB()

trainer.fit(X_train, y_train)

y_pred = trainer.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print(100 * accuracy)
