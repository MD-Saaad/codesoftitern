import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

card_df = pd.read_csv('fraudTrain.csv')
card_df.head()
card_df.info()

card_df["trans_date_trans_time"] = pd.to_datetime(card_df["trans_date_trans_time"])
card_df["dob"] = pd.to_datetime(card_df["dob"])
card_df

card_df.drop(columns=['Unnamed: 0','cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'],inplace=True)
card_df

card_df.dropna(ignore_index=True)
card_df

encoder = LabelEncoder()
card_df["merchant"] = encoder.fit_transform(card_df["merchant"])
card_df["category"] = encoder.fit_transform(card_df["category"])
card_df["gender"] = encoder.fit_transform(card_df["gender"])
card_df["job"] = encoder.fit_transform(card_df["job"])

exit_counts = card_df["is_fraud"].value_counts()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # Subplot for the pie chart
plt.pie(exit_counts, labels=["No", "YES"], autopct="%0.0f%%")
plt.title("is_fraud Counts")
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

X = card_df.drop(columns=["is_fraud"], inplace = False)
Y = card_df["is_fraud"]

model = SVC()
model = LogisticRegression()
model.fit(X, Y)

model.score(X, Y)

card_df = pd.read_csv("fraudTest.csv")
card_df

card_df.drop(columns=['Unnamed: 0','cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'],inplace=True)
card_df

encoder = LabelEncoder()
card_df["merchant"] = encoder.fit_transform(card_df["merchant"])
card_df["category"] = encoder.fit_transform(card_df["category"])
card_df["gender"] = encoder.fit_transform(card_df["gender"])
card_df["job"] = encoder.fit_transform(card_df["job"])

card_df

X_test = card_df.drop(columns=["is_fraud"], inplace = False)
Y_test = card_df["is_fraud"]

y_pred = model.predict(X_test)
y_pred

accuracy = accuracy_score(test_data['is_fraud'],y_pred)
accuracy

X = card_df.drop('is_fraud', axis=1)

Y = card_df['is_fraud']

X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state = 2)

model = LogisticRegression()
model.fit(X_train, Y_train)
ypredict = model.predict(X_test)

model.fit(X_train,Y_train)
ypredict = model.predict(X_test)

accuracy_score(ypredict, Y_test)
