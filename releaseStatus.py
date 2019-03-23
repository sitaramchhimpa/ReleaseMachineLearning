import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# read data file usng pandas
df = pd.read_csv("data/release_data.csv")

# print 5 first row
#df.head(5)

# print size of data file
df.shape

# check for null values in data file
df.isnull().values.any()

feature_col_names = ['qa_bug', 'uat_bugs', 'prod_bugs', 'total_count']
predicted_class_names = ['release_status']

X = df[feature_col_names].values  # predictor feature columns (8 X m)
y = df[predicted_class_names].values  # predicted class (1=true, 0=false) column (1 X m)
split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=55)


fill_0 = SimpleImputer(missing_values=0, strategy="mean")
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)


rf_model = RandomForestClassifier(random_state=42, n_estimators=10)      # Create random forest object
rf_model.fit(X_train, y_train.ravel())

filename = 'release_status.pkl'
pickle.dump(rf_model, open(filename, 'wb'))

# rf_predict_train = rf_model.predict(X_train)