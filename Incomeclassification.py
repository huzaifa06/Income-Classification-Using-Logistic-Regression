import numpy as np
import pandas as pd
import io
import requests
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
import os
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.base import TransformerMixin

columns = ["age", "census", "education", "education_num","capital_gain", "capital_loss", "hours_per_week", "income_level"]
dat = pd.read_csv("census_income.csv", names=columns, engine='python',sep=' *, *', na_values='?')

data  = pd.read_csv('census_income.csv', names=columns, engine='python',sep=' *, *', na_values='?')
data.info()

num_attributes = dat.select_dtypes(include=['int64'])
print(num_attributes.columns)

num_attributes.hist(figsize=(10,10))

# dat.describe()

cat_attributes = dat.select_dtypes(include=['object'])
print(cat_attributes.columns)

# sns.countplot(y='education', hue='income_level', data = cat_attributes)

# sns.countplot(y='hours_per_week', hue='income_level', data = cat_attributes)

class ColumnsSelector(BaseEstimator, TransformerMixin):
	def __init__(self, type):
		self.type = type
	def fit(self, X, y=None):
		return self
	def transform(self,X):
		return X.select_dtypes(include=[self.type])

num_pipeline = Pipeline(steps=[("num_attr_selector", ColumnsSelector(type='int')),("scaler", StandardScaler())])
print(num_pipeline)

l=1000
cat_pipeline = Pipeline(steps=[("cat_attr_selector", ColumnsSelector(type='object')),("encoder", preprocessing.LabelEncoder())])
print(cat_pipeline)
full_pipeline = FeatureUnion([("num_pipe", num_pipeline),("cat_pipeline", cat_pipeline)])

# data.info()
dat.info()


train_copy = dat.copy()
train_copy["income_level"] = train_copy["income_level"].apply(lambda x:0 if x=='<=50K' else 1)
X_train = train_copy.drop('income_level', axis =1)
Y_train = train_copy['income_level']
print(X_train)

model = LogisticRegression(random_state=0,solver='lbfgs')
model.fit(X_train, Y_train)

test_copy = data.copy()
test_copy["income_level"] = test_copy["income_level"].apply(lambda x:0 if x=='<=50K.' else 1)
X_test = test_copy.drop('income_level', axis =1)
Y_test = test_copy['income_level']


predicted_classes = model.predict(X_test)
a = (accuracy_score(predicted_classes, Y_test.values)*l)
print("Accuracy percentage is ",a,"%")

cfm = confusion_matrix(predicted_classes, Y_test.values)
sns.heatmap(cfm, annot=True)
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')

cross_val_model = LogisticRegression(random_state=0,solver='lbfgs')
scores = cross_val_score(cross_val_model, X_train,Y_train, cv=25)
print(np.mean(scores)*100)
penalty = ['l2']
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)

clf = GridSearchCV(estimator = model, param_grid = hyperparameters,cv=15, verbose=0)

best_model = clf.fit(X_train, Y_train)
print('Best Penalty:', best_model.best_estimator_.get_params() ['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

best_predicted_values = best_model.predict(X_test)
accuracy_score(best_predicted_values, Y_test.values)
print("Accuracy is ",a,"%")