# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 19:18:38 2023

@author: Rahul
"""

import pandas as pd
import numpy as np
from sklearn import datasets  
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing

df = pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\Random Forests\\Fraud_check.csv")
df.head()
df.tail()
df.dtypes

#Creating dummy vairables
df = pd.get_dummies(df,columns = ['Undergrad','Marital.Status','Urban'],drop_first=True)

#Creating new cols TaxInc and dividing 'Taxable.Income' 
df["TaxInc"] = pd.cut(df["Taxable.Income"], bins = [0,30000,99620], labels = ["Risky", "Good"])

print(df)

#After creation of new col. TaxInc also made its dummies var concating right side of df
df = pd.get_dummies(df,columns = ["TaxInc"],drop_first=True)


import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(data=df, hue = 'TaxInc_Good')

# Normalization function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

df_norm = norm_func(df.iloc[:,1:])
df_norm

# Splitting the variables
X = df_norm.drop(['TaxInc_Good'], axis=1)
y = df_norm['TaxInc_Good']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Converting the Taxable income variable to bucketing. 
df_norm["income"]="<=30000"
df_norm.loc[df["Taxable.Income"]>=30000,"income"]="Good"
df_norm.loc[df["Taxable.Income"]<=30000,"income"]="Risky"

#Droping the Taxable income variable
df.drop(["Taxable.Income"],axis=1,inplace=True)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass

#Splitting the data into featuers and labels
features = df.iloc[:,0:5]
labels = df.iloc[:,5]

# Collecting the column names
colnames = list(df.columns)
predictors = colnames[0:5]
target = colnames[5]

#Splitting the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)

#Model building
from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)

model.estimators_
model.classes_
model.n_features_
model.n_classes_

model.n_outputs_

model.oob_score_

#Predictions on train data
prediction = model.predict(x_train)


#Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)

np.mean(prediction == y_train)

#Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)

##Prediction on test data
pred_test = model.predict(x_test)

#Accuracy
acc_test =accuracy_score(y_test,pred_test)

# In random forest we can plot a Decision tree present in Random forest
from sklearn.tree import export_graphviz
# pip install pydotplus
import pydotplus
from six import StringIO

tree = model.estimators_[5]

dot_data = StringIO()
export_graphviz(tree,out_file = dot_data, filled = True,rounded = True, feature_names = predictors ,class_names = target,impurity =False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# Building Decision Tree Classifier using Entropy Criteria
model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)

from sklearn import tree

#PLot the decision tree
tree.plot_tree(model);

colnames = list(df.columns)
colnames

fn=['population','experience','Undergrad_YES','Marital.Status_Married','Marital.Status_Single','Urban_YES']
cn=['1', '0']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);

#Predicting on test data
preds = model.predict(x_test) 
pd.Series(preds).value_counts()

preds

pd.crosstab(y_test,preds)

# Accuracy 
np.mean(preds==y_test)

#Building Decision Tree Classifier (CART) using Gini Criteria
from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)

model_gini.fit(x_train, y_train)

#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)


# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

array = df.values
X = array[:,0:3]
y = array[:,3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

#Find the accuracy
model.score(X_test,y_test)

































