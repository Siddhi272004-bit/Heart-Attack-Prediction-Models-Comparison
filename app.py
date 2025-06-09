import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Random Forest
from sklearn.ensemble import RandomForestClassifier


# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier


# Calibrated Classifier (for probability improvement)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,accuracy_score,roc_auc_score,roc_curve,auc
import xgboost as xgb

# THIS TIME IT'S GONNA BE MULTIPLE MODELS IN COMPARISON
# load the dataset
df=pd.read_csv('./Medicaldataset.csv')

# FEATURE EXTRACTION:
# preprocessing(axis=1:for each row)
x=df.drop('Result',axis=1)
y=df['Result']

# train-test-split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# mapping the string values:
y_train = y_train.map({'negative': 0, 'positive': 1})
y_test = y_test.map({'negative': 0, 'positive': 1})

# TRAINING THE MODELS:
# XGBClassifer:
model1 = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='logloss')
model1.fit(x_train,y_train)

# Random Forest:
model2=RandomForestClassifier()
model2.fit(x_train,y_train)

# K-nearest neighbours
model3=KNeighborsClassifier(n_neighbors=5)
model3.fit(x_train,y_train)

# Decision Tree
model4=DecisionTreeClassifier(max_depth=4, random_state=42)
model4.fit(x_train,y_train)

# Gradient Boosting
model5 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model5.fit(x_train,y_train)

# Logisitic Regression
model6 = LogisticRegression(max_iter=1000)
model6.fit(x_train,y_train)

# saving the models for streamlit use:
import joblib

# Save models
joblib.dump(model1, 'model1_xgb.pkl')
joblib.dump(model2, 'model2_rf.pkl')
joblib.dump(model3, 'model3_knn.pkl')
joblib.dump(model4, 'model4_dt.pkl')
joblib.dump(model5, 'model5_gb.pkl')
joblib.dump(model6, 'model6_lr.pkl')
# FOR STORING ALL OF THE MODELS:
models = [model1, model2, model3, model4, model5, model6]

# MAKE PREDICTIONS
predictions1=model1.predict(x_test)
predictions2=model2.predict(x_test)
predictions3=model3.predict(x_test)
predictions4=model4.predict(x_test)
predictions5=model5.predict(x_test)
predictions6=model6.predict(x_test)

# evaluating the models
# accuracy
accuracy1=accuracy_score(y_test,predictions1)
accuracy2=accuracy_score(y_test,predictions2)
accuracy3=accuracy_score(y_test,predictions3)
accuracy4=accuracy_score(y_test,predictions4)
accuracy5=accuracy_score(y_test,predictions5)
accuracy6=accuracy_score(y_test,predictions6)

#confusion matrix and then calculating metrics and stuff
tn1, fp1, fn1, tp1 = confusion_matrix(y_test, predictions1).ravel()
tn2, fp2, fn2, tp2 = confusion_matrix(y_test, predictions2).ravel()
tn3, fp3, fn3, tp3 = confusion_matrix(y_test, predictions3).ravel()
tn4, fp4, fn4, tp4 = confusion_matrix(y_test, predictions4).ravel()
tn5, fp5, fn5, tp5 = confusion_matrix(y_test, predictions5).ravel()
tn6, fp6, fn6, tp6 = confusion_matrix(y_test, predictions6).ravel()



# Sensitivity / Recall
sensitivity1 = tp1 / (tp1 + fn1)
sensitivity2 = tp2 / (tp2 + fn2)
sensitivity3 = tp3 / (tp3 + fn3)
sensitivity4 = tp4 / (tp4 + fn4)
sensitivity5 = tp5 / (tp5 + fn5)
sensitivity6 = tp6 / (tp6 + fn6)

# Specificity
specificity1 = tn1 / (tn1 + fp1)
specificity2 = tn2 / (tn2 + fp2)
specificity3 = tn3 / (tn3 + fp3)
specificity4 = tn4 / (tn4 + fp4)
specificity5 = tn5 / (tn5 + fp5)
specificity6 = tn6 / (tn6 + fp6)

# Precision
precision1 = precision_score(y_test, predictions1)
precision2 = precision_score(y_test, predictions2)
precision3 = precision_score(y_test, predictions3)
precision4 = precision_score(y_test, predictions4)
precision5 = precision_score(y_test, predictions5)
precision6 = precision_score(y_test, predictions6)

# F1 Score
# STORING ALL THE F1 SCORES IN A DICTIONARY:(FOR SCALABILITY)
f1_scores = [f1_score(y_test, m.predict(x_test)) for m in models]



# ROC CURVES AND STUFF

# FUNCTION FOR PLOTTING ALL OF THE ROC CURVES IN A SINGLE PLOT
def plot_multiple_roc_curves(models, x_test, y_test, model_names):
    y_test = (y_test == 1).astype(int)  # binary labels

    plt.figure(figsize=(10, 8))

    for model, name in zip(models, model_names):
        y_pred_proba = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc_value = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_value:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.show()


# CALLING THE ROC-AUC CURVE FUNCTIONS

model_names = ['XGBoost', 'Random Forest', 'KNN', 'Decision Tree', 'Gradient Boosting', 'Logistic Regression']

plot_multiple_roc_curves(models, x_test, y_test, model_names)

