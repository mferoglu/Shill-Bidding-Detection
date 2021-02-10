#!/usr/bin/env python
# coding: utf-8

# In[5]:



import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import time
from time import perf_counter
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from IPython.display import Image  
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from time import perf_counter
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# In[6]:



raw_data = pd.read_csv('Shill Bidding Dataset.csv')
del raw_data["Record_ID"]

print(raw_data.head(5))
print("\n\n")
print(raw_data.dtypes)
print("\n\n")
ord_enc = OrdinalEncoder()
raw_data["Bidder_ID"] = ord_enc.fit_transform(raw_data[["Bidder_ID"]])


min_max_scaler = preprocessing.MinMaxScaler()
scaled = min_max_scaler.fit_transform(raw_data)
data_processed = pd.DataFrame(scaled)



X = raw_data.iloc[:,1:11]
Y = raw_data["Class"]

print(data_processed.head(5))
print("\n\n")
print("Missing values\n")

#raw_data['Auction_Duration'] = preprocessing.normalize(raw_data['Auction_Duration'])

# No missing values.
#print("Record_ID - Missing Values: "+str(raw_data["Record_ID"].isnull().sum())+", Max Value: "+str(raw_data["Record_ID"].max())+", Min Value: "+str(raw_data["Record_ID"].min()))
print("Auction_ID - Missing Values:"+str(raw_data["Auction_ID"].isnull().sum())+", Max Value: "+str(raw_data["Auction_ID"].max())+", Min Value: "+str(raw_data["Auction_ID"].min()))
print("Bidder_ID - Missing Values:"+str(raw_data["Bidder_ID"].isnull().sum())+", Max Value: "+str(raw_data["Bidder_ID"].max())+", Min Value: "+str(raw_data["Bidder_ID"].min()))
print("Bidder_Tendency - Missing Values:"+str(raw_data["Bidder_Tendency"].isnull().sum())+", Max Value: "+str(raw_data["Bidder_Tendency"].max())+", Min Value: "+str(raw_data["Bidder_Tendency"].min()))
print("Bidding_Ratio - Missing Values:"+str(raw_data["Bidding_Ratio"].isnull().sum())+", Max Value: "+str(raw_data["Bidding_Ratio"].max())+", Min Value: "+str(raw_data["Bidding_Ratio"].min()))
print("Successive_Outbidding - Missing Values:"+str(raw_data["Successive_Outbidding"].isnull().sum())+", Max Value: "+str(raw_data["Successive_Outbidding"].max())+", Min Value: "+str(raw_data["Successive_Outbidding"].min()))
print("Last_Bidding - Missing Values:"+str(raw_data["Last_Bidding"].isnull().sum())+", Max Value: "+str(raw_data["Last_Bidding"].max())+", Min Value: "+str(raw_data["Last_Bidding"].min()))
print("Auction_Bids - Missing Values:"+str(raw_data["Auction_Bids"].isnull().sum())+", Max Value: "+str(raw_data["Auction_Bids"].max())+", Min Value: "+str(raw_data["Auction_Bids"].min()))
print("Starting_Price_Average - Missing Values:"+str(raw_data["Starting_Price_Average"].isnull().sum())+" Max Value: "+str(raw_data["Starting_Price_Average"].max())+", Min Value: "+str(raw_data["Starting_Price_Average"].min()))
print("Early_Bidding - Missing Values:"+str(raw_data["Early_Bidding"].isnull().sum())+", Max Value: "+str(raw_data["Early_Bidding"].max())+", Min Value: "+str(raw_data["Early_Bidding"].min()))
print("Winning_Ratio - Missing Values:"+str(raw_data["Winning_Ratio"].isnull().sum())+", Max Value: "+str(raw_data["Winning_Ratio"].max())+", Min Value: "+str(raw_data["Winning_Ratio"].min()))
print("Auction_Duration - Missing Values:"+str(raw_data["Auction_Duration"].isnull().sum())+", Max Value: "+str(raw_data["Auction_Duration"].max())+", Min Value: "+str(raw_data["Auction_Duration"].min()))
print("Class - Missing Values:"+str(raw_data["Class"].isnull().sum())+", Max Value: "+str(raw_data["Class"].max())+", Min Value: "+str(raw_data["Class"].min()))
print("\n")

print(data_processed.dtypes)

#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # 67% training and 33% test


# In[16]:


#-------------------------------------Decision Tree With Gain---------------------------------------------------------------------------------------------------------

print("\n")
start_timer= perf_counter()
clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X_train,y_train)
y_pred_gain = clf.predict(X_test)
end_timer= perf_counter()
print("\n")
print(f"Execution time of Decision Tree With Gain Bagging Model {end_timer-start_timer:0.4f} seconds")
print("\n")
print("Decision Tree With Gain Accuracy Holdout Method:",metrics.accuracy_score(y_test, y_pred_gain))
print("\n")
accuracy = cross_val_score(clf, X, Y,scoring="accuracy", cv= model_selection.KFold(n_splits=9))
print("\n")
print( classification_report(y_test, y_pred_gain))

print("Decision Tree With Gain Accuracy (Cross Validation):" )
print("\n")
count = 0
for i in accuracy:
    count+=1
    print("% 3.2f"%(i*100),end="")
    if(count%3 == 0):
        print("\n")
print("Mean Of : %3.2f"%(np.mean(accuracy)*100))


#printing confusion matrix

# cf_matrix = confusion_matrix(y_test,y_pred)
# sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
#             fmt='.2%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"])

# plt.title("Default")
# plt.show()
# plt.close()


# # printing the tree into png file
tree.plot_tree(clf)
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)

tree.plot_tree(clf,
               feature_names = data_processed.columns, 
               class_names=["Normal Bidding","Shill Bidding"],
               filled = True)

fig.savefig('decision_Tree_With_Gain.png')


# In[8]:


#------------- Gain Bagging----------------------------------

start_timer= perf_counter()
bagging_model = BaggingClassifier(base_estimator=clf)
bagging_model.fit(X_train,y_train)
kfold = model_selection.KFold(n_splits=9)
accuracy = cross_val_score(bagging_model,X,Y,scoring = "accuracy",cv=kfold)
bagging_model_pr = bagging_model.predict(X_test)
end_timer= perf_counter()
print("\n")
print(f"Execution time of Decision Tree With Gain Bagging Model {end_timer-start_timer:0.4f} seconds")
print("\n")
print("Decision Tree With Gain Bagging Ensemble Accuracy: %5.4f"%(metrics.accuracy_score(y_test,bagging_model_pr)))
print("\n")
print( classification_report(y_test, bagging_model_pr))
print("\n")
print("Decision Tree With Gain Bagging Ensemble Accuracy (Cross Validation):\n" )

count = 0
for i in accuracy:
    count+=1
    print("% 3.2f"%(i*100),end="")
    if(count%3 == 0):
        print("\n")
print("Mean Of : %3.2f"%(np.mean(accuracy)*100))

#printing confusion matrix

# cf_matrix = confusion_matrix(y_test,bagging_model_pr)
# sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
#             fmt='.2%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"])

# plt.title("Bagging")
# plt.show()
# plt.close()


# In[9]:


# ------------Gain Boosting---------------------------------
start_timer= perf_counter()
kfold = model_selection.KFold(n_splits=9)
boosting_model = GradientBoostingClassifier(init = clf)
boosting_model.fit(X_train,y_train.values.ravel())
boosting_model_pr = boosting_model.predict(X_test)
accuracy_boosting = cross_val_score(boosting_model, X, Y,scoring="accuracy", cv=kfold)

end_timer= perf_counter()
print("\n")
print(f"Execution time of Decision Tree With Gain Bagging Model {end_timer-start_timer:0.4f} seconds")
print("\n")
print("Boosting Ensemble Accuracy: %5.4f"%(metrics.accuracy_score(y_test,boosting_model_pr)))
print("\n")
print("Decision Tree With Gain Bagging Ensemble Accuracy (Cross Validation):" )
print("\n")
count = 0
for i in accuracy_boosting:
    count+=1
    print("% 3.2f"%(i*100),end="")
    if(count%3 == 0):
        print("\n")
print("Mean Of : %3.2f"%(np.mean(accuracy)*100))

# printing confusion matrix

# cf_matrix = confusion_matrix(y_test,boosting_model_pr)
# sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
#             fmt='.2%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"],)

# plt.title("Boosting")
# plt.show()
# plt.close()


# In[11]:


#-------------------------------------Decision Tree With Gini---------------------------------------------------------------------------------------------------------
start_timer= perf_counter()
# Create Decision Tree classifer object
clf_gini = DecisionTreeClassifier(criterion="gini")
# Train Decision Tree Classifer
clf_gini = clf_gini.fit(X_train,y_train)
#Predict the response for test dataset
y_pred_gini = clf_gini.predict(X_test)
end_timer= perf_counter()
print("\n")
print(f"Execution time of Decision Tree With Gini Index {end_timer-start_timer:0.4f} seconds")
print("\n")
print("Decision Tree With Gini Index Accuracy:",metrics.accuracy_score(y_test, y_pred_gini))
print("\n")
print( classification_report(y_test, y_pred_gini))

# time.sleep(3)
# cf_matrix = confusion_matrix(y_test,y_pred_gini)
# sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
#             fmt='.2%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"])

# plt.title("Default Decision Tree (Gini Index)")
# plt.show()



# Tree plotting into png file

tree.plot_tree(clf)
plt.close()
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf,
               feature_names = data_processed.columns, 
               class_names=["Normal Bidding","Shill Bidding"],
               filled = True)

fig.savefig('decision_Tree_With_Gini.png')


# In[12]:


#-------------Bagging----------------------------------

start_timer= perf_counter()
bagging_model_gini = BaggingClassifier(base_estimator=clf)
bagging_model_gini.fit(X_train,y_train)
#cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 10,random_state = 1)
kfold = model_selection.KFold(n_splits=9)
accuracy = cross_val_score(bagging_model_gini,X,Y,scoring = "accuracy",cv=kfold)
bagging_model_gini_pr = bagging_model_gini.predict(X_test)
end_timer= perf_counter()


print("\n")
print(f"Execution time of Decision Tree With Gini Index Bagging Model {end_timer-start_timer:0.4f} seconds")
print("\n")
print("Decision Tree With Gini Index Bagging Ensemble Accuracy: "+str(metrics.accuracy_score(y_test,bagging_model_gini_pr)))
print( classification_report(y_test, bagging_model_gini_pr))
print("Decision Tree With Gini Index Bagging Ensemble Accuracy (Cross Validation):\n" )

count = 0
for i in accuracy:
    count+=1
    print("% 5.4f"%(i),end="")
    if(count%3 == 0 & count !=9):
        print("\n")
print("Mean Of :"+str(np.mean(accuracy)))
print("\n")
#printing confusion matrix
# time.sleep(3)
# cf_matrix = confusion_matrix(y_test,bagging_model_gini_pr)
# sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
#             fmt='.2%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"])

# plt.title("Bagging")
# plt.show()
# plt.close()


# In[13]:


# ------------Boosting---------------------------------
start_timer= perf_counter()
kfold = model_selection.KFold(n_splits=9)
boosting_model_gini = AdaBoostClassifier(base_estimator = clf)
boosting_model_gini.fit(X_train,y_train)
accuracy = cross_val_score(boosting_model_gini, X, Y,scoring="accuracy", cv=kfold)
boosting_model_gini_pr = boosting_model_gini.predict(X_test)
end_timer= perf_counter()
print(f"Execution time of Decision Tree With Gini Index Bagging Model {end_timer-start_timer:0.4f} seconds")
print("\n")
print(f"Decision Tree with Gini Index Boosting Ensemble Accuracy: {metrics.accuracy_score(y_test,boosting_model_gini_pr)}")                                                           
print( classification_report(y_test, boosting_model_gini_pr))
print("Decision Tree With Gini Index Boosting Ensemble Accuracy (Cross Validation):\n") 

count = 0
for i in accuracy:
    count+=1
    print("% 5.4f"%(i),end="")
    if(count%3 == 0):
        print("\n")
print("Mean Of :"+str(np.mean(accuracy)))
plt.close()


#printing confusion matrix
time.sleep(3)
cf_matrix = confusion_matrix(y_test,boosting_model_gini_pr)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"],)

plt.title("Boosting")
plt.show()
print("\n")


# In[81]:



start_timer= perf_counter()
clf = SVC(kernel="linear")
clf.fit(X_train,y_train)
svm_pr = clf.predict(X_test)
end_timer=perf_counter()
print("\nKernel : Linear")
print(f"Execution time of SVM Model {end_timer-start_timer:0.4f} seconds")
print("\n")
print(f"SVM Default Accuracy: {metrics.accuracy_score(y_test,svm_pr)}") 
print("\n")

cf_matrix = confusion_matrix(y_test,svm_pr)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"],)
plt.show()



clf_poly = SVC(kernel="poly")
clf_poly.fit(X_train,y_train)
svm_poly_pr = clf_poly.predict(X_test)
print("\nKernel : Polynomial")
print(f"SVM Default Accuracy: {metrics.accuracy_score(y_test,svm_poly_pr)}")


clf_rbf = SVC(kernel="rbf")
clf_rbf.fit(X_train,y_train)
svm_rbf_pr = clf_rbf.predict(X_test)
print("\nKernel : rbf")
print(f"SVM Default Accuracy: {metrics.accuracy_score(y_test,svm_rbf_pr)}")  


clf_sig = SVC(kernel="sigmoid")
clf_sig.fit(X_train,y_train)
svm_sig_pr = clf_sig.predict(X_test)
print("\nKernel : Sigmoid")
print(f"SVM Default Accuracy: {metrics.accuracy_score(y_test,svm_sig_pr)}")  


# Since the SVM is strong learner boosting ensembled is not applied on.


# # ------------Boosting---------------------------------


# boosting_model = AdaBoostClassifier(base_estimator =clf)
# print("asd")
# boosting_model.fit(X_train,y_train)

# kfold = model_selection.KFold(n_splits=9)
# accuracy = cross_val_score(boosting_model, X, Y,scoring="accuracy", cv=kfold)
# boosting_model_pr = boosting_model.predict(X_test)
# print(f"SVM Boosting Ensemble Accuracy: {metrics.accuracy_score(y_test,boosting_model_pr)}")

# print( classification_report(y_test, boosting_model_pr))
# print("SVM Boosting Ensemble Accuracy (Cross Validation):\n") 

# count = 0
# for i in accuracy:
#     count+=1
#     print("% 5.4f"%(i),end="")
#     if(count%3 == 0):
#         print("\n")
# print("Mean Of :"+str(np.mean(accuracy)))
# plt.close()


# #printing confusion matrix

# cf_matrix = confusion_matrix(y_test,boosting_model_pr)
# sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
#             fmt='.2%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"],)

# plt.title("Boosting")
# plt.show()
# print("\n")
# print("bitti")


# In[ ]:


#-------------Bagging----------------------------------
start_timer= perf_counter()
clf = SVC(kernel="linear")
clf.fit(X_train,y_train)
svm_pr = clf.predict(X_test)
end_timer=perf_counter()

print("Bagging")
start_timer= perf_counter()
bagging_model_cvm = BaggingClassifier(base_estimator=clf)
bagging_model_cvm.fit(X_train,y_train)
#cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 10,random_state = 1)
kfold = model_selection.KFold(n_splits=9)
accuracy = cross_val_score(bagging_model_cvm,X,Y,scoring = "accuracy",cv=kfold)
bagging_model_cvm_pr = bagging_model_cvm.predict(X_test)
end_timer=perf_counter()
print("\n")
print(f"Execution time of SVM Bagging Model {end_timer-start_timer:0.4f} seconds")
print("\n")
print("SVM Bagging Ensemble Accuracy: "+str(metrics.accuracy_score(y_test,bagging_model_cvm_pr)))
print( classification_report(y_test, bagging_model_pr))
print("SVM Bagging Ensemble Accuracy (Cross Validation):\n" )

count = 0
for i in accuracy:
    count+=1
    print("% 5.4f"%(i),end="")
    if(count%3 == 0 & count !=9):
        print("\n")
print("Mean Of :"+str(np.mean(accuracy)))
print("\n")
#printing confusion matrix

cf_matrix = confusion_matrix(y_test,bagging_model_cvm_pr)
print(cf_matrix)
print(f"asadasdasd{np.sum(cf_matrix)}")
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"])

plt.title("Bagging")
plt.show()
plt.close()


# In[67]:


print("\n")
print("----------------------- NAIVE BAYES -----------------------")
start_timer= perf_counter()
nb=GaussianNB()
nb.fit(X_train, y_train) #train data
y_pred=nb.predict(X_test)  #the result of training (after test) 
end_timer=perf_counter()
time.sleep(3)

cf_matrix1 = confusion_matrix(y_test,y_pred)

print("Holdout Method Naive Bayes Accuracy:", accuracy_score(y_test, y_pred, normalize=True))
print("Confusion Matrix of Naive Bayes:\n", cf_matrix1)
print(f"Execution time of Naive Bayes Bagging Model {end_timer-start_timer:0.4f} seconds")
print("\n")
print( classification_report(y_test, y_pred))


# sns.heatmap(cf_matrix1/np.sum(cf_matrix1), annot=True, 
#     fmt='.1%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"],)

# plt.show()
# plt.close()


# In[69]:


#-------------Bagging----------------------------------

start_timer= perf_counter()
bagging_model=BaggingClassifier( base_estimator=nb, n_estimators=9)
bagging_model.fit(X_train, y_train)
bagging_model_pr = bagging_model.predict(X_test)
end_timer=perf_counter()


time.sleep(3)
cf_matrix2 = confusion_matrix(y_test,bagging_model_pr)

print("Naive Bayes Bagging Model Accuracy:", accuracy_score(y_test,bagging_model_pr))
print("Confusion Matrix of Naive Bayes Bagging Model:\n", cf_matrix2)
print(f"Execution time of Naive Bayes Bagging Model {end_timer-start_timer:0.4f} seconds")
print("\n")
print(classification_report(y_test,bagging_model_pr))

# sns.heatmap(cf_matrix2/np.sum(cf_matrix2), annot=True, 
#     fmt='.1%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"],)

# plt.show()
# plt.close()



# In[71]:


# ------------Boosting---------------------------------

start_timer= perf_counter()
boosting_model=AdaBoostClassifier( base_estimator=nb, n_estimators=9)
boosting_model.fit(X_train,y_train)
boosting_model_pr=boosting_model.predict(X_test)
end_timer=perf_counter()

time.sleep(3)
cf_matrix3 = confusion_matrix(y_test,boosting_model_pr)

print("Naive Bayes Boosting Model Accuracy:", accuracy_score(y_test, boosting_model_pr))
print("Confusion Matrix of Naive Bayes Boosting Model:\n", confusion_matrix(y_test,boosting_model_pr))
print(f"Execution time of Naive Bayes Boosting Model {end_timer-start_timer:0.4f} seconds")
print("\n")
print(classification_report(y_test, boosting_model_pr))


sns.heatmap(cf_matrix3/np.sum(cf_matrix3), annot=True, 
    fmt='.1%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"],)

plt.show()
plt.close()


# In[73]:


#------------------------------------- Artificial Neural Network with 1 Hidden Layer ---------------------------------------------------------------------------------------------------------
print("Artificial Neural Network with 1 Hidden Layer\n***********************")
start_timer= perf_counter()
mlp = MLPClassifier(hidden_layer_sizes=(150,), activation = 'relu', solver='adam', max_iter=1000)
mlp.fit(X_train, y_train)
y_pred_nn = mlp.predict(X_test)
end_timer= perf_counter()

print(f"Execution time of ANN with 1 Hidden Layer {end_timer-start_timer:0.4f} seconds")
print("confusion matrix: \n", confusion_matrix(y_test,y_pred_nn))
print("accuracy: ", metrics.accuracy_score(y_test, y_pred_nn))

# time.sleep(3)
# cf_matrix = confusion_matrix(y_test,y_pred_nn)
# sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
#     fmt='.1%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"],)

# plt.show()



# In[75]:


# #-------------implementation of the Bagging ensemble method ---------------------------
start_timer= perf_counter()
bagging_model_nn = BaggingClassifier(base_estimator=mlp)
bagging_model_nn.fit(X_train, y_train)
RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#scores_nn = cross_val_score(bagging_model_nn, X, Y, scoring='accuracy', cv=cv_nn)
y_pred_bgg_nn = bagging_model_nn.predict(X_test)
end_timer= perf_counter()
print(f"Execution time of ANN with 1 Hidden Layer Bagging Model {end_timer-start_timer:0.4f} seconds")
print("\nBagging ensemble confusion matrix: \n", confusion_matrix(y_test, y_pred_bgg_nn))
print("Bagging ensemble accuracy: ", metrics.accuracy_score(y_test, y_pred_bgg_nn))

# time.sleep(3)
# cf_matrix = confusion_matrix(y_test,y_pred_bgg_nn)
# sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
#     fmt='.1%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"],)

# plt.show()



# In[76]:


# # ------------implementation of the Boosting ensemble method--------------------------
start_timer= perf_counter()
boosting_model_nn = GradientBoostingClassifier(init=mlp)
boosting_model_nn.fit(X_train, y_train)
RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#scores_bst_nn = cross_val_score(boosting_model_nn, X, Y, scoring='accuracy', cv=cv_bst_nn)
y_pred_bst_nn = boosting_model_nn.predict(X_test)
end_timer= perf_counter()
print(f"Execution time of ANN with 1 Hidden Layer Boosting Model {end_timer-start_timer:0.4f} seconds")
print("\nBoosting ensemble confusion matrix: \n", confusion_matrix(y_test, y_pred_bst_nn))
print("Boosting ensemble accuracy: ", metrics.accuracy_score(y_test, y_pred_bst_nn))

time.sleep(3)
cf_matrix = confusion_matrix(y_test,y_pred_bst_nn)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
    fmt='.1%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"],)

plt.show()


# In[77]:


#------------------------------------- Artificial Neural Network with 2 Hidden Layers ---------------------------------------------------------------------------------------------------------

print("\nArtificial Neural Network with 2 Hidden Layer\n***********************")
start_timer= perf_counter()
mlp = MLPClassifier(hidden_layer_sizes=(75,2), activation = 'relu', solver='adam', max_iter=1000)
mlp.fit(X_train, y_train)
y_pred_nn = mlp.predict(X_test)
end_timer= perf_counter()
print(f"Execution time of ANN with 2 Hidden Layer {end_timer-start_timer:0.4f} seconds")
print("confusion matrix: \n", confusion_matrix(y_test,y_pred_nn))
print("accuracy: ", metrics.accuracy_score(y_test, y_pred_nn))

time.sleep(3)
cf_matrix = confusion_matrix(y_test,y_pred_nn)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
    fmt='.1%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"],)

plt.show()
plt.close()



# In[78]:



# #-------------implementation of the Bagging ensemble method --------------------------
start_timer= perf_counter()
bagging_model_nn = BaggingClassifier(base_estimator=mlp)
bagging_model_nn.fit(X_train, y_train)
RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
y_pred_bgg_nn = bagging_model_nn.predict(X_test)
end_timer= perf_counter()
print(f"Execution time of ANN with 2 Hidden Layer Bagging Model {end_timer-start_timer:0.4f} seconds")
print("\nBagging ensemble confusion matrix: \n", confusion_matrix(y_test, y_pred_bgg_nn))
print("Bagging ensemble accuracy: ", metrics.accuracy_score(y_test, y_pred_bgg_nn))
time.sleep(3)
cf_matrix = confusion_matrix(y_test,y_pred_bgg_nn)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
    fmt='.1%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"],)

plt.show()
plt.close()


# In[79]:


# # ------------implementation of the Boosting ensemble method-------------------------
start_timer= perf_counter()
boosting_model_nn = GradientBoostingClassifier(init=mlp)
boosting_model_nn.fit(X_train, y_train)
RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#scores_bst_nn = cross_val_score(boosting_model_nn, X, Y, scoring='accuracy', cv=cv_bst_nn)
y_pred_bst_nn = boosting_model_nn.predict(X_test)
end_timer= perf_counter()
print(f"Execution time of ANN with 2 Hidden Layer Boosting Model {end_timer-start_timer:0.4f} seconds")
print("\nBoosting ensemble confusion matrix: \n", confusion_matrix(y_test, y_pred_bst_nn))
print("Boosting ensemble accuracy: ", metrics.accuracy_score(y_test, y_pred_bst_nn))
time.sleep(3)
cf_matrix = confusion_matrix(y_test,y_pred_bst_nn)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
    fmt='.1%', cmap='Blues',xticklabels=["Normal Bid","Shill Bid"],yticklabels=["Normal Bid","Shill Bid"],)

plt.show()
plt.close()


# In[ ]:




