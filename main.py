# %% [markdown]
# # 1. Importing Modules

# %%

#importing basic packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score


# Decision Tree model 
from sklearn.tree import DecisionTreeClassifier


# Random Forest model
from sklearn.ensemble import RandomForestClassifier


# Multilayer Perceptrons model
from sklearn.neural_network import MLPClassifier

#XGBoost Classification model
from xgboost import XGBClassifier


#importing required packages
import keras
from keras.layers import Input, Dense
from keras import regularizers
import tensorflow as tf
from keras.models import Model
from sklearn import metrics



#Support vector machine model
from sklearn.svm import SVC

# To save model
import pickle

# %% [markdown]
# # 2. Loading Data and checking data

# %%
df_1 = pd.read_csv('Dataset\dataset_full.csv')
df_2 = pd.read_csv('Dataset\dataset_small.csv', error_bad_lines=False)

df = df_1.merge(df_2)


# %%
df_1.head()


# %%
df_2.head()

# %%
df.head()

# %%

#Checking the shape of the dataset
df.shape
     

# %%

#Listing the features of the dataset
df.columns

# %%
#Information about the dataset
df.info()


# %% [markdown]
# # 3. Visualizing Data

# %%
#Plotting the data distribution
df.hist(bins = 50,figsize = (40,40))
plt.show()


# %% [markdown]
# # 4. Data Preprocessing & EDA

# %%
df.describe()

# %%
data = df

# %%

#checking the data for null or missing values
data.isnull().sum()

# %%
# No Missing Data

# %% [markdown]
#  # 5. Splitting the Data

# %%

# Sepratating & assigning features and target columns to X & y

y = data['phishing'] # Output columns

X = data.drop('phishing',axis=1)   # Input columns for training

X.shape, y.shape      # To find the shape of datasets
     

# %%
# Splitting the dataset into train and test sets: 80-20 split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)

X_train.shape, X_test.shape

# %% [markdown]
# # 6. Machine Learning Models & Training
# 
# From the dataset above, it is clear that this is a supervised machine learning task. There are two major types of supervised machine learning problems, called classification and regression.
# 
# This data set comes under classification problem, as the input URL is classified as phishing (1) or legitimate (0). The supervised machine learning models (classification) considered to train the dataset in this notebook are:
# 
# <ul>
# <li>Decision Tree</li>
# <li>Random Forest</li>
# <li>Multilayer Perceptrons</li>
# <li>XGBoost</li>
# <li>Autoencoder Neural Network</li>
# <li>Support Vector Machines</li>
# 
# 
# 
# 
# 
# 
# </ul><br>

# %%

# Creating holders to store the model performance results in list
ML_Model = []
acc_train = []
acc_test = []

# function to call for storing the results

def storeResults(model, a,b):
  ML_Model.append(model)
  acc_train.append(round(a, 3))
  acc_test.append(round(b, 3))

# %% [markdown]
# ## 6.1. Decision Tree Classifier
# Decision trees are widely used models for classification and regression tasks. Essentially, they learn a hierarchy of if/else questions, leading to a decision. Learning a decision tree means learning the sequence of if/else questions that gets us to the true answer most quickly.
# 
# In the machine learning setting, these questions are called tests (not to be confused with the test set, which is the data we use to test to see how generalizable our model is). To build a tree, the algorithm searches over all possible tests and finds the one that is most informative about the target variable.

# %%
# instantiate the model 
tree = DecisionTreeClassifier(max_depth = 5)

# fit the model 
tree.fit(X_train, y_train)

# %%

#predicting the target value from the model for the samples
y_test_tree = tree.predict(X_test)
y_train_tree = tree.predict(X_train)

# %% [markdown]
# ### Performence Evaluation : 

# %%

#computing the accuracy of the model performance 

acc_train_tree = accuracy_score(y_train,y_train_tree)  # from sklearn.metrics
acc_test_tree = accuracy_score(y_test,y_test_tree)

print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree))
print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree))

# %%
#storing the results. The below mentioned order of parameter passing is important.
storeResults('Decision Tree', acc_train_tree, acc_test_tree)

# %% [markdown]
# ## 6.2. Random Forest Classifier
# Random forests for regression and classification are currently among the most widely used machine learning methods.A random forest is essentially a collection of decision trees, where each tree is slightly different from the others. The idea behind random forests is that each tree might do a relatively good job of predicting, but will likely overfit on part of the data.
# 
# If we build many trees, all of which work well and overfit in different ways, we can reduce the amount of overfitting by averaging their results. To build a random forest model, you need to decide on the number of trees to build (the n_estimators parameter of RandomForestRegressor or RandomForestClassifier). They are very powerful, often work well without heavy tuning of the parameters, and don’t require scaling of the data.

# %%
# instantiate the model
forest = RandomForestClassifier(max_depth=5)

# fit the model 
forest.fit(X_train, y_train)

# %%
#predicting the target value from the model for the samples
y_test_forest = forest.predict(X_test)
y_train_forest = forest.predict(X_train)
     


#computing the accuracy of the model performance
acc_train_forest = accuracy_score(y_train,y_train_forest)
acc_test_forest = accuracy_score(y_test,y_test_forest)

print("Random forest: Accuracy on training Data: {:.3f}".format(acc_train_forest))
print("Random forest: Accuracy on test Data: {:.3f}".format(acc_test_forest))

# %%

#storing the results. The below mentioned order of parameter passing is important.
storeResults('Random Forest', acc_train_forest, acc_test_forest)

# %% [markdown]
# ## 6.3. Multilayer Perceptrons (MLPs): Deep Learning
# Multilayer perceptrons (MLPs) are also known as (vanilla) feed-forward neural networks, or sometimes just neural networks. Multilayer perceptrons can be applied for both classification and regression problems.
# 
# MLPs can be viewed as generalizations of linear models that perform multiple stages of processing to come to a decision.

# %%
# instantiate the model
mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=([100,100,100]))

# fit the model 
mlp.fit(X_train, y_train)
     

#predicting the target value from the model for the samples
y_test_mlp = mlp.predict(X_test)
y_train_mlp = mlp.predict(X_train)
     
#computing the accuracy of the model performance
acc_train_mlp = accuracy_score(y_train,y_train_mlp)
acc_test_mlp = accuracy_score(y_test,y_test_mlp)

print("Multilayer Perceptrons: Accuracy on training Data: {:.3f}".format(acc_train_mlp))
print("Multilayer Perceptrons: Accuracy on test Data: {:.3f}".format(acc_test_mlp))
     

# %%

#storing the results. The below mentioned order of parameter passing is important.
storeResults('Multilayer Perceptrons', acc_train_mlp, acc_test_mlp)

# %% [markdown]
# ## 6.4. XGBoost Classifier
# XGBoost is one of the most popular machine learning algorithms these days. XGBoost stands for eXtreme Gradient Boosting. Regardless of the type of prediction task at hand; regression or classification. XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.

# %%
# instantiate the model
xgb = XGBClassifier(learning_rate=0.4,max_depth=7)
#fit the model
xgb.fit(X_train, y_train)
     

# %%
#predicting the target value from the model for the samples
y_test_xgb = xgb.predict(X_test)
y_train_xgb = xgb.predict(X_train)

#computing the accuracy of the model performance
acc_train_xgb = accuracy_score(y_train,y_train_xgb)
acc_test_xgb = accuracy_score(y_test,y_test_xgb)

print("XGBoost: Accuracy on training Data: {:.3f}".format(acc_train_xgb))
print("XGBoost : Accuracy on test Data: {:.3f}".format(acc_test_xgb))


#storing the results. The below mentioned order of parameter passing is important.
storeResults('XGBoost', acc_train_xgb, acc_test_xgb)

# %% [markdown]
# ## 6.5. Autoencoder Neural Network
# An auto encoder is a neural network that has the same number of input neurons as it does outputs. The hidden layers of the neural network will have fewer neurons than the input/output neurons. Because there are fewer neurons, the auto-encoder must learn to encode the input to the fewer hidden neurons. The predictors (x) and output (y) are exactly the same in an auto encoder.

# %%
#building autoencoder model

input_dim = X_train.shape[1]
encoding_dim = input_dim

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="relu",
                activity_regularizer=regularizers.l1(10e-4))(input_layer)
encoder = Dense(int(encoding_dim), activation="relu")(encoder)

encoder = Dense(int(encoding_dim-2), activation="relu")(encoder)
code = Dense(int(encoding_dim-4), activation='relu')(encoder)
decoder = Dense(int(encoding_dim-2), activation='relu')(code)

decoder = Dense(int(encoding_dim), activation='relu')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()

# %%
#compiling the model
autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

#Training the model
history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, shuffle=True, validation_split=0.2) 
     

# %%
acc_train_auto = autoencoder.evaluate(X_train, X_train)[1]
acc_test_auto = autoencoder.evaluate(X_test, X_test)[1]

print('\nAutoencoder: Accuracy on training Data: {:.3f}' .format(acc_train_auto))
print('Autoencoder: Accuracy on test Data: {:.3f}' .format(acc_test_auto))
     
#storing the results. The below mentioned order of parameter passing is important.

storeResults('AutoEncoder', acc_train_auto, acc_test_auto)

# %% [markdown]
# ## 6.6. Support Vector Machines
# In machine learning, support-vector machines (SVMs, also support-vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier.

# %%
# instantiate the model
svm = SVC(kernel='linear', C=1.0, random_state=12)



# %%
#fit the model
svm.fit(X_train, y_train)


# %%

#predicting the target value from the model for the samples
y_test_svm = svm.predict(X_test)
y_train_svm = svm.predict(X_train)



# %%
#computing the accuracy of the model performance
acc_train_svm = accuracy_score(y_train,y_train_svm)
acc_test_svm = accuracy_score(y_test,y_test_svm)

print("SVM: Accuracy on training Data: {:.3f}".format(acc_train_svm))
print("SVM : Accuracy on test Data: {:.3f}".format(acc_test_svm))

#storing the results. The below mentioned order of parameter passing is important.
storeResults('SVM', acc_train_svm, acc_test_svm)

# %% [markdown]
# #7. Comparision of Models
# To compare the models performance, a dataframe is created. The columns of this dataframe are the lists created to store the results of the model.

# %%

#creating dataframe
results = pd.DataFrame({ 'ML Model': ML_Model,    
    'Train Accuracy': acc_train,
    'Test Accuracy': acc_test})
results

# %%

#Sorting the datafram on accuracy
results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False)

# %% [markdown]
# **From the above comparision, it is clear that the XGBoost Classifier works well with our dataset.**
# 
# 

# %%
# save XGBoost model to file
pickle.dump(xgb, open("phishing.pkl", "wb"))


