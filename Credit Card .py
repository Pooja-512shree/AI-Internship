#!/usr/bin/env python
# coding: utf-8

# In[7]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)


# In[8]:


#Import Libraries 
import pandas as pd
import os

file_path = os.path.join(path, "creditcard.csv")

df = pd.read_csv(file_path)


# In[9]:


df.head()


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.shape


# In[13]:


df.isnull().sum()


# In[14]:


df['Class'].value_counts()


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Class', data=df)

plt.title("Fraud vs Normal Transactions")

plt.show()


# In[16]:


corr_matrix = df.corr()
corr_matrix


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(18,14))

sns.heatmap(corr_matrix, cmap='coolwarm',annot=True,fmt=".2f")

plt.title("Feature Correlation Heatmap")

plt.show()


# In[21]:


X = df.drop('Class', axis=1)

y = df['Class']


# In[22]:


X.shape


# In[23]:


y.shape


# In[24]:


#Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[25]:


X_train.shape


# In[26]:


X_test.shape


# In[27]:


y_train.shape


# In[28]:


y_test.shape


# In[29]:


#Random Forest Baseline Model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


# In[30]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


# In[31]:


#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
grid_search.best_params_


# In[32]:


best_model = grid_search.best_estimator_

y_pred_tuned = best_model.predict(X_test)


# In[33]:


accuracy_score(y_test, y_pred_tuned)


# In[34]:


#Feature Importance Analysis
import pandas as pd

feature_importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

feature_importance.head(10)


# In[38]:


plt.figure(figsize=(10,6))

feature_importance.head(10).plot(kind='bar', color='teal')

plt.title("Top 10 Important Features")
plt.ylabel("Importance Score")
plt.xlabel("Features")

plt.show()


# In[37]:


# ROC Curve Evaluation
# Import required metrics
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get probability predictions for the positive class (fraud = 1)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Compute ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Compute AUC score
auc_score = roc_auc_score(y_test, y_prob)

# Plot ROC Curve
plt.figure(figsize=(8,6))

plt.plot(fpr, tpr, color='blue', label=f"Random Forest (AUC = {auc_score:.4f})")

# Plot diagonal reference line
plt.plot([0,1], [0,1], linestyle='--', color='red')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Credit Card Fraud Detection")

plt.legend()

plt.show()


# In[ ]:




