
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
np.random.seed(0)

# loading our dataset
# adding the dataset into data frame to work with
df = pd.read_csv("dataset.csv")

# dividing the dataset into training and testing parts 75% for training 25% for testing
# a random number between 0 and 1 if less than .75 is true else false --> for every row df
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

# creating data frames to the test and train rows
train, test = df[df['is_train'] == True], df[df['is_train'] == False]

# creating a list of feature names
features = df.columns[:31]

# getting the target column into y
y=train['Result']

# creating RF classifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)#fee atribute esmo max_features

# training the classifier
clf.fit(train[features], y)# adding a print will show every attribute in the algorithm feh max features used and other attribitues

# apply the trained classifier to the test ds
preds=clf.predict(test[features])
# print it out to see the result
# print(clf.predict_proba(test[features])[0:10]) shows the probability of each url awl 10

#cross tab bt compare el values l predicted bl actual w bttl3 el result
print(pd.crosstab(test['Result'], preds, rownames=['Actual Result'], colnames=['Predicted Result']))
# 97.32% accuracy
