import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


file = pd.read_csv('ResumeDataSet.csv')


train = file[file['Category'] != 'Testing']    
print( train.head(1) )

# x_train, x_val, y_train, y_val = train_test_split(
#     train['Category'], train['Resume'], test_size=0.2, random_state=0)

# print(train.columns)

# randomforest = RandomForestClassifier()

# # Fit the training data along with its output
# randomforest.fit(x_train, y_train)

# y_pred = randomforest.predict(x_val)

# # Find the accuracy score of the model
# acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
# print(acc_randomforest)