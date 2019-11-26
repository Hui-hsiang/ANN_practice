from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.preprocessing  import StandardScaler
from sklearn import preprocessing


col_names = ['age','workclass','fnlwht','education','education-num','marital-status'
				   ,'occupation','relationship','race','sex','capital-gain','capital-loss',
				   'hours-per-week','native-country','result' ]
data = pd.read_csv("adult.csv",names = col_names)

data_clean = data.replace(regex=[r'\?|\.|\$'],value=np.nan)

adult = data_clean.dropna(how='any')

label_encoder = preprocessing.LabelEncoder()
for col in col_names:
    if (col in ['fnlwht','education-num','capital-gain','capital-loss','hours-per-week','age'] ):
        continue
    encoded = label_encoder.fit_transform(adult[col])
    adult[col] = encoded

adult['age'] *= 500
print (adult['age'])

X_train , X_test , y_train , y_test = train_test_split(adult[col_names[:14]],adult[col_names[14]],test_size=0.3,random_state=1010)




sc=StandardScaler()

sc.fit(X_train)
x_train_nor=sc.transform(X_train)
x_test_nor=sc.transform(X_test)



                       
clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(3, 1),
              learning_rate='constant', learning_rate_init=0.0001,
              max_iter=800, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='sgd', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)

clf.fit(x_train_nor, y_train) 




y_test_predicted = clf.predict(x_test_nor)

accuracy = accuracy_score(y_test, y_test_predicted)
print('training 準確率:',accuracy)

test_data = pd.read_csv("test.csv",names = col_names)

data_clean = test_data.replace(regex=[r'\?|\$'],value=np.nan)

test = data_clean.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

label_encoder = preprocessing.LabelEncoder()
for col in col_names:
    if (col in ['fnlwht','education-num','capital-gain','capital-loss','hours-per-week','age'] ):
        continue
    encoded = label_encoder.fit_transform(test[col])
    test[col] = encoded

xtest = test[col_names[:14]]
ytest = test[col_names[14]]

p = clf.predict(xtest)
accuracy = accuracy_score(ytest, p)
print('testing 準確率:',accuracy)




