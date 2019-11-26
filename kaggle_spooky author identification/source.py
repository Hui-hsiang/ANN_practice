import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.preprocessing  import StandardScaler
from sklearn import preprocessing
import nltk


text = pd.read_csv("train.csv")

byauthor = text.groupby("author")

wordFreqByAuthor = nltk.probability.ConditionalFreqDist()

for name, group in byauthor:
   sentences = group['text'].str.cat(sep = ' ')
   sentences = sentences.lower()
   tokens = nltk.tokenize.word_tokenize(sentences)
   frequency = nltk.FreqDist(tokens)
   wordFreqByAuthor[name] = (frequency)

df_fdist = pd.DataFrame.from_dict(wordFreqByAuthor,orient='index')

col_name = df_fdist.columns


# alist = []
# for i in col_name:
#    dropable = True
#    for j in wordFreqByAuthor.keys():
#       wordFreq = wordFreqByAuthor[j].freq(i)
#       if wordFreq > 1e-3:
#          dropable = False

#    if dropable == False:
#       alist.append(i)



# xdf = pd.DataFrame(columns = alist)
# ydf = pd.DataFrame(columns = ['author'] )
xdf = pd.read_csv("train_tree.csv")
ydf = pd.read_csv("train_author.csv")


# for i in text.index :
   # sentences = text.loc[i,'text']
   # sentences = sentences.lower()
   # tokens = nltk.tokenize.word_tokenize(sentences)
   # xdf.loc[i] = 0
   # print(i)
   # for j in tokens:
   #    if j in alist:
   #       xdf.loc[i][j]+=1
   # stra = text.loc[i,'author']
   # ydf.loc[i] = stra

   
# xdf.to_csv("train_tree.csv")   
# ydf.to_csv("train_author.csv")

xdf.info()
ydf.info()


X_train , X_test , y_train , y_test = train_test_split(xdf,ydf['author'],test_size=0.3,random_state=1010)

print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

sc=StandardScaler()

sc.fit(X_train)
x_train_nor=sc.transform(X_train)
x_test_nor=sc.transform(X_test)

                       
clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(5, 2),
              learning_rate='constant', learning_rate_init=0.01,
              max_iter=200, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='sgd', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)

clf.fit(x_train_nor, y_train) 

y_test_predicted = clf.predict(x_test_nor)

accuracy = accuracy_score(y_test, y_test_predicted)
print('training 準確率:',accuracy)



testdf = pd.read_csv("test_dt.csv")


# predata = pd.read_csv("test.csv")
# for i in predata.index :
#    sentences = predata.loc[i,'text']
#    sentences = sentences.lower()
#    tokens = nltk.tokenize.word_tokenize(sentences)
#    testdf.loc[i] = 0
#    for j in tokens:
#       if j in testdf.columns:
#          testdf.loc[i][j]+=1
# testdf.to_csv("test_dt.csv")

output = clf.predict_proba(testdf)


submission = pd.read_csv("sample_submission.csv")
submission["EAP"] = output[:,0]
submission["HPL"] = output[:,1]
submission["MWS"] = output[:,2]
# submission[["EAP","HPL","MWS"]] = output[[:,0],[:,1],[0,2]]
submission.to_csv("submission.csv")

