from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier




lyrics_df=pd.read_csv("cleaned_lyrics.csv").astype(str)
lyrics_df=lyrics_df[lyrics_df["genre"]!="nan"]


train,test=train_test_split(lyrics_df,test_size=0.2,random_state=1)

dummy_clf = DummyClassifier(strategy="most_frequent")

dummy_clf.fit(train["lyrics"],train["genre"])

res=dummy_clf.predict(test["lyrics"])

print(classification_report(test["genre"],res))


#
# pipe=Pipeline([('count', CountVectorizer()),('tfidf',TfidfTransformer()),("NB",MultinomialNB())])
# parameter={"NB__alpha":(np.linspace(0.1,10,10))}


### SVM
pipe=Pipeline([('count', CountVectorizer()),('tfidf',TfidfTransformer()),("SVC",SVC())])
parameter={"SVC__kernel":("linear", "poly", "rbf", "sigmoid"),"SVC__C":np.linspace(1,10,10)}

clf=GridSearchCV(pipe,parameter,n_jobs=-1,verbose=10)
clf.fit(train["lyrics"],train["genre"])
p=pd.DataFrame(clf.cv_results_)
p.to_csv("svm.csv")


t=pd.read_csv("svm.csv")
sns.lineplot(data=t,x="param_SVC__C",y="mean_test_score",hue="param_SVC__kernel")
plt.show()


### deeper SVM
pipe=Pipeline([('count', CountVectorizer()),('tfidf',TfidfTransformer()),("SVC",SVC())])
parameter={"SVC__C":(np.linspace(1,4,31))}

clf=GridSearchCV(pipe,parameter,n_jobs=-1,verbose=10)
clf.fit(train["lyrics"],train["genre"])
p=pd.DataFrame(clf.cv_results_)
p.to_csv("svm_detail.csv")

t=pd.read_csv("svm_detail.csv")
sns.lineplot(data=t,x="param_SVC__C",y="mean_test_score")
plt.show()


### DT clf
pipe=Pipeline([('count', CountVectorizer()),('tfidf',TfidfTransformer()),("DTC",DecisionTreeClassifier())])
parameter={"DTC__criterion":("gini", "entropy"),"DTC__max_depth":(np.linspace(3,50,8,dtype="int"))}
clf=GridSearchCV(pipe,parameter,n_jobs=-1,verbose=10)
clf.fit(train["lyrics"],train["genre"])
p=pd.DataFrame(clf.cv_results_)
p.to_csv("DTC_1.csv")

t=pd.read_csv("DTC_1.csv")
sns.lineplot(data=t,x="param_DTC__max_depth",y="mean_test_score",hue="param_DTC__criterion")
plt.show()

### RF clf
pipe=Pipeline([('count', CountVectorizer()),('tfidf',TfidfTransformer()),("RF",RandomForestClassifier())])
parameter={"RF__n_estimators":np.linspace(100,1000,10,dtype="int")}
#,"RF__max_depth":(np.linspace(3,50,8,dtype="int"))
clf=GridSearchCV(pipe,parameter,n_jobs=-1,verbose=10)
clf.fit(train["lyrics"],train["genre"])
p=pd.DataFrame(clf.cv_results_)
p.to_csv("RF_3.csv")


t=pd.read_csv("RF_3.csv")
#sns.lineplot(data=t,x="param_RF__n_estimators",y="mean_test_score",hue="param_RF__max_depth")
sns.lineplot(data=t,x="param_RF__n_estimators",y="mean_test_score")

plt.show()


pipe=Pipeline([('count', CountVectorizer()),('tfidf',TfidfTransformer()),("Ada",AdaBoostClassifier(learning_rate=0.1))])
parameter={"Ada__n_estimators":np.linspace(1300,1800,6,dtype="int")}
#,"RF__max_depth":(np.linspace(3,50,8,dtype="int"))
clf=GridSearchCV(pipe,parameter,n_jobs=-1,verbose=10)
clf.fit(train["lyrics"],train["genre"])
p=pd.DataFrame(clf.cv_results_)
p.to_csv("Ada_2.csv")

t=pd.read_csv("Ada_2.csv")
sns.lineplot(data=t,x="param_Ada__n_estimators",y="mean_test_score")
plt.show()

# res=clf.predict(test["lyrics"])
# print(classification_report(test["genre"],res))
# from sklearn.metrics import plot_confusion_matrix
# plot_confusion_matrix(clf,test["lyrics"],test["genre"])

