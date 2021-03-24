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
from collections import Counter
data=pd.read_csv("cleaned_lyrics.csv").astype(str)
sns.countplot(x="genre",data=data)
plt.show()

train,test=train_test_split(data,test_size=0.2,random_state=1)
# majority baseline

dummy_clf = DummyClassifier(strategy="most_frequent")

dummy_clf.fit(train["lyrics"],train["genre"])

res=dummy_clf.predict(test["lyrics"])

print(classification_report(test["genre"],res))


### svm results
svm_res=pd.read_csv("svm.csv")
sns.lineplot(data=svm_res,x="param_SVC__C",y="mean_test_score",hue="param_SVC__kernel")
plt.show()

final_svm=pd.read_csv("svm_detail.csv")
sns.lineplot(data=final_svm,x="param_SVC__C",y="mean_test_score")
plt.show()

### C=1.4 kernel=rbf
svm=Pipeline([('count', CountVectorizer()),('tfidf',TfidfTransformer()),("SVC",SVC(C=1.4,kernel="rbf"))])
svm.fit(train["lyrics"],train["genre"])
svm_pred=svm.predict(test["lyrics"])
print(classification_report(test["genre"],svm_pred,digits=4))

svm_count=Counter(svm_pred)
plt.figure(figsize=(5,5))
plt.pie(x=svm_count.values(),labels=svm_count.keys(),autopct = '%3.2f%%')
plt.show()

## decision tree res
dt_res=pd.read_csv("DTC_1.csv")
sns.lineplot(data=dt_res,x="param_DTC__max_depth",y="mean_test_score",hue="param_DTC__criterion")
plt.show()


dt_pipe=Pipeline([('count', CountVectorizer()),('tfidf',TfidfTransformer()),("DTC",DecisionTreeClassifier())])
parameter={"DTC__criterion":("gini", "entropy")}
dt_clf=GridSearchCV(dt_pipe,parameter,n_jobs=-1,verbose=10)
dt_clf.fit(train["lyrics"],train["genre"])
p=pd.DataFrame(dt_clf.cv_results_)

## gini md=9 md=none lower than md=9
dt_pipe=Pipeline([('count', CountVectorizer()),('tfidf',TfidfTransformer()),("DTC",DecisionTreeClassifier())])

dt_pipe.fit(train["lyrics"],train["genre"])
dt_pred=dt_pipe.predict(test["lyrics"])
print(classification_report(test["genre"],dt_pred,digits=4))

dt_count=Counter(dt_pred)
plt.figure(figsize=(5,5))
plt.pie(x=dt_count.values(),labels=dt_count.keys(),autopct = '%3.2f%%')
plt.show()


rf_res1=pd.read_csv("RF_2.csv")
sns.lineplot(data=rf_res1,x="param_RF__n_estimators",y="mean_test_score",hue="param_RF__max_depth")
plt.show()

rf_res2=pd.read_csv("RF_3.csv")
sns.lineplot(data=rf_res2,x="param_RF__n_estimators",y="mean_test_score")
plt.show()

## rf n_estimator=800
rf_pipe=Pipeline([('count', CountVectorizer()),('tfidf',TfidfTransformer()),("RF",RandomForestClassifier(n_estimators=800))])
rf_pipe.fit(train["lyrics"],train["genre"])
#rf_pred_train=rf_pipe.predict(train["lyrics"])
rf_pred=rf_pipe.predict(test["lyrics"])

rf_count=Counter(rf_pred)
plt.figure(figsize=(5,5))
plt.pie(x=rf_count.values(),labels=rf_count.keys(),autopct = '%3.2f%%')
plt.show()

print(classification_report(test["genre"],rf_pred,digits=4))


acc_res={"Baseline":0.30,"Randomforest":0.6357,"Decision tree":0.5102,"SVM":0.6245,"GRU":0.5828,"LSTM":0.6063,"TextCNN":0.5073}
plt.bar(acc_res.keys(),acc_res.values())
plt.show()

comp_res=pd.DataFrame({"model":["randomforest","decision tree","svm"]})
filter_rf=rf_res2[rf_res2["rank_test_score"]==1].loc[:,["mean_fit_time","split0_test_score","split1_test_score","split2_test_score","split3_test_score","split4_test_score","mean_test_score"]]
filter_dt=dt_res[dt_res["rank_test_score"]==1].loc[:,["mean_fit_time","split0_test_score","split1_test_score","split2_test_score","split3_test_score","split4_test_score","mean_test_score"]]
filter_svm=svm_res[svm_res["rank_test_score"]==1].loc[:,["mean_fit_time","split0_test_score","split1_test_score","split2_test_score","split3_test_score","split4_test_score","mean_test_score"]]
filtered_res=pd.concat((filter_rf,filter_dt,filter_svm),ignore_index=True)
comp_res=pd.concat((comp_res,filtered_res),axis=1)










