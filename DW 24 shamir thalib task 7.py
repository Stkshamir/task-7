import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
db = pd.read_csv('testKNN.csv')
dn = pd.read_csv('trainKNN.csv')

dataset = pd.read_csv("glass.csv")
df = pd.read_csv('tested.csv')

dataset = dataset[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']]
X= dataset[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
Y = dataset['Type']




x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)

knn = KNeighborsClassifier(5)
knn.fit(x_train,y_train)
result = knn.predict(x_test)

print("*****KNN algorithem outcome*****")
print(classification_report(y_test,result))
print('**********svm algorithem outcome')

sv = SVC()
sv.fit(x_train,y_train)
prediction = sv.predict(x_test)

sns.scatterplot(data = dataset,x ='RI',y= 'Ba' ,hue='Type')

non_d = dataset[dataset['Type']==0]
d = dataset[dataset['Type']==1]
plt.scatter(non_d['RI'],non_d['Na'],non_d['Mg'],color = 'red')
plt.scatter(d['RI'],d['Na'],d['Mg'],color = 'blue')
plt.show()


import matplotlib.pyplot
from sklearn import tree
from sklearn.metrics import accuracy_score
X= dataset[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
Y = dataset['Type']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)
obj = DecisionTreeClassifier(max_depth=(10))
dtree = obj.fit(x_train,y_train)
predict = obj.predict(x_test)
predict_train = obj.predict(x_train)
print('Testing accuracy:',accuracy_score(y_test, predict))
print('Traning accuracy:',accuracy_score(y_train, predict_train))

dtree = obj.fit(x_train,y_train)
plt.figure(figsize=(20,10))
tree.plot_tree(dtree,class_names=X.columns,rounded=True,filled = True,fontsize=6)
print(obj.get_depth())
plt.show()
print('**************************')
from sklearn.ensemble import RandomForestClassifier
dtc = RandomForestClassifier(n_estimators= 10)
dtc.fit(x_train,y_train)
predict = dtc.predict(x_test)
predict_train = dtc.predict(x_train)
print('Testing accuracy:',accuracy_score(y_test, predict))
print('Traning accuracy:',accuracy_score(y_train, predict_train))





