from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from feature2 import getData
attr=['eye','mouth','label']
data=getData()
print(data[attr[:2]],data['label'])
X_train,X_test,Y_train,Y_test = train_test_split(data[attr[:3]],data['label'],train_size=0.80)
