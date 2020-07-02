import sklearn 
import sklearn.datasets
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pydotplus
from pydotplus import graphviz
from IPython.display import Image
from sklearn import tree
from graphviz import Graph

cancer=load_breast_cancer()
print(cancer)
x_train,x_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=3)


clf=DecisionTreeClassifier(max_depth=3)
model=clf.fit(x_train,y_train)





##print("accuracy train:",model.score(x_train,y_train)*100)
print("accuracy:",model.score(x_test,y_test)*100)
print("data:",cancer)



dot_data= tree.export_graphviz(clf,
                               out_file=None,
                               feature_names=cancer.feature_names,
                               class_names=cancer.target_names)

graph= pydotplus.graph_from_dot_data(dot_data)


##Image(graph.create_png())
##
##graph.write_png("breast_cancer.png")

