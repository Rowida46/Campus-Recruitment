# Campus-Recruitment

*Machine learning project
Academic Year / Level Fourth year, First Semeste.
CS major*


## Team Members

[George Azmy Fawzy](https://github.com/MrGaFs), [Rowida Nagah](https://github.com/Rowida46), [Abdelrahman Saeed](https://github.com/AbdelrahmanSaeed11) and [Makar Samer](https://github.com/makar132)


## Objective 
  
  Our goal is to predict whether a candidate will be hired or not. This is a classification problem and we will use KNN and SVM and compare how these perform individually.

[Used Data Set](https://www.kaggle.com/aayushmishra1512/campus-recruitment-logistic-knn-svm/data)


*We will work on*
-  Data Exploration & Cleaning
-  Look for null values in our data.
-  Determining factors that influence placement.
-  Drop Misleading or unnecessary column, `sl_no and` and `salary` column.
```python
df1.drop(['sl_no','salary'],axis = 1,inplace = True)
 ```
-  Split them and into training and testing sets.
```python
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 101)

```
-  Scale down our features in order to have better results
```python
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

- And then We will apply *KNN* and *SVM* on this data.

---


**KNN* Algorithm*
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
prediction = knn.predict(X_test)
```



**SVM Algorithm**
```python
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
svc_pred = svc.predict(X_test)
```

# Result
---

<dl>
  <dt>Using *SVM*</dt>
  <dd>Accuracy: 79.06976744186046
  <dd>Precision: 82.14285714285714</dd>
  <dd>Recall: 85.18518518518519</dd>
  <dt>Using *KKN*</dt>
  <dd>Accuracy: 74.4186046511628</dd>
  <dd>Precision: 80.76923076923077</dd>
  <dd>Recall: 77.77777777777779</dd>
</dl>

This model still has some scope for improvement. But we could see that Logistic Regression performed better than both KNN and SVM. So atleast we have an idea which model would give us the better results.


## Referance 

- [Predicting whether student gets hired or not *github Repo*](https://github.com/Nanasei878/Campus-Recruitment)
- [Campus Recruitment Kaggle Challenge](https://www.kaggle.com/benroshan/factors-affecting-campus-placement)
