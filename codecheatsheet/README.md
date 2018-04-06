# scikit-learn codecheatsheet

## 基本的な使用例
```python
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X, y = iris.data[:, :2], iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)
```

## データの読み込み
```python
import numpy as np
X = np.random.random((10,5))
y = np.array(['M','M','F','F','M','F','M','M','F','F','F'])
X[X < 0.7] = 0
```

## 前処理

### 標準化
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)
```

### 正規化
```python
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)
```
### one-hot エンコーディング(pandas)
```python
df = pd.get_dummies(df,columns=None)
columns : list-like, default None
```

## データ分割
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
```
## モデル作成

### 教師あり学習

#### SVM
```python
from sklearn.svm import SVC
svc = SVC(kernel='linear')
```
#### Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
```
#### KNN
```python
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
```

### 教師なし学習

#### 主成分分析
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
```
#### k-means
```python
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3, random_state=0)
```

## モデルフィッテイング
### 教師なし学習
```python
knn.fit(X_train, y_train)
svc.fit(X_train, y_train)
```
### 教師あり学習
```python
k_means.fit(X_train)
pca_model = pca.fit_transform(X_train)
```

## 推定
### 教師なし学習
```python
y_pred = svc.predict(np.random.random((2,5)))
```
```python
y_pred = knn.predict_proba(X_test))
```
### 教師あり学習
```python
y_pred = k_means.predict(X_test)
```

## モデルの評価

### 分類用の指標
#### Accuracy
```python
knn.score(X_test, y_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```

#### Classification Report
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred)))
```

#### Confusion Matrics（混同行列）
```python
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred)))
```

### 回帰用の指標
#### Mean Absolute Error（平均絶対誤差）
```python
from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2])
mean_absolute_error(y_true, y_pred))
```
#### Mean Squared Error（平均二乗誤差）
```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred))
```

#### R^2Score（決定係数）
```python
from sklearn.metrics import r2_score
r2_score(y_true, y_pred))
```

### 交差検証
```python
from sklearn.model_selection import cross_val_score
print(cross_val_score(knn, X_train, y_train, cv=4))
print(cross_val_score(lr, X, y, cv=2))
```

## モデルチューニング

### グリッドサーチ
```python
from sklearn.grid_search import GridSearchCV
params = {"n_neighbors": np.arange(1,3), "metric": ["euclidean", "cityblock"]}
grid = GridSearchCV(estimator=knn,param_grid=params)
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)
```
### ランダムサーチ
```python
from sklearn.grid_search import RandomizedSearchCV
params = {"n_neighbors": range(1,5), "weights": ["uniform", "distance"]}
rsearch = RandomizedSearchCV(estimator=knn,
   param_distributions=params,
   cv=4,
   n_iter=8,
   random_state=5)
rsearch.fit(X_train, y_train)
print(rsearch.best_score_)
```