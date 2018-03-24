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
### one-hot エンコーディング

## データ分割
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
```
## モデル作成

### 教師あり学習

- SVM
```python
from sklearn.svm import SVC
svc = SVC(kernel='linear')
```
- Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
```
- KNN
```python
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
```

### 教師なし学習

- 主成分分析
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
```
- k-means
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
- Accuracy
- Classification Report
- Confusion Matrics（混同行列）

### 回帰用の指標
- Mean Absolute Error（平均絶対誤差）
- Mean Squared Error（平均二乗誤差）
- R^2Score（決定係数）

### 交差検証

## モデルチューニング

### グリッドサーチ

### ランダムサーチ
