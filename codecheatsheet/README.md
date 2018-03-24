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

### 正規化
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)
```

### 標準化

### one-hot エンコーディング

## データ分割

## モデル作成

### 教師あり学習

- SVM
- Naive Bayes
- KNN

### 教師なし学習

- 主成分分析
- k-means

## モデルフィッテイング

## 推定

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
