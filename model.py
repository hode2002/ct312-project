from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

from data_transformation import data

"""### CHIA TẬP DỮ LIỆU"""
X = data[[
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
    'Extent', 'Solidity', 'roundness', 'Compactness',
    'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
]]
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

"""### XÂY DỰNG MÔ HÌNH DỰ ĐOÁN"""
nFold = 5

# Xây dựng mô hình với k = 14
model = KNeighborsClassifier(n_neighbors=14)
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')
