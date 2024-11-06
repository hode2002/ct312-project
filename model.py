from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from data_transformation import data, scaler

# Tách dữ liệu thành đầu vào và nhãn
X = data[[
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
    'Extent', 'Solidity', 'roundness', 'Compactness',
    'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
]]
y = data['Class']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=24)

# Xây dựng mô hình KNN với k=14
model = KNeighborsClassifier(n_neighbors=14)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra và tính độ chính xác
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy:.2f}")

# Lưu mô hình và scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
