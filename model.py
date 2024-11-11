from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from data_transformation import x_smote, y_smote

X_train, X_test, y_train, y_test = train_test_split(x_smote, y_smote,test_size = 0.2, random_state=42, shuffle=True)

scaler = MinMaxScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)

model = DecisionTreeClassifier(max_depth=8)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy:.2f}")

joblib.dump(model, 'model.pkl')
