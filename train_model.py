import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

DATASET_PATH = "cats_dogs_dataset/train"
IMG_SIZE = 64

X = []
y = []

for img_name in os.listdir(DATASET_PATH):
    img_path = os.path.join(DATASET_PATH, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    if "cat" in img_name.lower():
        X.append(features)
        y.append(0)
    elif "dog" in img_name.lower():
        X.append(features)
        y.append(1)

X = np.array(X)
y = np.array(y)

X, y = shuffle(X, y, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC(kernel="rbf", C=5, gamma="scale", probability=True)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ✅ SAVE FILES (IMPORTANT)
joblib.dump(svm, "svm_hog_model.pkl")
joblib.dump(scaler, "hog_scaler.pkl")

print("✅ HOG model saved successfully")
