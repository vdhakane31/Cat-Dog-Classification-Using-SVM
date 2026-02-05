import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.utils import shuffle

# =============================
# DATASET PATH
# =============================
DATASET_PATH = "cats_dogs_dataset/train"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

print("✅ Dataset path:", DATASET_PATH)

# =============================
# IMAGE SIZE
# =============================
IMG_SIZE = 32

X, y = [], []

# =============================
# LOAD IMAGES
# =============================
for img_name in os.listdir(DATASET_PATH):
    img_path = os.path.join(DATASET_PATH, img_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.flatten()

    if "cat" in img_name.lower():
        X.append(img)
        y.append(0)
    elif "dog" in img_name.lower():
        X.append(img)
        y.append(1)

X = np.array(X)
y = np.array(y)

X, y = shuffle(X, y, random_state=42)

print("✅ Dataset Loaded")
print("Total Samples:", len(X))

# =============================
# TRAIN TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# FEATURE SCALING
# =============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================
# TRAIN SVM MODEL
# =============================
svm = SVC(kernel="rbf", C=1, gamma="scale", probability=True)
svm.fit(X_train, y_train)

# Save model
joblib.dump(svm, "svm_cat_dog_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model Trained & Saved")

# =============================
# MODEL EVALUATION
# =============================
y_train_pred = svm.predict(X_train)
y_test_pred = svm.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("\n📊 MODEL PERFORMANCE")
print("Train Accuracy:", train_acc)
print("Test Accuracy :", test_acc)

print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_test_pred, target_names=["Cat", "Dog"]))

# =============================
# CONFUSION MATRIX
# =============================
cm = confusion_matrix(y_test, y_test_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Cat", "Dog"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# =============================
# ROC CURVE (BONUS)
# =============================
y_prob = svm.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# =============================
# INTERACTIVE PREDICTION
# =============================
def predict_user_image():
    img_path = input("\nEnter image path (or 'exit'): ")

    if img_path.lower() == "exit":
        return False

    if not os.path.exists(img_path):
        print("❌ Image not found")
        return True

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Unable to read image")
        return True

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_flat = img_resized.flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_flat)

    pred = svm.predict(img_scaled)[0]
    prob = svm.predict_proba(img_scaled)[0]
    confidence = max(prob) * 100

    label = "CAT 🐱" if pred == 0 else "DOG 🐶"
    text = f"{label} ({confidence:.2f}%)"

    cv2.putText(img, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    print("\n🔍 PREDICTION RESULT")
    print("Class     :", label)
    print(f"Confidence: {confidence:.2f}%")

    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return True

# =============================
# MENU
# =============================
while True:
    print("\n========== CAT vs DOG CLASSIFIER ==========")
    print("1. Predict new image")
    print("2. Show model accuracy")
    print("3. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        if not predict_user_image():
            break
    elif choice == "2":
        print("Train Accuracy:", train_acc)
        print("Test Accuracy :", test_acc)
    elif choice == "3":
        print("✅ Program exited")
        break
    else:
        print("❌ Invalid choice")
