import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog

# =============================
# Load trained HOG model
# =============================
svm = joblib.load("svm_hog_model.pkl")
scaler = joblib.load("hog_scaler.pkl")

IMG_SIZE = 64

# =============================
# Page config
# =============================
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

# =============================
# Title
# =============================
st.markdown("## 🐱🐶 Cat vs Dog Image Classifier")
st.write("**SVM with HOG Features (Grayscale, 64×64)**")

# =============================
# File uploader
# =============================
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

# =============================
# Prediction logic
# =============================
if uploaded_file is not None:
    # Load and show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to grayscale
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Extract HOG features
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    ).reshape(1, -1)

    features = scaler.transform(features)

    # Predict
    prediction = svm.predict(features)[0]
    probability = svm.predict_proba(features)[0]
    confidence = float(max(probability)) * 100

    label = "CAT 🐱" if prediction == 0 else "DOG 🐶"

    # =============================
    # Output Section
    # =============================
    st.markdown("---")
    st.subheader("🔍 Prediction Result")

    st.markdown(
        f"""
        <h2 style='text-align:center; color:#4CAF50;'>
            {label}
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.write(f"**Confidence:** {confidence:.2f}%")
    st.progress(int(confidence))

    if confidence >= 80:
        st.success("✅ High confidence prediction")
    elif confidence >= 60:
        st.warning("⚠️ Medium confidence prediction")
    else:
        st.error("❌ Low confidence – model is unsure")

# =============================
# Model details
# =============================
st.markdown("---")
st.subheader("📊 Model Details")
st.write("- **Algorithm:** Support Vector Machine (RBF Kernel)")
st.write("- **Image Size:** 64 × 64")
st.write("- **Features:** HOG (Histogram of Oriented Gradients)")
st.write("- **Color Mode:** Grayscale")
st.write("- **Feature Scaling:** StandardScaler")
st.write("- **Classes:** Cat (0), Dog (1)")

st.markdown("---")
st.caption("Internship Task 3 | SVM + HOG Image Classification")
