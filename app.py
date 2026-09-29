import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MedOpsAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = r"C:\Users\ASUS\My_Learning\medical mlops project"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "densenet121_brain_tumor_final.keras"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "Testing"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor"
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background-color: #ffffff;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    /* Make normal Streamlit text dark */
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown span,
    label,
    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #333333 !important;
    }

    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background-color: #f3f6fa;
        border-right: 1px solid #dce3eb;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    .sidebar-brand {
        text-align: center;
        padding-top: 5px;
        padding-bottom: 15px;
    }

    .sidebar-brand-icon {
        font-size: 38px;
        line-height: 1;
    }

    .sidebar-brand-title {
        color: #1559a6 !important;
        font-size: 22px;
        font-weight: 700;
        margin-top: 7px;
    }

    .sidebar-brand-subtitle {
        color: #666666 !important;
        font-size: 9px;
        margin-top: 2px;
    }

    /* Sidebar radio text */
    section[data-testid="stSidebar"]
    [data-testid="stRadio"] label {
        color: #333333 !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stRadio"] label p {
        color: #333333 !important;
    }


    /* ======================================================
       MAIN HEADER
       ====================================================== */

    .main-title {
        color: #1559a6 !important;
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .main-subtitle {
        color: #444444 !important;
        font-size: 16px;
        margin-bottom: 5px;
    }

    .main-description {
        color: #666666 !important;
        font-size: 13px;
        line-height: 1.5;
    }


    /* ======================================================
       SECTION TITLE
       ====================================================== */

    .section-title {
        color: #222222 !important;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }


    /* ======================================================
       DIVIDER
       ====================================================== */

    .custom-divider {
        height: 1px;
        background-color: #dfe5eb;
        margin-top: 20px;
        margin-bottom: 25px;
    }


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {
        background-color: #f6f9fc;
        border-radius: 7px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #f6f9fc !important;
        border: 1px dashed #b9c8d8 !important;
        border-radius: 7px !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #333333 !important;
    }


    /* ======================================================
       PREDICTION EMPTY BOX
       ====================================================== */

    .prediction-empty {
        background-color: #eaf4ff;
        border-radius: 8px;
        padding: 18px;
        color: #2678c5 !important;
        font-size: 14px;
        border: 1px solid #d6e8fa;
    }


    /* ======================================================
       PREDICTION RESULT
       ====================================================== */

    .prediction-title {
        color: #222222 !important;
        font-size: 15px;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 8px;
    }


    /* ======================================================
       CLASS PROBABILITIES
       ====================================================== */

    .probability-heading {
        color: #222222 !important;
        font-size: 15px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 12px;
    }

    .probability-label {
        color: #333333 !important;
        font-size: 13px;
        font-weight: 600;
        margin-top: 8px;
        margin-bottom: 2px;
    }

    .probability-value {
        color: #666666 !important;
        font-size: 11px;
        margin-top: -4px;
        margin-bottom: 4px;
    }


    /* ======================================================
       EXAMPLE IMAGE LABEL
       ====================================================== */

    .example-label {
        color: #333333 !important;
        text-align: center;
        font-size: 12px;
        font-weight: 600;
        margin-top: 4px;
    }


    /* ======================================================
       HOW IT WORKS
       ====================================================== */

    .how-title {
        color: #222222 !important;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .step-number {
        width: 34px;
        height: 34px;
        background-color: #e5f0ff;
        color: #1559a6 !important;
        border-radius: 50%;
        margin: auto;
        text-align: center;
        line-height: 34px;
        font-weight: 700;
        font-size: 14px;
    }

    .step-title {
        color: #222222 !important;
        text-align: center;
        font-size: 13px;
        font-weight: 700;
        margin-top: 8px;
    }

    .step-description {
        color: #777777 !important;
        text-align: center;
        font-size: 11px;
        margin-top: 3px;
    }


    /* ======================================================
       METRIC BOX
       ====================================================== */

    .metric-box {
        background-color: #f7fafc;
        border: 1px solid #dce5ee;
        border-radius: 7px;
        padding: 15px;
        text-align: center;
    }

    .metric-value {
        color: #1559a6 !important;
        font-size: 25px;
        font-weight: 700;
    }

    .metric-label {
        color: #666666 !important;
        font-size: 12px;
    }


    /* ======================================================
       IMAGE
       ====================================================== */

    [data-testid="stImage"] {
        border-radius: 7px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(
        MODEL_PATH
    )


model = load_model()


# ============================================================
# MODEL CHECK
# ============================================================

if model is None:

    st.error(
        "DenseNet121 model was not found."
    )

    st.write(
        "Expected model path:"
    )

    st.code(
        MODEL_PATH
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # Native Streamlit/Markdown branding
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">🧠</div>
            <div class="sidebar-brand-title">
                MedOpsAI
            </div>
            <div class="sidebar-brand-subtitle">
                Brain Tumor MRI Classification
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔍 Predict",
            "🧠 Model Information",
            "📊 Dataset",
            "📈 Model Evaluation",
            "⚙️ MLOps Pipeline",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        '<div class="main-title">MedOpsAI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Brain Tumor MRI Classification and MLOps Platform'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-description">'
        'Upload a brain MRI image and get an instant prediction '
        'using a deep learning model (DenseNet121). '
        'The system classifies MRI images into Glioma, '
        'Meningioma, No Tumor or Pituitary Tumor.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="custom-divider"></div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # UPLOAD + PREDICTION
    # ========================================================

    left_column, right_column = st.columns(
        [1, 1],
        gap="large"
    )


    # ========================================================
    # LEFT: UPLOAD MRI
    # ========================================================

    with left_column:

        st.markdown(
            '<div class="section-title">'
            'Upload MRI Image'
            '</div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose an MRI image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            label_visibility="collapsed"
        )


        # ----------------------------------------------------
        # SHOW UPLOADED IMAGE
        # ----------------------------------------------------

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.markdown(
                '<div class="probability-label">'
                'Uploaded MRI Image'
                '</div>',
                unsafe_allow_html=True
            )

            st.image(
                image,
                width=320
            )

        else:

            st.info(
                "Upload an MRI image to preview it here."
            )


        st.caption(
            "Supported formats: JPG, JPEG, PNG | "
            "Recommended size: 224 × 224"
        )


    # ========================================================
    # RIGHT: PREDICTION
    # ========================================================

    with right_column:

        st.markdown(
            '<div class="section-title">'
            'Prediction Result'
            '</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # NO IMAGE
        # ----------------------------------------------------

        if uploaded_file is None:

            st.markdown(
                """
                <div class="prediction-empty">
                    Upload an MRI image to see the prediction result.
                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # IMAGE UPLOADED
        # ----------------------------------------------------

        else:

            # =================================================
            # PREPROCESS IMAGE
            # =================================================

            resized_image = image.resize(
                IMG_SIZE
            )

            img_array = np.array(
                resized_image
            ).astype("float32")

            # SAME PREPROCESSING USED DURING TRAINING
            img_array = img_array / 255.0

            # Add batch dimension
            img_array = np.expand_dims(
                img_array,
                axis=0
            )


            # =================================================
            # PREDICTION
            # =================================================

            predictions = model.predict(
                img_array,
                verbose=0
            )[0]


            predicted_index = np.argmax(
                predictions
            )


            predicted_class = CLASS_NAMES[
                predicted_index
            ]


            confidence = (
                predictions[predicted_index]
                * 100
            )


            # =================================================
            # RESULT
            # =================================================

            st.success(
                f"Prediction: "
                f"{DISPLAY_NAMES[predicted_class]}"
            )


            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


            # =================================================
            # CLASS PROBABILITIES
            # =================================================

            st.markdown(
                '<div class="probability-heading">'
                'Class Probabilities'
                '</div>',
                unsafe_allow_html=True
            )


            for class_name, probability in zip(
                CLASS_NAMES,
                predictions
            ):

                percentage = float(
                    probability * 100
                )


                # Label
                st.markdown(
                    f"""
                    <div class="probability-label">
                        {DISPLAY_NAMES[class_name]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # Streamlit progress bar
                st.progress(
                    min(
                        max(
                            float(probability),
                            0.0
                        ),
                        1.0
                    )
                )


                # Percentage
                st.markdown(
                    f"""
                    <div class="probability-value">
                        {percentage:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # ========================================================
    # EXAMPLE MRI IMAGES
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Example MRI Images'
        '</div>',
        unsafe_allow_html=True
    )


    example_columns = st.columns(4)


    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        class_folder = os.path.join(
            TEST_DIR,
            class_name
        )

        selected_image = None


        # Find first image
        if os.path.exists(
            class_folder
        ):

            for filename in os.listdir(
                class_folder
            ):

                if filename.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png"
                    )
                ):

                    selected_image = os.path.join(
                        class_folder,
                        filename
                    )

                    break


        with example_columns[i]:

            if selected_image:

                st.image(
                    selected_image,
                    use_container_width=True
                )

                st.markdown(
                    f"""
                    <div class="example-label">
                        {DISPLAY_NAMES[class_name]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.warning(
                    f"No image found for "
                    f"{DISPLAY_NAMES[class_name]}"
                )


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="how-title">'
        'How It Works'
        '</div>',
        unsafe_allow_html=True
    )


    step1, step2, step3, step4 = st.columns(4)


    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    with step1:

        st.markdown(
            '<div class="step-number">1</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="step-title">'
            'Upload MRI Image'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="step-description">'
            'Select an MRI image'
            '</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    with step2:

        st.markdown(
            '<div class="step-number">2</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="step-title">'
            'Preprocess'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="step-description">'
            'Resize to 224 × 224'
            '</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    with step3:

        st.markdown(
            '<div class="step-number">3</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="step-title">'
            'DenseNet121'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="step-description">'
            'Classify the MRI'
            '</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    with step4:

        st.markdown(
            '<div class="step-number">4</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="step-title">'
            'Get Prediction'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="step-description">'
            'Class & probabilities'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# PREDICT PAGE
# ============================================================

elif page == "🔍 Predict":

    st.title("MRI Prediction")

    st.write(
        "Upload a brain MRI image to classify it "
        "using the trained DenseNet121 model."
    )


    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        col1, col2 = st.columns(2)


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        with col1:

            st.image(
                image,
                caption="Uploaded MRI",
                use_container_width=True
            )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with col2:

            resized = image.resize(
                IMG_SIZE
            )

            array = np.array(
                resized
            ).astype("float32")

            array = array / 255.0

            array = np.expand_dims(
                array,
                axis=0
            )


            predictions = model.predict(
                array,
                verbose=0
            )[0]


            index = np.argmax(
                predictions
            )


            predicted_class = CLASS_NAMES[
                index
            ]


            confidence = (
                predictions[index]
                * 100
            )


            st.subheader(
                "Prediction"
            )


            st.success(
                DISPLAY_NAMES[
                    predicted_class
                ]
            )


            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


            st.subheader(
                "Class Probabilities"
            )


            for class_name, probability in zip(
                CLASS_NAMES,
                predictions
            ):

                percentage = probability * 100

                st.write(
                    f"**{DISPLAY_NAMES[class_name]}**"
                )

                st.progress(
                    float(probability)
                )

                st.caption(
                    f"{percentage:.2f}%"
                )


# ============================================================
# MODEL INFORMATION PAGE
# ============================================================

elif page == "🧠 Model Information":

    st.title(
        "Model Information"
    )


    st.write(
        "MedOpsAI uses DenseNet121 with transfer learning "
        "for brain tumor MRI classification."
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Model",
            "DenseNet121"
        )


    with col2:

        st.metric(
            "Input Size",
            "224 × 224"
        )


    with col3:

        st.metric(
            "Classes",
            "4"
        )


    st.divider()


    st.subheader(
        "Classification Classes"
    )


    for class_name in CLASS_NAMES:

        st.write(
            f"• {DISPLAY_NAMES[class_name]}"
        )


    st.subheader(
        "Model Architecture"
    )


    st.code(
        """
Input MRI
    ↓
224 × 224 × 3
    ↓
DenseNet121
    ↓
Global Average Pooling
    ↓
Dense Layer (256)
    ↓
Dropout
    ↓
Softmax
    ↓
4 Classes
        """,
        language="text"
    )


# ============================================================
# DATASET PAGE
# ============================================================

elif page == "📊 Dataset":

    st.title(
        "Dataset"
    )


    st.write(
        "Brain MRI dataset used for training, validation "
        "and testing."
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Training",
            "4,760"
        )


    with col2:

        st.metric(
            "Validation",
            "840"
        )


    with col3:

        st.metric(
            "Testing",
            "1,600"
        )


    st.divider()


    st.subheader(
        "Classes"
    )


    for class_name in CLASS_NAMES:

        st.write(
            f"**{DISPLAY_NAMES[class_name]}** "
            f"— 400 test images"
        )


# ============================================================
# MODEL EVALUATION PAGE
# ============================================================

elif page == "📈 Model Evaluation":

    st.title(
        "Model Evaluation"
    )


    st.write(
        "Performance of the trained DenseNet121 model "
        "on the independent testing dataset."
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            """
            <div class="metric-box">

                <div class="metric-value">
                    90.69%
                </div>

                <div class="metric-label">
                    Test Accuracy
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="metric-box">

                <div class="metric-value">
                    1,600
                </div>

                <div class="metric-label">
                    Test Images
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.divider()


    st.subheader(
        "Evaluation Metrics"
    )


    st.write(
        "• Accuracy"
    )

    st.write(
        "• Precision"
    )

    st.write(
        "• Recall"
    )

    st.write(
        "• F1-Score"
    )

    st.write(
        "• Confusion Matrix"
    )


# ============================================================
# MLOPS PIPELINE PAGE
# ============================================================

elif page == "⚙️ MLOps Pipeline":

    st.title(
        "MLOps Pipeline"
    )


    st.write(
        "MedOpsAI integrates MLOps practices to support "
        "reproducible machine learning development."
    )


    st.code(
        """
Dataset
   ↓
DVC
   ↓
Preprocessing
   ↓
DenseNet121 Training
   ↓
MLflow Experiment Tracking
   ↓
Model Evaluation
   ↓
Model Versioning
   ↓
Streamlit Deployment
   ↓
Monitoring
   ↓
Retraining
        """,
        language="text"
    )


    st.subheader(
        "MLOps Components"
    )


    st.write(
        "**DVC** — Dataset versioning"
    )

    st.write(
        "**MLflow** — Experiment tracking and model management"
    )

    st.write(
        "**Git/GitHub** — Source code version control"
    )

    st.write(
        "**Streamlit** — Web application and model inference"
    )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About":

    st.title(
        "About MedOpsAI"
    )


    st.write(
        """
        MedOpsAI is a deep learning and MLOps platform
        designed for brain tumor classification from MRI images.

        The system uses DenseNet121 with transfer learning
        to classify MRI images into four categories:

        • Glioma
        • Meningioma
        • No Tumor
        • Pituitary Tumor

        The trained model achieved 90.69% accuracy on the
        independent testing dataset containing 1,600 images.
        """
    )


    st.divider()


    st.caption(
        "MedOpsAI | Brain Tumor MRI Classification and MLOps Platform"
    )