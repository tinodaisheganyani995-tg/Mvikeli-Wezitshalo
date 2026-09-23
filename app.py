# =========================================================
# MVIKELI WEZITSHALO
# TOMATO DISEASE RECOGNITION SYSTEM
#
# FULL IMPROVED APP.PY
#
# MEMORY-SAFE VERSION FOR RENDER / CPU DEPLOYMENT
#
# FEATURES
# ---------------------------------------------------------
# 1. Tomato / non-tomato gatekeeper
# 2. ResNet18 disease classification
# 3. Confidence score
# 4. Top-1 vs Top-2 margin uncertainty
# 5. Image quality checking
#    - Resolution
#    - Blur
#    - Brightness
# 6. Gatekeeper confidence threshold
# 7. Top-3 predictions
# 8. Professional agricultural disease names
# 9. Grad-CAM explainability
# 10. Disease information
# 11. Professional PDF report
# 12. External validation
# 13. Accuracy / Precision / Recall / F1
# 14. Confusion matrix
# 15. About page
# 16. Health endpoint
#
# MEMORY / DEPLOYMENT IMPROVEMENTS
# ---------------------------------------------------------
# 17. Uploaded images resized before processing
# 18. Large original images no longer used for Grad-CAM
# 19. Torch inference_mode for normal predictions
# 20. Better exception logging
#
# NOT INCLUDED YET
# ---------------------------------------------------------
# Prediction history / SQLite
# =========================================================


# =========================================================
# 1. IMPORTS
# =========================================================

from flask import (
    Flask,
    render_template,
    request,
    send_file
)

import torch
import torch.nn as nn

from torchvision import models, transforms

from PIL import Image

import os
import uuid
import cv2
import numpy as np
import traceback

from werkzeug.utils import secure_filename

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.pagesizes import A4

from reportlab.lib import colors

from reportlab.lib.enums import (
    TA_CENTER,
    TA_LEFT
)

from reportlab.lib.units import mm

from xml.sax.saxutils import escape

from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from disease_info import DISEASE_INFO


# =========================================================
# 2. FLASK APPLICATION
# =========================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)


# =========================================================
# 3. UPLOAD CONFIGURATION
# =========================================================

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# MEMORY-SAFETY SETTINGS
# =========================================================

# The AI model already resizes images to 224 x 224.
# There is therefore no benefit in keeping a huge
# 4000 x 3000 or 6000 x 4000 phone image during processing.

MAX_UPLOAD_DIMENSION = 1600

# JPEG quality for the normalized processing image.
NORMALIZED_JPEG_QUALITY = 90


# =========================================================
# 4. DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("DEVICE:", device)
print("=" * 60)


# =========================================================
# 5. DISEASE CLASS NAMES
#
# IMPORTANT:
# These MUST remain exactly the same as the
# classes used when training the model.
# =========================================================

CLASS_NAMES = [

    "Tomato___Bacterial_spot",

    "Tomato___Early_blight",

    "Tomato___Late_blight",

    "Tomato___Leaf_Mold",

    "Tomato___Septoria_leaf_spot",

    "Tomato___Spider_mites Two-spotted_spider_mite",

    "Tomato___Target_Spot",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",

    "Tomato___Tomato_mosaic_virus",

    "Tomato___healthy"

]


# =========================================================
# 6. GATEKEEPER CLASSES
# =========================================================

GATEKEEPER_CLASS_NAMES = [

    "not_tomato",

    "tomato"

]


# =========================================================
# 7. MODEL PATHS
# =========================================================

MODEL_PATH = "best_resnet18_new.pth"

GATEKEEPER_PATH = (
    "best_gatekeeper_resnet18.pth"
)


# =========================================================
# 8. AI THRESHOLDS
# =========================================================

UNCERTAINTY_THRESHOLD = 70.0

MARGIN_THRESHOLD = 10.0

GATEKEEPER_THRESHOLD = 80.0


# =========================================================
# 9. IMAGE QUALITY THRESHOLDS
# =========================================================

MIN_IMAGE_WIDTH = 224

MIN_IMAGE_HEIGHT = 224

BLUR_THRESHOLD = 80.0

MIN_BRIGHTNESS = 35.0

MAX_BRIGHTNESS = 220.0


# =========================================================
# 10. EXTERNAL VALIDATION DATASET
# =========================================================

EXTERNAL_VALIDATION_DIR = (
    "external_validation"
)


# =========================================================
# 11. DISEASE DISPLAY NAME HELPER
# =========================================================

def get_display_name(class_name):

    disease = DISEASE_INFO.get(
        class_name
    )

    if disease:

        return disease.get(
            "name",
            class_name
        )

    return class_name


# =========================================================
# 12. LOAD DISEASE MODEL
# =========================================================

print("Loading disease model...")


model = models.resnet18(
    weights=None
)


model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


if isinstance(checkpoint, dict) and \
        "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(device)

model.eval()


print(
    "Disease ResNet18 loaded successfully."
)


# =========================================================
# 13. LOAD GATEKEEPER MODEL
# =========================================================

print("Loading gatekeeper model...")


gatekeeper = models.resnet18(
    weights=None
)


gatekeeper.fc = nn.Linear(
    gatekeeper.fc.in_features,
    len(GATEKEEPER_CLASS_NAMES)
)


gatekeeper_checkpoint = torch.load(
    GATEKEEPER_PATH,
    map_location=device
)


if isinstance(
    gatekeeper_checkpoint,
    dict
) and "model_state_dict" in gatekeeper_checkpoint:

    gatekeeper.load_state_dict(
        gatekeeper_checkpoint[
            "model_state_dict"
        ]
    )

else:

    gatekeeper.load_state_dict(
        gatekeeper_checkpoint
    )


gatekeeper = gatekeeper.to(device)

gatekeeper.eval()


print(
    "Gatekeeper ResNet18 loaded successfully."
)


# =========================================================
# 14. IMAGE TRANSFORMATION
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])


# =========================================================
# 15. LATEST RESULT
# =========================================================

latest_result = None


# =========================================================
# 16. NORMALIZE UPLOADED IMAGE
#
# IMPORTANT MEMORY FIX
#
# Large phone images such as:
#
# 4032 x 3024
# 4608 x 3456
# 6000 x 4000
#
# are reduced before OpenCV, prediction and Grad-CAM.
# =========================================================

def normalize_uploaded_image(
    input_path,
    output_path
):

    try:

        with Image.open(
            input_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            original_width, original_height = (
                image.size
            )

            print("=" * 60)
            print("IMAGE NORMALIZATION")
            print("=" * 60)

            print(
                "Original resolution:",
                original_width,
                "x",
                original_height
            )

            # -------------------------------------------------
            # Resize only when necessary.
            # Smaller images are NOT enlarged.
            # -------------------------------------------------

            image.thumbnail(

                (
                    MAX_UPLOAD_DIMENSION,
                    MAX_UPLOAD_DIMENSION
                ),

                Image.Resampling.LANCZOS

            )

            new_width, new_height = (
                image.size
            )

            print(
                "Processing resolution:",
                new_width,
                "x",
                new_height
            )

            image.save(

                output_path,

                format="JPEG",

                quality=NORMALIZED_JPEG_QUALITY,

                optimize=True

            )

            print(
                "Normalized image saved successfully."
            )

            return {

                "original_width":
                    original_width,

                "original_height":
                    original_height,

                "processed_width":
                    new_width,

                "processed_height":
                    new_height

            }

    except Exception as e:

        print(
            "IMAGE NORMALIZATION ERROR:",
            repr(e)
        )

        traceback.print_exc()

        raise


# =========================================================
# 17. IMAGE QUALITY CHECK
# =========================================================

def check_image_quality(image_path):

    messages = []

    image = cv2.imread(
        image_path
    )

    if image is None:

        return {

            "valid": False,

            "messages": [
                "The uploaded image could not be read."
            ],

            "blur_score": 0.0,

            "brightness": 0.0,

            "width": 0,

            "height": 0

        }


    height, width = image.shape[:2]


    if width < MIN_IMAGE_WIDTH or \
            height < MIN_IMAGE_HEIGHT:

        messages.append(

            f"Image resolution is too low. "
            f"Please upload an image of at least "
            f"{MIN_IMAGE_WIDTH} × "
            f"{MIN_IMAGE_HEIGHT} pixels."

        )


    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY

    )


    blur_score = float(

        cv2.Laplacian(

            gray,

            cv2.CV_64F

        ).var()

    )


    if blur_score < BLUR_THRESHOLD:

        messages.append(

            "The image appears blurry. "
            "Please capture a sharper image "
            "of the tomato leaf."

        )


    brightness = float(
        np.mean(gray)
    )


    if brightness < MIN_BRIGHTNESS:

        messages.append(

            "The image is too dark. "
            "Please use better lighting."

        )

    elif brightness > MAX_BRIGHTNESS:

        messages.append(

            "The image is too bright or overexposed. "
            "Please reduce direct glare or strong light."

        )


    valid = len(messages) == 0


    return {

        "valid":
            valid,

        "messages":
            messages,

        "blur_score":
            round(
                blur_score,
                2
            ),

        "brightness":
            round(
                brightness,
                2
            ),

        "width":
            width,

        "height":
            height

    }


# =========================================================
# 18. GRAD-CAM CLASS
# =========================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.target_layer = target_layer

        self.activations = None

        self.gradients = None


        self.forward_handle = (

            target_layer.register_forward_hook(

                self.save_activation

            )

        )


        self.backward_handle = (

            target_layer.register_full_backward_hook(

                self.save_gradient

            )

        )


    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output


    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]


    def generate(
        self,
        image_tensor,
        class_index
    ):

        self.model.zero_grad(
            set_to_none=True
        )


        output = self.model(
            image_tensor
        )


        target_score = output[
            0,
            class_index
        ]


        target_score.backward()


        activations = self.activations

        gradients = self.gradients


        if activations is None or \
                gradients is None:

            raise RuntimeError(

                "Grad-CAM activations or gradients "
                "were not captured."

            )


        weights = torch.mean(

            gradients,

            dim=(2, 3),

            keepdim=True

        )


        cam = torch.sum(

            weights * activations,

            dim=1

        )


        cam = torch.relu(
            cam
        )


        cam = cam.squeeze(
            0
        ).detach().cpu().numpy()


        cam_min = cam.min()

        cam_max = cam.max()


        if cam_max - cam_min < 1e-8:

            cam = np.zeros_like(
                cam
            )

        else:

            cam = (

                cam - cam_min

            ) / (

                cam_max - cam_min

            )


        return cam


    def close(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


# =========================================================
# 19. CREATE GRAD-CAM
# =========================================================

target_layer = (
    model.layer4[-1].conv2
)


gradcam = GradCAM(
    model,
    target_layer
)


# =========================================================
# 20. CREATE GRAD-CAM IMAGE
# =========================================================

def create_gradcam(
    image_path,
    class_index,
    output_path
):

    original_image = cv2.imread(
        image_path
    )


    if original_image is None:

        raise ValueError(
            "Could not read image for Grad-CAM."
        )


    # -----------------------------------------------------
    # The uploaded image has already been normalized to
    # MAX_UPLOAD_DIMENSION.
    #
    # This additional protection prevents Grad-CAM from
    # creating a huge heatmap if a large image somehow
    # reaches this function.
    # -----------------------------------------------------

    height, width = original_image.shape[:2]

    if max(
        height,
        width
    ) > MAX_UPLOAD_DIMENSION:

        scale = (

            MAX_UPLOAD_DIMENSION
            /
            max(
                height,
                width
            )

        )

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        original_image = cv2.resize(

            original_image,

            (
                new_width,
                new_height
            ),

            interpolation=cv2.INTER_AREA

        )


    rgb_image = cv2.cvtColor(

        original_image,

        cv2.COLOR_BGR2RGB

    )


    pil_image = Image.fromarray(
        rgb_image
    )


    image_tensor = transform(
        pil_image
    ).unsqueeze(
        0
    ).to(device)


    image_tensor.requires_grad_()


    cam = gradcam.generate(

        image_tensor,

        class_index

    )


    height, width = (
        original_image.shape[:2]
    )


    cam = cv2.resize(

        cam,

        (
            width,
            height
        ),

        interpolation=cv2.INTER_LINEAR

    )


    heatmap = np.uint8(

        255 * cam

    )


    heatmap = cv2.applyColorMap(

        heatmap,

        cv2.COLORMAP_JET

    )


    overlay = cv2.addWeighted(

        original_image,

        0.60,

        heatmap,

        0.40,

        0

    )


    success = cv2.imwrite(

        output_path,

        overlay

    )


    if not success:

        raise RuntimeError(
            "Grad-CAM image could not be saved."
        )


# =========================================================
# 21. GATEKEEPER PREDICTION
# =========================================================

def check_tomato(
    image_path
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    image_tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(device)


    with torch.inference_mode():

        outputs = gatekeeper(
            image_tensor
        )

        probabilities = torch.softmax(

            outputs,

            dim=1

        )


    tomato_index = (
        GATEKEEPER_CLASS_NAMES.index(
            "tomato"
        )
    )


    tomato_probability = (

        probabilities[
            0,
            tomato_index
        ].item() * 100

    )


    predicted_index = torch.argmax(

        probabilities,

        dim=1

    ).item()


    predicted_class = (
        GATEKEEPER_CLASS_NAMES[
            predicted_index
        ]
    )


    if tomato_probability >= GATEKEEPER_THRESHOLD:

        gate_result = "tomato"

    else:

        gate_result = "not_tomato"


    return (

        gate_result,

        round(
            tomato_probability,
            2
        ),

        predicted_class

    )


# =========================================================
# 22. DISEASE PREDICTION
# =========================================================

def predict(
    image_path
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    image_tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(device)


    with torch.inference_mode():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(

            outputs,

            dim=1

        )


    top3_probabilities, top3_indices = torch.topk(

        probabilities,

        3,

        dim=1

    )


    top1_index = (
        top3_indices[
            0,
            0
        ].item()
    )


    top1_confidence = (

        top3_probabilities[
            0,
            0
        ].item() * 100

    )


    top2_confidence = (

        top3_probabilities[
            0,
            1
        ].item() * 100

    )


    margin = (

        top1_confidence
        -
        top2_confidence

    )


    disease = CLASS_NAMES[
        top1_index
    ]


    confidence = round(
        top1_confidence,
        2
    )


    margin = round(
        margin,
        2
    )


    top3 = []


    for probability, index in zip(

        top3_probabilities[0],

        top3_indices[0]

    ):

        index = index.item()


        probability = round(

            probability.item() * 100,

            2

        )


        internal_class_name = (
            CLASS_NAMES[index]
        )


        display_name = get_display_name(
            internal_class_name
        )


        top3.append({

            "name":
                display_name,

            "class_name":
                internal_class_name,

            "confidence":
                probability

        })


    low_confidence = (

        confidence
        <
        UNCERTAINTY_THRESHOLD

    )


    small_margin = (

        margin
        <
        MARGIN_THRESHOLD

    )


    is_uncertain = (

        low_confidence
        or
        small_margin

    )


    if low_confidence and small_margin:

        uncertainty_message = (

            "The model has low confidence and "
            "the top two disease predictions are "
            "very close. Please capture a clearer "
            "image or consult an agricultural expert."

        )


    elif low_confidence:

        uncertainty_message = (

            "The model confidence is below the "
            "recommended threshold. Please capture "
            "a clearer image or consult an "
            "agricultural expert."

        )


    elif small_margin:

        uncertainty_message = (

            "The model's top two disease predictions "
            "are close to each other. The result "
            "should be interpreted with caution."

        )


    else:

        uncertainty_message = (

            "The model produced a relatively "
            "clear prediction for this image."

        )


    return {

        "disease":
            disease,

        "confidence":
            confidence,

        "top2_confidence":
            round(
                top2_confidence,
                2
            ),

        "margin":
            margin,

        "top3":
            top3,

        "predicted_index":
            top1_index,

        "is_uncertain":
            is_uncertain,

        "uncertainty_message":
            uncertainty_message

    }


# =========================================================
# 23. HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# 24. PREDICTION ROUTE
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def prediction():

    global latest_result


    try:

        # =================================================
        # CHECK UPLOAD
        # =================================================

        if "image" not in request.files:

            return (
                "No image uploaded.",
                400
            )


        file = request.files["image"]


        if file.filename == "":

            return (
                "No image selected.",
                400
            )


        original_filename = secure_filename(
            file.filename
        )


        extension = os.path.splitext(
            original_filename
        )[1].lower()


        allowed_extensions = {

            ".jpg",
            ".jpeg",
            ".png",
            ".webp"

        }


        if extension not in allowed_extensions:

            return (

                "Unsupported image format. "
                "Please upload JPG, JPEG, PNG "
                "or WEBP.",

                400

            )


        # =================================================
        # SAVE TEMPORARY ORIGINAL
        # =================================================

        temporary_filename = (

            "temp_"

            + str(uuid.uuid4())

            + extension

        )


        temporary_path = os.path.join(

            app.config[
                "UPLOAD_FOLDER"
            ],

            temporary_filename

        )


        file.save(
            temporary_path
        )


        # =================================================
        # NORMALIZE IMAGE
        #
        # IMPORTANT:
        # All later processing uses the normalized image.
        # =================================================

        unique_filename = (

            str(uuid.uuid4())

            + ".jpg"

        )


        filepath = os.path.join(

            app.config[
                "UPLOAD_FOLDER"
            ],

            unique_filename

        )


        try:

            normalization_info = (
                normalize_uploaded_image(

                    temporary_path,

                    filepath

                )
            )

        finally:

            # Remove temporary original after normalization.
            try:

                if os.path.exists(
                    temporary_path
                ):

                    os.remove(
                        temporary_path
                    )

            except Exception as cleanup_error:

                print(
                    "Temporary file cleanup error:",
                    cleanup_error
                )


        print("=" * 60)
        print("UPLOAD PROCESSING")
        print("=" * 60)

        print(
            "Original resolution:",
            normalization_info[
                "original_width"
            ],
            "x",
            normalization_info[
                "original_height"
            ]
        )

        print(
            "Processing resolution:",
            normalization_info[
                "processed_width"
            ],
            "x",
            normalization_info[
                "processed_height"
            ]
        )


        # =================================================
        # 1. IMAGE QUALITY CHECK
        # =================================================

        quality = check_image_quality(
            filepath
        )


        print("=" * 60)
        print("IMAGE QUALITY")
        print("=" * 60)

        print(
            "Resolution:",
            quality["width"],
            "x",
            quality["height"]
        )

        print(
            "Blur score:",
            quality["blur_score"]
        )

        print(
            "Brightness:",
            quality["brightness"]
        )

        print(
            "Quality valid:",
            quality["valid"]
        )


        if not quality["valid"]:

            quality_message = (
                " ".join(
                    quality["messages"]
                )
            )


            disease_info = {

                "name":
                    "Image Quality Problem",

                "description":
                    quality_message,

                "symptoms": [],

                "treatment": [

                    "Retake the photograph "
                    "under suitable lighting "
                    "with the leaf in focus."

                ],

                "prevention": [

                    "Hold the camera steady.",
                    "Avoid strong glare.",
                    "Capture the leaf clearly."

                ]

            }


            latest_result = {

                "disease":
                    "Image Quality Problem",

                "display_name":
                    "Image Quality Problem",

                "confidence":
                    0.0,

                "top3": [],

                "image":
                    filepath,

                "image_filename":
                    unique_filename,

                "disease_info":
                    disease_info,

                "is_uncertain":
                    True,

                "uncertainty_message":
                    quality_message,

                "gradcam_filename":
                    None,

                "quality":
                    quality

            }


            return render_template(

                "result.html",

                image=filepath,

                image_filename=
                    unique_filename,

                prediction=
                    "Image Quality Problem",

                confidence=0.0,

                disease_info=
                    disease_info,

                description=
                    quality_message,

                symptoms=[],

                treatment=
                    disease_info[
                        "treatment"
                    ],

                prevention=
                    disease_info[
                        "prevention"
                    ],

                top3=[],

                is_uncertain=True,

                uncertainty_message=
                    quality_message,

                gradcam_filename=None

            )


        # =================================================
        # 2. TOMATO GATEKEEPER
        # =================================================

        (

            gate_result,

            gate_confidence,

            gate_predicted_class

        ) = check_tomato(
            filepath
        )


        print("=" * 60)
        print("GATEKEEPER")
        print("=" * 60)

        print(
            "Predicted class:",
            gate_predicted_class
        )

        print(
            "Tomato probability:",
            gate_confidence,
            "%"
        )

        print(
            "Gatekeeper threshold:",
            GATEKEEPER_THRESHOLD,
            "%"
        )

        print(
            "Decision:",
            gate_result
        )


        if gate_result == "not_tomato":

            disease_info = {

                "name":
                    "Tomato Leaf Not Detected",

                "description":

                    "The image did not meet the "
                    "tomato-image confidence "
                    "requirement. Please upload a "
                    "clear image of a tomato leaf.",

                "symptoms": [],

                "treatment": [

                    "Upload a clear tomato leaf image."

                ],

                "prevention": [

                    "Make sure the leaf occupies "
                    "a reasonable portion of the image."

                ]

            }


            latest_result = {

                "disease":
                    "Tomato Leaf Not Detected",

                "display_name":
                    "Tomato Leaf Not Detected",

                "confidence":
                    gate_confidence,

                "top3": [],

                "image":
                    filepath,

                "image_filename":
                    unique_filename,

                "disease_info":
                    disease_info,

                "is_uncertain":
                    True,

                "uncertainty_message":

                    "The image did not meet the "
                    "tomato-image confidence "
                    "requirement.",

                "gradcam_filename":
                    None,

                "quality":
                    quality

            }


            return render_template(

                "result.html",

                image=filepath,

                image_filename=
                    unique_filename,

                prediction=
                    "Tomato Leaf Not Detected",

                confidence=
                    gate_confidence,

                disease_info=
                    disease_info,

                description=
                    disease_info[
                        "description"
                    ],

                symptoms=[],

                treatment=
                    disease_info[
                        "treatment"
                    ],

                prevention=
                    disease_info[
                        "prevention"
                    ],

                top3=[],

                is_uncertain=True,

                uncertainty_message=

                    "The image did not meet the "
                    "tomato-image confidence "
                    "requirement.",

                gradcam_filename=None

            )


        # =================================================
        # 3. DISEASE MODEL
        # =================================================

        result = predict(
            filepath
        )


        disease = result[
            "disease"
        ]


        confidence = result[
            "confidence"
        ]


        top2_confidence = result[
            "top2_confidence"
        ]


        margin = result[
            "margin"
        ]


        top3 = result[
            "top3"
        ]


        predicted_index = result[
            "predicted_index"
        ]


        is_uncertain = result[
            "is_uncertain"
        ]


        uncertainty_message = result[
            "uncertainty_message"
        ]


        display_name = get_display_name(
            disease
        )


        print("=" * 60)
        print("DISEASE PREDICTION")
        print("=" * 60)

        print(
            "Internal class:",
            disease
        )

        print(
            "Display name:",
            display_name
        )

        print(
            "Top-1 confidence:",
            confidence,
            "%"
        )

        print(
            "Top-2 confidence:",
            top2_confidence,
            "%"
        )

        print(
            "Top-1 / Top-2 margin:",
            margin,
            "%"
        )

        print(
            "Uncertain:",
            is_uncertain
        )


        # =================================================
        # 4. DISEASE INFORMATION
        # =================================================

        disease_info = DISEASE_INFO.get(

            disease,

            {

                "name":
                    display_name,

                "description":
                    "No information available.",

                "symptoms": [],

                "treatment": [],

                "prevention": []

            }

        )


        disease_info = dict(
            disease_info
        )


        disease_info["name"] = (
            display_name
        )


        # =================================================
        # 5. GRAD-CAM
        # =================================================

        gradcam_filename = (

            "gradcam_"

            + str(uuid.uuid4())

            + ".jpg"

        )


        gradcam_path = os.path.join(

            app.config[
                "UPLOAD_FOLDER"
            ],

            gradcam_filename

        )


        try:

            create_gradcam(

                filepath,

                predicted_index,

                gradcam_path

            )


            print(
                "Grad-CAM generated successfully."
            )


        except Exception as e:

            print(
                "Grad-CAM error:",
                repr(e)
            )

            traceback.print_exc()


            gradcam_filename = None


        # =================================================
        # 6. SAVE LATEST RESULT
        # =================================================

        latest_result = {

            "disease":
                disease,

            "display_name":
                display_name,

            "confidence":
                confidence,

            "top2_confidence":
                top2_confidence,

            "margin":
                margin,

            "top3":
                top3,

            "image":
                filepath,

            "image_filename":
                unique_filename,

            "disease_info":
                disease_info,

            "predicted_index":
                predicted_index,

            "is_uncertain":
                is_uncertain,

            "uncertainty_message":
                uncertainty_message,

            "gradcam_filename":
                gradcam_filename,

            "quality":
                quality,

            "gatekeeper_confidence":
                gate_confidence

        }


        # =================================================
        # 7. RESULT PAGE
        # =================================================

        return render_template(

            "result.html",

            image=filepath,

            image_filename=
                unique_filename,

            prediction=
                display_name,

            confidence=
                confidence,

            disease_info=
                disease_info,

            description=
                disease_info.get(
                    "description",
                    ""
                ),

            symptoms=
                disease_info.get(
                    "symptoms",
                    []
                ),

            treatment=
                disease_info.get(
                    "treatment",
                    []
                ),

            prevention=
                disease_info.get(
                    "prevention",
                    []
                ),

            top3=
                top3,

            is_uncertain=
                is_uncertain,

            uncertainty_message=
                uncertainty_message,

            gradcam_filename=
                gradcam_filename

        )


    except Exception as e:

        print("=" * 60)
        print("PREDICTION ROUTE ERROR")
        print("=" * 60)

        print(
            "Error:",
            repr(e)
        )

        traceback.print_exc()

        return (

            "Prediction failed because of an internal "
            "server error. Please try another image.",

            500

        )


# =========================================================
# 25. ABOUT
# =========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# =========================================================
# 26. DISEASE INFORMATION PAGE
# =========================================================

@app.route(
    "/disease/<path:disease_name>"
)
def disease_page(
    disease_name
):

    disease_info = DISEASE_INFO.get(
        disease_name
    )


    if disease_info is None:

        return (

            "Disease information not found.",

            404

        )


    return render_template(

        "disease.html",

        disease=disease_name,

        disease_info=disease_info

    )


# =========================================================
# 27. PROFESSIONAL PDF REPORT
# =========================================================

@app.route("/download")
@app.route("/download_report")
def download_report():

    global latest_result


    if latest_result is None:

        return (

            "No prediction available.",

            404

        )


    disease = latest_result[
        "disease"
    ]


    display_name = latest_result.get(

        "display_name",

        get_display_name(
            disease
        )

    )


    confidence = latest_result[
        "confidence"
    ]


    image_path = latest_result[
        "image"
    ]


    disease_info = latest_result[
        "disease_info"
    ]


    top3 = latest_result[
        "top3"
    ]


    top2_confidence = latest_result.get(

        "top2_confidence",

        0.0

    )


    margin = latest_result.get(

        "margin",

        0.0

    )


    is_uncertain = latest_result.get(

        "is_uncertain",

        False

    )


    gradcam_filename = latest_result.get(

        "gradcam_filename",

        None

    )


    report_filename = (
        "tomato_disease_report.pdf"
    )


    report_path = os.path.join(

        "static",

        report_filename

    )


    CREAM = colors.HexColor(
        "#FFF9DC"
    )

    DARK_GREEN = colors.HexColor(
        "#173B2B"
    )

    GREEN = colors.HexColor(
        "#0B5D3B"
    )

    LIGHT_GREEN = colors.HexColor(
        "#E8F3E8"
    )

    LIGHT_YELLOW = colors.HexColor(
        "#F3EDB3"
    )

    WHITE = colors.white

    BORDER = colors.HexColor(
        "#D8D09A"
    )

    GREY = colors.HexColor(
        "#5F6B63"
    )


    def draw_pdf_page(
        canvas,
        document
    ):

        canvas.saveState()


        canvas.setFillColor(
            CREAM
        )

        canvas.rect(

            0,
            0,
            A4[0],
            A4[1],

            fill=1,
            stroke=0

        )


        canvas.setFillColor(
            DARK_GREEN
        )

        canvas.rect(

            0,
            A4[1] - 12 * mm,

            A4[0],
            12 * mm,

            fill=1,
            stroke=0

        )


        canvas.setFillColor(
            WHITE
        )

        canvas.setFont(
            "Helvetica-Bold",
            9
        )

        canvas.drawString(

            20 * mm,

            A4[1] - 7.5 * mm,

            "MVIKELI WEZITSHALO"

        )


        canvas.setFont(
            "Helvetica",
            8
        )

        canvas.drawRightString(

            A4[0] - 20 * mm,

            A4[1] - 7.5 * mm,

            "Plant Disease Recognition System"

        )


        canvas.setStrokeColor(
            BORDER
        )

        canvas.setLineWidth(
            0.6
        )

        canvas.line(

            20 * mm,
            14 * mm,

            A4[0] - 20 * mm,
            14 * mm

        )


        canvas.setFillColor(
            GREY
        )

        canvas.setFont(
            "Helvetica",
            7.5
        )

        canvas.drawString(

            20 * mm,
            9 * mm,

            "Mvikeli Wezitshalo • AI-assisted tomato disease recognition"

        )


        canvas.drawRightString(

            A4[0] - 20 * mm,
            9 * mm,

            f"Page {document.page}"

        )


        canvas.restoreState()


    doc = SimpleDocTemplate(

        report_path,

        pagesize=A4,

        rightMargin=20 * mm,

        leftMargin=20 * mm,

        topMargin=25 * mm,

        bottomMargin=20 * mm

    )


    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(

        "ProfessionalTitle",

        parent=styles["Title"],

        alignment=TA_CENTER,

        fontName="Helvetica-Bold",

        fontSize=23,

        leading=28,

        textColor=DARK_GREEN,

        spaceAfter=5

    )


    subtitle_style = ParagraphStyle(

        "ProfessionalSubtitle",

        parent=styles["BodyText"],

        alignment=TA_CENTER,

        fontName="Helvetica",

        fontSize=10,

        leading=14,

        textColor=GREY,

        spaceAfter=12

    )


    section_style = ParagraphStyle(

        "ProfessionalSection",

        parent=styles["Heading2"],

        fontName="Helvetica-Bold",

        fontSize=14,

        leading=18,

        textColor=DARK_GREEN,

        spaceBefore=12,

        spaceAfter=7

    )


    normal_style = ParagraphStyle(

        "ProfessionalNormal",

        parent=styles["BodyText"],

        fontName="Helvetica",

        fontSize=9.5,

        leading=14,

        textColor=DARK_GREEN,

        spaceAfter=5

    )


    small_style = ParagraphStyle(

        "ProfessionalSmall",

        parent=styles["BodyText"],

        fontName="Helvetica",

        fontSize=8,

        leading=11,

        textColor=GREY

    )


    center_style = ParagraphStyle(

        "ProfessionalCenter",

        parent=styles["BodyText"],

        alignment=TA_CENTER,

        fontName="Helvetica",

        fontSize=9,

        leading=13,

        textColor=GREY

    )


    white_bold_style = ParagraphStyle(

        "WhiteBold",

        parent=styles["BodyText"],

        fontName="Helvetica-Bold",

        fontSize=11,

        leading=15,

        textColor=WHITE

    )


    confidence_style = ParagraphStyle(

        "ConfidenceStyle",

        parent=styles["BodyText"],

        alignment=TA_CENTER,

        fontName="Helvetica-Bold",

        fontSize=19,

        leading=23,

        textColor=GREEN

    )


    story = []


    story.append(
        Spacer(
            1,
            5 * mm
        )
    )


    story.append(

        Paragraph(

            "Mvikeli Wezitshalo",

            title_style

        )

    )


    story.append(

        Paragraph(

            "Tomato Disease Recognition Report",

            subtitle_style

        )

    )


    current_date = datetime.now().strftime(

        "%d %B %Y, %H:%M"

    )


    meta_data = [

        [

            Paragraph(
                "<b>Report Date</b>",
                normal_style
            ),

            Paragraph(
                escape(current_date),
                normal_style
            )

        ],

        [

            Paragraph(
                "<b>System</b>",
                normal_style
            ),

            Paragraph(
                "ResNet18 + Tomato Gatekeeper + Grad-CAM",
                normal_style
            )

        ]

    ]


    meta_table = Table(

        meta_data,

        colWidths=[

            38 * mm,
            125 * mm

        ]

    )


    meta_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                WHITE
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                BORDER
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.3,
                BORDER
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5
            )

        ])

    )


    story.append(
        meta_table
    )


    story.append(
        Spacer(
            1,
            7 * mm
        )
    )


    # =====================================================
    # ORIGINAL IMAGE
    # =====================================================

    story.append(

        Paragraph(

            "Original Image",

            section_style

        )

    )


    if os.path.exists(
        image_path
    ):

        try:

            report_image = PDFImage(

                image_path,

                width=95 * mm,

                height=80 * mm

            )


            report_image.hAlign = "CENTER"


            image_table = Table(

                [[report_image]],

                colWidths=[
                    105 * mm
                ]

            )


            image_table.setStyle(

                TableStyle([

                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, -1),
                        WHITE
                    ),

                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        1,
                        BORDER
                    ),

                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        5
                    ),

                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        5
                    ),

                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        5
                    ),

                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        5
                    )

                ])

            )


            image_table.hAlign = "CENTER"


            story.append(
                image_table
            )


            story.append(
                Spacer(
                    1,
                    6 * mm
                )
            )


        except Exception as e:

            print(
                "PDF image error:",
                e
            )


    # =====================================================
    # PREDICTION SUMMARY
    # =====================================================

    story.append(

        Paragraph(

            "AI Prediction",

            section_style

        )

    )


    prediction_white_style = ParagraphStyle(

        "PredictionWhite",

        parent=normal_style,

        fontName="Helvetica-Bold",

        fontSize=11,

        textColor=WHITE,

        alignment=TA_CENTER

    )


    prediction_data = [

        [

            Paragraph(

                "Detected Condition",

                white_bold_style

            ),

            Paragraph(

                escape(display_name),

                prediction_white_style

            )

        ],

        [

            Paragraph(

                "AI Confidence",

                normal_style

            ),

            Paragraph(

                f"{confidence:.2f}%",

                confidence_style

            )

        ],

        [

            Paragraph(

                "Second-best Confidence",

                normal_style

            ),

            Paragraph(

                f"{top2_confidence:.2f}%",

                normal_style

            )

        ],

        [

            Paragraph(

                "Top-1 / Top-2 Margin",

                normal_style

            ),

            Paragraph(

                f"{margin:.2f}%",

                normal_style

            )

        ]

    ]


    prediction_table = Table(

        prediction_data,

        colWidths=[

            70 * mm,
            93 * mm

        ]

    )


    prediction_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                GREEN
            ),

            (
                "BACKGROUND",
                (0, 1),
                (-1, -1),
                WHITE
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                1,
                BORDER
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.4,
                BORDER
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "ALIGN",
                (1, 0),
                (1, -1),
                "CENTER"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )

        ])

    )


    story.append(
        prediction_table
    )


    story.append(
        Spacer(
            1,
            5 * mm
        )
    )


    # =====================================================
    # PREDICTION STATUS
    # =====================================================

    if is_uncertain:

        status_text = (

            "UNCERTAIN PREDICTION — "
            "The result should be interpreted with caution. "
            "A clearer image or agricultural expert "
            "confirmation is recommended."

        )

        status_background = colors.HexColor(
            "#FFF0C2"
        )

        status_color = colors.HexColor(
            "#7A5A00"
        )

    else:

        status_text = (

            "RELATIVELY CLEAR PREDICTION — "
            "The model produced a relatively clear "
            "prediction for this image."

        )

        status_background = LIGHT_GREEN

        status_color = GREEN


    status_style = ParagraphStyle(

        "StatusStyle",

        parent=normal_style,

        fontName="Helvetica-Bold",

        fontSize=9,

        leading=13,

        textColor=status_color,

        alignment=TA_CENTER

    )


    status_table = Table(

        [[

            Paragraph(

                status_text,

                status_style

            )

        ]],

        colWidths=[

            163 * mm

        ]

    )


    status_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                status_background
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.8,
                BORDER
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])

    )


    story.append(
        status_table
    )


    story.append(
        Spacer(
            1,
            5 * mm
        )
    )


    # =====================================================
    # GRAD-CAM
    # =====================================================

    if gradcam_filename:

        gradcam_path = os.path.join(

            app.config[
                "UPLOAD_FOLDER"
            ],

            gradcam_filename

        )


        if os.path.exists(
            gradcam_path
        ):

            story.append(

                Paragraph(

                    "AI Visual Explanation",

                    section_style

                )

            )


            story.append(

                Paragraph(

                    "The Grad-CAM visualization highlights "
                    "image regions that contributed to the "
                    "model's prediction. It is an explanatory "
                    "visualization and should not be interpreted "
                    "as a clinical or agricultural diagnosis by itself.",

                    small_style

                )

            )


            try:

                gradcam_image = PDFImage(

                    gradcam_path,

                    width=110 * mm,

                    height=82 * mm

                )


                gradcam_image.hAlign = "CENTER"


                gradcam_table = Table(

                    [[gradcam_image]],

                    colWidths=[
                        120 * mm
                    ]

                )


                gradcam_table.setStyle(

                    TableStyle([

                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, -1),
                            WHITE
                        ),

                        (
                            "BOX",
                            (0, 0),
                            (-1, -1),
                            1,
                            BORDER
                        ),

                        (
                            "LEFTPADDING",
                            (0, 0),
                            (-1, -1),
                            5
                        ),

                        (
                            "RIGHTPADDING",
                            (0, 0),
                            (-1, -1),
                            5
                        ),

                        (
                            "TOPPADDING",
                            (0, 0),
                            (-1, -1),
                            5
                        ),

                        (
                            "BOTTOMPADDING",
                            (0, 0),
                            (-1, -1),
                            5
                        )

                    ])

                )


                gradcam_table.hAlign = "CENTER"


                story.append(
                    gradcam_table
                )


                story.append(
                    Spacer(
                        1,
                        5 * mm
                    )
                )


            except Exception as e:

                print(
                    "PDF Grad-CAM error:",
                    e
                )


    # =====================================================
    # DESCRIPTION
    # =====================================================

    description = disease_info.get(

        "description",

        ""

    )


    story.append(

        Paragraph(

            "About the Condition",

            section_style

        )

    )


    description_table = Table(

        [[

            Paragraph(

                escape(
                    str(description)
                ),

                normal_style

            )

        ]],

        colWidths=[

            163 * mm

        ]

    )


    description_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                WHITE
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                BORDER
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])

    )


    story.append(
        description_table
    )


    # =====================================================
    # HELPER FOR PDF LIST SECTIONS
    # =====================================================

    def create_pdf_list_table(items):

        content = []


        if items:

            for item in items:

                content.append(

                    Paragraph(

                        "• "
                        + escape(
                            str(item)
                        ),

                        normal_style

                    )

                )

        else:

            content.append(

                Paragraph(

                    "
