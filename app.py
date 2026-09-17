from flask import Flask, render_template, request, send_file

import torch
import torch.nn as nn
from torchvision import models, transforms

import numpy as np
from PIL import Image

import os
import uuid

from werkzeug.utils import secure_filename

# ---------------------------------------------------------
# REPORTLAB - PDF GENERATION
# ---------------------------------------------------------

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
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm

from xml.sax.saxutils import escape
from datetime import datetime

from disease_info import DISEASE_INFO


# =========================================================
# FLASK APP
# =========================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)


# =========================================================
# UPLOAD FOLDER
# =========================================================

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# DISEASE CLASS NAMES
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
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# =========================================================
# LOAD DISEASE RESNET18
# =========================================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)


# =========================================================
# LOAD DISEASE MODEL
# =========================================================

MODEL_PATH = "best_resnet18_new.pth"

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print(
    "ResNet18 disease model loaded successfully."
)


# =========================================================
# GATEKEEPER
# =========================================================

GATEKEEPER_CLASS_NAMES = [
    "not_tomato",
    "tomato"
]

GATEKEEPER_PATH = (
    "best_gatekeeper_resnet18.pth"
)


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

gatekeeper.load_state_dict(
    gatekeeper_checkpoint[
        "model_state_dict"
    ]
)

gatekeeper = gatekeeper.to(device)

gatekeeper.eval()

print(
    "Gatekeeper model loaded successfully."
)


# =========================================================
# IMAGE TRANSFORMATION
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
# STORE LATEST RESULT
# =========================================================

latest_result = None


# =========================================================
# GATEKEEPER PREDICTION
# =========================================================

def check_tomato(image_path):

    # -----------------------------------------------------
    # OPEN IMAGE
    # -----------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    # -----------------------------------------------------
    # TRANSFORM IMAGE
    # -----------------------------------------------------

    image_tensor = transform(
        image
    ).unsqueeze(0)


    # -----------------------------------------------------
    # MOVE TO GPU / CPU
    # -----------------------------------------------------

    image_tensor = image_tensor.to(
        device
    )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    with torch.no_grad():

        outputs = gatekeeper(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


    # -----------------------------------------------------
    # GET HIGHEST PROBABILITY
    # -----------------------------------------------------

    confidence, predicted_index = torch.max(
        probabilities,
        1
    )

    predicted_index = (
        predicted_index.item()
    )


    # -----------------------------------------------------
    # ROUND CONFIDENCE
    # -----------------------------------------------------

    confidence = round(
        probabilities[
            0,
            predicted_index
        ].item() * 100,
        2
    )


    # -----------------------------------------------------
    # CLASS NAME
    # -----------------------------------------------------

    result = GATEKEEPER_CLASS_NAMES[
        predicted_index
    ]


    return (
        result,
        confidence
    )


# =========================================================
# DISEASE MODEL PREDICTION
# =========================================================

def predict(image_path):

    # -----------------------------------------------------
    # OPEN IMAGE
    # -----------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    # -----------------------------------------------------
    # TRANSFORM IMAGE
    # -----------------------------------------------------

    image_tensor = transform(
        image
    ).unsqueeze(0)


    image_tensor = image_tensor.to(
        device
    )


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


    # -----------------------------------------------------
    # TOP PREDICTION
    # -----------------------------------------------------

    confidence, predicted_index = torch.max(
        probabilities,
        1
    )

    predicted_index = (
        predicted_index.item()
    )


    # -----------------------------------------------------
    # ROUND MAIN CONFIDENCE
    # -----------------------------------------------------

    confidence = round(
        probabilities[
            0,
            predicted_index
        ].item() * 100,
        2
    )


    # -----------------------------------------------------
    # DISEASE NAME
    # -----------------------------------------------------

    disease = CLASS_NAMES[
        predicted_index
    ]


    # -----------------------------------------------------
    # TOP 3 PREDICTIONS
    # -----------------------------------------------------

    top3_probabilities, top3_indices = torch.topk(
        probabilities,
        3,
        dim=1
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

        top3.append({

            "name": CLASS_NAMES[
                index
            ],

            "confidence": probability
        })


    # -----------------------------------------------------
    # RETURN RESULTS
    # -----------------------------------------------------

    return (
        disease,
        confidence,
        top3
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# PREDICTION ROUTE
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def prediction():

    global latest_result


    # -----------------------------------------------------
    # CHECK IMAGE
    # -----------------------------------------------------

    if "image" not in request.files:

        return "No image uploaded."


    file = request.files[
        "image"
    ]


    # -----------------------------------------------------
    # CHECK FILE NAME
    # -----------------------------------------------------

    if file.filename == "":

        return "No image selected."


    # -----------------------------------------------------
    # SECURE FILE NAME
    # -----------------------------------------------------

    original_filename = secure_filename(
        file.filename
    )


    # -----------------------------------------------------
    # FILE EXTENSION
    # -----------------------------------------------------

    extension = os.path.splitext(
        original_filename
    )[1].lower()


    # -----------------------------------------------------
    # GENERATE UNIQUE FILE NAME
    # -----------------------------------------------------

    unique_filename = (
        str(uuid.uuid4())
        + extension
    )


    # -----------------------------------------------------
    # FILE PATH
    # -----------------------------------------------------

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )


    # -----------------------------------------------------
    # SAVE IMAGE
    # -----------------------------------------------------

    file.save(
        filepath
    )


    # -----------------------------------------------------
    # GATEKEEPER
    # -----------------------------------------------------

    gate_result, gate_confidence = check_tomato(
        filepath
    )


    # =====================================================
    # NOT TOMATO
    # =====================================================

    if gate_result == "not_tomato":

        latest_result = {

            "disease":
                "Image Not Recognized as Tomato",

            "confidence":
                round(
                    gate_confidence,
                    2
                ),

            "top3": [],

            "image":
                filepath,

            "image_filename":
                unique_filename,

            "disease_info": {

                "name":
                    "Not a Tomato Image",

                "description": (
                    "The uploaded image "
                    "was identified as not "
                    "being a tomato image."
                ),

                "symptoms": [],

                "treatment": [],

                "prevention": []
            }
        }


        return render_template(

            "result.html",

            image=filepath,

            image_filename=unique_filename,

            prediction=(
                "Image Not Recognized "
                "as Tomato"
            ),

            confidence=round(
                gate_confidence,
                2
            ),

            disease_info={

                "name":
                    "Not a Tomato Image",

                "description": (
                    "The uploaded image "
                    "was identified as not "
                    "being a tomato image."
                ),

                "symptoms": [],

                "treatment": [],

                "prevention": []
            },

            description=(
                "The gatekeeper model did "
                "not identify the uploaded "
                "image as a tomato image. "
                "Please upload a clear "
                "photograph of a tomato leaf."
            ),

            symptoms=[],

            treatment=[],

            prevention=[],

            top3=[]
        )


    # =====================================================
    # DISEASE PREDICTION
    # =====================================================

    disease, confidence, top3 = predict(
        filepath
    )


    # =====================================================
    # GET DISEASE INFORMATION
    # =====================================================

    disease_info = DISEASE_INFO.get(

        disease,

        {

            "name":
                disease,

            "description":
                "No information available.",

            "symptoms": [],

            "treatment": [],

            "prevention": []
        }
    )


    # =====================================================
    # STORE RESULT
    # =====================================================

    latest_result = {

        "disease":
            disease,

        "confidence":
            round(
                confidence,
                2
            ),

        "top3":
            top3,

        "image":
            filepath,

        "image_filename":
            unique_filename,

        "disease_info":
            disease_info
    }


    # =====================================================
    # RESULT PAGE
    # =====================================================

    return render_template(

        "result.html",

        image=filepath,

        image_filename=unique_filename,

        prediction=disease,

        confidence=round(
            confidence,
            2
        ),

        disease_info=disease_info,

        description=disease_info.get(
            "description",
            ""
        ),

        symptoms=disease_info.get(
            "symptoms",
            []
        ),

        treatment=disease_info.get(
            "treatment",
            []
        ),

        prevention=disease_info.get(
            "prevention",
            []
        ),

        top3=top3
    )


# =========================================================
# ABOUT PAGE
# =========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# =========================================================
# DISEASE INFORMATION PAGE
# =========================================================

@app.route(
    "/disease/<disease_name>"
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
# DOWNLOAD / GENERATE PDF REPORT
# =========================================================

@app.route("/download")
@app.route("/download_report")
def download_report():

    global latest_result


    # -----------------------------------------------------
    # CHECK RESULT
    # -----------------------------------------------------

    if latest_result is None:

        return "No prediction available."


    # -----------------------------------------------------
    # RESULT DATA
    # -----------------------------------------------------

    disease = latest_result[
        "disease"
    ]

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


    # -----------------------------------------------------
    # REPORT FILE
    # -----------------------------------------------------

    report_filename = (
        "tomato_disease_report.pdf"
    )

    report_path = os.path.join(
        "static",
        report_filename
    )


    # =====================================================
    # PAGE SETUP
    # =====================================================

    doc = SimpleDocTemplate(

        report_path,

        pagesize=A4,

        rightMargin=20 * mm,

        leftMargin=20 * mm,

        topMargin=20 * mm,

        bottomMargin=20 * mm
    )


    # =====================================================
    # STYLES
    # =====================================================

    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(

        "ReportTitle",

        parent=styles["Title"],

        alignment=TA_CENTER,

        fontSize=22,

        leading=28,

        spaceAfter=12
    )


    heading_style = ParagraphStyle(

        "ReportHeading",

        parent=styles["Heading2"],

        fontSize=15,

        leading=19,

        spaceBefore=12,

        spaceAfter=8
    )


    normal_style = ParagraphStyle(

        "ReportNormal",

        parent=styles["BodyText"],

        fontSize=10,

        leading=15,

        spaceAfter=6
    )


    center_style = ParagraphStyle(

        "ReportCenter",

        parent=styles["BodyText"],

        alignment=TA_CENTER,

        fontSize=10,

        leading=14
    )


    # =====================================================
    # PDF CONTENT
    # =====================================================

    story = []


    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    story.append(

        Paragraph(

            "Mvikeli Wezitshalo",

            title_style
        )
    )


    story.append(

        Paragraph(

            "Tomato Disease Recognition Report",

            center_style
        )
    )


    story.append(

        Spacer(
            1,
            10
        )
    )


    # -----------------------------------------------------
    # DATE
    # -----------------------------------------------------

    current_date = datetime.now().strftime(
        "%d %B %Y, %H:%M"
    )


    story.append(

        Paragraph(

            f"<b>Report Date:</b> "
            f"{current_date}",

            normal_style
        )
    )


    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    if os.path.exists(
        image_path
    ):

        try:

            report_image = PDFImage(

                image_path,

                width=100 * mm,

                height=100 * mm
            )

            report_image.hAlign = "CENTER"


            story.append(
                report_image
            )


            story.append(

                Spacer(
                    1,
                    10
                )
            )


        except Exception as e:

            print(
                "Could not add image to PDF:",
                e
            )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    story.append(

        Paragraph(

            "Prediction",

            heading_style
        )
    )


    story.append(

        Paragraph(

            f"<b>Detected condition:</b> "
            f"{escape(disease)}",

            normal_style
        )
    )


    story.append(

        Paragraph(

            f"<b>Confidence:</b> "
            f"{confidence:.2f}%",

            normal_style
        )
    )


    # -----------------------------------------------------
    # DESCRIPTION
    # -----------------------------------------------------

    description = disease_info.get(
        "description",
        ""
    )


    story.append(

        Paragraph(

            "Description",

            heading_style
        )
    )


    story.append(

        Paragraph(

            escape(
                str(description)
            ),

            normal_style
        )
    )


    # -----------------------------------------------------
    # SYMPTOMS
    # -----------------------------------------------------

    story.append(

        Paragraph(

            "Symptoms",

            heading_style
        )
    )


    symptoms = disease_info.get(
        "symptoms",
        []
    )


    if symptoms:

        for symptom in symptoms:

            story.append(

                Paragraph(

                    "• " + escape(
                        str(symptom)
                    ),

                    normal_style
                )
            )

    else:

        story.append(

            Paragraph(

                "No symptoms information available.",

                normal_style
            )
        )


    # -----------------------------------------------------
    # TREATMENT
    # -----------------------------------------------------

    story.append(

        Paragraph(

            "Treatment",

            heading_style
        )
    )


    treatment = disease_info.get(
        "treatment",
        []
    )


    if treatment:

        for item in treatment:

            story.append(

                Paragraph(

                    "• " + escape(
                        str(item)
                    ),

                    normal_style
                )
            )

    else:

        story.append(

            Paragraph(

                "No treatment information available.",

                normal_style
            )
        )


    # -----------------------------------------------------
    # PREVENTION
    # -----------------------------------------------------

    story.append(

        Paragraph(

            "Prevention",

            heading_style
        )
    )


    prevention = disease_info.get(
        "prevention",
        []
    )


    if prevention:

        for item in prevention:

            story.append(

                Paragraph(

                    "• " + escape(
                        str(item)
                    ),

                    normal_style
                )
            )

    else:

        story.append(

            Paragraph(

                "No prevention information available.",

                normal_style
            )
        )


    # -----------------------------------------------------
    # TOP 3 PREDICTIONS
    # -----------------------------------------------------

    story.append(

        Paragraph(

            "Top 3 Model Predictions",

            heading_style
        )
    )


    table_data = [

        [

            Paragraph(
                "<b>Rank</b>",
                normal_style
            ),

            Paragraph(
                "<b>Condition</b>",
                normal_style
            ),

            Paragraph(
                "<b>Confidence</b>",
                normal_style
            )
        ]
    ]


    for rank, item in enumerate(
        top3,
        start=1
    ):

        table_data.append(

            [

                str(rank),

                Paragraph(

                    escape(
                        item["name"]
                    ),

                    normal_style
                ),

                f'{item["confidence"]:.2f}%'
            ]
        )


    top3_table = Table(

        table_data,

        colWidths=[

            20 * mm,

            100 * mm,

            35 * mm
        ]
    )


    top3_table.setStyle(

        TableStyle([

            (

                "GRID",

                (0, 0),

                (-1, -1),

                0.5,

                colors.grey
            ),

            (

                "VALIGN",

                (0, 0),

                (-1, -1),

                "TOP"
            ),

            (

                "ALIGN",

                (0, 0),

                (0, -1),

                "CENTER"
            ),

            (

                "ALIGN",

                (-1, 0),

                (-1, -1),

                "CENTER"
            ),

            (

                "BACKGROUND",

                (0, 0),

                (-1, 0),

                colors.lightgrey
            )
        ])
    )


    story.append(
        top3_table
    )


    # =====================================================
    # BUILD PDF
    # =====================================================

    doc.build(
        story
    )


    # =====================================================
    # SEND PDF
    # =====================================================

    return send_file(

        report_path,

        as_attachment=True
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return {

        "status": "ok",

        "disease_model":
            "ResNet18",

        "gatekeeper_model":
            "ResNet18"
    }


# =========================================================
# ERROR HANDLERS
# =========================================================

@app.errorhandler(404)
def page_not_found(error):

    return (
        "Page not found.",
        404
    )


@app.errorhandler(500)
def internal_server_error(error):

    return (
        "Internal server error. "
        "Please try again.",
        500
    )


# =========================================================
# RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=False,
        use_reloader=False
    )