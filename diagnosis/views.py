import joblib
import cv2
import numpy as np

from rest_framework.decorators import api_view
from rest_framework.response import Response

from django.http import HttpResponse
from django.core.files.storage import default_storage

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime


# ================= LOAD MODELS =================

diabetes_model = joblib.load("ml_models/diabetes_rf.pkl")
diabetes_scaler = joblib.load("ml_models/diabetes_scaler.pkl")

heart_model = joblib.load("ml_models/heart_rf.pkl")
heart_scaler = joblib.load("ml_models/heart_scaler.pkl")

liver_model = joblib.load("ml_models/liver_rf.pkl")
liver_scaler = joblib.load("ml_models/liver_scaler.pkl")


# ================= RECOMMENDATIONS =================

def get_recommendation(disease, risk_level):
    if disease == "heart":
        return {
            "High": [
                "Consult a cardiologist within 7 days",
                "Reduce cholesterol and fatty foods",
                "Schedule ECG and blood pressure tests"
            ],
            "Medium": [
                "Monitor heart health regularly",
                "Adopt a low-salt diet",
                "Engage in light physical activity"
            ],
            "Low": [
                "Maintain healthy lifestyle",
                "Annual heart checkup"
            ]
        }[risk_level]

    if disease == "liver":
        return {
            "High": [
                "Consult a hepatologist immediately",
                "Avoid alcohol completely",
                "Schedule liver function tests"
            ],
            "Medium": [
                "Maintain balanced diet",
                "Reduce oily foods",
                "Re-test liver parameters in 3 months"
            ],
            "Low": [
                "Healthy liver function",
                "Continue balanced nutrition"
            ]
        }[risk_level]

    if disease == "diabetes":
        return {
            "High": [
                "Consult an endocrinologist",
                "Monitor blood glucose daily",
                "Reduce sugar intake"
            ],
            "Medium": [
                "Regular glucose monitoring",
                "Increase physical activity",
                "Low-carb diet"
            ],
            "Low": [
                "Maintain healthy lifestyle",
                "Annual diabetes screening"
            ]
        }[risk_level]


# ================= ECG ANALYZER (IMAGE-BASED, NO DATASET) =================

def analyze_ecg_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return {
            "prediction": "Invalid ECG Image",
            "risk_percentage": 0,
            "risk_level": "Unknown",
            "recommendations": ["Please upload a clear ECG image"]
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    signal_strength = np.mean(edges)

    if signal_strength < 20:
        prediction = "Low Heart Activity (Possible Bradycardia)"
        risk_level = "Medium"
    elif signal_strength <= 60:
        prediction = "Normal ECG Pattern"
        risk_level = "Low"
    else:
        prediction = "High Heart Activity (Possible Tachycardia)"
        risk_level = "High"

    risk_percentage = min(round(signal_strength, 2), 100)

    return {
        "prediction": prediction,
        "risk_percentage": risk_percentage,
        "risk_level": risk_level,
        "recommendations": [
            "Consult cardiologist if symptoms persist",
            "This is not a medical diagnosis"
        ]
    }


# ================= DIABETES API =================

@api_view(["POST"])
def diabetes_predict_api(request):
    d = request.data

    features = [
        float(d["pregnancies"]),
        float(d["glucose"]),
        float(d["bp"]),
        float(d["skin"]),
        float(d["insulin"]),
        float(d["bmi"]),
        float(d["dpf"]),
        float(d["age"]),
    ]

    scaled = diabetes_scaler.transform([features])
    pred = diabetes_model.predict(scaled)[0]
    prob = diabetes_model.predict_proba(scaled)[0][1]

    risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"

    return Response({
        "prediction": "Diabetes Detected" if pred else "No Diabetes",
        "risk_percentage": round(prob * 100, 2),
        "risk_level": risk_level,
        "recommendations": get_recommendation("diabetes", risk_level)
    })


# ================= HEART API =================

@api_view(["POST"])
def heart_predict_api(request):
    d = request.data

    age = float(d["age"])
    cp = float(d["cp"])
    chol = float(d["chol"])
    thalach = float(d["thalach"])
    oldpeak = float(d["oldpeak"])

    sex = 1
    restbp = 120
    exang = 0
    ca = 0
    thal = 2

    features = [
        age, sex, cp, restbp, chol,
        thalach, exang, oldpeak, ca, thal
    ]

    scaled = heart_scaler.transform([features])
    pred = heart_model.predict(scaled)[0]
    prob = heart_model.predict_proba(scaled)[0][1]

    risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"

    return Response({
        "prediction": "Heart Disease Detected" if pred else "No Heart Disease",
        "risk_percentage": round(prob * 100, 2),
        "risk_level": risk_level,
        "recommendations": get_recommendation("heart", risk_level)
    })


# ================= LIVER API =================

@api_view(["POST"])
def liver_predict_api(request):
    d = request.data

    features = [
        float(d["bilirubin"]),
        float(d["albumin"]),
        float(d["alkphos"]),
    ]

    scaled = liver_scaler.transform([features])
    pred = liver_model.predict(scaled)[0]
    prob = liver_model.predict_proba(scaled)[0][1]

    risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"

    return Response({
        "prediction": "Liver Disease Detected" if pred else "No Liver Disease",
        "risk_percentage": round(prob * 100, 2),
        "risk_level": risk_level,
        "recommendations": get_recommendation("liver", risk_level)
    })


# ================= ECG API =================

@api_view(["POST"])
def ecg_predict_api(request):
    if "ecg_image" not in request.FILES:
        return Response({"error": "ECG image is required"}, status=400)

    ecg_file = request.FILES["ecg_image"]

    file_path = default_storage.save(
        f"ecg_uploads/{ecg_file.name}", ecg_file
    )
    full_path = default_storage.path(file_path)

    result = analyze_ecg_image(full_path)
    return Response(result)


# ================= PDF REPORT =================

@api_view(["POST"])
def generate_report(request):
    data = request.data

    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = "attachment; filename=Medical_Report.pdf"

    pdf = canvas.Canvas(response, pagesize=A4)
    width, height = A4

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, height - 50, "Smart Healthcare AI – Medical Report")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, height - 90, f"Date: {datetime.now().strftime('%d-%m-%Y')}")

    y = height - 130

    for disease, result in data.items():
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(50, y, disease.upper())
        y -= 20

        pdf.setFont("Helvetica", 11)
        pdf.drawString(60, y, f"Prediction: {result['prediction']}")
        y -= 15
        pdf.drawString(60, y, f"Risk Percentage: {result['risk_percentage']}%")
        y -= 15
        pdf.drawString(60, y, f"Risk Level: {result['risk_level']}")
        y -= 20

        pdf.drawString(60, y, "Recommendations:")
        y -= 15

        for rec in result.get("recommendations", []):
            pdf.drawString(80, y, f"- {rec}")
            y -= 15

        y -= 20

    pdf.showPage()
    pdf.save()
    return response

from reportlab.lib.colors import red, lightcoral
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageDraw


@api_view(["POST"])
def generate_ecg_report(request):
    prediction = request.data.get("prediction")
    risk_level = request.data.get("risk_level")
    risk_percentage = request.data.get("risk_percentage")
    recommendations = request.data.get("recommendations", [])
    image_path = request.data.get("image_path")  # optional future use

    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = "attachment; filename=ECG_Report.pdf"

    pdf = canvas.Canvas(response, pagesize=A4)
    width, height = A4

    # ================= HEADER =================
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, height - 40, "ECG ANALYSIS REPORT")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(
        50,
        height - 65,
        f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}"
    )

    # ================= ECG IMAGE SECTION =================
    y = height - 110

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "ECG Image")
    y -= 10

    # ---- CREATE ECG GRID IMAGE (TEMP) ----
    grid_width = 450
    grid_height = 180

    img = Image.new("RGB", (grid_width, grid_height), "white")
    draw = ImageDraw.Draw(img)

    # Small grid
    for x in range(0, grid_width, 10):
        draw.line((x, 0, x, grid_height), fill=(255, 200, 200))
    for y2 in range(0, grid_height, 10):
        draw.line((0, y2, grid_width, y2), fill=(255, 200, 200))

    # Bold grid
    for x in range(0, grid_width, 50):
        draw.line((x, 0, x, grid_height), fill=(255, 100, 100), width=2)
    for y2 in range(0, grid_height, 50):
        draw.line((0, y2, grid_width, y2), fill=(255, 100, 100), width=2)

    # ---- SIMULATED ECG TRACE (OPTIONAL VISUAL) ----
    draw.line((20, 90, 120, 90), fill="black", width=2)
    draw.line((120, 90, 130, 40), fill="black", width=2)
    draw.line((130, 40, 150, 120), fill="black", width=2)
    draw.line((150, 120, 180, 90), fill="black", width=2)
    draw.line((180, 90, 300, 90), fill="black", width=2)

    # ---- RED ABNORMAL REGION ----
    draw.rectangle(
        (150, 50, 300, 130),
        outline="red",
        width=3
    )

    # Save temp image
    temp_img_path = "temp_ecg.png"
    img.save(temp_img_path)

    pdf.drawImage(
        ImageReader(temp_img_path),
        50,
        y - 200,
        width=450,
        height=180
    )

    # ================= ECG RESULTS =================
    y -= 230

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "ECG Findings")
    y -= 20

    pdf.setFont("Helvetica", 11)
    pdf.drawString(60, y, f"Prediction: {prediction}")
    y -= 15
    pdf.drawString(60, y, f"Risk Level: {risk_level}")
    y -= 15
    pdf.drawString(60, y, f"Risk Percentage: {risk_percentage}%")
    y -= 25

    # ================= RECOMMENDATIONS =================
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Recommendations")
    y -= 18

    pdf.setFont("Helvetica", 11)
    for rec in recommendations:
        pdf.drawString(70, y, f"- {rec}")
        y -= 15

    y -= 20

    # ================= DISCLAIMER =================
    pdf.setFont("Helvetica-Oblique", 10)
    pdf.drawString(
        50,
        y,
        "Disclaimer: This ECG analysis is AI-assisted and not a medical diagnosis."
    )

    pdf.showPage()
    pdf.save()

    return response

from django.http import JsonResponse
import random
from datetime import datetime

def google_fit_data(request):
    data = {
        "source": "Google Fit",
        "heart_rate": random.randint(70, 105),
        "steps": random.randint(3000, 9000),
        "sleep_hours": round(random.uniform(4.5, 8.0), 1),
        "last_sync": datetime.now().strftime("%H:%M:%S")
    }
    return JsonResponse(data)

from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def register_user(request):
    if request.method == "POST":
        data = json.loads(request.body)
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return JsonResponse(
                {"error": "Username and password required"}, status=400
            )

        if User.objects.filter(username=username).exists():
            return JsonResponse(
                {"error": "User already exists"}, status=400
            )

        User.objects.create_user(username=username, password=password)

        return JsonResponse(
            {"message": "User registered successfully"}
        )

    # ✅ VERY IMPORTANT: handle GET (or any other method)
    return JsonResponse(
        {"message": "Register API endpoint. Use POST method."},
        status=405
    )

@csrf_exempt
def login_user(request):
    if request.method == "POST":
        data = json.loads(request.body)
        username = data.get("username")
        password = data.get("password")

        try:
            user = User.objects.get(username=username)
            if user.check_password(password):
                return JsonResponse(
                    {"user_id": user.id, "username": user.username}
                )
            return JsonResponse({"error": "Invalid password"}, status=400)
        except User.DoesNotExist:
            return JsonResponse({"error": "User not found"}, status=400)

    return JsonResponse(
        {"message": "Login API endpoint. Use POST method."},
        status=405
    )

