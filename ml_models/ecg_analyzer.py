import cv2
import numpy as np

def analyze_ecg(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return {"error": "Invalid ECG image"}

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise removal
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Signal strength
    signal_strength = np.mean(edges)

    # Heart activity
    if signal_strength < 20:
        heart_status = "Low Heart Activity (Possible Bradycardia)"
    elif signal_strength <= 60:
        heart_status = "Normal Heart Activity"
    else:
        heart_status = "High Heart Activity (Possible Tachycardia)"

    # Rhythm check
    projection = np.sum(edges, axis=0)
    variation = np.std(projection)

    if variation > 500:
        rhythm = "Irregular Rhythm Detected"
    else:
        rhythm = "Rhythm Appears Regular"

    return {
        "heart_status": heart_status,
        "rhythm": rhythm,
        "note": "This is not a medical diagnosis"
    }
