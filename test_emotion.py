import cv2, numpy as np, base64, urllib.request

# ── 1. Download a real human face image ──────────────────────────────────────
url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
urllib.request.urlretrieve(url, "test_face.jpg")
img = cv2.imread("test_face.jpg")
print("1. Image loaded:", img.shape if img is not None else "FAILED")

# ── 2. Face detection ─────────────────────────────────────────────────────────
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
print("2. Faces detected:", len(faces))

if len(faces) == 0:
    print("   !! Haar cascade found NO face - trying looser params")
    faces = cascade.detectMultiScale(gray, 1.05, 2, minSize=(20, 20))
    print("   Faces with loose params:", len(faces))

# ── 3. ONNX emotion inference ─────────────────────────────────────────────────
net = cv2.dnn.readNetFromONNX("emotion_model.onnx")
LABELS = ['neutral','happy','surprise','sad','angry','disgust','fear','contempt']

if len(faces) > 0:
    x, y, w, h = faces[0]
    print(f"3. Face region: x={x} y={y} w={w} h={h}")
    face_gray = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_gray, (64, 64)).astype(np.float32)
    blob = face_resized.reshape(1, 1, 64, 64)
    net.setInput(blob)
    scores = net.forward()[0]
    e_x = np.exp(scores - scores.max())
    probs = e_x / e_x.sum()
    print("4. Emotion scores:")
    for l, p in sorted(zip(LABELS, probs), key=lambda x: -x[1]):
        print(f"   {l:12s}: {p:.3f}")
    print("   Dominant:", LABELS[probs.argmax()])
else:
    print("3. SKIPPED - no face found")

# ── 4. Simulate what the app does with a base64 frame ────────────────────────
print("\n5. Simulating base64 frame pipeline...")
_, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
img_bytes = base64.b64decode(b64.split(',')[1])
arr = np.frombuffer(img_bytes, np.uint8)
decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
print("   Decoded shape:", decoded.shape if decoded is not None else "FAILED")
gray2 = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
faces2 = cascade.detectMultiScale(gray2, 1.1, 3, minSize=(30, 30))
print("   Faces in decoded frame:", len(faces2))
