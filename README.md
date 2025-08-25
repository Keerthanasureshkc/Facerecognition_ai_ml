# Facerecognition_ai_ml
This project implements a real-time face recognition application using Python, OpenCV, and deep learning (CNN with TensorFlow/Keras). It supports image preprocessing, dataset training, and real-time identity detection with optimized performance. The repository includes code, trained models, and documentation for setup and usage.
- Upload an image via Tkinter file dialog  
- Real-time face detection with Haar Cascade  
- Template matching for recognition  
- Bounding boxes + labels for matched faces
##  CODE  

```python
import cv2
from tkinter import Tk, filedialog

def load_image():
    Tk().withdraw()  
    file_path = filedialog.askopenfilename()  
    if file_path:
        image_name = file_path.split('/')[-1].split('.')[0]  
        return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE), image_name
    return None, None

uploaded_image, uploaded_image_name = load_image()

if uploaded_image is None:
    print("Error: No image selected.")
else:
    correlation_threshold = 0.09
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            uploaded_image_resized = cv2.resize(uploaded_image, (w, h))
            result = cv2.matchTemplate(gray_frame[y:y+h, x:x+w], uploaded_image_resized, cv2.TM_CCOEFF_NORMED)
            _, correlation, _, _ = cv2.minMaxLoc(result)

            if correlation > correlation_threshold:
                cv2.putText(frame, f"{uploaded_image_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                print("Correlation:", correlation)

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

### blame

load_image() → Opens file dialog, loads selected image in grayscale, extracts file name.

cv2.CascadeClassifier → Uses Haar Cascade for face detection.

while True loop → Captures live video frames from webcam.

cv2.cvtColor → Converts frames to grayscale for faster detection.

detectMultiScale → Detects faces in the current frame.

cv2.matchTemplate → Compares uploaded image with detected face region.

if correlation > 0.09 → Checks if match is strong enough to label.

cv2.putText & cv2.rectangle → Draws bounding box and name of matched person.

Exit condition → Press q to quit webcam feed.
