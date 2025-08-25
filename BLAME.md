
---

### BLAME.md
```md
# Code Explanation 

- **load_image()** → Opens file dialog, loads selected image in grayscale, extracts file name.  
- **cv2.CascadeClassifier** → Uses Haar Cascade model for real-time face detection.  
- **while True loop** → Continuously captures live video frames from the webcam.  
- **cv2.cvtColor** → Converts frames to grayscale for faster detection.  
- **detectMultiScale** → Detects faces within the current video frame.  
- **cv2.matchTemplate** → Compares uploaded image with detected face regions using template matching.  
- **if correlation > 0.09** → Validates match strength before labeling.  
- **cv2.putText & cv2.rectangle** → Draw bounding box and add label (person’s name).  
- **Exit condition (q)** → Stops webcam feed and closes all windows.  
