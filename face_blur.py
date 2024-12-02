import cv2
import numpy as np

# Function to apply Gaussian blur
def blur(img, k):
    h, w = img.shape[:2]
    kh, kw = h // k, w // k
    kh = kh if kh % 2 == 1 else kh - 1  # Ensure kernel size is odd
    kw = kw if kw % 2 == 1 else kw - 1
    return cv2.GaussianBlur(img, ksize=(kh, kw), sigmaX=0)

# Function to pixelate a face region
def pixelate_face(image, blocks=10):
    (h, w) = image.shape[:2]
    x_steps = np.linspace(0, w, blocks + 1, dtype="int")
    y_steps = np.linspace(0, h, blocks + 1, dtype="int")
    for i in range(1, len(y_steps)):
        for j in range(1, len(x_steps)):
            start_x, end_x = x_steps[j - 1], x_steps[j]
            start_y, end_y = y_steps[i - 1], y_steps[i]
            roi = image[start_y:end_y, start_x:end_x]
            mean_color = np.mean(roi, axis=(0, 1)).astype(int)  # Mean color
            image[start_y:end_y, start_x:end_x] = mean_color  # Fill block
    return image

# Main function
def main():
    factor = 3
    blocks = 10
    cap = cv2.VideoCapture(0)

    # Load Haar Cascade
    cascade_path = r'D:\pycharmCodes\barcodeScanner\haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Verify if the cascade is loaded
    if face_cascade.empty():
        raise IOError(f"Failed to load cascade classifier from {cascade_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        for (x, y, w, h) in faces:
            if w > 0 and h > 0:  # Valid face region
                face_roi = frame[y:y + h, x:x + w]
                face_roi = blur(face_roi, factor)
                face_roi = pixelate_face(face_roi, blocks)
                frame[y:y + h, x:x + w] = face_roi  # Replace face region

        # Display the video feed
        cv2.imshow('Live', frame)
        if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
