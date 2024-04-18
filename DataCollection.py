#collect data from various sources ..asking people to come infront of webcam,click 20 each pictures each
#store the part of the image containing the face
import cv2
import numpy as np
import os

# Check if the cascade file exists
cascade_file = "haarcascade_frontalface_alt.xml"
if not os.path.exists(cascade_file):
    print("Cascade file not found!")
    exit(1)

cam = cv2.VideoCapture(0)
filename = input("Enter name: ")
dataset_path = "C:\\Users\\SHAMBHAVI MISHRA\\Desktop\\face_detection\\data"
offset = 20

# Load the Haar Cascade classifier
model = cv2.CascadeClassifier(cascade_file)

# Create a list to save data
facedata = []
skip = 0

while True:
    success, img = cam.read()
    if not success:
        print("Reading failed")
        break

    # Convert the image to grayscale
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = model.detectMultiScale(grayimg, 1.4, 5)

    # Sort faces by size
    faces = sorted(faces, key=lambda f: f[2] * f[3])

    # Pick the largest face
    if len(faces) > 0:
        f = faces[-1]
        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop and resize the face region
        cropped_face = img[y - offset:y + h + offset, x - offset:x + w + offset]
        cropped_face = cv2.resize(cropped_face, (100, 100))

        skip += 1
        if skip % 10 == 0:
            facedata.append(cropped_face)
            print("Saved so far: " + str(len(facedata)))

    cv2.imshow("Image window", img)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# Convert facedata to numpy array
facedata = np.asarray(facedata)
m = facedata.shape
facedata = facedata.reshape((m[0], -1))

# Save facedata to disk as numpy array
file_path = os.path.join(dataset_path, filename + ".npy")
np.save(file_path, facedata)
print("Data saved: " + file_path)

cam.release()
cv2.destroyAllWindows()