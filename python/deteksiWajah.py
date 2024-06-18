import numpy as np
import cv2

def writeText(image, text):
    cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

def detectEdge(insert_images):
    source_image = insert_images.copy()
    medium_value = np.median(source_image)
    lower_value = int(max(0, 0.7 * medium_value))
    upper_value = int(min(255, 1.3 * medium_value))
    edge = cv2.Canny(source_image, threshold1=lower_value, threshold2=upper_value)
    source_image[edge == 255] = [0, 255, 0]
    return writeText(source_image, 'Edge Detection')

def detectContour(source):
    canny = cv2.Canny(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), 30, 200)
    contour, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    desired_contour = np.zeros(source.shape)
    cv2.drawContours(desired_contour, contour, -1, (0, 255, 0), 3)
    return writeText(desired_contour, 'Contour Detection')

def detectFace(source):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise IOError('Cannot load Haar cascade xml file')
    image = source.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in face:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return writeText(image, 'Face Detection')

def detectCorner(source):
    gray = np.float32(cv2.cvtColor(source.copy(), cv2.COLOR_BGR2GRAY))
    destination = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)
    destination = cv2.dilate(destination, None)
    source[destination > 0.01 * destination.max()] = [0, 255, 0]
    return writeText(source, 'Corner Detection')

def applyWatershed(source):
    source = source.astype(np.uint8)
    image = cv2.cvtColor(cv2.medianBlur(source, 5), cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
    _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    clear_threshold_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_background = cv2.dilate(clear_threshold_image, kernel, iterations=3)
    distance_transform = cv2.distanceTransform(clear_threshold_image, cv2.DIST_L2, 5)
    _, sure_foreground = cv2.threshold(distance_transform, 0.7 * distance_transform.max(), 255, 0)
    sure_foreground = np.uint8(sure_foreground)
    unknown = cv2.subtract(sure_background, sure_foreground)
    _, marker = cv2.connectedComponents(sure_foreground)
    marker += 1
    marker[unknown == 255] = 0
    watershed_marker = cv2.watershed(image, marker)
    contours, hierarchy = cv2.findContours(watershed_marker, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 0, 0), 5)
    return writeText(image, 'Watershed Algorithm')

videoCapture = cv2.VideoCapture(0)
system = 1

while True:
    res, frame = videoCapture.read()
    if not res:
        break
    if system == 1:
        cv2.imshow('Camera', detectFace(frame))
    elif system == 2:
        cv2.imshow('Camera', detectCorner(frame))
    elif system == 3:
        cv2.imshow('Camera', detectContour(frame))
    elif system == 4:
        cv2.imshow('Camera', detectEdge(frame))
    elif system == 5:
        cv2.imshow('Camera', applyWatershed(frame))
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to break
        break
    elif key == ord('1'):
        system = 1
    elif key == ord('2'):
        system = 2
    elif key == ord('3'):
        system = 3
    elif key == ord('4'):
        system = 4
    elif key == ord('5'):
        system = 5

videoCapture.release()
cv2.destroyAllWindows()
