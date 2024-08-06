import cv2

# loading the required, trained classifier
detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

# capture frames from an image or camera
imp_img = cv2.VideoCapture("elon.jpg")

# res - true/false depending on if image has been read successfully, img - the pixel resolution of the image
res, img = imp_img.read()

# converting image to greyscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces of different sizes on the image - returns (x,y) of bottom left face-sqaure, w - width and h- height
faces = detect.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    # To draw a rectangle in a face
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)         # (image, vertex of rectangle, opposite vertex, thickness)

cv2.imshow("Elon Image", img)
cv2.waitKey(0)

# Close the window
imp_img.release()

# unallocate any associated memory usage
cv2.destroyAllWindows()