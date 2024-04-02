import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from skimage import morphology
import math
import os
import keras_ocr
from imutils import contours

pipeline = keras_ocr.pipeline.Pipeline()

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

def inpaint_text(img_path, pipeline):
    # read image
    img = keras_ocr.tools.read(img_path)
    # generate (word, box) tuples 
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
                 
    return(img)

IMG_PATH = "schematic/"
images  = [f for f in os.listdir(IMG_PATH) if os.path.isfile(os.path.join(IMG_PATH, f))]
Image = np.random.randint(85, size=1)
leimg = IMG_PATH+images[int(Image)]
original = cv2.imread(leimg)
plt.imshow(original)
cleaned_img = inpaint_text(leimg, pipeline)

plt.imshow(cleaned_img)

gray = cv2.cvtColor(cleaned_img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
blurImg = cv2.blur(edges,(15,15))
plt.imshow(blurImg)

thresh = cv2.threshold(blurImg, 40, 255, cv2.THRESH_BINARY)[1]
plt.imshow(thresh)

# thresheroded = cv2.erode(thresh, None, iterations=1)
# threshfinal = cv2.dilate(thresheroded, None, iterations=1)


threshdilated = cv2.dilate(thresh, None, iterations=1)
threshfinal = cv2.erode(threshdilated, None, iterations=1)

plt.imshow(threshfinal)

cnts = cv2.findContours(threshfinal.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]
# loop over the contours
for (i, c) in enumerate(cnts):
	# draw the bright spot on the image
	(x, y, w, h) = cv2.boundingRect(c)
	((cX, cY), radius) = cv2.minEnclosingCircle(c)
	cv2.circle(image, (int(cX), int(cY)), int(radius),
		(0, 0, 255), 3)
	cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
# show the output image
plt.imshow(image)