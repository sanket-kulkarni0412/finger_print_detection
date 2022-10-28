import numpy as np
import cv2
import imutils
import scipy.ndimage.interpolation as inter
# step 1 - load the model

net = cv2.dnn.readNetFromONNX('last.onnx')

# step 2 - feed a 640x640 image to get predictions

def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result
def define_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = 255 - gray
    # thresh_val = 5
    # gray[gray < thresh_val] = 0
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imshow('image',thresh)
    cv2.waitKey(0)
    # Compute rotated bounding box
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    print(angle)
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

def extreme_points(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = 255 - gray
    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    # find contours in thresholded image, then grab the largest
    # one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    cont=[]
    extLeft = list(c[c[:, :, 0].argmin()][0])
    extRight = list(c[c[:, :, 0].argmax()][0])
    extTop = list(c[c[:, :, 1].argmin()][0])
    extBot = list(c[c[:, :, 1].argmax()][0])
    cont.append(extLeft)
    cont.append(extRight)
    cont.append(extTop)
    cont.append(extBot)
    print(cont)
    cv2.drawContours(image, [c],-1,(0, 255, 255), 1)
    cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
    cv2.circle(image, extRight, 8, (0, 255, 0), -1)
    cv2.circle(image, extTop, 8, (255, 0, 0), -1)
    cv2.circle(image, extBot, 8, (255, 255, 0), -1)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    # Rotate image to deskew
def fill_image(image):
    th, im_th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)
# Copy the thresholded image.
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    # Display images.
    cv2.imshow("Thresholded Image", im_th)
    cv2.imshow("Floodfilled Image", im_floodfill)
    # cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    # cv2.imshow("Foreground", im_out)
    cv2.waitKey(0)


    cv2.waitKey(0)
image = cv2.imread('input\LEFT_r_50.jpg')
input_image = format_yolov5(image) # making the image square
blob = cv2.dnn.blobFromImage(input_image , 1/255.0, (640, 640), swapRB=True)
net.setInput(blob)
predictions = net.forward()

# step 3 - unwrap the predictions to get the object detections 

class_ids = []
confidences = []
boxes = []

output_data = predictions[0]

image_width, image_height, _ = input_image.shape
x_factor = image_width / 640
y_factor =  image_height / 640

for r in range(25200):
    row = output_data[r]
    confidence = row[4]
    if confidence >= 0.4:

        classes_scores = row[5:]
        _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
        class_id = max_indx[1]
        if (classes_scores[class_id] > .25):

            confidences.append(confidence)

            class_ids.append(class_id)

            x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
            left = int((x - 0.5 * w) * x_factor)
            top = int((y - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            box = np.array([left, top, width, height])
            boxes.append(box)

class_list = []
with open("classes.txt", "r") as f:
    class_list = [cname.strip() for cname in f.readlines()]

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

result_class_ids = []
result_confidences = []
result_boxes = []

for i in indexes:
    result_confidences.append(confidences[i])
    result_class_ids.append(class_ids[i])
    result_boxes.append(boxes[i])

for i in range(len(result_class_ids)):

    box = result_boxes[i]
    class_id = result_class_ids[i]

    cv2.rectangle(image, box, (0, 255, 255), 2)
    cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1]), (0, 255, 255), -1)
    cv2.putText(image, class_list[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,220,0))
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[0] + box[2])
    y2 = int(box[1] + box[3])
    cropped = image[y1:y2, x1:x2]
    fill_image(cropped)
    # draw_con(cropped)
    # skewCorrect(cropped)
# cv2.imwrite("out_detection.png",cropped)
image_ = imutils.resize(image, width=1200)
cv2.imwrite("out_detection.png",image_)

# plt.imshow(image)
# plt.imshow(image)
# plt.show()