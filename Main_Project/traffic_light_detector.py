import cv2
import numpy as np
import imutils


def colorDetect(image):
    returnData = [0, 0, 0, 0]  # 0: Red,  1: Yellow,  2: Green
    data = cv2.medianBlur(image, 5)
    cv2.imshow("Original", data)

    dataHsv = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)

    red_min = np.array([0,5,150])
    red_max = np.array([8,255,255])
    red_min2 = np.array([175,5,150])
    red_max2 = np.array([180,255,255])

    yellow_min = np.array([20,5,150])
    yellow_max = np.array([30,255,255])

    green_min = np.array([35,5,150])
    green_max = np.array([90,255,255])

    # region Red
    mask = cv2.inRange(dataHsv,red_min,red_max)+cv2.inRange(dataHsv,red_min2,red_max2)
    mask = cv2.dilate(mask, (3, 3), iterations=3)
    contour = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)

    if len(contour) > 0:
        c = max(contour, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 5:
            print("red")
            returnData[0] = 1
        else:
            returnData[0] = 0
    # endregion Red

    # region Yellow
    mask = cv2.inRange(dataHsv, yellow_min, yellow_max)
    mask = cv2.dilate(mask, (3, 3), iterations=3)
    contour = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)

    if len(contour) > 0:
        c = max(contour, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 5:
            print("yellow")
            returnData[1] = 1
        else:
            returnData[1] = 0
    # endregion Yellow

    # region Green
    mask = cv2.inRange(dataHsv, green_min, green_max)
    mask = cv2.dilate(mask, (3, 3), iterations=3)
    contour = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)

    if len(contour) > 0:
        c = max(contour, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 5:
            print("green")
            returnData[2] = 1
        else:
            returnData[2] = 0
    # endregion Green

    return returnData
