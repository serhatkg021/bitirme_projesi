import numpy as np
import cv2

IMAGE_PATH = "images/"
RED_LIGHT_PATH = "kirmizi-isik/"
YELLOW_LIGHT_PATH = "sari-isik/"
GREEN_LIGHT_PATH = "yesil-isik/"

class LightDetection:

    def __init__(self, path):
        self.image = cv2.imread(path)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.blured = cv2.blur(self.hsv, (3, 3))

    def red_check(self):
        lower = np.array([160, 155, 84])
        upper = np.array([179, 255, 255])
        self.mask = cv2.inRange(self.hsv, lower, upper)

        detected_circles = cv2.HoughCircles(self.mask,
                        cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                    param2 = 30, minRadius = 0, maxRadius = 0)

        if detected_circles is not None:
            print("buldu")

            detected_circles = np.round(detected_circles[0, :]).astype("int")
            for (x, y, r) in detected_circles:
                cv2.circle(self.image, (x, y), r, (0, 128, 255), 4)
                cv2.rectangle(self.image, (x - 5, y - 5),
                            (x + 5, y + 5), (0, 128, 255), -1)
        else:
            print("bulmadı")
        cv2.imshow('image',self.image)
        cv2.imshow('yellow-mask',self.mask)
        # cv2.imshow('result',self.result)

    def yellow_check(self):
        lower = np.array([15, 93, 0])
        upper = np.array([45, 255, 255])
        self.mask = cv2.inRange(self.hsv, lower, upper)

        detected_circles = cv2.HoughCircles(self.mask,
                        cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                    param2 = 30, minRadius = 0, maxRadius = 0)

        if detected_circles is not None:
            print("buldu")

            detected_circles = np.round(detected_circles[0, :]).astype("int")
            for (x, y, r) in detected_circles:
                cv2.circle(self.image, (x, y), r, (0, 128, 255), 4)
                cv2.rectangle(self.image, (x - 5, y - 5),
                            (x + 5, y + 5), (0, 128, 255), -1)
        else:
            print("bulmadı")
        cv2.imshow('image',self.image)
        cv2.imshow('yellow-mask',self.mask)
        # cv2.imshow('result',self.result)

    def green_check(self):
        lower = np.array([60, 52, 72])
        upper = np.array([80, 255, 255])
        self.mask = cv2.inRange(self.hsv, lower, upper)

        detected_circles = cv2.HoughCircles(self.mask,
                        cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                    param2 = 30, minRadius = 0, maxRadius = 0)

        if detected_circles is not None:
            print("buldu")

            detected_circles = np.round(detected_circles[0, :]).astype("int")
            for (x, y, r) in detected_circles:
                cv2.circle(self.image, (x, y), r, (0, 128, 255), 4)
                cv2.rectangle(self.image, (x - 5, y - 5),
                            (x + 5, y + 5), (0, 128, 255), -1)
        else:
            print("bulmadı")
        cv2.imshow('image',self.image)
        cv2.imshow('green-mask',self.mask)
        # cv2.imshow('result',self.result)


# light_detection = LightDetection(path=IMAGE_PATH + RED_LIGHT_PATH + '5.webp')
# light_detection = LightDetection(path=IMAGE_PATH + YELLOW_LIGHT_PATH + '3.jpg')
light_detection = LightDetection(path=IMAGE_PATH + GREEN_LIGHT_PATH + '3.jpg')

# light_detection.red_check()
# light_detection.yellow_check()
light_detection.green_check()

cv2.waitKey(0)
cv2.destroyAllWindows()

