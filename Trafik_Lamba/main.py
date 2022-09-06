import numpy as np
import cv2

IMAGE_PATH = "Trafik_Lamba/images/"
RED_LIGHT_PATH = "kirmizi-isik/"
YELLOW_LIGHT_PATH = "sari-isik/"
GREEN_LIGHT_PATH = "yesil-isik/"

class LightDetection:

    def __init__(self, path):
        print(path)
        self.image = cv2.imread(path)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def red_check(self):
        # sub_object = self.image[5:550, 350:600].copy()  # 1.jpg'de Görüntü üzerinde algılanan trafik lambasını görüntüden ayırıyoruz
        # sub_object_hsv = cv2.cvtColor(sub_object, cv2.COLOR_BGR2HSV)

        lower = np.array([160, 155, 84])
        upper = np.array([179, 255, 255])

        self.mask = cv2.inRange(self.hsv, lower, upper)
        # self.mask = cv2.inRange(sub_object_hsv, lower, upper) 

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
                # cv2.circle(sub_object, (x, y), r, (0, 128, 255), 4)
                # cv2.rectangle(sub_object, (x - 5, y - 5),
                #             (x + 5, y + 5), (0, 128, 255), -1)
        else:
            print("bulmadı")
        cv2.imshow('image',self.image)
        cv2.imshow('yellow-mask',self.mask)
        # cv2.imshow("cropped", sub_object)

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


light_detection = LightDetection(path=IMAGE_PATH + RED_LIGHT_PATH + '1.jpg')
# light_detection = LightDetection(path=IMAGE_PATH + YELLOW_LIGHT_PATH + '3.jpg')
# light_detection = LightDetection(path=IMAGE_PATH + GREEN_LIGHT_PATH + '1.webp')

light_detection.red_check()
# light_detection.yellow_check()
# light_detection.green_check()

cv2.waitKey(0)
cv2.destroyAllWindows()

