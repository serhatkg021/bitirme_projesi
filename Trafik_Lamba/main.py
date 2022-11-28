import numpy as np
import cv2

IMAGE_PATH = "images/"
RED_LIGHT_PATH = "kirmizi-isik/"
YELLOW_LIGHT_PATH = "sari-isik/"
GREEN_LIGHT_PATH = "yesil-isik/"

class LightDetection:

    def __init__(self, path):
        print(path)
        self.image = cv2.imread(path)
        # self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def red_check(self):
        sub_object = self.image[5:550, 350:600].copy()  # 1.jpg'de Görüntü üzerinde algılanan trafik lambasını görüntüden ayırıyoruz
        sub_object_hsv = cv2.cvtColor(sub_object, cv2.COLOR_BGR2HSV)

        lower = np.array([160, 155, 84])
        upper = np.array([179, 255, 255])

        # self.mask = cv2.inRange(self.hsv, lower, upper)
        self.mask = cv2.inRange(sub_object_hsv, lower, upper) 

        detected_circles = cv2.HoughCircles(self.mask,
                        cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                    param2 = 30, minRadius = 0, maxRadius = 0)

        if detected_circles is not None:
            print("buldu")

            detected_circles = np.round(detected_circles[0, :]).astype("int")
            for (x, y, r) in detected_circles:
                # cv2.circle(self.image, (x, y), r, (0, 128, 255), 4)
                # cv2.rectangle(self.image, (x - 5, y - 5),
                #             (x + 5, y + 5), (0, 128, 255), -1)
                cv2.circle(sub_object, (x, y), r, (0, 128, 255), 4)
                cv2.rectangle(sub_object, (x - 5, y - 5),
                            (x + 5, y + 5), (0, 128, 255), -1)
        else:
            print("bulmadı")
        cv2.imshow('image',self.image)
        cv2.imshow('red-mask',self.mask)
        cv2.imshow("cropped", sub_object)

    def yellow_check(self):
        sub_object = self.image[100:600, 400:768].copy()  # 1.jpg'de Görüntü üzerinde algılanan trafik lambasını görüntüden ayırıyoruz
        sub_object_hsv = cv2.cvtColor(sub_object, cv2.COLOR_BGR2HSV)

        lower = np.array([15, 93, 0])
        upper = np.array([45, 255, 255])
        self.mask = cv2.inRange(sub_object_hsv, lower, upper)

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
        cv2.imshow("cropped", sub_object)

    def green_check(self):
        sub_object = self.image[100:560, 590:768].copy()  # 1.jpg'de Görüntü üzerinde algılanan trafik lambasını görüntüden ayırıyoruz
        sub_object_hsv = cv2.cvtColor(sub_object, cv2.COLOR_BGR2HSV)

        lower = np.array([60, 52, 72])
        upper = np.array([80, 255, 255])
        self.mask = cv2.inRange(sub_object_hsv, lower, upper)

        detected_circles = cv2.HoughCircles(self.mask,
                        cv2.HOUGH_GRADIENT, 1, 10, param1 = 50,
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
        cv2.imshow("cropped", sub_object)

    def colorDetect2(self):
        import cv2 as cv
        import imutils

        returnData = [0, 0, 0]  # 0: Green,  1: Yellow,  2: Red
        data = cv.medianBlur(self.image, 5)
        dataHsv = cv.cvtColor(data, cv.COLOR_BGR2HSV)


        colors = [[np.array([36, 0, 0]), np.array([70, 255, 255])],   # 0: Green,  1: Yellow,  2: Red
                [np.array([20,5,150]), np.array([30, 255, 255])],
                [np.array([0,5,150]), np.array([8,255,255])]]
                        

        for index, color  in enumerate(colors):
            print(index, color[0])
            mask = cv.inRange(dataHsv, color[0], color[1])
            mask = cv.dilate(mask, (3, 3), iterations=3)
            contour = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contour = imutils.grab_contours(contour)

            if len(contour) > 0:
                c = max(contour, key = cv.contourArea)
                ((x, y), radius) = cv.minEnclosingCircle(c)
                M = cv2.moments(c)
                xc=int(M["m10"] / M["m00"])
                yc=int(M["m01"] / M["m00"])
                center = (xc,yc)
                if radius > 15:
                    cv2.rectangle(data, (int(x-radius), int(y-radius)),
                                (int(x+radius), int(y+radius)), (0, 255, 255), 3)
                    # cisme sarı çerçeve çizdik.
                    # cismin merkezine kırmızı nokta koyduk
                    cv2.circle(data, center, 5, (0, 0, 255), -1)
                    cv2.imshow("Original", data)
                    returnData[index] = 1
        print(returnData)
        return returnData

    def colorDetect(self):
        import cv2 as cv

        data = cv.medianBlur(self.image,5)
        dataHsv=cv.cvtColor(data,cv.COLOR_BGR2HSV)

        # lower_color = np.array([38, 100, 100])  # Green Color
        # upper_color = np.array([75, 255, 255])

        # lower_color = np.array([22, 100, 100])  # Yellow Color
        # upper_color = np.array([38, 255, 255])
        
        lower_color = np.array([160, 100, 100])  # Red Color
        upper_color = np.array([179, 255, 255])

        mask = cv.inRange(dataHsv, lower_color,upper_color)
        mask = cv.dilate(mask,(3,3),iterations=3)

        contour = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        contour = imutils.grab_contours(contour)
        
        if len(contour) > 0:
            c = max(contour, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            xc=int(M["m10"] / M["m00"])
            yc=int(M["m01"] / M["m00"])
            center = (xc,yc)

            if radius > 50: #eğer kırmızı renk tespit edildiyse
                cv2.rectangle(data,(int(x-radius),int(y-radius)) , (int(x+radius),int(y+radius)), (0, 255, 255), 3)
                #cisme sarı çerçeve çizdik.
                cv2.circle(data, center, 5, (0, 0, 255), -1) #cismin merkezine kırmızı nokta koyduk

                print(xc,yc) 

        for i in range(len(contour)):
            print(i)
            cv.drawContours(data,contour,i,(0,0,255),4)

        cv.imshow("Original",data)
        cv.imshow("Hsv",dataHsv)
        cv.imshow("Mask",mask)

        cv.waitKey(0)
        cv.destroyAllWindows()

light_detection = LightDetection(path=IMAGE_PATH + '2.jpg')
# light_detection = LightDetection(path=IMAGE_PATH + YELLOW_LIGHT_PATH + '4.jpg')
# light_detection = LightDetection(path=IMAGE_PATH + GREEN_LIGHT_PATH + '2.jpg')

# light_detection.red_check()
# light_detection.yellow_check()
light_detection.colorDetect2()

cv2.waitKey(0)
cv2.destroyAllWindows()

