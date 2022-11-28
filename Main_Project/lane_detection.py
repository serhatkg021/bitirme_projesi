from turtle import right, st
import global_variables
import cv2
import numpy as np
import random
def average_slope_intercept(image_shapes, lines):
    left_fit = []
    right_fit = []
    
    left_lines = []
    right_lines = []

    left_line = []
    right_line = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # en küçük kare polinom uyumunu bulmaya yarar yani karelerin toplamını en aza indirerek belirli bir nokta kümesine en uygun eğriyi bulmaktır
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            # parametreler x noktları y noktaları ve derece
            # derecesi 1 olduğu için 2 elemanlı bir dizi döner
            slope = parameters[0]  # slope (eğim) dizinin 1. elemanı
            intercept = parameters[1]  # intercept (kesişim) dizinin 2. elemanı
            if slope < 0:
                left_fit.append((slope, intercept))  # append diziye ekliyor
                left_lines.append([x1,y1,x2,y2])
            else:
                right_fit.append((slope, intercept))
                right_lines.append([x1,y1,x2,y2])

        if left_fit:
            # average belirtilen eksen boyunca ağırlıklı ortalamayı hesaplar
            left_fit_average = np.average(left_fit, axis=0)
            left_lines = np.average(left_lines, axis=0)
            # print("left -- " + str(left_lines))
            # parametreler ortalaması alınacak dizi, axis 0 ise sütun boyunca axis 1 ise satır boyunca oluyor
            left_line = make_coordinates(image_shapes, left_fit_average)
        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
            right_lines = np.average(right_lines, axis=0)
            # print("right -- " + str(right_lines))
            right_line = make_coordinates(image_shapes, right_fit_average)   

        line_navigation = line_find([left_lines,right_lines])
        data = []
        if line_navigation != 2:
            data = [left_line, right_line, None,line_navigation]
        else:
            data = [left_lines,right_lines, None, line_navigation]        
        return data

def line_find(lines):
    serit = 2 # 0 sol, 1 sağ, 2 orta   --şerit durumu
    if(len(lines[0])>0 and len(lines[1])>0):
        line_left = abs(lines[0][1] - lines[0][3])
        line_right = abs(lines[1][1] - lines[1][3])
        if(line_left < line_right):
            serit = 1   
        else:
            serit = 0
    return serit  




def make_coordinates(image_shapes, line_parameters):
    # 2 elemanlı dizi line_parameters ilk eleman slope ikinci eleman intercept oluyor
    slope, intercept = line_parameters
    # y1 image.shape[0] yani y eksenini alıyor shape de y,x olarak geliyor 0 y ekseni oluyor
    y1 = image_shapes[0]
    y2 = int(y1*3/5)
    x1 = int(y1 - intercept)/slope
    x2 = int(y2 - intercept)/slope
    return np.array([x1, y1, x2, y2])    

def line_center(averaged_lines):
    line_left_center = [(averaged_lines[0][0]+averaged_lines[0][2])/2,
                        (averaged_lines[0][1]+averaged_lines[0][3])/2]  # x,y
    # sol çizginin ortalaması alınıyor x eksenleri toplanıp 2 ye bölünüyor aynı şekilde y eksenkleri toplanıp 2 ye bölünüyor
    line_right_center = [(averaged_lines[1][0]+averaged_lines[1][2])/2,
                        (averaged_lines[1][1]+averaged_lines[1][3])/2]
    lines_distance = line_right_center[0] - line_left_center[0]
    # aynı şekilde sağ çizginin ortalaması alınıyor
    # sol ve sağ çizgilerin x değerlerinin ortalaması alınıyor toplanıp 2 ye bölünerek
    x_center = (line_left_center[0] + line_right_center[0]) / 2
    # sol ve sağ çizginin y değerlerinin ortalaması alınıyor toplanıp 2 ye bölünerek
    y_center = (line_left_center[1] + line_right_center[1]) / 2
    # ortalama kordinatları döndürülüyor array olarak
    return np.array([x_center, y_center, lines_distance, averaged_lines[2]])
  

def turn_way(middle_value, middle_const):  # middle value şeritlerin orta noktası(x,y elemanlı array olarak geliyor) hareketli nokta middle cost (sadece x noktası geliyor) ise çalışma alanının orta noktası yani sabit nokta
    if middle_value[3] != 2:
        # 480-520 roi orta merkez 500
        middle_value_x = middle_value[0]  # x değeri array in 0. elemanı oluyor
        #               385  -  500 = 115
        status = middle_value_x - middle_const # Şeritin orta noktası ile ROI orta noktası arası fark
        #           115 * 100  /  (330/2) / 100 = 0,69
        status = status * 100 / (middle_value[2]/2) / 100 # Aradaki farkın -1 ile 1 arasında karşılık gelen değeri

        # Şerit genişliğine göre gelen -1 ile 1 arasındaki status değeri hata payı aralıgındaysa düz gitmesi için 0 gönder
        if (status > -global_variables.LANE_SAFELY_THRESH and status < global_variables.LANE_SAFELY_THRESH):  # eğer sabit nokta x değeri hareketli noktanın x değerinde büyükse araba solda kalmış olur sağa gitmesi gerekir
            return str(0)
        else:                   
            return str(round(status, 2)) # Hata payı aralığında değilse direk -1 ile 1 arasında bir değer olan statusu gönder
    else:
        if(len(middle_value[0]) == 0 and len(middle_value[1]) > 0):
            status = (middle_const - ((middle_value[1][0] + middle_value[1][2])/2))/(middle_const/2)
            return str((1 - abs(status)) * -1)

        elif(len(middle_value[1]) == 0 and len(middle_value[0]) > 0):
            status = (middle_const - ((middle_value[0][0] + middle_value[0][2])/2))/(middle_const/2)
            return str((1 - abs(status)))

        elif(len(middle_value[0]) > 0 and len(middle_value[1]) > 0):
            line_len = abs(middle_value[0][0] - middle_value[1][2])
            if(line_len < global_variables.ROI_AREA_CENTER_X*0.1):
                left_line_average = (middle_value[0][0] + middle_value[0][2])/2
                right_line_average = (middle_value[1][0] + middle_value[1][2])/2
                line_average = (left_line_average + right_line_average)/2
                if(global_variables.ROI_AREA_CENTER_X<line_average):
                    return str(-1)
                else:
                    return str(1)
        else:
            return " "


def lane_detec(image):
    cv2.imshow("camera", image)

    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale kopya
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # rgb kopya
    cv2.imshow("gray", im)

    # masked_white = cv2.inRange(im,180,255)
    # cv2.imshow("masked-white", masked_white)

    # Gaussian Blur

    blurred = cv2.GaussianBlur(im, (5, 5), 0.8)
    cv2.imshow("blured", blurred)

    edge_image = cv2.Canny(blurred, 50, 150)
    cv2.imshow("edge", edge_image)

    # ROI
    mask = np.zeros_like(edge_image)
    vertices = np.array(
        [global_variables.ROI_AREA], np.int32)

    cost_middle_value = global_variables.ROI_AREA_CENTER_X
    # print(vertices)
    cv2.fillPoly(mask, vertices, 255)
    cv2.imshow("rip-area", mask)

    # print (edge_image.shape, mask.shape)
    masked = cv2.bitwise_and(edge_image, mask)
    cv2.imshow("lane-area", masked)

    lines = cv2.HoughLinesP(masked, 1, np.pi/180, 60, np.array([]), minLineLength=50, maxLineGap=200)
    zeros = np.zeros_like(img)
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(zeros, (x1, y1), (x2, y2), (0,0,255), 4)

    combo_image = cv2.addWeighted(img, 0.8, zeros, 1, 1)
    # çalışma alanının orta noktasını ekrana basmak için
    averaged_lines = average_slope_intercept(im.shape, lines)
    # print(averaged_lines)
    if averaged_lines is not None:
        if averaged_lines[3] != 2:
            middle_value = line_center(averaged_lines)
            cv2.rectangle(combo_image, 
                                (int(cost_middle_value - middle_value[2] * global_variables.LANE_SAFELY_THRESH), 370),
                                (int(cost_middle_value + middle_value[2] * global_variables.LANE_SAFELY_THRESH), 410),
                                (255, 255, 255), -1)
            cv2.rectangle(combo_image, (int(middle_value[0]-5), 380), (int(middle_value[0]+5), 400), (0, 255, 0), -1)
            # çalışma alanının orta noktasının x noktası kamera açısına göre ayarlandığı için sabit oluyor direk
            cv2.imshow('result', combo_image)  
        else:
            middle_value = averaged_lines
        return turn_way(middle_value, cost_middle_value)
    else:
        return " "
    # parametreler 1.image, 1.image in ağırlığı, 2. image, 2.image in ağırlığı, çıkış görüntüsünün ağırlığı
    # cv2.imshow("image", img)
