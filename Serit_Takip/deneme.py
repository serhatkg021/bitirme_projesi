import cv2
import numpy as np

vid = cv2.VideoCapture('deneme2.mp4') # video alınıyor

def processImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # fotoyu gri fortmata çeviriyor
    # masked_white = cv2.inRange(gray,180,255)
    blur = cv2.GaussianBlur(gray, (5,5), 0) # gri fotoğrafa gaussianBlur uyguluyor Gauss gürültüsünü gidermek için
    #parametreler img , çıkış arrayi(output array), sigma x
    canny = cv2.Canny(blur, 50, 150) # cany filterisi uyguluyor kenar bulmak için
    # parametreler image, düşük eşik değeri (low treshold), yüksek eşik değeri (high treshold)
    return canny

def region_of_interest(image):
    height = image.shape[0] # shape de ilk parametre y kısmı yani height oluyor
    polygons = np.array([
    [(500,height), (1000,600), (1150,600),(1500,height)] #sol alt sol üst sağ üst sağ alt olarak 4 elemanlı ve her biri x,y değeri olan dizi tanımladık
    ])
    mask = np.zeros_like(image) # image boyutunda sıfırlardan oluşan matris oluşturuyor
    cv2.fillPoly(mask, polygons, 255) # resme çokgen çizmek için kullanılır yani çalışma alanını belirliyoruz
    #parametreler resim, çokgen noktaları , renk
    cv2.imshow("alan",mask)
    masked_image = cv2.bitwise_and(image, mask) #görüntünün belirli bir bölgesinin değiştirilmesini sağlar image ile mask parametresi alarak sadece mask alanının kalmasını sağlıyor image de 
    cv2.imshow("roi",masked_image)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None: #line yoksa
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4) # dizi içindeki 4 elemanlı diziyi direk tek bir 4 elemanlı dizi olarak dönüştürüyor ve sırasıyla değişkenlere atanıyor
            cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 10) # line methodu görüntü üzerine çizgi çekmek için kullanılır.
            #parametreler image, başlangıç noktası, bitiş noktası , renk , çizgi kalınlığı
    return line_image

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1) # en küçük kare polinom uyumunu bulmaya yarar yani karelerin toplamını en aza indirerek belirli bir nokta kümesine en uygun eğriyi bulmaktır
            #parametreler x noktları y noktaları ve derece
            # derecesi 1 olduğu için 2 elemanlı bir dizi döner 
            slope = parameters[0] # slope (eğim) dizinin 1. elemanı
            intercept = parameters[1] # intercept (kesişim) dizinin 2. elemanı
            if slope < 0:
                left_fit.append((slope, intercept)) # append diziye ekliyor
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis=0) # average belirtilen eksen boyunca ağırlıklı ortalamayı hesaplar
        # parametreler ortalaması alınacak dizi, axis 0 ise sütun boyunca axis 1 ise satır boyunca oluyor
        right_fit_average = np.average(right_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters #2 elemanlı dizi line_parameters ilk eleman slope ikinci eleman intercept oluyor
    y1 = image.shape[0] # y1 image.shape[0] yani y eksenini alıyor shape de y,x olarak geliyor 0 y ekseni oluyor
    y2 = int(y1*3/5)
    x1 = int(y1 - intercept)/slope
    x2 = int(y2 - intercept)/slope
    return np.array([x1, y1, x2, y2])


def line_center(averaged_lines):
    line_left_center = [(averaged_lines[0][0]+averaged_lines[0][2])/2, (averaged_lines[0][1]+averaged_lines[0][3])/2] #x,y
    # sol çizginin ortalaması alınıyor x eksenleri toplanıp 2 ye bölünüyor aynı şekilde y eksenkleri toplanıp 2 ye bölünüyor
    line_right_center = [(averaged_lines[1][0]+averaged_lines[1][2])/2, (averaged_lines[1][1]+averaged_lines[1][3])/2]
    # aynı şekilde sağ çizginin ortalaması alınıyor 
    x_center = (line_left_center[0] + line_right_center[0]) / 2 # sol ve sağ çizgilerin x değerlerinin ortalaması alınıyor toplanıp 2 ye bölünerek
    y_center = (line_left_center[1] + line_right_center[1]) / 2 # sol ve sağ çizginin y değerlerinin ortalaması alınıyor toplanıp 2 ye bölünerek
    return np.array([x_center, y_center]) # ortalama kordinatları döndürülüyor array olarak

def turn_way(middle_value,middle_const) : # middle value şeritlerin orta noktası(x,y elemanlı array olarak geliyor) hareketli nokta middle cost (sadece x noktası geliyor) ise çalışma alanının orta noktası yani sabit nokta
    middle_value_x = middle_value[0] # x değeri array in 0. elemanı oluyor
    if(middle_const > middle_value_x): # eğer sabit nokta x değeri hareketli noktanın x değerinde büyükse araba solda kalmış olur sağa gitmesi gerekir
        print("turn right")
    elif(middle_const < middle_value_x): # eğer sabit nokta x değeri hareketli noktanın x değerinden küçükse araba sağda kalmış olur sola gitmesi gerekir
        print("turn left")
    else:                   # eğer noktalar birbirine eşitse düz gitmesi gerekir ortalamış olur
        print("straight")        
 
 # buraya bir eşik değer konabilir daha optimize etmek amacıyla örneğin hareketli nokta x değeri ile sabit nokta x değeri birbirine eşitse düz git kısmında 
 # sabit noktanın x değerinin belli bir değerde altında ve üstündeyse de düz git şeklinde yapılabilir 
 # yani if (middle_cost+50 < middle_value_x  && middle_cost-50 > middle_value_x) ise düz git şeklinde


while True:
    ret, frame = vid.read() # frame okunan video ret ise okunuyorsa true oluyor
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gri filtre uygulanıyor videoya
    processed_image = processImage(frame) 
    cropped_image = region_of_interest(processed_image)
    cv2.rectangle(cropped_image, (1030,750), (1070,770),(255, 255, 255), -1) # çalışma alanının orta noktasını ekrana basmak için
    #parametreler image ,x1,y1 , x2,y2, renk, 
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=10) # görüntüde şekil algılamak için kullanılır
    # parametreler image, r (piksel cinsinden uzaklık çözünürlüğü), teta (radyan cinsinden açı çözünürlüğü) , treshold (geçerli satır için minimum seçim sayısı ,
    #  minLineLength (izin verilen minimum satır uzunluğu) , maxLineGap (birleştirmek için çizgiler arasında izin verilen maksimum boşluk)
    averaged_lines = average_slope_intercept(grayFrame, lines)
    line_image = display_lines(cropped_image,lines) 
    combo_image = cv2.addWeighted(grayFrame, .6, line_image, 1, 1) # İki dizinin (görüntünün) ağırlıklı toplamını hesaplar.
    # parametreler 1.image, 1.image in ağırlığı, 2. image, 2.image in ağırlığı, çıkış görüntüsünün ağırlığı
    middle_value = line_center(averaged_lines)
    cv2.rectangle(combo_image, (int(middle_value[0]-5),750), (int(middle_value[0]+5),770),(0, 255, 0), -1)
    cv2.imshow('result', combo_image)
    turn_way(middle_value,1050) # çalışma alanının orta noktasının x noktası kamera açısına göre ayarlandığı için sabit oluyor direk
    #print(lines)
    if cv2.waitKey(30) & 0xFF == ord('q'): # q ya basınca while dan çık yani kapat
        break

vid.release() # videoyu bırakıyor
cv2.destroyAllWindows() # açılan ekranları kapatıyor