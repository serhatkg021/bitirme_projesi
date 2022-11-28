import cv2
import numpy as np
import json

from tcp_socket import TCP_Socket
from lane_detection import lane_detec
from yolo.yolov5live import ObjectDetection

# objDetect = ObjectDetection(model_name="./yolo/trafic_lights_v2.pt") 
objDetect = ObjectDetection(path="yolo/ultralytics_yolov5_master/", model_name="yolov5s.pt") 

socket = TCP_Socket()
while True:
    if socket.c is None:
        socket.set_client_accept()

    image = socket.get_client_image()
    lane_status = lane_detec(image)
    object_status = objDetect.detect_object(image)
    # print(lane_status)
    # cv2.imshow("camera", image)

    # send_data = {"lane_status": lane_status,
    #              "traffic_light": }
    # data = json.dumps(send_data)
    socket.send_client_data(lane_status + "_" + str(object_status[0]) + "_" + str(object_status[1]))

    if cv2.waitKey(10) == 13:
        socket.close_client_connection()
        cv2.destroyAllWindows()
        break
