import cv2
import numpy as np
import json

from tcp_socket import TCP_Socket
from lane_detection import lane_detec

socket = TCP_Socket()
while True:
    if socket.c is None:
        socket.set_client_accept()

    image = socket.get_client_image()
    lane_status = lane_detec(image)
    print(lane_status)
    # cv2.imshow("camera", image)

    # send_data = {"lane_status": lane_status,
    #              "traffic_light": }
    # data = json.dumps(send_data)
    socket.send_client_data(lane_status)

    if cv2.waitKey(10) == 13:
        socket.close_client_connection()
        cv2.destroyAllWindows()
        break
