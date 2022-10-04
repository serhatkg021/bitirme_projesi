import cv2 
import numpy as np
import json

from tcp_socket import TCP_Socket

socket = TCP_Socket()
while True:
    if socket.c is None:
        socket.set_client_accept()

    image = socket.get_client_image()
    cv2.imshow("camera", image)   
    send_data = {"id": 2, "name": "abc"}
    data = json.dumps(send_data)
    socket.send_client_data(data)

    if cv2.waitKey(10) == 13:
        socket.close_client_connection()
        cv2.destroyAllWindows()
        break
