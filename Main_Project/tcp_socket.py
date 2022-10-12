import global_variables
import socket

import cv2 
import numpy as np

class TCP_Socket:
    def __init__(self, host=global_variables.HOST, port=global_variables.PORT):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print("socket oluşturuldu.")
            self.socket.bind((host, port))
            self.socket.listen(1)
            print("socket dinleniyor")
            self.c = None
            self.addr = None
        except socket.error as msg:
            print("Hata:", msg)

    def set_client_accept(self):
        self.c, self.addr = self.socket.accept()
        print('Gelen bağlantı:', self.addr)

    def get_client_image(self):
        dataFromClient = self.c.recv(global_variables.RECV)
        image_array = np.frombuffer(dataFromClient, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image

    def send_client_data(self, data):
        self.c.send(bytes(data, encoding="utf-8"))

    def close_client_connection(self):
        self.c.close()

