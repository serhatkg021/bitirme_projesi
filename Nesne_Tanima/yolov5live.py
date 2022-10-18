import torch
import numpy as np
import cv2
import time


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using OpenCV.
    """

    def __init__(self, model_name):  # model name i parametre verdik
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        # path = 'best.pt'
        # model = torch.hub.load(path, 'yolov5s', source='local', pretrained=True)
        if (model_name):
            model = torch.hub.load(
                'ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5',
                                   'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.6:  # eşik değer
                x1, y1, x2, y2 = int(
                    row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                # tanıdığı  modeli kare içine alıyor
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                # kare üstüne label ve güven skorunu yazıyor
                cv2.putText(frame, self.class_to_label(
                    labels[i]) + " " + str(row[4]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def detect_object(self, image):  # instance olarak çalışıyor
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        # cap = cv2.VideoCapture(0)

        # while cap.isOpened():

        start_time = time.perf_counter()
        frame = image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.score_frame(rgb_frame)
        frame = self.plot_boxes(results, frame)
        end_time = time.perf_counter()
        fps = 1 / np.round(end_time - start_time, 3)
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow("img", frame)


# Create a new object and execute.
# kendi modelimizi veriyoruz burada init çalışıyor kurucu fonk
detection = ObjectDetection(model_name='best.pt')
detection.detect_object()
