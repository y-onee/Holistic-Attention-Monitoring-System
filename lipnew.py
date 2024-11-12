import cv2
import dlib
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import time

class LipReadingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lip Reading App")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(self.predictor_path)

        self.cap = None
        self.video_frame = None
        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.speaking_label = tk.Label(root, text="Speaking Status: Not Detected", font=("Helvetica", 14))
        self.speaking_label.pack()

        self.start_button = tk.Button(root, text="Start Video", command=self.start_video)
        self.start_button.pack()

        self.stop_button = tk.Button(root, text="Stop Video", command=self.stop_video, state=tk.DISABLED)
        self.stop_button.pack()

        self.lip_opening_threshold = 20
        self.lip_movement_threshold = 18
        self.consecutive_frames_threshold = 15
        self.speaking_frames = 0

        self.inference_times = []

    def start_video(self):
        self.cap = cv2.VideoCapture(0)  # Use this line for webcam input
        # self.cap = cv2.VideoCapture("Vid 2.mp4")  # Use this line for video file input

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.show_frame()

    def show_frame(self):
        start_time = time.time()  # Record start time for inference

        ret, frame = self.cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.detector(gray)

            for face in faces:
                shape = self.predictor(gray, face)

                lip_opening = self.calculate_lip_opening(shape)
                lip_movement = self.calculate_lip_movement(shape)

                if lip_opening > self.lip_opening_threshold and lip_movement > self.lip_movement_threshold:
                    self.speaking_frames += 1
                else:
                    self.speaking_frames = 0

                if self.speaking_frames > self.consecutive_frames_threshold:
                    self.speaking_label.config(text="Speaking Status: Speaking", fg="red")
                else:
                    self.speaking_label.config(text="Speaking Status: Not Speaking", fg="green")

                cv2.putText(frame, f"Lip Opening: {lip_opening}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Lip Movement: {lip_movement}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            self.video_label.after(10, self.show_frame)

            end_time = time.time()  # Record end time for inference
            inference_time = end_time - start_time
            self.inference_times.append(inference_time)

        else:
            self.stop_video()

    def stop_video(self):
        if self.cap is not None:
            self.cap.release()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

            self.video_label.configure(image='')
            self.speaking_label.config(text="Speaking Status: Not Detected", fg="black")

            if self.inference_times:
                average_inference_time = sum(self.inference_times) / len(self.inference_times)
                fps = 1 / average_inference_time
                print(f"Average Inference Time: {average_inference_time:.4f} seconds")
                print(f"Frames Per Second (FPS): {fps:.2f}")

    def calculate_lip_opening(self, shape):
        upper_lip_top = shape.part(51).y
        lower_lip_bottom = shape.part(57).y
        lip_opening = lower_lip_bottom - upper_lip_top
        return lip_opening

    def calculate_lip_movement(self, shape):
        lip_left = shape.part(48).x
        lip_right = shape.part(54).x
        lip_movement = lip_right - lip_left
        return lip_movement

if __name__ == "__main__":
    root = tk.Tk()
    app = LipReadingApp(root)
    root.mainloop()
