import cv2
import numpy as np
import dlib
from imutils import face_utils
import tkinter as tk
from tkinter import messagebox  # Import messagebox module for alerts
from PIL import Image, ImageTk

class LipReadingDrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.attributes('-fullscreen', True)  # Make the window full-screen
        self.root.title("Lip Reading, Drowsiness Detection, and Mouse Tracker")
        self.root.configure(bg="#f0f0f0")  # Set background color

        self.cap = cv2.VideoCapture(0)

        # Drowsiness Detection
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.sleep = 0
        self.drowsy = 0
        self.active = 0
        self.movement = 0
        self.status = ""
        self.color = (0, 0, 0)

        # Lip Reading
        self.lip_opening_threshold = 20
        self.lip_movement_threshold = 18
        self.consecutive_frames_threshold = 15
        self.speaking_frames = 0

        # Mouse Tracker
        self.mouse_x, self.mouse_y = 0, 0

        # Create main frame
        self.main_frame = tk.Frame(self.root, bg="pink")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.label = tk.Label(self.main_frame, bg="pink")
        self.label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)  # Add padding

        self.mouse_label = tk.Label(self.main_frame, text="Mouse coordinates: (0, 0)", bg="pink", font=("Helvetica", 12))
        self.mouse_label.pack(pady=10)  # Add padding

        self.start_button = tk.Button(self.main_frame, text="Start Detection", command=self.start_detection, bg="black", fg="white", font=("Helvetica", 14))
        self.start_button.pack(pady=10)  # Add padding

        # Bind window events
        self.root.bind("<Motion>", self.track_mouse)
        self.root.bind("<FocusIn>", self.window_focus_in)
        self.root.bind("<FocusOut>", self.window_focus_out)
        self.root.bind("<Escape>", self.exit_fullscreen)  # Bind the Escape key to exit full-screen

        # Alert timer
        self.alert_duration = 5000  # 5 seconds
        self.alert_timer = None

        # Split-screen detection timer
        self.split_screen_duration = 5000  # 5 seconds
        self.split_screen_warning_shown = False
        self.split_screen_timer = None

        # Flag to track if the video should start
        self.should_start_video = False

        # Timer for checking focus periodically
        self.focus_check_interval = 1000  # 1 second
        self.focus_check_timer = None

        self.warning_label = tk.Label(self.main_frame, text="", bg="pink", font=("Helvetica", 12))
        self.warning_label.pack(pady=10)  # Add padding
        self.long_duration_threshold = 100

    def start_detection(self):
        self.start_button.config(state=tk.DISABLED)
        self.should_start_video = True
        self.show_frame()

    def show_frame(self):
        if self.should_start_video:
            ret, frame = self.cap.read()

            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = self.detector(gray)
                if faces:
                    for face in faces:
                        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

                        face_frame = frame.copy()  # Assigning face_frame here
                        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        landmarks = self.predictor(gray, face)
                        landmarks = face_utils.shape_to_np(landmarks)

                        # Drowsiness Detection
                        self.detect_drowsiness_and_gaze(landmarks, face_frame)

                        # Lip Reading
                        self.detect_lip_reading(landmarks, face_frame)

                        for n in range(0, 68):
                            (x, y) = landmarks[n]
                            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

                    self.display_frame(face_frame)  # Moved this line inside the loop

            # Update mouse coordinates
            self.mouse_label.config(text=f"Mouse coordinates: ({self.mouse_x}, {self.mouse_y})")

        self.root.after(10, self.show_frame)

    def detect_drowsiness_and_gaze(self, landmarks, frame):
        alignment = self.detect_alignment(landmarks)
        gaze_direction = self.detect_gaze(landmarks)

        left_blink = self.blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = self.blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if left_blink == 0 or right_blink == 0:
            self.sleep += 1
            self.drowsy = 0
            self.active = 0
            if self.sleep > 6:
                self.status = "Not Active"
                self.color = (255, 255, 255)
                self.font = ("Helvetica", 12)
        else:
            self.drowsy = 0
            self.sleep = 0
            self.active += 1
            if self.active > 6:
                self.status = "Active"
                self.color = (255, 255, 255)
                self.font = ("Helvetica", 12)

        if gaze_direction != "Center":
            self.drowsy += 1
            self.movement += 1
            if self.movement > 3:
                self.status = "Not Focused"
                self.color = (255, 255, 255)
                self.font = ("Helvetica", 12)

        cv2.putText(frame, self.status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.color, 3)

    def detect_alignment(self, landmarks):
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        nose_bridge = landmarks[27]
        if nose_bridge[0] < left_eye_center[0] - 10:
            return "Right"
        elif nose_bridge[0] > right_eye_center[0] + 10:
            return "Left"
        else:
            return "Center"

    def detect_gaze(self, landmarks):
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        eye_center = (left_eye_center + right_eye_center) // 2
        nose_bridge = landmarks[27]
        if eye_center[0] < nose_bridge[0] - 20:
            return "Left"
        elif eye_center[0] > nose_bridge[0] + 20:
            return "Right"
        else:
            return "Center"

    def blinked(self, a, b, c, d, e, f):
        up = self.compute(b, d) + self.compute(c, e)
        down = self.compute(a, f)
        ratio = up / (2.0 * down)
        if ratio > 0.25:
            return 2
        elif ratio > 0.21 and ratio <= 0.25:
            return 1
        else:
            return 0

    def compute(self, ptA, ptB):
        dist = np.linalg.norm(ptA - ptB)
        return dist

    def detect_lip_reading(self, landmarks, frame):
        upper_lip_top = landmarks[51][1]
        lower_lip_bottom = landmarks[57][1]
        lip_opening = lower_lip_bottom - upper_lip_top
        lip_left = landmarks[48][0]
        lip_right = landmarks[54][0]
        lip_movement = lip_right - lip_left
        if lip_opening > self.lip_opening_threshold and lip_movement > self.lip_movement_threshold:
            self.speaking_frames += 1
        else:
            self.speaking_frames = 0

        # Display message on the GUI if speaking duration exceeds threshold
        if self.speaking_frames > self.consecutive_frames_threshold:
            self.warning_label.config(text="Please stop talking, otherwise you will be removed from the class")

        # Exit the application if speaking duration exceeds long duration
        if self.speaking_frames > self.long_duration_threshold:
            self.warning_label.config(text="Speaking duration exceeded long duration. Exiting application.")
            self.root.quit()

        if self.speaking_frames > self.consecutive_frames_threshold:
            cv2.putText(frame, "Speaking", (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        else:
            cv2.putText(frame, "Not Speaking", (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.config(image=imgtk)
        self.label.imgtk = imgtk

    def track_mouse(self, event):
        self.mouse_x, self.mouse_y = event.x, event.y

    def window_focus_in(self, event):
        print("Window focused in")
        # Cancel the alert timer if the window regains focus
        if self.alert_timer:
            self.root.after_cancel(self.alert_timer)
            self.alert_timer = None
        # Set the flag to start the video
        self.should_start_video = True

        # Start the focus check timer
        if self.focus_check_timer is None:
            self.focus_check_timer = self.root.after(self.focus_check_interval, self.check_focus)

    def window_focus_out(self, event):
        print("Window focused out")

    def exit_fullscreen(self, event):
        # Exit full-screen mode when the Escape key is pressed
        self.root.attributes('-fullscreen', False)

    def check_focus(self):
        # Check if the window is focused
        if not self.root.focus_get():
            self.window_focus_out(None)
        else:
            # Restart the focus check timer
            self.focus_check_timer = self.root.after(self.focus_check_interval, self.check_focus)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.cap.release()
            self.root.destroy()

    def check_split_screen(self, event):
        current_width = self.root.winfo_width()
        current_height = self.root.winfo_height()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        if current_width < screen_width // 2 or current_height < screen_height // 2:
            if not self.split_screen_warning_shown:
                self.split_screen_warning_shown = True
                self.split_screen_timer = self.root.after(self.split_screen_duration, self.show_split_screen_warning)
        else:
            if self.split_screen_warning_shown:
                self.split_screen_warning_shown = False
                if self.split_screen_timer:
                    self.root.after_cancel(self.split_screen_timer)
                    self.split_screen_timer = None

    def show_split_screen_warning(self):
        messagebox.showwarning("Split Screen Warning", "The application is in split screen mode for too long! Exiting application.")
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = LipReadingDrowsinessApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.bind("<Configure>", app.check_split_screen)  # Bind the split screen check to the window resize event
    root.mainloop()

