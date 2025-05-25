import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque

class HandTracker:
    def __init__(self, mode=False, max_hands=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=False):
        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        self.results = self.hands.process(img_rgb)
        
        # Draw hand landmarks if hands are detected (disabled for middle finger only)
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
        return img

    def find_position(self, img, hand_no=0):
        landmark_list = []
        z_list = []
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_no:
                hand = self.results.multi_hand_landmarks[hand_no]
                for id, lm in enumerate(hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([id, cx, cy])
                    z_list.append(lm.z)
        return landmark_list, z_list

def main():
    # Initialize webcam with lower resolution for better performance
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam!")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    tracker = HandTracker()

    # Variables for relative movement
    prev_x, prev_y, prev_z = None, None, None
    # Get initial mouse position
    mouse_x, mouse_y = pyautogui.position()

    # Sensitivity factors for cursor speed
    sensitivity_xy = 1.0  # For x and y movement

    # Z thresholds for different fingers
    middle_finger_threshold = -80  # For cursor movement
    index_finger_threshold = -50   # For click detection

    # Click detection variables
    click_time = 0
    click_count = 0
    last_click_time = 0
    index_finger_up = False
    index_finger_down = False

    # Frame rate control
    frame_rate = 30
    frame_delay = 1 / frame_rate

    # Wait for initial hand detection and show coordinates
    print("Please place your hand in front of the camera...")
    while True:
        success, img = cap.read()
        if not success:
            break

        # Flip the image horizontally and vertically
        img = cv2.flip(img, -1)

        # Find hands
        img = tracker.find_hands(img, draw=False)  # Disabled hand drawing
        
        # Get hand landmarks and z values
        landmark_list, z_list = tracker.find_position(img)
        
        if landmark_list and len(z_list) > 12:
            middle_finger = landmark_list[12]  # Middle finger tip
            index_finger = landmark_list[8]    # Index finger tip
            current_x, current_y = middle_finger[1], middle_finger[2]
            middle_z = z_list[12]  # Get z value for middle finger
            index_z = z_list[8]    # Get z value for index finger
            
            # Display current coordinates
            print(f"Middle finger Z: {middle_z}, Index finger Z: {index_z}")
            print("Press 'q' to start tracking...")
            
            # Show coordinates on screen
            cv2.putText(img, f"Middle Z: {int(middle_z * 1000)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Index Z: {int(index_z * 1000)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Draw circles at the finger tips
            cv2.circle(img, (current_x, current_y), 10, (0, 255, 0), -1)  # Middle finger
            cv2.circle(img, (index_finger[1], index_finger[2]), 10, (255, 0, 0), -1)  # Index finger
            
            cv2.imshow('Initial Position', img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print("Starting hand tracking...")
    cv2.destroyWindow('Initial Position')

    while True:
        start_time = time.time()

        success, img = cap.read()
        if not success:
            break

        # Flip the image horizontally and vertically
        img = cv2.flip(img, -1)

        # Find hands
        img = tracker.find_hands(img, draw=False)  # Disabled hand drawing
        
        # Get hand landmarks and z values
        landmark_list, z_list = tracker.find_position(img)
        
        if landmark_list and len(z_list) > 12:  # Make sure we have z values
            middle_finger = landmark_list[12]  # Middle finger tip
            index_finger = landmark_list[8]    # Index finger tip
            current_x, current_y = middle_finger[1], middle_finger[2]
            middle_z = z_list[12]  # Get z value for middle finger
            index_z = z_list[8]    # Get z value for index finger
            
            # Convert z to a more readable value
            middle_z_display = int(middle_z * 1000)  # Scale up for better visibility
            index_z_display = int(index_z * 1000)
            
            # Draw circles at the finger tips
            cv2.circle(img, (current_x, current_y), 10, (0, 255, 0), -1)  # Middle finger
            cv2.circle(img, (index_finger[1], index_finger[2]), 10, (255, 0, 0), -1)  # Index finger
            
            # Check if middle finger z is less than threshold
            is_active = middle_z > -0.08  # -0.08 is equivalent to -80 in scaled values
            
            # Display z values and status on screen
            status_color = (0, 255, 0) if is_active else (0, 0, 255)
            cv2.putText(img, f"Middle Z: {middle_z_display}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(img, f"Index Z: {index_z_display}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(img, "ACTIVE" if is_active else "INACTIVE", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Handle mouse movement
            if is_active:  # Only move if middle finger z is greater than threshold
                if prev_x is not None and prev_y is not None:
                    # Calculate how much the middle finger moved in x and y
                    dx = (current_x - prev_x) * sensitivity_xy
                    dy = (current_y - prev_y) * sensitivity_xy
                    
                    # Move mouse relative to its current position
                    mouse_x += dx
                    mouse_y += dy
                    pyautogui.moveTo(mouse_x, mouse_y)
            
            # Handle click detection
            current_time = time.time()
            
            # Check if index finger is up or down using its own threshold
            if index_z < -0.05:  # Index finger is up (using -50 as threshold)
                if not index_finger_up:
                    index_finger_up = True
                    index_finger_down = False
                    click_count += 1
                    if click_count == 1:
                        click_time = current_time
                    elif click_count == 2:
                        if current_time - click_time < 1.0:  # Double click within 1 second
                            pyautogui.click()
                            print("Click!")
                        click_count = 0
            else:  # Index finger is down
                if not index_finger_down:
                    index_finger_down = True
                    index_finger_up = False
            
            # Reset click count if more than 1 second has passed
            if click_count > 0 and current_time - click_time > 1.0:
                click_count = 0
            
            # Update previous position
            prev_x, prev_y = current_x, current_y
        else:
            # Reset previous position when hand is not detected
            prev_x, prev_y = None, None
            click_count = 0

        # Show the webcam feed
        cv2.imshow('Hand Tracking', img)

        # Control frame rate
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_delay:
            time.sleep(frame_delay - elapsed_time)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 