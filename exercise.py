import sys
import cv2
import time
import numpy as np
import mediapipe as mp
import base64

SQUAT_REFRACTORY_PERIOD = 1.0 # Seconds

class ExerciseTracker:
    def __init__(self, exercise_type="bicep_curl"):
        # Initialize MediaPipe Pose with higher confidence thresholds
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,  # Increased from 0.5
            min_tracking_confidence=0.7,   # Increased from 0.5
            model_complexity=1,
            smooth_landmarks=True
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Set the selected exercise
        self.exercise_type = exercise_type

        # Exercise counters and states
        self.rep_counter = 0

        # Exercise state
        self.position = None  # General state like 'up', 'down', 'center', 'left', 'right'

        # Time trackers
        self.start_time = time.time()
        self.last_time = time.time()
        self.last_rep_time = 0

    def calculate_angle(self, a, b, c):
        """
        Calculate angle between three points.
        """
        a = np.array(a)  # First point
        b = np.array(b)  # Mid point (joint)
        c = np.array(c)  # End point

        # Calculate vectors
        ba = a - b
        bc = c - b

        # Calculate cosine of angle
        cosine_angle = np.dot(ba, bc) / \
            (np.linalg.norm(ba) * np.linalg.norm(bc))
        # Ensure value is in range [-1, 1] to avoid domain errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)

        # Convert to degrees
        angle = np.degrees(angle)

        return angle

    def detect_bicep_curl(self, landmarks_mp):
        """
        Detect and count bicep curls based on arm angle (right arm).
        """
        try:
            # Get coordinates for right arm
            shoulder = [landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle
            angle = self.calculate_angle(shoulder, elbow, wrist)

            # Check bicep curl state
            if angle > 160:  # Arm extended
                current_position = "down"
            elif angle < 60: # Arm flexed
                current_position = "up"
            else:
                current_position = self.position

            # Count repetition when arm goes from up to down (or down to up, let's be consistent)
            # Standard count: curl up (flexion) then down (extension)
            current_time = time.time()
            if self.position == "up" and current_position == "down":
                if current_time - self.last_rep_time > SQUAT_REFRACTORY_PERIOD: # Generic refractory period for now
                    self.rep_counter += 1
                    self.last_rep_time = current_time

            self.position = current_position

            return angle, self.position
        except Exception as e:
            return 0, "Error"

    def detect_dumbbell_overhead_press(self, landmarks_mp):
        """
        Detect and count dumbbell overhead presses (right arm).
        """
        try:
            # Get coordinates for right arm
            shoulder = [landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate elbow angle
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)

            # Determine position based on elbow angle and wrist height relative to shoulder
            # Down: wrist near shoulder, elbow bent
            # Up: wrist above shoulder, arm extended
            is_arm_extended = elbow_angle > 150
            is_wrist_high = landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y < landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y - 0.05 # Wrist y is smaller when higher

            if is_arm_extended and is_wrist_high:
                current_position = "up"
            elif elbow_angle < 90 : # Arm bent, likely at shoulder level or lower
                current_position = "down"
            else:
                current_position = self.position

            # Count repetition: transition from down to up
            current_time = time.time()
            if self.position == "down" and current_position == "up":
                if current_time - self.last_rep_time > SQUAT_REFRACTORY_PERIOD: # Generic refractory period for now
                    self.rep_counter += 1
                    self.last_rep_time = current_time

            self.position = current_position
            return elbow_angle, self.position
        except Exception as e:
            return 0, "Error"

    def detect_squat(self, landmarks_mp):
        """
        Detect and count squats based on knee angle (left leg).
        """
        try:
            # Get coordinates for left leg
            hip = [landmarks_mp[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks_mp[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks_mp[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks_mp[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks_mp[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks_mp[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate knee angle
            angle = self.calculate_angle(hip, knee, ankle)

            # Check squat state
            if angle > 160: # Standing up
                current_position = "up"
            elif angle < 100: # Squatting down
                current_position = "down"
            else:
                current_position = self.position

            # Count repetition
            current_time = time.time()
            if self.position == "down" and current_position == "up":
                if current_time - self.last_rep_time > SQUAT_REFRACTORY_PERIOD:
                    self.rep_counter += 1
                    self.last_rep_time = current_time

            self.position = current_position
            return angle, self.position
        except Exception as e:
            return 0, "Error"

    def detect_lateral_raise(self, landmarks_mp):
        """
        Detect and count lateral raises (left arm).
        Arm abduction relative to torso.
        """
        try:
            # Landmarks for left arm and torso
            left_shoulder = [landmarks_mp[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks_mp[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks_mp[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks_mp[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks_mp[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks_mp[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks_mp[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks_mp[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]

            # Angle of the arm (shoulder-elbow) relative to a vertical line downwards from the shoulder
            # Define a point directly below the shoulder to form a reference for a 0-degree (arm down) angle
            vertical_ref_point = [left_shoulder[0], left_shoulder[1] + 0.5] # Y increases downwards
            
            # Calculate angle of abduction
            # Using shoulder, hip, and elbow to represent arm relative to torso might be more stable
            # angle = self.calculate_angle(left_hip, left_shoulder, left_elbow) 
            # Simpler: wrist y relative to shoulder y
            
            wrist_y = landmarks_mp[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
            shoulder_y = landmarks_mp[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            elbow_y = landmarks_mp[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y

            # Ensure arm is relatively straight
            arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            is_arm_straight = arm_angle > 150

            if not is_arm_straight: # If arm is bent, it's not a proper lateral raise
                # self.position remains, return a neutral angle or previous
                return 0, self.position if self.position else "Error"


            # Position detection based on wrist height relative to shoulder
            if wrist_y < shoulder_y - 0.05 : # Wrist is above shoulder height (adjust threshold as needed)
                current_position = "up"
            elif wrist_y > shoulder_y + 0.05: # Wrist is below shoulder (arm down)
                current_position = "down"
            else:
                current_position = self.position

            current_time = time.time()
            if self.position == "down" and current_position == "up":
                if current_time - self.last_rep_time > SQUAT_REFRACTORY_PERIOD: # Generic refractory period for now
                    self.rep_counter += 1
                    self.last_rep_time = current_time
            
            self.position = current_position
            # For display, we can return the arm straightness angle or a conceptual abduction angle if calculated
            return arm_angle, self.position 
        except Exception as e:
            return 0, "Error"

    def detect_torso_twist(self, landmarks_mp):
        """
        Detect torso twists. Assumes arms are held out to the sides (T-pose).
        Uses z-coordinates of wrists to determine twist.
        """
        try:
            # Get relevant landmarks
            l_shoulder = landmarks_mp[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_shoulder = landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_elbow = landmarks_mp[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            r_elbow = landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            l_wrist = landmarks_mp[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            r_wrist = landmarks_mp[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

            # Check if arms are held out (T-pose)
            left_arm_angle = self.calculate_angle(
                [l_shoulder.x, l_shoulder.y], [l_elbow.x, l_elbow.y], [l_wrist.x, l_wrist.y])
            right_arm_angle = self.calculate_angle(
                [r_shoulder.x, r_shoulder.y], [r_elbow.x, r_elbow.y], [r_wrist.x, r_wrist.y])

            # Check if wrists are roughly at shoulder height
            left_wrist_at_shoulder_height = abs(l_wrist.y - l_shoulder.y) < 0.15
            right_wrist_at_shoulder_height = abs(r_wrist.y - r_shoulder.y) < 0.15

            if not (left_arm_angle > 150 and right_arm_angle > 150 and \
                    left_wrist_at_shoulder_height and right_wrist_at_shoulder_height):
                # Not in T-pose, so can't reliably detect twist this way
                # Keep previous position or set to a neutral waiting state
                return 0, self.position if self.position else "Hold T-Pose"


            # Use z-coordinates of wrists for twist detection
            # z is depth: smaller z is closer to camera
            diff_z_wrists = l_wrist.z - r_wrist.z
            
            twist_threshold = 0.15 # Tunable parameter for twist sensitivity

            current_position = self.position
            if diff_z_wrists < -twist_threshold: # Left wrist significantly forward
                current_position = "left_twist"
            elif diff_z_wrists > twist_threshold: # Right wrist significantly forward
                current_position = "right_twist"
            elif abs(diff_z_wrists) < twist_threshold / 2: # Centered
                current_position = "center"
            
            # Count a rep when moving from one side to the other
            current_time = time.time()
            if (self.position == "left_twist" and current_position == "right_twist") or \
               (self.position == "right_twist" and current_position == "left_twist"):
                if current_time - self.last_rep_time > SQUAT_REFRACTORY_PERIOD: # Generic refractory period for now
                    self.rep_counter += 1
                    self.last_rep_time = current_time

            self.position = current_position
            # No specific angle to display, use 0 or a conceptual value
            return abs(diff_z_wrists * 100), self.position # Display scaled Z diff as "angle"

        except Exception as e:
            return 0, "Error"

    def process_frame_for_server(self, frame_bytes):
        """
        Process a single image frame (bytes) and return landmark data and exercise status.
        This method is designed to be called by the server.
        """
        try:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_np is None:
                return None, None, {"error": "Failed to decode image"}

            # No flipping here, assume client sends it in correct orientation or handles it
            # img_np = cv2.flip(img_np, 1) 
            
            rgb_frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            landmarks_data = []
            exercise_angle = 0
            current_exercise_position = "N/A"

            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmarks_data.append({
                        "id": idx,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility if landmark.HasField('visibility') else 0.0
                    })
                
                mp_landmarks = results.pose_landmarks.landmark

                if self.exercise_type == "bicep_curl":
                    exercise_angle, current_exercise_position = self.detect_bicep_curl(mp_landmarks)
                elif self.exercise_type == "squat":
                    exercise_angle, current_exercise_position = self.detect_squat(mp_landmarks)
                elif self.exercise_type == "lateral_raise":
                    exercise_angle, current_exercise_position = self.detect_lateral_raise(mp_landmarks)
                elif self.exercise_type == "overhead_press":
                    exercise_angle, current_exercise_position = self.detect_dumbbell_overhead_press(mp_landmarks)
                elif self.exercise_type == "torso_twist":
                    exercise_angle, current_exercise_position = self.detect_torso_twist(mp_landmarks)
            
            exercise_status = {
                "exercise_type": self.exercise_type,
                "reps": self.rep_counter,
                "position": current_exercise_position,
                "angle": int(exercise_angle) 
            }
            
            # For drawing on client, we can return the original frame with landmarks drawn by mp
            # Or client can draw based on raw landmarks_data
            # Let's return an annotated frame for now (optional for client to use)
            annotated_frame = img_np.copy() # Work on a copy
            if results.pose_landmarks:
                 self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            annotated_frame_bytes = base64.b64encode(buffer).decode('utf-8')

            return landmarks_data, exercise_status, annotated_frame_bytes

        except Exception as e:
            print(f"Error in process_frame_for_server: {e}")
            return None, {"error": str(e)}, None

    def process_frame(self, frame):
        """
        Process video frame and detect exercises (for local cv2.imshow loop).
        Kept for standalone testing if needed.
        """
        if frame is None:
            return frame
            
        frame_height, frame_width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            results = self.pose.process(rgb_frame)
            
            angle = 0
            current_position_display = "N/A"
            
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
                
                landmarks = results.pose_landmarks.landmark
                
                if self.exercise_type == "bicep_curl":
                    angle, current_position_display = self.detect_bicep_curl(landmarks)
                elif self.exercise_type == "squat":
                    angle, current_position_display = self.detect_squat(landmarks)
                elif self.exercise_type == "lateral_raise":
                    angle, current_position_display = self.detect_lateral_raise(landmarks)
                elif self.exercise_type == "overhead_press":
                    angle, current_position_display = self.detect_dumbbell_overhead_press(landmarks)
                elif self.exercise_type == "torso_twist":
                    angle, current_position_display = self.detect_torso_twist(landmarks)

                exercise_name_display = self.exercise_type.replace("_", " ").title()
                cv2.putText(frame, f'{exercise_name_display}: {self.rep_counter} reps', 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                if current_position_display and current_position_display not in ["Error", "Hold T-Pose"]:
                    status_color = (
                        0, 255, 0) if current_position_display == "up" or "right" in current_position_display else (
                        (0, 165, 255) if current_position_display == "down" or "left" in current_position_display else (200,200,200)
                    )

                    cv2.putText(frame, f'Status: {current_position_display.upper()}', 
                                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
                elif current_position_display == "Hold T-Pose":
                     cv2.putText(frame, 'Status: HOLD T-POSE', 
                                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                if self.exercise_type not in ["torso_twist"] and current_position_display != "Error":
                     cv2.putText(frame, f'Angle: {int(angle)}', 
                                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                elif self.exercise_type == "torso_twist" and current_position_display != "Error":
                     cv2.putText(frame, f'Twist Factor: {int(angle)}', 
                                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

            else:
                cv2.putText(frame, "No person detected. Stand in frame.", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        except Exception as e:
            cv2.putText(frame, f"Error: {str(e)[:30]}...",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.putText(frame, 'q: Quit | r: Reset',
                    (20, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (200, 200, 200), 2, cv2.LINE_AA)
        
        return frame
    
    def reset_counter(self):
        """
        Reset exercise counter.
        """
        self.rep_counter = 0
        self.position = None
        self.last_rep_time = 0
        
    def close(self):
        """
        Release resources.
        """
        if self.pose:
            self.pose.close()

def select_exercise():
    """
    Display a menu for exercise selection.
    """
    print("\n==== Exercise Tracker ====")
    print("Select an exercise to track:")
    print("1. Bicep Curls")
    print("2. Squats")
    print("3. Lateral Raises")
    print("4. Dumbbell Overhead Press")
    print("5. Torso Twists (T-Pose)")
    
    exercise_map = {
        '1': "bicep_curl",
        '2': "squat",
        '3': "lateral_raise",
        '4': "overhead_press",
        '5': "torso_twist"
    }
    
    while True:
        try:
            choice = input(f"\nEnter selection (1-{len(exercise_map)}): ")
            if choice in exercise_map:
                return exercise_map[choice]
            else:
                print("Invalid selection. Please try again.")
        except EOFError:
             print("\nExiting due to input error.")
             sys.exit(0)
        except Exception as e:
            print(f"Invalid input: {e}. Please enter a number.")

def main(): # For standalone testing
    exercise_type = select_exercise()
    if not exercise_type: return

    tracker = ExerciseTracker(exercise_type)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check if it's used by another app.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print(f"\nTracking {exercise_type.replace('_', ' ').title()}...")
    print("Ensure you are clearly visible in the camera frame.")
    print("Controls: 'q' to quit, 'r' to reset counter.")
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from webcam. Retrying...")
                time.sleep(0.5)
                continue
            
            processed_frame = tracker.process_frame(frame) # Uses the original process_frame
            cv2.imshow('Exercise Tracker', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                tracker.reset_counter()
                print("Counter reset.")
    
    except KeyboardInterrupt:
        print("\nExercise Tracker stopped by user.")
    except Exception as e:
        print(f"An error occurred during tracking: {e}")
    finally:
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        if 'tracker' in locals() and tracker: # Ensure tracker exists
            tracker.close()
        print("Exercise Tracker closed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error starting application: {e}")
        sys.exit(1)
