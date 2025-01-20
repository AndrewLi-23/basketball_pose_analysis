import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

@dataclass
class MovementMetrics:
    frame_number: int
    elbow_angle: float
    knee_angle: float
    jump_height: float
    ball_hand_distance: Optional[float]
    movement_type: str

class BasketballPoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.metrics_history: List[MovementMetrics] = []
        self.base_height = None
        
    def calculate_angle(self, point1, point2, point3) -> float:
        """Calculate the angle between three points."""
        if not all(point.visibility > 0.5 for point in [point1, point2, point3]):
            return 0.0
            
        a = np.array([point1.x, point1.y])
        b = np.array([point2.x, point2.y])
        c = np.array([point3.x, point3.y])
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
        
    def detect_shooting_form(self, landmarks) -> Tuple[float, str]:
        """Analyze shooting form based on elbow and shoulder alignment."""
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        if 80 <= elbow_angle <= 100:
            return elbow_angle, "Good shooting form"
        else:
            return elbow_angle, "Adjust elbow angle"
            
    def analyze_dribbling(self, landmarks) -> Tuple[float, str]:
        """Analyze dribbling mechanics based on knee bend and hand position."""
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        if 130 <= knee_angle <= 150:
            return knee_angle, "Good dribbling stance"
        else:
            return knee_angle, "Adjust knee bend"
            
    def measure_jump_height(self, landmarks) -> Tuple[float, str]:
        """Measure vertical jump height based on hip position."""
        hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        
        if self.base_height is None:
            self.base_height = hip.y
            return 0.0, "Calibrating height"
            
        # Convert to relative height in pixels
        current_height = (self.base_height - hip.y) * 100
        
        if current_height > 10:
            return current_height, "Jumping"
        else:
            return current_height, "Standing"
            
    def process_video(self, video_path: str, duration_minutes: float = None):
        """
        Process a basketball video and extract pose information
        
        Args:
            video_path (str): Path to the video file
            duration_minutes (float): Number of minutes to analyze from the start of the video.
                                    If None, analyzes the entire video.
        """
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Calculate total frames to process based on duration
        total_frames = None
        if duration_minutes is not None:
            total_frames = int(duration_minutes * 60 * fps)
            print(f"Analyzing first {duration_minutes} minutes ({total_frames} frames)")
        
        # Create output video writer
        output_path = str(Path(video_path).parent / f"analyzed_{Path(video_path).name}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_number = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_number += 1
            
            # Check if we've reached the duration limit
            if total_frames is not None and frame_number > total_frames:
                print(f"Reached specified duration of {duration_minutes} minutes")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect poses
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                
                # Analyze movements
                elbow_angle, shooting_feedback = self.detect_shooting_form(results.pose_landmarks)
                knee_angle, dribbling_feedback = self.analyze_dribbling(results.pose_landmarks)
                jump_height, jump_feedback = self.measure_jump_height(results.pose_landmarks)
                
                # Store metrics
                metrics = MovementMetrics(
                    frame_number=frame_number,
                    elbow_angle=elbow_angle,
                    knee_angle=knee_angle,
                    jump_height=jump_height,
                    ball_hand_distance=None,  # TODO: Implement ball detection
                    movement_type=f"{shooting_feedback} | {dribbling_feedback} | {jump_feedback}"
                )
                self.metrics_history.append(metrics)
                
                # Display feedback on frame
                cv2.putText(frame, f"Shooting: {shooting_feedback}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Dribbling: {dribbling_feedback}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Jump: {jump_feedback}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write the frame to output video
            out.write(frame)
            
            # Display the frame
            cv2.imshow('Basketball Pose Analysis', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Generate analysis report
        self.generate_analysis_report(output_path)
        
    def generate_analysis_report(self, video_path: str):
        """Generate a detailed analysis report with graphs and statistics."""
        if not self.metrics_history:
            return
            
        # Convert metrics to DataFrame
        df = pd.DataFrame([vars(m) for m in self.metrics_history])
        
        # Create report directory
        report_dir = Path(video_path).parent / "analysis_reports"
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"analysis_report_{timestamp}"
        
        # Plot metrics
        plt.figure(figsize=(15, 10))
        
        # Elbow angle plot
        plt.subplot(3, 1, 1)
        plt.plot(df['frame_number'], df['elbow_angle'])
        plt.title('Elbow Angle Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        
        # Knee angle plot
        plt.subplot(3, 1, 2)
        plt.plot(df['frame_number'], df['knee_angle'])
        plt.title('Knee Angle Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        
        # Jump height plot
        plt.subplot(3, 1, 3)
        plt.plot(df['frame_number'], df['jump_height'])
        plt.title('Jump Height Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Relative Height')
        
        plt.tight_layout()
        plt.savefig(f"{report_path}_graphs.png")
        plt.close()
        
        # Generate statistics report
        stats = {
            'avg_elbow_angle': df['elbow_angle'].mean(),
            'max_jump_height': df['jump_height'].max(),
            'avg_knee_angle': df['knee_angle'].mean()
        }
        
        with open(f"{report_path}_stats.txt", 'w') as f:
            f.write("Basketball Movement Analysis Statistics\n")
            f.write("=====================================\n\n")
            f.write(f"Average Elbow Angle: {stats['avg_elbow_angle']:.2f} degrees\n")
            f.write(f"Maximum Jump Height: {stats['max_jump_height']:.2f}\n")
            f.write(f"Average Knee Angle: {stats['avg_knee_angle']:.2f} degrees\n")

if __name__ == "__main__":
    analyzer = BasketballPoseAnalyzer()
    # Example usage:
    # analyzer.process_video("path/to/your/basketball/video.mp4")
