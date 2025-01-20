from pose_analyzer import BasketballPoseAnalyzer
from video_downloader import VideoDownloader
import os
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Basketball Pose Analysis')
    parser.add_argument('--video', type=str, help='Path to local video file')
    parser.add_argument('--url', type=str, help='YouTube video URL')
    parser.add_argument('--duration', type=float, help='Duration in minutes to analyze (default: entire video)')
    args = parser.parse_args()
    
    # Initialize the pose analyzer
    analyzer = BasketballPoseAnalyzer()
    
    # Get video path
    video_path = None
    
    if args.url:
        # Download YouTube video
        print("Downloading YouTube video...")
        downloader = VideoDownloader()
        video_path = downloader.download_video(args.url, args.duration)
    elif args.video:
        video_path = args.video
    else:
        # Use default video path
        video_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        video_path = os.path.join(video_dir, 'sample_video.mp4')
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print("Please provide a valid video file path or YouTube URL")
        return
    
    print(f"Processing video: {video_path}")
    print("Press 'q' to quit the video analysis")
    
    # Process the video
    analyzer.process_video(video_path, args.duration)

if __name__ == "__main__":
    main()
