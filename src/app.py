import streamlit as st
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from pose_analyzer import BasketballPoseAnalyzer
from video_downloader import VideoDownloader
import cv2
from PIL import Image
import numpy as np

st.set_page_config(page_title="Basketball Pose Analysis", layout="wide")

def load_latest_results(analysis_dir):
    """Load the most recent analysis results."""
    if not os.path.exists(analysis_dir):
        return None, None, None
        
    # Get the latest report files
    report_files = list(Path(analysis_dir).glob("analysis_report_*"))
    if not report_files:
        return None, None, None
        
    # Group files by timestamp
    reports_by_timestamp = {}
    for file in report_files:
        timestamp = file.stem.split('analysis_report_')[1].split('_')[0]
        reports_by_timestamp.setdefault(timestamp, []).append(file)
        
    if not reports_by_timestamp:
        return None, None, None
        
    # Get the latest timestamp
    latest_timestamp = max(reports_by_timestamp.keys())
    latest_files = reports_by_timestamp[latest_timestamp]
    
    stats_file = None
    graphs_file = None
    
    for file in latest_files:
        if file.name.endswith('_stats.txt'):
            stats_file = file
        elif file.name.endswith('_graphs.png'):
            graphs_file = file
            
    return latest_timestamp, stats_file, graphs_file

def main():
    st.title("Basketball Pose Analysis")
    
    # Sidebar for input
    st.sidebar.header("Analysis Input")
    input_type = st.sidebar.radio("Select Input Type", ["YouTube URL", "Upload Video"])
    
    duration = st.sidebar.number_input(
        "Analysis Duration (minutes)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Duration of video to analyze in minutes"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Video Input")
        if input_type == "YouTube URL":
            url = st.text_input("Enter YouTube URL")
            process_button = st.button("Process Video")
            
            if process_button and url:
                with st.spinner("Downloading and analyzing video..."):
                    try:
                        # Download and analyze video
                        downloader = VideoDownloader()
                        video_path = downloader.download_video(url, duration)
                        
                        analyzer = BasketballPoseAnalyzer()
                        analyzer.process_video(video_path, duration)
                        
                        st.success("Analysis complete!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        
        else:
            uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
            process_button = st.button("Process Video")
            
            if process_button and uploaded_file:
                with st.spinner("Analyzing video..."):
                    try:
                        # Save uploaded file
                        video_path = os.path.join("data", uploaded_file.name)
                        with open(video_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            
                        # Analyze video
                        analyzer = BasketballPoseAnalyzer()
                        analyzer.process_video(video_path, duration)
                        
                        st.success("Analysis complete!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Results section
    st.header("Analysis Results")
    
    # Load latest results
    analysis_dir = os.path.join("data", "analysis_reports")
    timestamp, stats_file, graphs_file = load_latest_results(analysis_dir)
    
    if timestamp and stats_file and graphs_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Statistics")
            with open(stats_file, 'r') as f:
                stats_text = f.read()
            st.text(stats_text)
            
        with col2:
            st.subheader("Movement Graphs")
            st.image(str(graphs_file))
            
        # Find and display the analyzed video
        video_files = list(Path("data").glob("analyzed_*.mp4"))
        if video_files:
            latest_video = max(video_files, key=os.path.getctime)
            st.subheader("Analyzed Video")
            st.video(str(latest_video))
    else:
        st.info("No analysis results available. Please process a video first.")
        
    # Additional insights
    if timestamp and stats_file:
        st.header("Movement Insights")
        
        # Parse stats file for insights
        with open(stats_file, 'r') as f:
            stats_lines = f.readlines()
            
        stats_dict = {}
        for line in stats_lines:
            if ':' in line:
                key, value = line.split(':')
                try:
                    stats_dict[key.strip()] = float(value.split()[0])
                except:
                    continue
        
        if stats_dict:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Shooting Form",
                    f"{stats_dict.get('Average Elbow Angle', 0):.1f}°",
                    "Optimal: 80-100°"
                )
                
            with col2:
                st.metric(
                    "Dribbling Stance",
                    f"{stats_dict.get('Average Knee Angle', 0):.1f}°",
                    "Optimal: 130-150°"
                )
                
            with col3:
                st.metric(
                    "Max Jump Height",
                    f"{stats_dict.get('Maximum Jump Height', 0):.1f}",
                    "Relative height"
                )

if __name__ == "__main__":
    main()
