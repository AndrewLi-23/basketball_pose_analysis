from pytubefix import YouTube
from pathlib import Path
import os

class VideoDownloader:
    def __init__(self, download_dir: str = None):
        """
        Initialize the video downloader.
        
        Args:
            download_dir (str): Directory to save downloaded videos. 
                              If None, uses the default data directory.
        """
        if download_dir is None:
            # Use default data directory
            download_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'data'
            )
        
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
    def get_video_length(self, yt: YouTube) -> float:
        """Get video length in minutes, returns 0 if unavailable."""
        try:
            length_seconds = float(yt.length)
            return length_seconds / 60 if length_seconds else 0
        except:
            return 0
        
    def download_video(self, url: str, duration_minutes: float = None) -> str:
        """
        Download a video from YouTube.
        
        Args:
            url (str): YouTube video URL
            duration_minutes (float): If specified, only download first X minutes of the video
            
        Returns:
            str: Path to the downloaded video file
        
        Raises:
            Exception: If video download fails
        """
        try:
            # Create YouTube object
            yt = YouTube(url)
            
            # Get video duration in minutes
            video_length_minutes = self.get_video_length(yt)
            if video_length_minutes > 0 and duration_minutes and duration_minutes < video_length_minutes:
                print(f"Note: Will analyze only first {duration_minutes} minutes of {video_length_minutes:.1f} minute video")
            
            # Get the highest resolution progressive stream
            # (streams that contain both video and audio)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            
            if not stream:
                raise Exception("No suitable video stream found")
            
            # Download the video
            print(f"Downloading: {yt.title}")
            video_path = stream.download(output_path=str(self.download_dir))
            
            # Rename the file to remove special characters
            clean_title = "".join(c for c in yt.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            new_path = os.path.join(self.download_dir, f"{clean_title}.mp4")
            
            # Rename if necessary
            if video_path != new_path:
                if os.path.exists(new_path):
                    os.remove(new_path)
                os.rename(video_path, new_path)
                video_path = new_path
            
            print(f"Download completed: {os.path.basename(video_path)}")
            return video_path
            
        except Exception as e:
            raise Exception(f"Failed to download video: {str(e)}")

if __name__ == "__main__":
    # Example usage
    downloader = VideoDownloader()
    # video_path = downloader.download_video("https://www.youtube.com/watch?v=EXAMPLE")
