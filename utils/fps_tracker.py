import matplotlib.pyplot as plt
import time
from collections import deque
import os
import csv


class FPSTracker:
    def __init__(self, max_samples=300, save_dir="saves"):
        self.fps_history = deque(maxlen=max_samples)
        self.time_history = deque(maxlen=max_samples)
        self.start_time = time.time()
        self.save_dir = save_dir
        self.csv_file = os.path.join(save_dir, 'fps_data.csv')
        self.plot_file = os.path.join(save_dir, 'fps_plot.png')
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def update(self, fps):
        """Add a new FPS measurement"""
        current_time = time.time() - self.start_time
        self.fps_history.append(fps)
        self.time_history.append(current_time)
    
    def save_csv(self):
        """Save FPS data to CSV file"""
        if len(self.fps_history) == 0:
            return
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time (seconds)', 'FPS'])
            for time_val, fps_val in zip(self.time_history, self.fps_history):
                writer.writerow([f'{time_val:.2f}', f'{fps_val:.1f}'])
        
        print(f"FPS data saved: {self.csv_file}")
    
    def save_plot(self):
        """Generate and save FPS plot"""
        if len(self.fps_history) < 2:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(list(self.time_history), list(self.fps_history), 'b-', linewidth=1)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('FPS', fontsize=12)
        plt.title('FPS Performance Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        min_fps = min(self.fps_history)
        max_fps = max(self.fps_history)
        
        stats_text = f'Avg: {avg_fps:.1f} | Min: {min_fps:.1f} | Max: {max_fps:.1f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        plt.savefig(self.plot_file, dpi=100)
        plt.close()
        print(f"FPS plot saved: {self.plot_file}")
    
    def finalize(self):
        """Save CSV and plot at the end of session"""
        self.save_csv()
        self.save_plot()
    
    def get_average_fps(self):
        """Get current average FPS"""
        if len(self.fps_history) == 0:
            return 0
        return sum(self.fps_history) / len(self.fps_history)
