#!/usr/bin/env python3
import subprocess
import time
import os

# Clean up old files
os.system("rm -f /home/vedant/suspicious_ai/saves/processed_video_*.mkv")

# Run main.py with logging
starttime = time.time()
proc = subprocess.Popen(
    ["conda", "run", "-n", "project_env", "python", "main.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Monitor output
print("Starting video export process...\n")
try:
    for line in iter(proc.stdout.readline, ''):
        if line:
            if any(x in line for x in ['Starting', 'exported', 'completed', 'Error', 'error']):
                print(f"[{time.time()-starttime:.1f}s] {line.strip()}")
        
        # Let it run for up to 150 seconds
        if time.time() - starttime > 150:
            proc.terminate()
            print(f"\n[{time.time()-starttime:.1f}s] Timeout - terminating process")
            break
    
    proc.wait(timeout=5)
    print(f"\n[{time.time()-starttime:.1f}s] Process completed")
except Exception as e:
    proc.kill()
    print(f"Process error: {e}")

elapsed = time.time() - starttime
print(f"\nTotal elapsed: {elapsed:.1f}s")

# Check output file
import glob
mkv_files = glob.glob("/home/vedant/suspicious_ai/saves/processed_video_*.mkv")
if mkv_files:
    print(f"\n✓ Export file created:")
    for f in mkv_files:
        size = os.path.getsize(f) / (1024*1024)
        print(f"  {os.path.basename(f)} ({size:.1f} MB)")
else:
    print("\n✗ No export file found")
