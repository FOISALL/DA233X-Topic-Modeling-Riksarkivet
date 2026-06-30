

import subprocess
import sys

# List scripts 
scripts = [
    "runbertopicbase.py",
    "runbertopicnoiseparams.py",    
    "runbertopicreassignment.py",      
    "runctm.py",
    "runetm.py",
    "runlda.py"
]

for script in scripts:
    print(f"\n{'='*40}")
    print(f"LAUNCHING: {script}")
    print(f"{'='*40}\n")
    
    # This runs the script and waits for it to finish before moving to the next
    result = subprocess.run([sys.executable, script])
    
    if result.returncode != 0:
        print(f"CRITICAL ERROR: {script} failed. Stopping the pipeline.")
        break

print("\nAll experiments finished successfully!")
