import subprocess
import time
from datetime import datetime

def run_kubectl_get_pods():
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\nCurrent Time: {current_time}")
        subprocess.run(["kubectl", "get", "pods"])
        time.sleep(3)  # Wait for 3 seconds before running the command again

run_kubectl_get_pods()
