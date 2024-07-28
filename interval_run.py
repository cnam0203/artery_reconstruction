import subprocess
import time

def run_command():
    while True:
        # Run the command
        process = subprocess.run(['python', 'diameter_analyze.py'])
        
        # Check if the command completed successfully
        if process.returncode == 0:
            print("Command finished successfully, re-running...")
        else:
            print(f"Command failed with return code {process.returncode}. Re-running...")
        
        # Optionally, wait for a short period before re-running
        time.sleep(1)

if __name__ == "__main__":
    run_command()