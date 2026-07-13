import webview
import subprocess
import threading
import time
import socket
import sys
import os

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def run_streamlit():
    """Runs the Streamlit app in a background process."""
    if getattr(sys, 'frozen', False):
        # If running as a compiled PyInstaller bundle
        cmd = [
            sys.executable, "run_streamlit_internal"
        ]
    else:
        # If running normally via python
        cmd = [
            sys.executable, "-m", "streamlit", "run", "combined_app.py",
            "--server.headless", "true",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ]
    subprocess.Popen(cmd)

if __name__ == '__main__':
    # PYINSTALLER TRICK: Intercept subprocess call to run Streamlit
    if getattr(sys, 'frozen', False) and len(sys.argv) > 1 and sys.argv[1] == "run_streamlit_internal":
        from streamlit.web import cli as stcli
        # The combined_app.py script will be extracted in the PyInstaller temp folder
        script_path = os.path.join(sys._MEIPASS, "combined_app.py")
        sys.argv = ["streamlit", "run", script_path, "--server.headless", "true", "--server.port", "8501", "--global.developmentMode", "false"]
        sys.exit(stcli.main())

    # Normal desktop wrapper code
    port = 8501
    
    # Check if streamlit is already running on the port
    if not is_port_in_use(port):
        print("Starting Streamlit server...")
        server_thread = threading.Thread(target=run_streamlit, daemon=True)
        server_thread.start()
        
        print("Waiting for server to initialize...")
        max_retries = 30
        while not is_port_in_use(port) and max_retries > 0:
            time.sleep(1)
            max_retries -= 1
            
        if max_retries == 0:
            print("Error: Could not start Streamlit server.")
            sys.exit(1)
    else:
        print(f"Streamlit server already running on port {port}.")

    print("Opening Desktop Window...")
    # Create the desktop window pointing to the local Streamlit server
    window = webview.create_window(
        'FEBUS DTSS Viewer & Assistant',
        f'http://localhost:{port}',
        width=1280,
        height=850,
        min_size=(1000, 700)
    )
    
    # Start the webview application
    webview.start()
