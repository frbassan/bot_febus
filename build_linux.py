import os
import streamlit
import PyInstaller.__main__

# Find where streamlit is installed to copy the "static" (frontend) folder
streamlit_path = os.path.dirname(streamlit.__file__)

PyInstaller.__main__.run([
    'run_desktop.py',
    '--name=FEBUS_Viewer',
    '--onefile',
    '--windowed',
    f'--add-data={streamlit_path}:streamlit',
    '--add-data=combined_app.py:.',
    '--copy-metadata=streamlit',
    '--copy-metadata=google-generativeai',
    '--hidden-import=streamlit',
    '--hidden-import=google.generativeai',
    '--hidden-import=plotly',
    '--hidden-import=pandas',
    '--hidden-import=h5py'
])
