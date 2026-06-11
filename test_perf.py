import time
import app
import pandas as pd

t0 = time.time()
meta = app.get_sensor_metadata(app.HDF5_FILE_PATH)
t1 = time.time()
print(f"get_sensor_metadata took {t1-t0:.4f} seconds")

core = app.LlmBotCore(app.HDF5_FILE_PATH)
t2 = time.time()
dist, temp, defo, idx = core.query_profile('temperature', 59)
t3 = time.time()
print(f"query_profile took {t3-t2:.4f} seconds")

df = pd.DataFrame({"Distance": dist, "Temp": temp})
t4 = time.time()
print(f"dataframe creation took {t4-t3:.4f} seconds")
