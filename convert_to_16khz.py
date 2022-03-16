from concurrent.futures import ProcessPoolExecutor
import os
def downsample(file):
    if os.path.exists(os.path.join("/home/rupnik/macocu/task8/data_16000", file)):
        return
    os.system(f"ffmpeg -i /home/rupnik/macocu/task8/data/{file} -ar 16000 /home/rupnik/macocu/task8/data_16000/{file}")
with ProcessPoolExecutor(max_workers=100) as executor:
    executor.map(downsample, os.listdir("/home/rupnik/macocu/task8/data"))