import time

def time_full():
    """Funn format of time, for example: '2023/3/20, 14:52:04'"""
    return f"{time.localtime().tm_year}/{time.localtime().tm_mon}/{time.localtime().tm_mday}, {time.localtime().tm_hour:02d}:{time.localtime().tm_min:02d}:{time.localtime().tm_sec:02d}"


## TODO
# Job 1: preprocess reference images, save
# Job 1-2: segmentation, submit to GPU node

# Job 2: preprocess other images, save
# Job 3: spot fitting
