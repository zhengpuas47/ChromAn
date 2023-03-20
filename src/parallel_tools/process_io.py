import time

def time_full():
    """Funn format of time, for example: '2023/3/20, 14:52:04'"""
    f"{time.localtime().tm_year}/{time.localtime().tm_mon}/{time.localtime().tm_mday}, {time.localtime().tm_hour:02d}:{time.localtime().tm_min:02d}:{time.localtime().tm_sec:02d}"