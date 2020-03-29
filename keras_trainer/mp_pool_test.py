from multiprocessing import Pool
from collections import deque

def f(val, val2):
    return 10*val*val2

if __name__ == "__main__":
    p = Pool(processes=8)
    
    res = p.starmap(f, deque([1, 2, 3,4, 5, 6, 7,8 , 9, 10]), deque([1, 2, 3,4, 5, 6, 7,8 , 9, 10]))
    print(res)