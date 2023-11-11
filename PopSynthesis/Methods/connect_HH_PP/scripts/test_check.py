from multiprocessing import Process, Queue, Lock
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(q):
    q.put([42, None, 'hello'])

def multi_process():
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    print(q.get())
    p.join()


if __name__ ==  "__main__":
    multi_process()