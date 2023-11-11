from multiprocessing import Process, Queue, , Lock
import os


def test(infer_model, rela):
    print(f"Doing {rela}")
    evidences = [State(var='age_main', state=96), State(var='sex_main', state='F'), State(var='persinc_main', state='$400-599 p.w.'), State(var='nolicence_main', state='Some Licence'), State(var='anywork_main', state='N')]
    syn = infer_model.rejection_sample(evidence=evidences, size=2, show_progress=True)
    remove_cols = [x for x in syn.columns if "_main" in x]
    syn = syn.drop(columns=remove_cols)
    print(syn)


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