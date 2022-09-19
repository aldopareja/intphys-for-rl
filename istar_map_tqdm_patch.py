# istarmap.py for Python 3.8+
import multiprocessing.pool as mpp

from tqdm import tqdm


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap

from multiprocessing import Pool, cpu_count


def array_apply(func, args, parallel, cpu_frac=1, use_tqdm=True, chunksize=None, total=None, unpack=True):
    progress = tqdm if use_tqdm else lambda x, total: x
    total = len(args) if total is None else total

    if not parallel:
        results = []
        for a in progress(args, total=total):
            r = func(*a) if unpack else func(a)
            results.append(r)
        return results
    else:
        num_processes = cpu_count() // cpu_frac
        chunksize = total // num_processes if chunksize is None else chunksize
        with Pool(num_processes) as p:
            applier = p.istarmap if unpack else p.imap
            return [a for a in progress(applier(func, args, chunksize), total=total)]

if __name__ == "__main__":
    def h(v):
        return v ** 2


    x = array_apply(h, [[e] for e in range(1000)], True)
    print(x)