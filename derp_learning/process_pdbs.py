import sys
import os
import glob
import _pickle
import concurrent.futures
import threading
import random

import tqdm
import pyrosetta

from derp_learning.pdbdata import pdbdata

pyrosetta.init("-beta_nov16 -mute all")


class InProcessExecutor:
    def __init__(self, *args, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, fn, *args, **kw):
        return NonFuture(fn, *args, **kw)

    # def map(self, func, *iterables):
    # return map(func, *iterables)
    # return (NonFuture(func(*args) for args in zip(iterables)))


class NonFuture:
    def __init__(self, fn, *args, dummy=None, **kw):
        self.fn = fn
        self.dummy = not callable(fn) if dummy is None else dummy
        self.args = args
        self.kw = kw
        self._condition = threading.Condition()
        self._state = "FINISHED"
        self._waiters = []

    def result(self):
        if self.dummy:
            return self.fn
        return self.fn(*self.args, **self.kw)


def pdbs_from_file(fname):
    # print("pdbs_from_file", fname)
    if fname.endswith((".pdb", ".pdb.gz")):
        return [fname]
    else:
        print("ERROR don't know now to handle file", fname)
        return []


def pdbs_from_glob(pattern):
    # print("pdbs_from_glob", pattern)
    pdbs = list()
    for path in glob.glob(pattern):
        if os.path.isdir(path):
            pdbs += pdbs_from_directory(path)
        else:
            pdbs += pdbs_from_file(path)
    return pdbs


def pdbs_from_directory(path):
    # print("pdbs_from_directory", path)
    pdbs = list()
    pdbs += pdbs_from_glob(path + "/*.pdb")
    pdbs += pdbs_from_glob(path + "/*.pdb.gz")
    return pdbs


def pdbs_from_arg_list(args):
    # print("pdbs_from_arg_list", args)
    pdbs = list()
    for arg in args:
        if os.path.isdir(arg):
            pdbs += pdbs_from_directory(arg)
        elif arg.count("*"):
            pdbs += pdbs_from_glob(arg)
        else:
            pdbs += pdbs_from_file(arg)
    return pdbs


def process_pdb(fname, storage_dir="."):
    try:
        # print(fname, "read pose")
        pose = pyrosetta.pose_from_file(fname)
    except:
        # print("ERROR pyrosetta can't read", fname)
        return fname

    try:
        dat = pdbdata(pose, fname)
    except Exception as e:
        print("ERROR can't process pose from", fname)
        print("   ", e)
        return fname

    outfile = fname.lstrip("/") + ".pickle"
    outfile = os.path.join(storage_dir, outfile)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "wb") as out:
        _pickle.dump(dat, out)


def process_parallel(pdbs, max_workers):
    exe = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    # exe = InProcessExecutor()
    errors = list()
    with exe as pool:
        futures = list()
        for pdb in pdbs:
            futures.append(pool.submit(process_pdb, pdb))
        iter = concurrent.futures.as_completed(futures)
        iter = tqdm.tqdm(iter, "processing pdb files", total=len(futures))
        for f in iter:
            err = f.result()
            if err is not None:
                errors.append(err)


def main():
    pdbs = pdbs_from_arg_list(a for a in sys.argv[1:] if a[0] != "-")
    random.shuffle(pdbs)
    print("pdbs", len(pdbs))
    errors = process_parallel(pdbs, max_workers=8)
    if errors:
        print("pdb files not processed:")
        for e in errors:
            print("   ", e)
    else:
        print("processed all files")


if __name__ == "__main__":
    main()
