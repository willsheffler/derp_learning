import os
import _pickle
import random
import numpy as np


def load_which_takes_forever():
    with open("pdb_res_pair_data.pickle", "rb") as inp:
        return _pickle.load(inp)


if hasattr(os, "a_very_unique_name"):
    rp = os.a_very_unique_name
else:
    rp = load_which_takes_forever()
    os.a_very_unique_name = rp


def main():

    pdbs = rp.pdb[np.random.choice(len(rp.pdb), 10)].values
    print(type(pdbs[0]))
    print("foo bar ars")
    print(np.sum(rp.pdb in pdbs))


if __name__ == "__main__":
    main()
