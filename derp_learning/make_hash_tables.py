import _pickle


def main():
    with open("pdb_res_pair_data.pickle", "rb") as inp:
        rp = _pickle.load(inp)

    print(rp)


if __name__ == "__main__":
    main()
