import pickle, os

testdatadir = os.path.join(os.path.dirname(__file__), 'testdata')

def save_test_data(fname, data):
   with open(os.path.join(testdatadir, fname + '.pickle'), 'wb') as out:
      pickle.dump(data, out)

def load_test_data(fname):
   with open(os.path.join(testdatadir, fname + '.pickle'), 'rb') as inp:
      return pickle.load(inp)
