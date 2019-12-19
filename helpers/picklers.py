import pickle

# Pickle functions
def pickle_me(weights, name):
    filehandler = open(name,"wb")
    pickle.dump(weights,filehandler)
    filehandler.close()

def load_me(name):
    file = open(name,'rb')
    object_file = pickle.load(file)
    file.close()
    return object_file