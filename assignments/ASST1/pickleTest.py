import pickle




pickleFile = open("params.pickle", 'rb')
n_hidden = pickle.load(pickleFile)
w1 = pickle.load(pickleFile)
w2 = pickle.load(pickleFile)
lambdaval = pickle.load(pickleFile)
pickleFile.close()

print ("The n_hidden is :")
print (n_hidden)

print ("The w1 is :")
print (w1)

print ("The w2 is :")
print (w2)

print ("The lambdaval is :")
print (lambdaval)

