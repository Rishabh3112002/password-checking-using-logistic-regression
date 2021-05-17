def sigmoid(z):
    g = np.zeros(len(z))
    for i in range(0, len(z)):
        g = np.ones(len(z))*np.transpose(1/(1.0+np.exp(-z)))
        g = np.transpose(g)
    print(g)