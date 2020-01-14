import numpy as np
import networkx as nx

def randomNPGraph(n, p, diagonal = True, undirected = True):
    adjM = np.random.binomial(1, p, (n, n))
    if diagonal:
        for i in range(len(adjM)):
            adjM[i,i] = 1
    if undirected:
        xy = np.mgrid[0:n:1, 0:n:1].reshape(2,-1).T.reshape(n,n,2)
        adjM = np.where(xy[..., 1] > xy[..., 0], adjM, adjM.T)
    return adjM

def randomGraphColoring(n, m):
    coloring = np.zeros((n,m))
    indices = list((np.arange(n), np.random.randint(m, size=n)))
    coloring[tuple(indices)] = 1
    return coloring

def checkGraphColoringError(adjM, coloring):
    neighbours = [np.where(adjM[i] == 1) for i in range(len(adjM))]
    errors=np.array([[np.sum(coloring[i]*coloring[j]) if i!=j and j in neighbours[i][0] else 0 for j in range(len(adjM)) ] for i in range(len(adjM))])
    sum_of_errors = np.sum(errors)/2
    return sum_of_errors

def checkIfGraphConnected(adjM):
    G = nx.from_numpy_matrix(adjM)
    return nx.is_connected(G)

n = 7 # nodes
m = 4 # colors
p = 0.4 # edge probability
NPGraph = randomNPGraph(n, p)
coloring = randomGraphColoring(n, m)

connected = checkIfGraphConnected(NPGraph)
coloringError = checkGraphColoringError(NPGraph, coloring)

print(connected)
print(coloringError)
