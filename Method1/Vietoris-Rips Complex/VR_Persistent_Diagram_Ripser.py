import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics.pairwise import pairwise_distances
from scipy import sparse
import tadasets
from ripser import ripser
from persim import plot_diagrams

def makeSparseDM(X, thresh):
    N = X.shape[0]
    D = pairwise_distances(X, metric='euclidean')
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    I = I[D <= thresh]
    J = J[D <= thresh]
    V = D[D <= thresh]
    return sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()

if __name__ == '__main__':
    data = [[198, 408], [202, 463], [212, 517], [226, 570], [248, 620], [280, 662], [320, 698], [364, 726], [411, 733], [459, 726], [503, 698], [543, 663], [575, 620], [595, 570], [607, 516], [614, 461], [616, 405], [229, 355], [255, 327], [294, 317], [334, 323], [370, 341], [442, 339], [479, 320], [519, 314], [558, 323], [584, 351], [406, 401], [407, 442], [407, 482], [407, 523], [370, 553], [389, 558], [408, 563], [428, 557], [446, 552], [270, 408], [295, 391], [327, 392], [353, 418], [324, 425], [292, 424], [461, 417], [486, 391], [518, 389], [543, 406], [522, 422], [489, 424], [339, 624], [366, 611], [391, 601], [408, 607], [425, 601], [451, 613], [480, 626], [451, 648], [426, 658], [408, 660], [389, 658], [365, 647], [353, 625], [391, 625], [408, 627], [425, 625], [466, 627], [425, 626], [408, 628], [390, 625]]
    data = np.array(data)
    thresh = 1000000000
    D = makeSparseDM(data, thresh)
    results1 = ripser(D, distance_matrix=True)
    print("%i edges added in the sparse filtration"%results1['num_edges'])

    plt.title("Sparse Filtration")
    plot_diagrams(results1['dgms'], show=False)
    plt.tight_layout()
    plt.show()