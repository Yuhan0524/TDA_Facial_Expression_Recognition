import numpy as np
import numpy.random as rd
import gudhi as gd
import gudhi.representations
import matplotlib.pyplot as plt
import random
rd.seed(1)

dgms_normal = []
normal_pcs = [[128, 120], [128, 147], [132, 174], [138, 200], [147, 226], [160, 250], [176, 271], [195, 287], [221, 292], [247, 288], [270, 274], [289, 255], [305, 231], [316, 206], [322, 178], [327, 149], [328, 120], [144, 104], [156, 94], [173, 92], [191, 96], [207, 103], [238, 103], [256, 95], [274, 90], [293, 92], [307, 102], [222, 120], [221, 138], [220, 156], [219, 174], [200, 185], [210, 190], [221, 193], [232, 190], [243, 185], [162, 120], [172, 114], [186, 114], [197, 123], [185, 125], [172, 125], [252, 122], [263, 113], [276, 112], [287, 119], [277, 124], [264, 124], [186, 225], [198, 218], [211, 214], [221, 217], [231, 215], [245, 219], [260, 226], [245, 238], [231, 243], [220, 244], [210, 243], [197, 238], [192, 226], [211, 224], [221, 225], [231, 224], [254, 226], [231, 230], [220, 231], [210, 230]]
rips = gudhi.RipsComplex(points=normal_pcs).create_simplex_tree(max_dimension=1)
rips.compute_persistence()
dgms_normal.append(rips.persistence_intervals_in_dimension(0))
gd.plot_persistence_barcode(dgms_normal[0])
plt.show()



