from gtda.diagrams import Amplitude
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
import matplotlib.pyplot as plt
import numpy as np

data = [[198, 408], [202, 463], [212, 517], [226, 570], [248, 620], [280, 662], [320, 698], [364, 726], [411, 733],
        [459, 726], [503, 698], [543, 663], [575, 620], [595, 570], [607, 516], [614, 461], [616, 405], [229, 355],
        [255, 327], [294, 317], [334, 323], [370, 341], [442, 339], [479, 320], [519, 314], [558, 323], [584, 351],
        [406, 401], [407, 442], [407, 482], [407, 523], [370, 553], [389, 558], [408, 563], [428, 557], [446, 552],
        [270, 408], [295, 391], [327, 392], [353, 418], [324, 425], [292, 424], [461, 417], [486, 391], [518, 389],
        [543, 406], [522, 422], [489, 424], [339, 624], [366, 611], [391, 601], [408, 607], [425, 601], [451, 613],
        [480, 626], [451, 648], [426, 658], [408, 660], [389, 658], [365, 647], [353, 625], [391, 625], [408, 627],
        [425, 625], [466, 627], [425, 626], [408, 628], [390, 625]]
data = np.array(data).reshape(1,68,2)
VR_persistence = VietorisRipsPersistence(n_jobs=-1)
PD = VR_persistence.fit_transform(data)

#following for wasserstein and bottleneck
# Reshape single diagram to (n_samples, n_features, 3) format
diagram = PD[0][None, :, :]
print(Amplitude(metric='wasserstein').fit_transform(diagram))
print(Amplitude(metric='bottleneck').fit_transform(diagram))



