
from scipy.spatial.distance import directed_hausdorff
import numpy as np

a = np.ones((2, 3))

b = a

c = directed_hausdorff(a, b)

d = directed_hausdorff(b, a)

print(c)

print(d)

task = 'lobe'

for phase in ['valid']:
    if task == 'lobe':
        labels = [0, 4, 5, 6, 7, 8]
        data_dir = '/data/jjia/mt/data/lobe/' + phase + '/ori_ct/GLUCOLD'
        gdth_dir = '/data/jjia/mt/data/lobe/' + phase + '/gdth_ct/GLUCOLD'
        preds_dir = '/data/jjia/e2e_new/results/lobe/preds/' + phase + '/GLUCOLD/' + \
                    model_name.split('/')[-1][:8]
        # print (preds_dir)
    elif task == 'vessel':
        labels = [0, 1]
        data_dir = '/data/jjia/mt/data/vessel/' + phase + '/ori_ct'
        gdth_dir = '/data/jjia/mt/data/vessel/' + phase + '/gdth_ct'
        preds_dir = '/data/jjia/e2e_new/results/vessel/' + \
                    model_name.split('/')[-1][:8]
