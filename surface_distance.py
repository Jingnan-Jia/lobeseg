import numpy as np
from scipy.ndimage import morphology


def surfd(input1, input2, sampling=1, connectivity=1):
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    input1_erosion = morphology.binary_erosion(input_1, conn).astype(int)
    input2_erosion = morphology.binary_erosion(input_2, conn).astype(int)
    S = input_1 - input1_erosion
    Sprime = input_2 - input2_erosion
    S = S.astype(np.bool)
    Sprime = Sprime.astype(np.bool)

    vs = ~S
    vsprime = ~Sprime
    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)
    ds1 = np.ravel(dta[Sprime != 0])
    ds2 = np.ravel(dtb[S != 0])

    sds = np.concatenate([ds1, ds2])

    return sds

# test_seg = np.array([[1,1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1,1],
#                    [1,1,0,0,0,1,1,1,1,1],
#                    [1,1,0,0,0,1,1,1,1,1],
#                    [1,1,0,0,0,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1,1]])
#
# GT_seg = np.array([[1,1,1,1,1,1,1,1,1,1],
#                    [1,0,0,0,1,1,1,1,1,1],
#                    [1,0,0,0,1,1,1,1,1,1],
#                    [1,0,0,0,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1,1]])


GT_seg = np.array([[1,1,1],
                   [0,0,0],
                   [0,0,0]])
test_seg = np.array([[0,0,0],
                     [1,1,0],
                     [0,0,1]])
surface_distance = surfd(test_seg, GT_seg, [1, 1],1)

print(surface_distance)

msd = surface_distance.mean()
rms = np.sqrt((surface_distance ** 2).mean())
hd = surface_distance.max()

print('ok')