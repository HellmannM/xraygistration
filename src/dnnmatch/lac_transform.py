from enum import IntEnum
import numpy as np
import multiprocessing

class TUBE_POTENTIAL(IntEnum):
    TB13000EV = 0
    TB13500EV = 1
    TB14000EV = 2

LAC_LUT = np.array([
[ # 13000 eV
[0,    0.0],
[21,   0.000289],
[282,  0.0747],
[443,  0.11205],
[909,  0.13708],
[911,  0.09568],
[970,  0.18216],
[1016, 0.153],
[1027, 0.230945],
[1029, 0.159355],
[1113, 0.17135],
[1114, 0.29268],
[1143, 0.1888],
[1228, 0.3744],
[1301, 0.5544],
[1329, 1.06485],
[1571, 0.90852],
[2034, 1.6224],
[2516, 2.484]
],
[ # 13500 eV
[0,    0.0],
[21,   0.00026],
[282,  0.0672],
[443,  0.1008],
[909,  0.1242],
[911,  0.087492],
[970,  0.16434],
[1016, 0.13872],
[1027, 0.207955],
[1029, 0.14413],
[1113, 0.1514],
[1114, 0.26244],
[1143, 0.1711],
[1228, 0.33813],
[1301, 0.4984],
[1329, 0.958365],
[1571, 0.8174],
[2034, 1.45704],
[2516, 2.244]
],
[ # 14000 eV
[0,    0.0],
[21,   0.000235],
[282,  0.0609],
[443,  0.09135],
[909,  0.11316],
[911,  0.080224],
[970,  0.14949],
[1016, 0.12648],
[1027, 0.1881],
[1029, 0.130935],
[1113, 0.1403],
[1114, 0.2376],
[1143, 0.15576],
[1228, 0.30537],
[1301, 0.45024],
[1329, 0.864475],
[1571, 0.73834],
[2034, 1.31508],
[2516, 2.02]
]
])


def attenuation_lookup(density, tb=TUBE_POTENTIAL.TB13000EV):
    #clamp to [0, 2516]
    density = max(0, min(density, 2516))

    index = 0
    length = len(LAC_LUT[tb,:,:])
    while (index < length) and (LAC_LUT[tb, index, 0] <= density):
        index += 1

    return np.float32(                                              \
            LAC_LUT[tb, index-2, 1]                                 \
            + (density - LAC_LUT[tb, index-2, 0])                   \
            / (LAC_LUT[tb, index-1, 0] - LAC_LUT[tb, index-2, 0])   \
            * (LAC_LUT[tb, index-1, 1] - LAC_LUT[tb, index-2, 1])   \
           )

def transform_to_lac_3d(data_array):
    for i in range(0, data_array.shape[0]):
        for j in range(0, data_array.shape[1]):
            for k in range(0, data_array.shape[2]):
                data_array[i,j,k] = attenuation_lookup(data_array[i,j,k], TUBE_POTENTIAL.TB13000EV)
    return data_array

def transform_to_lac_sequential(data_array):
    for i in range(0, len(data_array)):
        data_array[i] = attenuation_lookup(data_array[i], TUBE_POTENTIAL.TB13000EV)
    return data_array

def transform_to_lac_multiproc(data_array):
    datashape = data_array.shape
    #data = data_array.ravel()
    data = data_array.flatten()
    cores = multiprocessing.cpu_count()
    procs = []
    for i in range(0, cores):
        r = [int(i * len(data)/cores), min(int((i+1) * len(data)/cores), len(data))]
        p = multiprocessing.Process(target=transform_to_lac_sequential, args=[data[r[0]:r[1]]])
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    return data.reshape(datashape)

if __name__ == '__main__':
    print("Testing attenuation_lookup")
    densities = [0, 1, 21, 2516]
    expected = [0, 0.000289/21, 0.000289, 2.484]
    result = []
    for d in densities:
        result.append(attenuation_lookup(d))
    print("expected:\n", expected)
    print("result:\n", result)
    print("Testing lac transform...")
    result = transform_to_lac_sequential(densities)
    print("expected:\n", expected)
    print("result:\n", result)

