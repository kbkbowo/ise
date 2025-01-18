import numpy

file = 'down_dataset/push_bar/dinov2/all_feats.npy'

data = numpy.load(file, allow_pickle=True).item()

# print(data.items())

keys = list(data.keys())
values = list(data.values())

keys = keys[:10]
values = values[:10]

for i in range(len(keys)):
    print(keys[i], len(values[i]))
# print(keys, values.shape)

