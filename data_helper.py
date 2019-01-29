import numpy as np
import os

# target: cut out 7 days' data from the origin dataset


data_path = 'data/METR-LA/'
train_data_file = data_path + 'train.npz'
val_data_file = data_path + 'val.npz'
test_data_file = data_path + 'test.npz'

train_data = np.load(train_data_file)
x = train_data['x']
y = train_data['y']
print(x.shape)
print(y.shape)

num_samples = 7 * 288
num_test = round(num_samples * 0.2)
num_train = round(num_samples * 0.7)
num_val = num_samples - num_test - num_train

p = x.shape[0] / 2
# train
x_train, y_train = x[:num_train], y[:num_train]
# val
x_val, y_val = (
    x[num_train: num_train + num_val],
    y[num_train: num_train + num_val],
)
# test
x_test, y_test = x[num_samples-num_test:num_samples], y[num_samples-num_test:num_samples]

for cat in ["train", "val", "test"]:
    _x, _y = locals()["x_" + cat], locals()["y_" + cat]
    print(cat, "x: ", _x.shape, "y:", _y.shape)
    np.savez_compressed(
        os.path.join(data_path, "%s.npz" % (cat+'_1')),
        x=_x,
        y=_y,
    )


