import scipy.io as sio
import numpy as np

def load_data():
    data = sio.loadmat('data/YaleBCrop025.mat')
    img = data['Y']
    I = []
    Label = []
    for i in range(img.shape[2]):
        for j in range(img.shape[1]):
            temp = np.reshape(img[:, j, i], [42, 48])
            Label.append(i)
            I.append(temp)
    I = np.array(I)
    label = np.array(Label[:])
    Img = np.transpose(I, [0, 2, 1])
    Img = Img / 255
    Img = np.expand_dims(Img[:], 3)
    Img = np.transpose(Img, [0, 3, 1, 2])
    Img = Img[:640, :, :, :]
    label = label[:640]
    return Img, label