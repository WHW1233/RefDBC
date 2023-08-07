import numpy as np
import os

def H_compress(H, h=128, w=128, type='x2', idx=0):
    H1 = H.reshape(h,w)
    if type=='x2':
        Code = np.zeros((h//2,w//2))
        for i in range(h//2):
            for j in range(w//2):
                H1[2*i, 2*j + 1] = H1[2*i, 2*j] + 1
                H1[2*i + 1, 2*j] = H1[2*i, 2*j] + 128
                H1[2*i + 1, 2*j + 1] = H1[2*i, 2*j] + 1 + 128
                Code[i,j] = H1[2*i,2*j]
        np.savez_compressed('./compression_data/encode/H_{}'.format(idx), Code)
        Code = np.load('./compression_data/encode/H_{}.npz'.format(idx))

    elif type=='x1_2':
        Code = np.zeros((h,w//2))
        for i in range(h):
            for j in range(w//2):
                Code[i, j] = H1[ i, 2 * j]
                H1[i, 2*j + 1] = H1[i, 2*j] + 1
        np.savez_compressed('./compression_data/encode/H_{}'.format(idx), Code)
        Code = np.load('./compression_data/encode/H_{}.npz'.format(idx))
    elif type == 'x1':
        np.savez_compressed('./compression_data/encode/H_{}'.format(idx), H1)

    elif type=='x4':
        Code = np.zeros((h // 4, w // 4))
        for i in range(h//4):
            for j in range(w//4):
                H1[4*i, 4*j + 1] = H1[4*i, 4*j] + 1
                H1[4 * i, 4 * j + 2] = H1[4 * i, 4 * j] + 2
                H1[4 * i, 4 * j + 3] = H1[4 * i, 4 * j] + 3

                H1[4*i + 1, 4*j] = H1[4*i, 4*j] + 128
                H1[4 * i + 1, 4 * j] = H1[4 * i, 4 * j] + 128*2
                H1[4 * i + 1, 4 * j] = H1[4 * i, 4 * j] + 128*3

                H1[4*i + 1, 4*j + 1] = H1[4*i, 4*j] + 1 + 128
                H1[4*i + 1, 4*j + 2] = H1[4*i, 4*j] + 2 + 128
                H1[4*i + 1, 4*j + 3] = H1[4*i, 4*j] + 3 + 128

                H1[4*i + 2, 4*j + 1] = H1[4*i, 4*j] + 1 + 128*2
                H1[4*i + 2, 4*j + 2] = H1[4*i, 4*j] + 2 + 128 * 2
                H1[4*i + 2, 4*j + 3] = H1[4*i, 4*j] + 3 + 128 * 2

                H1[4*i + 3, 4*j + 1] = H1[4*i, 4*j] + 1 + 128 * 3
                H1[4 * i + 3, 4 * j + 2] = H1[4 * i, 4*j] + 2 + 128 * 3
                H1[4 * i + 3, 4 * j + 3] = H1[4 * i, 4*j] + 3 + 128 * 3
                Code[i, j] = H1[4 * i, 4 * j]
        np.savez_compressed('./compression_data/encode/H_{}'.format(idx), Code)
        Code = np.load('./compression_data/encode/H_{}.npz'.format(idx))
    bits = os.path.getsize('./compression_data/encode/H_{}.npz'.format(idx))
    print('H所占的字节', bits)
    H2 = H1.reshape(1,h*w)
    H2 = np.minimum(H2, h*w-1)
    return H2, bits


if __name__ == '__main__':
    L = [i for i in range(16384)]
    X = np.array([L])
    print(X[0,-20:])
    H2 = H_compress(X,type='x4')
    print(H2[0,-20:])
    idx = np.minimum(H2, 128*128-10)
    print(idx)
