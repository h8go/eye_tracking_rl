import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import matplotlib.pyplot as plt
import pickle
import math
import pdb
import cv2
"""
for layer in range(4):
    print('layer', layer)
    arr = state_list[layer]
    plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
    plt.show()
"""

"""
#### MASK

sigma = 5
i = 5
j = 20
mask = np.zeros((84,84))
for x in range(84):
    for y in range(84):
        #mask[x][y] = 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-( ( (x-i)**2 + (y-j)**2 ) / ( 2.0 * sigma**2 ) ) )
        mask[x][y] = np.exp(-( ( (x-i)**2 + (y-j)**2 ) / ( 2.0 * sigma**2 ) ) )

print(mask)
plt.imshow(mask, cmap='gray', vmin=0, vmax=1/(sigma * np.sqrt(2*np.pi)))
plt.show()


##### GAUSSIAN BLUR
mask_blur = 2*mask
blur = cv2.GaussianBlur(state_list[0],(5,5),1,1)
blur2 = np.multiply(blur, mask_blur)
#print(mask)
print(np.max(blur2))


plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
plt.show()
"""

def phi_one_frame(frame, i, j):
    ### MASK
    sigma_mask = 5
    M = np.zeros((84,84))
    for x in range(84):
        for y in range(84):
            M[x][y] = np.exp(-( ( (x-i)**2 + (y-j)**2 ) / ( 2.0 * sigma_mask**2 ) ) )
            #M[x][y] = 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-( ( (x-i)**2 + (y-j)**2 ) / ( 2.0 * sigma**2 ) ) )

    sigma_blur = 3
    A = cv2.GaussianBlur(frame,(5,5), sigma_blur)
    # plt.imshow(A, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    return np.multiply(frame, np.ones((84,84)) - M) + np.multiply(A, M)

def phi(state, i, j):
    # print(state)
    state_prime = np.zeros(state.shape)
    for idx in range(4):

        frame = state[0,:,:,idx]


        frame_prime = phi_one_frame(frame, i, j)

        state_prime[0,:,:,idx] = frame_prime

    return state_prime

"""
with open('/home/hugo/dopamine/state_saves3.pickle', 'rb') as f:
    state = pickle.load(f)

print(state.shape)
i = 78
j = 73
state_prime = phi(state, i, j)

for idx in range(4):
    plt.imshow(state_prime[0,:,:,idx], cmap='gray', vmin=0, vmax=255)
    plt.show()
"""





"""
with open('/home/hugo/dopamine/state_saves.pickle', 'rb') as f:
    state_list = pickle.load(f)
i = 22
j = 72
perturbation_image = phi_one_frame(state_list[0], i ,j)
plt.imshow(perturbation_image, cmap='gray', vmin=0, vmax=255)
plt.show()

# plt.imshow(state_list[0], cmap='gray', vmin=0, vmax=255)
# plt.show()
"""
