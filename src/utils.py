import numpy as np

def adjustData(v,num_class=4):
    if len(v)==2:
        volume, mask = v
        if volume.dtype=='uint8':
            volume = volume / 255
        new_volume = np.zeros(volume.shape + (1,))
        new_volume[:,:,0] = volume
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        mask = new_mask
        volume = new_volume
        return (volume,mask)
    else:
        volume = v
        if volume.dtype=='uint8':
            volume = volume / (255*255)
        elif volume.dtype=='float64':
            volume = volume / 255
        new_volume = np.zeros(volume.shape + (1,))
        new_volume[:,:,0] = volume
        volume = new_volume
        return volume
