import numpy as np
import struct


def loadimg(imgfilename):
    with open(imgfilename, 'rb') as imgfile:
        datastr = imgfile.read()
    
    index = 0
    mgc_num, img_num, row_num, col_num = struct.unpack_from('>IIII', datastr, index)
    index += struct.calcsize('>IIII')
    
    image_array = np.zeros((img_num, row_num, col_num))
    for img_idx in range(img_num):
        img = struct.unpack_from('>784B', datastr, index)
        index += struct.calcsize('>784B')
        image_array[img_idx,:,:] = np.reshape(np.array(img), (28,28))
    image_array = image_array/255.0
    np.save(imgfilename[:6]+'image-py', image_array)
    return None

def loadlabel(labelfilename):
    with open(labelfilename, 'rb') as labelfile:
        datastr = labelfile.read()
    
    index = 0
    mgc_num, label_num = struct.unpack_from('>II', datastr, index)
    index += struct.calcsize('>II')
    
    label = struct.unpack_from('{}B'.format(label_num), datastr, index)
    index += struct.calcsize('{}B'.format(label_num))
    
    label_array = np.array(label)
    
    np.save(labelfilename[:5]+'label-py', label_array)
    return None

def main():
    """Run main function."""
    img_fs= ["t10k-images-idx3-ubyte", "train-images-idx3-ubyte"]
    label_fs= ["t10k-labels-idx1-ubyte", "train-labels-idx1-ubyte"]
    for imgf in img_fs: loadimg(imgf)
    for label_f in label_fs: loadlabel(label_f)

if __name__ == "__main__":
    main()
