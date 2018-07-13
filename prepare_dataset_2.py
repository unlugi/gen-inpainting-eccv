import argparse
import shutil
import numpy as np
from matplotlib import image,pyplot
import os
import cv2
import json

parser = argparse.ArgumentParser()
parser.add_argument('--MaskedImageFolder', type=str)
parser.add_argument('--FullImageFolder', type=str)
args = parser.parse_args()

pathX = os.path.join(args.MaskedImageFolder, "X/")
pathY = os.path.join(args.MaskedImageFolder, "Y/")

savepathX =  os.path.join(args.FullImageFolder, "X/")
savepathY =  os.path.join(args.FullImageFolder, "Y/")

flist = os.path.join(args.FullImageFolder, "imagefiles.flist")
jsonfile = os.path.join(args.MaskedImageFolder, "maskdata.json")
shutil.copy(jsonfile, args.FullImageFolder)

filesX = sorted(os.listdir(pathX))

# flist file creation#
if not os.path.exists(flist):
    os.mknod(flist)

fo = open(flist,"w")
with open(jsonfile, encoding='utf-8') as jfile:
    mask_info = json.loads(jfile.read())
    for fileX in filesX:
        imgX = cv2.imread(os.path.join(pathX,fileX), -1) #masked image
        imgY = cv2.imread(os.path.join(pathY, fileX), -1) #mask
        if imgX.ndim == 3:
            origIm = imgX + imgY
            mask = np.zeros_like(imgX[:, :, 0])
            for i in range(len(mask_info[fileX])):
                x, y, size = mask_info[fileX][i]
                mask[x:x+size, y:y+size] = 1
            # Save original image and the corresponding mask
            cv2.imwrite(savepathX + fileX, origIm)
            pyplot.imsave(savepathY + fileX, mask, cmap='gray')
            # Write the file paths to flist file
            fo.write("%s\n" % (savepathX + fileX))
        elif imgX.ndim == 2:
            if imgY.ndim == 3:
                imgY = imgY[:,:,0]
            origIm = imgX + imgY
            mask = np.zeros_like(imgX)
            for i in range(len(mask_info[fileX])):
                x, y, size = mask_info[fileX][i]
                mask[x:x + size, y:y + size] = 1
            # Save original image and the corresponding mask
            pyplot.imsave(savepathX + fileX, origIm, cmap='gray')
            pyplot.imsave(savepathY + fileX, mask, cmap='gray')
            # Write the file paths to flist file
            fo.write("%s\n" % (savepathX + fileX))

