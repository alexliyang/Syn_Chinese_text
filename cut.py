# coding="utf-8"
import codecs
from PIL import Image
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2
from common import colorize, Color
from math import *
import math
db_fname = 'results/SynthText.h5'
db = h5py.File(db_fname, 'r')
dsets = sorted(db['data'].keys())
count = 0
print "total number of images : ", colorize(Color.RED, len(dsets), highlight=True)

def rotate(img,pt1,pt2,pt3,pt4):

   # print (pt1,pt2,pt3,pt4)
    withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)
    heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) **2)
    #print (withRect,heightRect)
    if(withRect):
        angle = acos(abs(pt4[0] - pt1[0]) / withRect) * (180 / math.pi)
        #print (angle)

        if pt4[1]<=pt1[1]:

            angle=-angle

        height = img.shape[0]
        width = img.shape[1]
        rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
        widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

        rotateMat[0, 2] += (widthNew - width) / 2
        rotateMat[1, 2] += (heightNew - height) / 2
        imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
        # cv2.imshow('rotateImg2',  imgRotation)
        # cv2.waitKey(0)


        [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
        [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
        [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
        [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))


        if pt2[1]>pt4[1]:
            pt2[1],pt4[1]=pt4[1],pt2[1]
        if pt1[0]>pt3[0]:
            pt1[0],pt3[0]=pt3[0],pt1[0]

        imgOut = imgRotation[int(pt2[1]+1):int(pt4[1]+1),int(pt1[0]+1):int(pt3[0]+1)]
        # cv2.imshow("imgOut", imgOut)
        # cv2.waitKey(0)
        return imgOut  # rotated image

with codecs.open('result.csv', 'w', encoding='utf-8') as csv:

    for k in dsets:
        m = 0
        space = 0
        space_count = 0
        rgb = db['data'][k][...]
        charBB = db['data'][k].attrs['charBB']
        wordBB = db['data'][k].attrs['wordBB']
        txt = db['data'][k].attrs['txt']

        image = rgb


        txt_len = len(txt)

        for i in xrange(wordBB.shape[-1]):
            bb = wordBB[:, :, i]
            print(bb.shape)
            bb = np.c_[bb, bb[:, 0]]
            #print(bb.shape)
            print bb[0, :]
            print bb[1, :]
            # visualize the indiv vertices:

            leftx = 1000
            lefty = 1000
            rightx = -1000
            righty = -1000
            pt=[]
            pt_order=[]
            ptx=[]
            pty=[]
            for j in xrange(4):
                pt.append([bb[0,j],bb[1,j]])
                ptx.append(bb[0,j])
                pty.append(bb[1,j])
            ptx_order=sorted(ptx)
            pty_order=sorted(pty)
            min_x0=ptx.index(ptx_order[0])
            min_x1 = ptx.index(ptx_order[1])

            if pty[min_x0]<pty[min_x1]:
                pt_order.append(pt[min_x1])
                pt_order.append(pt[min_x0])
            else:
                pt_order.append(pt[min_x0])
                pt_order.append(pt[min_x1])
            min_x2 = ptx.index(ptx_order[2])
            min_x3 = ptx.index(ptx_order[3])
            if pty[min_x2] < pty[min_x3]:
                pt_order.append(pt[min_x2])
                pt_order.append(pt[min_x3])
            else:
                pt_order.append(pt[min_x3])
                pt_order.append(pt[min_x2])
            print(pt)
            print(pt_order)
            region =rotate(image,pt_order[0],pt_order[1],pt_order[2],pt_order[3])
            cv2.imwrite('cut-pics/' + str(count) + '.jpg',region)
            #region.save('cut-pics/' + str(count) + '.jpg')
            for a in txt:
                a = a.decode('utf-8')
                print a
            list = txt[m].split('\n')
            if len(list) > 1:
                if space_count == 0:
                    space = len(list)
                if m < txt_len - 1:
                    if space_count < space:
                        csv.write('%s %s\n' % (str(count) + '.jpg', list[space_count].decode('utf-8')))
                        space_count += 1
                        count += 1
                        if space_count == space:
                            m += 1
                            space_count = 0
                else:
                    if space_count < space:
                        csv.write('%s %s\n' % (str(count) + '.jpg', list[space_count].decode('utf-8')))
                        space_count += 1
                        count += 1
                        if space_count == space:
                            m = 0
                            space_count = 0
            else:
                if m < txt_len - 1:
                    csv.write('%s %s\n' % (str(count) + '.jpg', txt[m].decode('utf-8')))
                    m += 1
                elif m == txt_len - 1:
                    csv.write('%s %s\n' % (str(count) + '.jpg', txt[m].decode('utf-8')))
                    m = 0
                count += 1


            # box = (0, 41, 150, 131)
            # region = image.crop(box)
            # region.save(str(count) + '.jpg')
            # count += 1
db.close()
