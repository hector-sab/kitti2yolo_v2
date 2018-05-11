"""
Converts data from KITTI (.txt) labels format to yolo format.

Each example will be saved as a dict containing the image and the labels

File ex1.p will contain:
ex1 = {'im':image,'lbs':labels}

File ex2.p will contain:
ex2 = {'im':image,'lbs':labels}
.
.
.

All labels for a single image are contained in a tensor of shape:
  [G_H,G_W,N_A,BCP] [13,13,5,6]

  G_H: Grid height
  G_W: Grid widht
  N_A: Number of anchors
  BCP: Bounding box specs (4), Class (1), and Prob of being object (1)
"""


import os
import re
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from glob import glob

class Kitti2Yolo:
  """
  This class expects that the order of the images is the same order
  as of the labels
  """
  def __init__(self,ims_dir='./',lbs_dir='./'):
    self.ims_dir = None
    self.lbs_dir = None
    self.ims_paths = None
    self.lbs_paths = None

    self.set_new_dir(ims_dir,lbs_dir)


    self.im_out_h = 416
    self.im_out_w = 416

    self.grid_h = 13
    self.grid_w = 13

    # Each anchor is defined by [height, width]
    self.anchors_normalized = np.array(
      [
        [0.05210654, 0.04405615],
        [0.15865615, 0.14418923],
        [0.42110308, 0.25680231],
        [0.27136769, 0.60637077],
        [0.70525231, 0.75157846]
      ])
    self.anchors = self.anchors_normalized*np.array([self.grid_h,self.grid_w])
    self.num_anchors = self.anchors.shape[0]



  def convert(self,out_dir='./data/'):
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    print(len(self.ims_paths))

    with tqdm(total=len(self.ims_paths)) as pbar:
      for i in range(len(self.ims_paths)):
        im_path = self.ims_paths[i]
        lb_path = self.lbs_paths[i]

        #im_tmp = [m.start() for m in re.finditer(' ',im_path)]
        #lb_tmp = [m.start() for m in re.finditer(' ',lb_path)]
        
        im = self.__get_image(im_path)
        targets = self.__get_targets(lb_path)
        label = self.__targets2label(targets)

        out = {'image':im,'label':label}

        tmp = [m.start() for m in re.finditer('/',im_path)]
        out_path = os.path.join(out_dir,im_path[tmp[-1]+1:-3]+'p')
        out_path = os.path.abspath(out_path)

        with open(out_path,'wb') as f:
          
          pickle.dump(out,f)

        pbar.update(1)



  def __targets2label(self,targets):
    label = np.zeros((self.grid_h,self.grid_w,self.num_anchors,6))

    for target in targets:
      target_class = target[4]

      # Maps from [0,1] space to [0,13] space
      bbox = target[0:4]*np.array([self.grid_h,self.grid_w,self.grid_h,self.grid_w])

      # Get grid index for bbox center
      idx_y = int(float(bbox[0]))
      idx_x = int(float(bbox[1]))

      iou_best, idx_anchor_best = 0.,0
      for idx_anchor, anchor in enumerate(self.anchors):
        iou = self.__get_iou(bbox[2:4],anchor)

        if iou>iou_best:
          iou_best = iou
          idx_anchor_best = idx_anchor

      # Update labels
      if iou_best>0.:
        label[idx_y, idx_x,idx_anchor_best] = np.array([
                  bbox[0] - idx_y, # offset of box_center from top-left corner of grid containing box_center
                  bbox[1] - idx_x,
                  bbox[2]/self.anchors[idx_anchor_best,0], # scale of anchor box so as to fit the bbox
                  bbox[3]/self.anchors[idx_anchor_best,1],
                  target_class,
                  1.0 # prob_object (object is present with prob=1)
                  ])

    return(label)
  def __get_iou(self,hw1,hw2):
    """
    hw : (h,w)
    assumption: both boxes have same center
    """

    # Get extremes of boxes
    #print('hw: ',hw1,hw2)
    hw1_max, hw2_max = hw1/2., hw2/2.
    hw1_min, hw2_min = -hw1_max, -hw2_max

    # Get intersection area
    intersection_min = np.maximum(hw1_min, hw2_min)
    intersection_max = np.minimum(hw1_max, hw2_max)
    hw_intersection = np.maximum(intersection_max-intersection_min, 0.)
    area_intersection = hw_intersection[0] * hw_intersection[1]

    # Get union area
    area_hw1 = hw1[0] * hw1[1]
    area_hw2 = hw2[0] * hw2[1]
    area_union = area_hw1 + area_hw2 - area_intersection

    iou = area_intersection / area_union

    return(iou)

  def __get_targets(self,path):
    # TODO: Add conversion from cls_txt to cls_lbl for multiple classes
    with open(path,'r') as f:
      tmp = f.readlines()
      lines = []

      for l in tmp:
        l = l.strip('\n')
        lines.append(l)

    spaces = []
    for line in lines:
      tmp = [m.start() for m in re.finditer(' ',line)]
      spaces.append(tmp)

    bboxs = []
    clss_txt = []
    for i in range(len(spaces)):
      clss =line[i][:spaces[i][0]]
      x_min = int(lines[i][spaces[i][3]+1:spaces[i][4]])
      y_min = int(lines[i][spaces[i][4]+1:spaces[i][5]])
      x_max = int(lines[i][spaces[i][5]+1:spaces[i][6]])
      y_max = int(lines[i][spaces[i][6]+1:spaces[i][7]])

      y_center,x_center = (y_min + y_max)/2., (x_min + x_max)/2.
      bbox_h, bbox_w = y_max - y_min, x_max - x_min

      y_center /= self.tmp_im_orig_hw[0]
      x_center /= self.tmp_im_orig_hw[1]
      bbox_h /= self.tmp_im_orig_hw[0]
      bbox_w /= self.tmp_im_orig_hw[1]

      clss_txt.append(clss)
      # TODO: Add multi class support
      cls_lbl = 15
      bboxs.append([y_center,x_center,bbox_h,bbox_w,15])

    return(np.array(bboxs,dtype=np.float32))

  def __get_image(self,path):
    im = cv2.imread(path)
    im_h = im.shape[0]
    im_w = im.shape[1]

    self.tmp_im_orig_hw = (im_h,im_w)

    im = cv2.resize(im,(self.im_out_w,self.im_out_h))
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    return(im)

  def __get_ims_paths(self,direct):
    paths = self.__get_ext_paths(direct,'jpg')
    return(paths)

  def __get_lbs_paths(self,direct):
    paths = self.__get_ext_paths(direct,'txt')
    return(paths)    

  def __get_ext_paths(self,direct,ext):
    """
    direct: directory where the files are located
    ext: extension of the files of interest. I.E. 'txt'
    """
    files = glob(os.path.join(direct,'*.'+ext))
    files.sort()
    return(files)

  def set_new_dir(self,ims_dir,lbs_dir):
    """
    Sets New directories to look up for files
    """
    self.ims_dir = ims_dir
    self.lbs_dir = lbs_dir

    self.ims_paths = self.__get_ims_paths(self.ims_dir)
    self.lbs_paths = self.__get_lbs_paths(self.lbs_dir)

  def set_new_hw(self,height,width):
    """
    Sets new height and width of the output images
    """
    self.im_out_h = height
    self.im_out_w = width



if __name__=='__main__':
  ims_folder = '../../../test_ppl_extraction/00000000_PPL/'
  lbs_folder = '../../../test_ppl_extraction/00000000_PPL/'

  conv = Kitti2Yolo(ims_folder,lbs_folder)
  conv.convert(out_dir='../data/')