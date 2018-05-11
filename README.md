# KITTI to YOLO 9000 label converter
Converts txt kitti labels to yolo v2 labels.

# Inputs
Path of the folders where the images and labels are saved, and the output directory

# Outputs
pickle files containing a dictionary with the image and its corresponding labels for yolo v2.

´´´python
 file#.p: {'image':image,'label':labels}
´´´

The output labels are contained on a numpy array with zero shape of [13,13,5,6], where the only non zeros are those anchors with an object in them.

NOTE: It will create a file for each image.

# How it works
First of all, we have to make sure that the number of images and labels files are the same, and the order of the images correspond to the order of the labels. For example:

´´´
                OR
Folder_1        |   Folder_1        |   Folder_1        |   Folder_1
.               |   .               |   .               |   .        
├─ im1.jpg      |   ├─ im1.jpg      |   ├─ im1.jpg      |   ├─ im1.jpg
├─ im2.jpg      |   ├─ im2.jpg      |   ├─ im2.jpg      |   ├─ im2.jpg
├─ im3.jpg      |   ├─ im3.jpg      |   ├─ im3.jpg      |   ├─ im3.jpg
└─ im4.jpg      |   └─ im4.jpg      |   ├─ im4.jpg      |   ├─ im4.jpg
                |                   |   ├─ lb1.txt      |   ├─ asd.tx
Folder_2        |   Folder_2        |   ├─ lb2.txt      |   ├─ bda.txt
.               |   .               |   ├─ lb3.txt      |   ├─ crew.txt
├─ lb1.txt      |   ├─ asd.txt      |   └─ lb4.txt      |   └─ das.txt
├─ lb2.txt      |   ├─ bda.txt      |
├─ lb3.txt      |   ├─ crew.txt     |
└─ lb4.txt      |   └─ das.txt      |
´´´

You got the idea.

´´´python
from data_transform import Kitti2Yolo

ims_folder = '../ims/'
lbs_folder = '../lbs/'
out_folder = '../data/'

conv = Kitti2Yolo(ims_folder,lbs_folder)
conv.convert(out_dir=out_folder)
´´´


# To Do
Add saving support for multiple class. Right now it saves all the images with the class label 15, which indicates a person. It's not hard, it actually already reads the class on the file, it just need to compare it to some list containing the right class and assign it.
