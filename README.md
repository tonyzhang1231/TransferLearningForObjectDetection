# Transfer Learning for Object Detection Task, using InceptionV3 and YOLO
The goal is to predict all the bounding boxes that containing an object and the category of the object for each bounding box.

We used InceptionV3 as the base feature extractor and trained a new classifier, we implemented the loss function proposed in YOLO's paper as the loss function.


# Clone The Git
`mkdir project_dir && cd project_dir`

`git clone https://github.com/tonyzhang1231/TransferLearningForObjectDetection`

# Get The Pascal VOC Data

`mkdir image_dir && cd image_dir`

`wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar`

`wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar`

`wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar`

`tar xf VOCtrainval_11-May-2012.tar`

`tar xf VOCtrainval_06-Nov-2007.tar`

`tar xf VOCtest_06-Nov-2007.tar`


# Generate Labels for VOC
We need to generate the label files that our model uses. we need a .txt file for each image with a line for each ground truth object in the image that looks like:

`<object-class> <x> <y> <width> <height>`

x, y are the coordinate of the bounding box. all x, y, w, h are between 0 and 1, relative to the size of the image.

Copy the __voc_label.py__ to image_dir and run this file

`cp project_dir/voc_label.py image_dir/`

`cd image_dir && python voc_label.py`

In the image_dir, run the following command to get the train, val and test data

`cat 2007_train.txt 2012_*.txt > train.txt`

`cat 2007_val.txt > validation.txt`

`cat 2007_test.txt > test.txt`

# Change the settings in the run_retrain_InceptionV3_YOLO.sh file
Change the IMAGE_DIR and result_store_dir and anything else if you want

# Retrain the model
`sh run_retrain_InceptionV3_YOLO.sh`

# Requirements
- tensorflow-gpu==1.4.1
- python 2.7.12

highly recommend to use python environment


# reference
- darknet/yolo  https://pjreddie.com/darknet/yolo/
- darkflow   https://github.com/thtrieu/darkflow
