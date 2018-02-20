# Transfer Learning for Object Detection Task, using InceptionV3 and YOLO
The goal is to predict all the bounding boxes that containing an object and the category of the object for each bounding box.

We used InceptionV3 as the base feature extractor and trained a new classifier, we implemented the loss function proposed in YOLO's paper as the loss function.


# clone the git
`mkdir project_dir && cd project_dir`

`git clone `


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


`python voc_label.py`
