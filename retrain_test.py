# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=g-bad-import-order,unused-import
"""Tests the graph freezing tool."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import matplotlib.pyplot as plt

import cv2


from retrain_inceptionV3_yolo import *
import retrain_inceptionV3_yolo as retrain
from tensorflow.python.framework import test_util

"""
tf.test document
https://www.tensorflow.org/api_docs/python/tf/test/TestCase
"""

IMAGE_DIR = '/home/tonyzhang1231/Desktop/CVdata/PascalVOCData'
result_store_dir='/home/tonyzhang1231/Desktop/tf-tutorial/inceptionV3_yolo/retrain_inceptionV3_YOLO_output'
OUTPUT_GRAPH = os.path.join(result_store_dir,'output_graph.pb')
BOTTLENECK_DIR = os.path.join(result_store_dir,'bottleneck')
ONE_IMAGE_PATH = '/home/tonyzhang1231/Desktop/CVdata/PascalVOCData/VOCdevkit/VOC2007/JPEGImages/000012.jpg'


class ImageRetrainingTest(test_util.TensorFlowTestCase):

  def getImageLists(self):
    sample_image_lists={
        'training': [
            '/home/tonyzhang1231/Desktop/CVdata/PascalVOCData/VOCdevkit/VOC2007/JPEGImages/000012.jpg',
            '/home/tonyzhang1231/Desktop/CVdata/PascalVOCData/VOCdevkit/VOC2007/JPEGImages/000017.jpg'
        ],
        'validation':[
            '/home/tonyzhang1231/Desktop/CVdata/PascalVOCData/VOCdevkit/VOC2007/JPEGImages/000023.jpg',
            '/home/tonyzhang1231/Desktop/CVdata/PascalVOCData/VOCdevkit/VOC2007/JPEGImages/000026.jpg'
        ],
        'testing':[
            '/home/tonyzhang1231/Desktop/CVdata/PascalVOCData/VOCdevkit/VOC2007/JPEGImages/000032.jpg',
            '/home/tonyzhang1231/Desktop/CVdata/PascalVOCData/VOCdevkit/VOC2007/JPEGImages/000033.jpg'
        ]
    }
    return sample_image_lists

  def testGetLabelPathByImagePath(self):
    image_lists = self.getImageLists()
    for key in image_lists:
        image_list = image_lists[key]
        for image_path in image_list:
            label_path = get_label_path_by_image_path(image_path)
            self.assertTrue(tf.gfile.Exists(label_path))

  def testGetBottleneckPath(self):
    image_path = ONE_IMAGE_PATH
    bottleneck_path = get_bottleneck_path(image_path, BOTTLENECK_DIR)
    self.assertEqual(bottleneck_path, os.path.join(BOTTLENECK_DIR, '000012.txt'))

  @tf.test.mock.patch.object(retrain, 'FLAGS', S=7,B=2,C=20)
  def testCalculateConvertedLabel(self, flags_mock):
      image_path = ONE_IMAGE_PATH
      label_path = get_label_path_by_image_path(image_path)
      loss_feed_val = calculate_converted_label(label_path)
      self.assertIsNotNone(loss_feed_val['probs'])
      self.assertIsNotNone(loss_feed_val['confs'])
      self.assertIsNotNone(loss_feed_val['coord'])
      self.assertIsNotNone(loss_feed_val['proid'])
      self.assertIsNotNone(loss_feed_val['areas'])
      self.assertIsNotNone(loss_feed_val['upleft'])
      self.assertIsNotNone(loss_feed_val['botright'])

  def testReadLabel(self):
      """get all the objects in the label file, given a label path
      """
      image_path = ONE_IMAGE_PATH
      label_path = get_label_path_by_image_path(image_path)
      allobj = read_label(label_path)
      self.assertGreater(len(allobj), 0)

  @tf.test.mock.patch.object(retrain, 'FLAGS', S=7,B=2,C=20)
  def testConstructYoloBox(self, flags_mock):
      image_path = ONE_IMAGE_PATH
      label_path = get_label_path_by_image_path(image_path)
      true_cxywhs = read_label(label_path)

      label_path = get_label_path_by_image_path(image_path)
      loss_input = calculate_converted_label(label_path)



      # we fake a prediction which is same as converted true label
      # so if changing back, the result should be the same as the ground_truth
      probs = loss_input['probs']
      confs = loss_input['confs']
      coord = loss_input['coord']

      probs = probs.flatten()
      confs = confs.flatten()
      coord = coord.flatten()

      # print (probs.shape, confs.shape, coord.shape)
      prediction = np.concatenate([probs, confs, coord], 0)
      # print (prediction.shape)

      # so we expect there will be B boxes for each true obj
      boxes = construct_yolo_box(prediction, threshold = 0.2, sqrt = 2)
      B = int(len(boxes)/len(true_cxywhs))
      j = 0
      for cxywh in true_cxywhs:
          for i in range(B):
              self.assertEqual(cxywh[0], boxes[j*B + i].category)
              self.assertEqual(cxywh[1], boxes[j*B + i].x)
              self.assertEqual(cxywh[2], boxes[j*B + i].y)
              self.assertEqual(cxywh[3], boxes[j*B + i].w)
              self.assertEqual(cxywh[4], boxes[j*B + i].h)
          j += 1


  def testDrawBox(self):
      image_path = ONE_IMAGE_PATH
      label_path = get_label_path_by_image_path(image_path)
      true_cxywhs = read_label(label_path)
      true_boxes = [Box(c,1.0,x,y,w,h) for c,x,y,w,h in true_cxywhs]
      draw_box(image_path, true_boxes)

  def testPredcit(self):
      model_graph_file = OUTPUT_GRAPH
      image_paths = self.getImageLists()['testing']
      predict(model_graph_file, image_paths)

  # def testCalcYoloLoss(self):
  #     image_path = ONE_IMAGE_PATH
  #     label_path = get_label_path_by_image_path(image_path)
  #     loss_feed_val = calculate_converted_label(label_path)



  # def testShouldDistortImage(self):
  #   self.assertEqual(False, retrain.should_distort_images(False, 0, 0, 0))
  #   self.assertEqual(True, retrain.should_distort_images(True, 0, 0, 0))
  #   self.assertEqual(True, retrain.should_distort_images(False, 10, 0, 0))
  #   self.assertEqual(True, retrain.should_distort_images(False, 0, 1, 0))
  #   self.assertEqual(True, retrain.should_distort_images(False, 0, 0, 50))
  #
  # def testAddInputDistortions(self):
  #   with tf.Graph().as_default():
  #     with tf.Session() as sess:
  #       retrain.add_input_distortions(True, 10, 10, 10, 299, 299, 3, 128, 128)
  #       self.assertIsNotNone(sess.graph.get_tensor_by_name('DistortJPGInput:0'))
  #       self.assertIsNotNone(sess.graph.get_tensor_by_name('DistortResult:0'))
  #
  # @tf.test.mock.patch.object(retrain, 'FLAGS', learning_rate=0.01)
  # def testAddFinalTrainingOps(self, flags_mock):
  #   with tf.Graph().as_default():
  #     with tf.Session() as sess:
  #       bottleneck = tf.placeholder(tf.float32, [1, 1024], name='bottleneck')
  #       # Test creating final training op with quantization
  #       retrain.add_final_training_ops(5, 'final', bottleneck, 1024, False)
  #       self.assertIsNotNone(sess.graph.get_tensor_by_name('final:0'))
  #
  # @tf.test.mock.patch.object(retrain, 'FLAGS', learning_rate=0.01)
  # def testAddFinalTrainingOpsQuantized(self, flags_mock):
  #   with tf.Graph().as_default():
  #     with tf.Session() as sess:
  #       bottleneck = tf.placeholder(tf.float32, [1, 1024], name='bottleneck')
  #       # Test creating final training op with quantization
  #       retrain.add_final_training_ops(5, 'final', bottleneck, 1024, True)
  #       self.assertIsNotNone(sess.graph.get_tensor_by_name('final:0'))
  #
  # def testAddEvaluationStep(self):
  #   with tf.Graph().as_default():
  #     final = tf.placeholder(tf.float32, [1], name='final')
  #     gt = tf.placeholder(tf.int64, [1], name='gt')
  #     self.assertIsNotNone(retrain.add_evaluation_step(final, gt))
  #
  # def testAddJpegDecoding(self):
  #   with tf.Graph().as_default():
  #     jpeg_data, mul_image = retrain.add_jpeg_decoding(10, 10, 3, 0, 255)
  #     self.assertIsNotNone(jpeg_data)
  #     self.assertIsNotNone(mul_image)
  #
  # def testCreateModelInfo(self):
  #   did_raise_value_error = False
  #   try:
  #     retrain.create_model_info('no_such_model_name')
  #   except ValueError:
  #     did_raise_value_error = True
  #   self.assertTrue(did_raise_value_error)
  #   model_info = retrain.create_model_info('inception_v3')
  #   self.assertIsNotNone(model_info)
  #   self.assertEqual(299, model_info['input_width'])
  #
  # def testCreateModelInfoQuantized(self):
  #   # Test for mobilenet_quantized
  #   model_info = retrain.create_model_info('mobilenet_1.0_224')
  #   self.assertIsNotNone(model_info)
  #   self.assertEqual(224, model_info['input_width'])


if __name__ == '__main__':
  tf.test.main()
