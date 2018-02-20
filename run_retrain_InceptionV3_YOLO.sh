# where to find images and where to store the results
IMAGE_DIR=${HOME}/Desktop/CVdata/PascalVOCData
result_store_dir=${HOME}/Desktop/tf-tutorial/inceptionV3_yolo/retrain_inceptionV3_YOLO_output


# arguments
OUTPUT_GRAPH=${result_store_dir}/output_graph.pb
INTERMEDIATE_OUTPUT_GRAPHS_DIR=${result_store_dir}/intermediate_graph/
INTERMEDIATE_STORE_FREQUENCY=0
OUTPUT_LABELS=${result_store_dir}/output_labels.txt
SUMMARIES_DIR=${result_store_dir}/retrain_logs
HOW_MANY_TRAINING_STEPS=40
LEARNING_RATE=0.01
EVAL_STEP_INTERVAL=10
TRAIN_BATCH_SIZE=100
TEST_BATCH_SIZE=-1
VALIDATION_BATCH_SIZE=100
PRINT_MISCLASSIFIED_TEST_IMAGES=False
MODEL_DIR=${result_store_dir}/image_net
BOTTLENECK_DIR=${result_store_dir}/bottleneck
FINAL_TENSOR_NAME=final_result
FLIP_LEFT_RIGHT=False
RANDOM_CROP=0
RANDOM_SCALE=0
RANDOM_BRIGHTNESS=0
ARCHITECTURE=inception_v3
CONVERTED_LABEL_DIR=${result_store_dir}/converted_label

# run the python script
python retrain_inceptionV3_yolo.py --image_dir ${IMAGE_DIR} \
      --result_store_dir ${result_store_dir}\
      --output_graph ${OUTPUT_GRAPH} \
      --intermediate_output_graphs_dir ${INTERMEDIATE_OUTPUT_GRAPHS_DIR} \
      --intermediate_store_frequency ${INTERMEDIATE_STORE_FREQUENCY}\
      --output_labels ${OUTPUT_LABELS}\
      --summaries_dir ${SUMMARIES_DIR}\
      --how_many_training_steps ${HOW_MANY_TRAINING_STEPS}\
      --learning_rate ${LEARNING_RATE}\
      --eval_step_interval ${EVAL_STEP_INTERVAL}\
      --train_batch_size ${TRAIN_BATCH_SIZE}\
      --test_batch_size ${TEST_BATCH_SIZE}\
      --validation_batch_size ${VALIDATION_BATCH_SIZE}\
      --model_dir ${MODEL_DIR}\
      --bottleneck_dir ${BOTTLENECK_DIR}\
      --final_tensor_name ${FINAL_TENSOR_NAME}\
      --random_crop ${RANDOM_CROP}\
      --random_scale ${RANDOM_SCALE}\
      --random_brightness ${RANDOM_BRIGHTNESS}\
      --architecture ${ARCHITECTURE}\
      --converted_label_dir ${CONVERTED_LABEL_DIR}


# tensorboard --logdir=${HOME}/Desktop/tf-tutorial/inceptionV3_yolo/retrain_inceptionV3_YOLO_output/retrain_logs

#   --image_dir IMAGE_DIR
#                         Path to folders of labeled images.
#   --output_graph OUTPUT_GRAPH
#                         Where to save the trained graph.
#   --intermediate_output_graphs_dir INTERMEDIATE_OUTPUT_GRAPHS_DIR
#                         Where to save the intermediate graphs.
#   --intermediate_store_frequency INTERMEDIATE_STORE_FREQUENCY
#                         How many steps to store intermediate graph. If "0"
#                         then will not store.
#   --output_labels OUTPUT_LABELS
#                         Where to save the trained graph's labels.
#   --summaries_dir SUMMARIES_DIR
#                         Where to save summary logs for TensorBoard.
#   --how_many_training_steps HOW_MANY_TRAINING_STEPS
#                         How many training steps to run before ending.
#   --learning_rate LEARNING_RATE
#                         How large a learning rate to use when training.
#   --testing_percentage TESTING_PERCENTAGE
#                         What percentage of images to use as a test set.
#   --validation_percentage VALIDATION_PERCENTAGE
#                         What percentage of images to use as a validation set.
#   --eval_step_interval EVAL_STEP_INTERVAL
#                         How often to evaluate the training results.
#   --train_batch_size TRAIN_BATCH_SIZE
#                         How many images to train on at a time.
#   --test_batch_size TEST_BATCH_SIZE
#                         How many images to test on. This test set is only used
#                         once, to evaluate the final accuracy of the model
#                         after training completes. A value of -1 causes the
#                         entire test set to be used, which leads to more stable
#                         results across runs.
#   --validation_batch_size VALIDATION_BATCH_SIZE
#                         How many images to use in an evaluation batch. This
#                         validation set is used much more often than the test
#                         set, and is an early indicator of how accurate the
#                         model is during training. A value of -1 causes the
#                         entire validation set to be used, which leads to more
#                         stable results across training iterations, but may be
#                         slower on large training sets.
#   --print_misclassified_test_images
#                         Whether to print out a list of all misclassified test
#                         images.
#   --model_dir MODEL_DIR
#                         Path to classify_image_graph_def.pb,
#                         imagenet_synset_to_human_label_map.txt, and
#                         imagenet_2012_challenge_label_map_proto.pbtxt.
#   --bottleneck_dir BOTTLENECK_DIR
#                         Path to cache bottleneck layer values as files.
#   --final_tensor_name FINAL_TENSOR_NAME
#                         The name of the output classification layer in the
#                         retrained graph.
#   --flip_left_right     Whether to randomly flip half of the training images
#                         horizontally.
#   --random_crop RANDOM_CROP
#                         A percentage determining how much of a margin to
#                         randomly crop off the training images.
#   --random_scale RANDOM_SCALE
#                         A percentage determining how much to randomly scale up
#                         the size of the training images by.
#   --random_brightness RANDOM_BRIGHTNESS
#                         A percentage determining how much to randomly multiply
#                         the training image input pixels up or down by.
#   --architecture ARCHITECTURE
#                         Which model architecture to use. 'inception_v3' is the
#                         most accurate, but also the slowest. For faster or
#                         smaller models, chose a MobileNet with the form
#                         'mobilenet_<parameter size>_<input_size>[_quantized]'.
#                         For example, 'mobilenet_1.0_224' will pick a model
#                         that is 17 MB in size and takes 224 pixel input
#                         images, while 'mobilenet_0.25_128_quantized' will
#                         choose a much less accurate, but smaller and faster
#                         network that's 920 KB on disk and takes 128x128
#                         images. See https://research.googleblog.com/2017/06
#                         /mobilenets-open-source-models-for.html for more
#                         information on Mobilenet.
