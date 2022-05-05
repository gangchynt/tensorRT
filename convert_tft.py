# Re-run after Kernel restart
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

# model = InceptionV3(weights='imagenet')

def show_predictions(model):

  img_path = './data/img0.JPG'  # golden_retriever
  img = image.load_img(img_path, target_size=(299, 299))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  x = tf.constant(x)

  labeling = model(x)
  preds = labeling['predictions'].numpy()
  
  # decode the results into a list of tuples (class, description, probability)
  # (one such list for each sample in the batch)
  print('{} - Predicted: {}'.format(img_path, decode_predictions(preds, top=3)[0]))
  plt.subplot(2,2,1)
  plt.imshow(img)
  plt.axis('off')
  plt.title(decode_predictions(preds, top=3)[0][0][1])






def batch_input(batch_size=8):
  batched_input=np.zeros((batch_size,299,299,3),dtype=np.float32)

  for i in range(batch_size):
    img_path = './data/img%d.JPG' %(i%4)
    img=image.load_img(img_path,target_size=(299,299))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)
    batched_input[i, :]=x

    batched_input = tf.constant(batched_input)

    return batched_input




def load_tf_saved_model(input_saved_model_dir):
  print(f'Loading saved model{input_saved_model_dir}..')
  saved_model_loaded=tf.saved_model.load(input_saved_model_dir,tags=[tag_constants.SERVING])
  return saved_model_loaded


def predict_and_benchmark_throughput(batched_input, infer, N_warmup_run=50, N_run=1000):

  elapsed_time = []
  all_preds = []
  batch_size = batched_input.shape[0]

  for i in range(N_warmup_run):
    labeling = infer(batched_input)
    preds = labeling['predictions'].numpy()

  for i in range(N_run):
    start_time = time.time()

    labeling = infer(batched_input)

    preds = labeling['predictions'].numpy()

    end_time = time.time()

    elapsed_time = np.append(elapsed_time, end_time - start_time)
    
    all_preds.append(preds)

    if i % 50 == 0:
      print('Steps {}-{} average: {:4.1f}ms'.format(i, i+50, (elapsed_time[-50:].mean()) * 1000))

  print('Throughput: {:.0f} images/s'.format(N_run * batch_size / elapsed_time.sum()))
  return all_preds



batched_input = batch_input(batch_size=32)


def convert_to_trt_graph_and_save(precision_mode='float32',
                                  input_saved_model_dir='inceptionv3_saved_model',
                                  calibration_data=batched_input):
  if precision_mode=='float32':
    precision_mode=trt.TrtPrecisionMode.FP32
    converted_save_suffix='_TFTRT_FP32'

    output_saved_model_dir = input_saved_model_dir + converted_save_suffix

    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=precision_mode,
    max_workspace_size_bytes=8000000000)

    converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=input_saved_model_dir,
    conversion_params=conversion_params)

    print(f'Converting {input_saved_model_dir} to TF_TRT graph precision mode{precision_mode}')
    
    converter.convert()

    print(f'Saving Converted model to {output_saved_model_dir}')
    converter.save(output_saved_model_dir=output_saved_model_dir)
    print('Complete')


# batched_input = batch_input(batch_size=32)
# saved_model=load_tf_saved_model('inceptionv3_saved_model')
# infer = saved_model.signatures['serving_default']
# # Save the entire model as a TensorFlow SavedModel.
# tf.saved_model.save(model,'inceptionv3_saved_model')