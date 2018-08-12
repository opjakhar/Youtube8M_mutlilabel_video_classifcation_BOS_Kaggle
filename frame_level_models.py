# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags
from tensorflow import logging	
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")

class BiLstmModel_sep(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """


    model_input_list = tf.split(model_input, [1024, 128], axis=2)
    with tf.variable_scope('rgb_lstm'):
      rgb_lstm_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(1024, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
      rgb_lstm_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(512, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
      outputs_rgb, state_fwv, state_bwv = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([rgb_lstm_fw], [rgb_lstm_bw], model_input_list[0], sequence_length=num_frames, dtype=tf.float32)
    with tf.variable_scope('audio_lstm'):
      audio_lstm_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(1024, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
      audio_lstm_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(512, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
      outputs_audio, state_fwa, state_bwa = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([audio_lstm_fw], [audio_lstm_bw], model_input_list[1], sequence_length=num_frames, dtype=tf.float32)
    
    state_v = tf.concat([state_fwv[-1].h, state_bwv[-1].h], axis=1)
    state_a = tf.concat([state_fwa[-1].h, state_bwa[-1].h], axis=1)
    final_feature=tf.multiply(state_v,state_a)
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=final_feature,
        vocab_size=vocab_size,
        **unused_params)

class LstmModel_late(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """


    model_input_list = tf.split(model_input, [1024, 128], axis=2)
    with tf.variable_scope('rgb_lstm'):
      rgb_lstm_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(512, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
      rgb_lstm_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(512, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
      outputs_rgb, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([rgb_lstm_fw], [rgb_lstm_bw], model_input_list[0], sequence_length=num_frames, dtype=tf.float32)
    with tf.variable_scope('audio_lstm'):
      audio_lstm_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(150, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
      audio_lstm_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(150, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
      outputs_audio, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([audio_lstm_fw], [audio_lstm_bw], model_input_list[1], sequence_length=num_frames, dtype=tf.float32)

    outputs_hidden = tf.concat([outputs_rgb, outputs_audio], axis=2)
    with tf.variable_scope('hidden_lstm'):
      hidden_lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(1200, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
      outputs, state = tf.nn.dynamic_rnn(hidden_lstm, outputs_hidden,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state.h,
        vocab_size=vocab_size,
        **unused_params)


