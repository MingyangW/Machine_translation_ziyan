# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 09:59:23 2018

@author: mingyang.wang
"""

import tensorflow as tf
import numpy as np
import os

from tensor2tensor import problems
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import registry

# Enable TF Eager execution
tfe = tf.contrib.eager
tfe.enable_eager_execution()

# Other setup
Modes = tf.estimator.ModeKeys

# Setup some directories
data_dir = os.path.expanduser("E:/data/tensor2tensor/test_enzh_t2t/self_data")
tmp_dir = os.path.expanduser("E:/data/tensor2tensor/test_enzh_t2t/rawdata")
train_dir = os.path.expanduser("E:/data/tensor2tensor/test_enzh_t2t/train")
checkpoint_dir = os.path.expanduser("E:/data/tensor2tensor/test_enzh_t2t/train")


# Fetch the problem
ende_problem = problems.problem("my_problem_enzh_test")

# Get the encoders from the problem
encoders = ende_problem.feature_encoders(data_dir)
#decoders = ende_problem.feature_decoders()

# Setup helper functions for encoding and decoding
def encode(input_str, output_str=None):
  """Input str to features dict, ready for inference"""
  inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
  batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
  return {"inputs": batch_inputs}

def decode(integers):
  """List of ints to str"""
  integers = list(np.squeeze(integers))
  if 1 in integers:
    integers = integers[:integers.index(1)]
  return encoders["targets"].decode(np.squeeze(integers))

model_name = "transformer"
hparams_set = "transformer_base"

hparams = trainer_lib.create_hparams(hparams_set, data_dir=data_dir, problem_name="my_problem_enzh_test")

# NOTE: Only create the model once when restoring from a checkpoint; it's a
# Layer and so subsequent instantiations will have different variable scopes
# that will not match the checkpoint.
translate_model = registry.model(model_name)(hparams, Modes.EVAL)

ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
print('ckpt_path:',ckpt_path)

def translate(inputs):
    encoded_inputs = encode(inputs)
    print('encoded_inputs:',encoded_inputs)
    with tfe.restore_variables_on_create(ckpt_path):
        model_output = translate_model.infer(encoded_inputs)["outputs"]
    print('model_output:',model_output)
    return decode(model_output)

inputs = "He'll shoulder the task You can't wish such a task on me."
outputs = translate(inputs)

print("Inputs: %s" % inputs)
print("Outputs: %s" % outputs)
