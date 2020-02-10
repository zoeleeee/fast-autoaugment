from combined_test import target_model


import numpy as np

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

idx=10000
model = target_model('cifar100_pyramid272_30outputs_500epochs.pth', 30)
imgs = np.load('cifar100_advs_{}.npy'.format(idx))
label_path = 'cifar100_labels_{}.npy'.format(idx)
dummy_input = torch.from_numpy(imgs[0])
dummy_output = model(dummy_input)
print(dummy_output)

torch.onnx.export(model, dummy_input, 'cifar100.onnx', input_names=['input'], output_names=['output'])

model_onnx = onnx.load('cifar100.onnx')
tf_rep = prepare(model_onnx)
tf_rep.export_graph('cifar100.pb')

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

tf_graph = load_pb('cifar100.pb')
sess = tf.Session(graph=tf_graph)

# Show tensor names in graph
for op in tf_graph.get_operations():
  print(op.values())

output_tensor = tf_graph.get_tensor_by_name('test_output:0')
input_tensor = tf_graph.get_tensor_by_name('test_input:0')

output = sess.run(output_tensor, feed_dict={input_tensor: dummy_input})
print(output)