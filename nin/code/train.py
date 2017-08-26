from gen_utils import *
from nin_model import NiNModel

cifar_nin_conf = Config(lr=0.01, bsz=64, n_epochs=50, decay_rate=0.5, decay_epochs=20, optimizer='Momentum')
model = NiNModel(cifar_nin_conf)

'''
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config=tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    model.train(sess)
'''
with tf.Session() as sess:    
    model.train(sess)
