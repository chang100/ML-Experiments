from gen_utils import *
from nin_model import NiNModel
from utils.parser import parse

FLAGS = parse()

model = NiNModel(FLAGS)

with tf.Session() as sess:    
    model.train(sess)
