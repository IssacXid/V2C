import os
import glob
import sys
import pickle

import numpy as np
import torch
from torch.utils import data

# Root directory of the project
ROOT_DIR = os.path.abspath("/content/V2C/")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from v2c.model import *
from v2c import utils
from v2c.config import *
from datasets import iit_v2c

# Configuration for hperparameters
class TestConfig(Config):
    """Configuration for training with IIT-V2C.
    """
    NAME = 'v2c_IIT-V2C'
    MODE = 'test'
    ROOT_DIR = ROOT_DIR
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    DATASET_PATH = os.path.join(ROOT_DIR, 'datasets', 'IIT-V2C')
    MAXLEN = 10

# Setup configuration class
config = TestConfig()
# Setup tf.dataset object
vocab = pickle.load(open(os.path.join(config.CHECKPOINT_PATH, 'vocab.pkl'), 'rb'))
annotation_file = config.MODE + '.txt'
action_file = config.MODE+'_actions.txt'
clips, targets, actions, _, config, ind2lab = iit_v2c.parse_dataset(config, annotation_file, action_file, vocab=vocab)
test_dataset = iit_v2c.FeatureDataset(clips, targets, actions)
test_loader = data.DataLoader(test_dataset, 
                              batch_size=config.BATCH_SIZE, 
                              shuffle=False, 
                              num_workers=config.WORKERS)
config.display()

# # Setup and build video2command training inference
# v2c_model = Video2Command(config)
# v2c_model.build()

# # Safely create prediction dir if non-exist
# if not os.path.exists(os.path.join(config.CHECKPOINT_PATH, 'prediction')):
#     os.makedirs(os.path.join(config.CHECKPOINT_PATH, 'prediction'))

# # Start evaluating
# checkpoint_files = sorted(glob.glob(os.path.join(config.CHECKPOINT_PATH, 'saved', '*.pth')))
# for checkpoint_file in checkpoint_files:
#     epoch = int(checkpoint_file.split('_')[-1][:-4])
#     v2c_model.load_weights(checkpoint_file)
#     y_pred, y_true, ac_pred, ac_true = v2c_model.evaluate(test_loader, vocab)

#     # Save to evaluation file
#     f = open(os.path.join(config.CHECKPOINT_PATH, 'prediction', 'prediction_{}.txt'.format(epoch)), 'w')

#     for i in range(len(y_pred)):
#         #print(y_pred[i])
#         pred_command = utils.sequence_to_text(y_pred[i], vocab)
#         #print(y_true[i])
#         true_command = utils.sequence_to_text(y_true[i], vocab)
#         pred_action = ind2lab[ac_pred[i]]
#         true_action = ind2lab[ac_true[i]]
#         f.write('------------------------------------------\n')
#         f.write(str(i) + '\n')
#         f.write(pred_command + '\n')
#         f.write(true_command + '\n')
#         f.write(pred_action + '\n')
#         f.write(true_action + '\n')

#     print('Ready for cococaption.')

# Setup and build video2command training inference
ednet_model = EDNet(config)
ednet_model.build()

AClass_model = ActionClassification(config)
AClass_model.build()
# Safely create prediction dir if non-exist
if not os.path.exists(os.path.join(config.CHECKPOINT_PATH, 'prediction')):
    os.makedirs(os.path.join(config.CHECKPOINT_PATH, 'prediction'))

checkpoint_file = os.path.join(config.CHECKPOINT_PATH, 'saved/', 'v2c_epoch_150_CMD.pth')

# Start evaluating
print(checkpoint_file.split('_')[-2:-1][0])
epoch = int(checkpoint_file.split('_')[-2:-1][0])
ednet_model.load_weights(checkpoint_file)
y_pred, y_true = ednet_model.evaluate(test_loader, vocab)

checkpoint_file = os.path.join(config.CHECKPOINT_PATH, 'saved/', 'v2c_epoch_150_action.pth')

# Start evaluating
epoch = int(checkpoint_file.split('_')[-2:-1][0])
AClass_model.load_weights(checkpoint_file)
ac_pred, ac_true = AClass_model.evaluate(test_loader,)

    # Save to evaluation file
f = open(os.path.join(config.CHECKPOINT_PATH, 'prediction', 'prediction_{}.txt'.format(epoch)), 'w')

for i in range(len(y_pred)):
    #print(y_pred[i])
    pred_command = utils.sequence_to_text(y_pred[i], vocab)
    #print(y_true[i])
    true_command = utils.sequence_to_text(y_true[i], vocab)
    pred_action = ind2lab[ac_pred[i]]
    true_action = ind2lab[ac_true[i]]
    f.write('------------------------------------------\n')
    f.write(str(i) + '\n')
    f.write(pred_command + '\n')
    f.write(true_command + '\n')
    f.write(pred_action + '\n')
    f.write(true_action + '\n')

print('Ready for cococaption.')