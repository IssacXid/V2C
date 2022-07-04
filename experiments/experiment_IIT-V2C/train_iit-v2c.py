import os
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
class TrainConfig(Config):
    """Configuration for training with IIT-V2C.
    """
    NAME = 'v2c_IIT-V2C'
    MODE = 'train'
    ROOT_DIR = ROOT_DIR
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    DATASET_PATH = os.path.join(ROOT_DIR, 'datasets', 'IIT-V2C')
    MAXLEN = 10

# Setup configuration class
config = TrainConfig()
# Setup tf.dataset object
annotation_file = config.MODE + '.txt'
action_file = config.MODE+'_actions.txt'
clips, targets, actions, vocab, config, ind2lab = iit_v2c.parse_dataset(config, annotation_file, action_file)
config.display()
train_dataset = iit_v2c.FeatureDataset(clips, targets, actions)
train_loader = data.DataLoader(train_dataset, 
                               batch_size=config.BATCH_SIZE, 
                               shuffle=True, 
                               num_workers=config.WORKERS)
bias_vector = vocab.get_bias_vector() if config.USE_BIAS_VECTOR else None

# # Setup and build video2command training inference
# v2c_model = Video2Command(config)
# v2c_model.build(bias_vector)

# if os.path.exists(os.path.join(config.CHECKPOINT_PATH, 'saved')):
#   checkpoint_file = os.path.join(config.CHECKPOINT_PATH, 'saved/', 'v2c_epoch_50.pth')
#   v2c_model.load_weights(checkpoint_file)

# # Save vocabulary at last
# with open(os.path.join(config.CHECKPOINT_PATH, 'vocab.pkl'), 'wb') as f:
#     pickle.dump(vocab, f)

# # Start training
# v2c_model.train(train_loader)

# Setup and build EDNet training inference
ednet_model = EDNet(config)
ednet_model.build(bias_vector)

if os.path.exists(os.path.join(config.CHECKPOINT_PATH, 'saved/', 'v2c_epoch_150_CMD.pth')):
  checkpoint_file = os.path.join(config.CHECKPOINT_PATH, 'saved/', 'v2c_epoch_150_CMD.pth')
  ednet_model.load_weights(checkpoint_file)

# Save vocabulary at last
with open(os.path.join(config.CHECKPOINT_PATH, 'vocab.pkl'), 'wb') as f:
    pickle.dump(vocab, f)

# Start training
ednet_model.train(train_loader)

# Setup and build Action classification training inference
AClass_model = ActionClassification(config)
AClass_model.build()

if os.path.exists(os.path.join(config.CHECKPOINT_PATH, 'saved/', 'v2c_epoch_150_action.pth')):
  checkpoint_file = os.path.join(config.CHECKPOINT_PATH, 'saved/', 'v2c_epoch_150_action.pth')
  AClass_model.load_weights(checkpoint_file)

# Start training
AClass_model.train(train_loader)