import os
import glob
import sys
import pickle

import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("/content/V2C/")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library

def read_prediction_file(prediction_file):
    """Helper function to read generated prediction files.
    """
    # Create dicts for ground truths and predictions
    gts_dict, pds_dict = {}, {}
    f = open(prediction_file, 'r')
    lines = f.read().split('\n')
    f.close()

    for i in range(0, len(lines) - 6, 6):
        id_line = lines[i+1]
        pd_action = lines[i+4]
        gt_action = lines[i+5]
        gts_dict[id_line] = gt_action
        pds_dict[id_line] = pd_action
    
    return gts_dict, pds_dict

def test_iit_v2c():
    """Helper function to test on IIT-V2C dataset.
    """
    # Get all generated predicted files
    prediction_files = sorted(glob.glob(os.path.join(ROOT_DIR, 'checkpoints', 'prediction', '*.txt')))
    
    max_scores = 0
    max_file = None
    for prediction_file in prediction_files:
        gts_dict, pds_dict = read_prediction_file(prediction_file)
        ids = list(gts_dict.keys())
        count = 0
        for Id in ids:
          if gts_dict[Id] == pds_dict[Id]:
            count+=1
        accuracy = count/len(ids) * 100
        if accuracy > max_scores:
            max_scores = accuracy
            max_file = prediction_file
        
        print('Maximum Score with file', prediction_file)
        print('accuracy: %0.3f' % accuracy)

    print('Maximum Score with file', max_file)
    print('accuracy: %0.3f' % max_scores)


if __name__ == '__main__':
  test_iit_v2c()