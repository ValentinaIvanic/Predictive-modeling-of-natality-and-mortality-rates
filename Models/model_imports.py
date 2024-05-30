import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from Data_preprocessing import Data_extracting

import warnings
warnings.filterwarnings('ignore')