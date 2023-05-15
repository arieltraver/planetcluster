import os
import pandas as pd
import numpy as np

def create_frame(filename):
    frame = pd.read_csv(filename)
    return frame


