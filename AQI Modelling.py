import pandas as pd
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex2 import *

# save filepath to variable for easier access
air_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
air_data = pd.read_csv(air_file_path) 

air_features = ['County', 'Days with AQI', 'Max AQI']