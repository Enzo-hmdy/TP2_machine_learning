import csv
import numpy
import sklearn
from datetime import *

# Chargement des données
data = csv.reader(open("SeoulBikedata", "r"),
                  delimiter=",")
