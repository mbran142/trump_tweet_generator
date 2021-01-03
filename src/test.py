from train import build_and_train_model
from clean import clean_data
import pdb

pdb.set_trace()
clean_data('data.json', 'new.json')

build_and_train_model('default.json', '')