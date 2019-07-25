import numpy as np
import pandas as pd
import unicodedata,re,math,random
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

#加载数据集
train_df = pd.read_csv('train.tsv', sep = '\t')

#显示数据集的信息
train_df.describe()   

#对数据集做描述性分析
train_df.describe()   
