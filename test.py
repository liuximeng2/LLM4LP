from GLEM_util import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt

data_name = 'arxiv'
data = torch.load(f'data/{data_name}/{data_name}_fixed_sbert.pt', map_location='cpu')
#print(data.raw_text[0].split())

length = []
for i in range(len(data.raw_texts)):
    length.append(len(data.raw_texts[i].split()))
length = np.array(length)
print(length.mean())
print(f"std: {length.std()}")
print((length < 100).sum())

percentiles = np.percentile(length, [25, 50, 90])
print(f"25th Percentile: {percentiles[0]}")
print(f"50th Percentile (Median): {percentiles[1]}")
print(f"90th Percentile: {percentiles[2]}")

plt.hist(length, bins=20)
plt.title(data_name)
plt.xlabel('word count')
plt.savefig(f'img/{data_name}')