import torch
from torch.utils.data import Dataset
from torchtext.datasets import WikiText2

class WikiDataset(Dataset):
	def __init__(self, tokenizer, ctg='train'):
		raw_iter = WikiText2(split=ctg)
		self.data = [s.strip() for s in raw_iter if len(s.strip()) != 0]
#		self.data = tokenizer(data, padding='max_length', truncation=True)

	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, index):
		data = self.data[index]
	
		return data
	
