from patient import Patient
from torch.utils.data import Dataset
import torch
from data_loader import DataLoader
from PIL import Image

class LhypDataset(Dataset):

	def __init__(self, dirpath, transform = None):
		dl = DataLoader()
		patients = dl.unpicklePatients(dirpath)
		self.data = []

		for patient in patients:
			if patient.pathology == 'Normal':
				label = 0
			else:
				label = 1
			for i in range(len(patient.dy_images)):
				self.data.append({'img': Image.fromarray(patient.dy_images[i]), 'label': label})
				self.data.append({'img': Image.fromarray(patient.sy_images[i]), 'label': label})
				
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		
		sample = self.data[idx]['img']

		if self.transform:
			sample = self.transform(sample)

		return sample, torch.tensor(self.data[idx]['label'])