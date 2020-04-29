from patient import Patient
import torch
import torchvision
import torchvision.transforms as transforms
from lhyp_dataset import LhypDataset
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

def train(batch_size, num_of_epochs, learning_rate):

	writer = SummaryWriter('runs/pretrained')

	trans = transforms.Compose([
		transforms.Resize((250,250)),
		transforms.RandomRotation(90, fill=(0,)),
		transforms.RandomCrop((230,230), pad_if_needed=True),
		transforms.ToTensor(),
		transforms.Normalize([0.485], [0.229]),
		transforms.Lambda(lambda x: x.repeat(3, 1, 1))
	])

	val_trans = transforms.Compose([
		transforms.Resize((250,250)),
		transforms.ToTensor(),
		transforms.Normalize([0.485], [0.229]),
		transforms.Lambda(lambda x: x.repeat(3, 1, 1))
	])
	
	dataset = LhypDataset('/mnt/c/cucc/LHYP/data', trans)
	val_dataset = LhypDataset('/mnt/c/cucc/LHYP/data', val_trans)

	dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

	model = torchvision.models.densenet161(pretrained=True, progress=True, memory_efficient=True)

	num_features = model.classifier.in_features
	model.classifier = nn.Linear(num_features, 2)
	
	if torch.cuda.is_available():
		model = model.cuda()

	criterion = nn.CrossEntropyLoss()

	optimizer = optim.Adam(model.parameters(), lr = learning_rate )
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_of_epochs, eta_min= learning_rate*10e-4)
	
	for iteration in range(num_of_epochs):
		running_loss = 0
		correct = 0.00
		total = 0.00

		model.train()
		for i, input_datas in enumerate(dataloader, 0):
			datas, labels = input_datas

			optimizer.zero_grad()
			
			outputs = model(datas)
			loss = criterion(outputs, labels)
			running_loss += loss.item()
			loss.backward()
			optimizer.step()

			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()
			writer.add_scalar('Training loss', running_loss, iteration)
			writer.add_scalar('Training accuracy', correct/total, iteration)
		print('Train loss: ',running_loss,' Train acc ',correct/total)

		val_running_loss = 0
		val_correct = 0.00
		val_total = 0.00

		model.eval()
		for i, val_input_datas in enumerate(val_dataloader, 0):
			val_datas, val_labels = val_input_datas
			
			val_outputs = model(val_datas)
			val_loss = criterion(val_outputs, val_labels)

			val_running_loss += val_loss.item()
			_, val_predicted = torch.max(val_outputs, 1)
			val_total += val_labels.size(0)
			val_correct += val_predicted.eq(val_labels).sum().item()
			writer.add_scalar('Validation loss', val_running_loss, iteration)
			writer.add_scalar('Validation accuracy', val_correct/val_total, iteration)

		print('Val loss: ',val_running_loss,' Val acc ',val_correct/val_total)

		scheduler.step()
	
	#torch.save(model.state_dict(), 'models/base.pth')

def main():
	train(8, 40, 1e-4)

if __name__ == "__main__":
	main()