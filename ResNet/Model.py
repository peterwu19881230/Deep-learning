import os
import time
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from tqdm import tqdm

from NetWork import ResNet
from Block import standard_block,batch_norm_relu_layer,bottleneck_block,stack_layer,output_layer
from ImageUtils import parse_record


#This script defines the training, validation and testing process.

class Cifar(nn.Module):
	def __init__(self, config):
		super(Cifar, self).__init__()
		self.config = config
		self.network = ResNet(
			self.config.resnet_version,
			self.config.resnet_size,
			self.config.num_classes,
			self.config.first_num_filters,
			self.config.batch_size			
		)

		# define cross entropy loss and optimizer
		self.loss = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(nn.Module.parameters(self), lr=0.001, weight_decay=self.config.weight_decay)
		#personal note: L2 regularization is often referred to as weight decay since it makes the weights smaller 
		scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.25) #Decays the learning rate of each parameter group by gamma every step_size epochs


	def train(self, x_train, y_train, max_epoch):
		self.network.train()
		# Determine how many batches in an epoch
		num_samples = x_train.shape[0]
		num_batches = num_samples // self.config.batch_size

		print('### Training... ###')
		for epoch in range(1, max_epoch+1):
			start_time = time.time()
			print('doing epoch {}'.format(epoch))
		
			# Shuffle
			shuffle_index = np.random.permutation(num_samples)
			curr_x_train = x_train[shuffle_index]
			curr_y_train = y_train[shuffle_index]
		
			for i in range(num_batches):
				
				x_batch=[]
				for j in range(self.config.batch_size*i,self.config.batch_size*(i+1)):
					processed_image=parse_record(curr_x_train[j], training=True)				
					x_batch.append(processed_image)
				x_batch=np.array(x_batch)
				y_batch = curr_y_train[self.config.batch_size*i:self.config.batch_size*(i+1)]
			
			
				self.batch_inputs=torch.from_numpy(x_batch).cuda()
				self.labels=torch.from_numpy(y_batch).cuda()
			
				logits=self.network(self.batch_inputs)                
				loss = self.loss(logits,self.labels)
			
				self.optimizer.zero_grad() 
				loss.backward()
				self.optimizer.step()

				print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
		
			duration = time.time() - start_time
			print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))
			if epoch % self.config.save_interval == 0:
				self.save(epoch)


	def test_or_validate(self, x, y, checkpoint_num_list):
		self.network.eval()
		print('### Test or Validation ###')
		for checkpoint_num in checkpoint_num_list:
			checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(checkpoint_num))
			self.load(checkpointfile)
		
			preds = []
			for i in tqdm(range(x.shape[0])):                
				processed_test_image=torch.from_numpy(np.array(parse_record(x[i], training=False))).cuda()
				processed_test_image=torch.unsqueeze(processed_test_image, dim=0)
				   
				pred=self.network(processed_test_image).cpu().data.numpy().argmax()
				preds.append(pred)
			
			accu=np.mean(np.all(np.array([y,preds]), axis=0))
			print('Test accuracy: {:.4f}'.format(accu))

	def save(self, epoch):
		checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
		os.makedirs(self.config.modeldir, exist_ok=True)
		torch.save(self.network.state_dict(), checkpoint_path)
		print("Checkpoint has been created.")

	def load(self, checkpoint_name):
		ckpt = torch.load(checkpoint_name, map_location="cpu")
		self.network.load_state_dict(ckpt, strict=True)
		print("Restored model parameters from {}".format(checkpoint_name))