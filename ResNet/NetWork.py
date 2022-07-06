import torch
from torch.functional import Tensor
import torch.nn as nn
from Block import standard_block,batch_norm_relu_layer,bottleneck_block,stack_layer,output_layer

#This script defines the network.


class ResNet(nn.Module):
	def __init__(self,
			resnet_version,
			resnet_size,
			num_classes,
			first_num_filters,
			batch_size
		):
		"""
		1. Define hyperparameters.
		Args:
			resnet_version: 1 or 2, If 2, use the bottleneck blocks.
			resnet_size: A positive integer (n).
			num_classes: A positive integer. Define the number of classes.
			first_num_filters: An integer. The number of filters to use for the
				first block layer of the model. This number is then doubled
				for each subsampling block layer.
	
		2. Classify a batch of input images.

		Architecture (first_num_filters = 16):
		layer_name      | start | stack1 | stack2 | stack3 | output      |
		output_map_size | 32x32 | 32X32  | 16x16  | 8x8    | 1x1         |
		#layers         | 1     | 2n/3n  | 2n/3n  | 2n/3n  | 1           |
		#filters        | 16    | 16(*4) | 32(*4) | 64(*4) | num_classes |

		n = #residual_blocks in each stack layer = self.resnet_size
		The standard_block has 2 layers each.
		The bottleneck_block has 3 layers each.
	
		Example of replacing:
		standard_block      conv3-16 + conv3-16
		bottleneck_block    conv1-16 + conv3-16 + conv1-64

		Args:
			inputs: A Tensor representing a batch of input images.
	
		Returns:
			A logits Tensor of shape [<batch_size>, self.num_classes].
		"""
		super(ResNet, self).__init__()
		self.resnet_version = resnet_version
		self.resnet_size = resnet_size
		self.num_classes = num_classes
		self.first_num_filters = first_num_filters
		self.batch_size=batch_size
		
		
		self.start_layer=nn.Sequential(
		nn.Conv2d(3,self.first_num_filters,kernel_size=3,padding='same'), #first 2 arguments: input channel, output channel
		nn.ReLU()
		)



		# We do not include batch normalization or activation functions in V2
		# for the initial conv1 because the first block unit will perform these
		# for both the shortcut and non-shortcut paths as part of the first
		# block's projection.
		
		
		if self.resnet_version == 1:
			self.batch_norm_relu_start = batch_norm_relu_layer(
				num_features=self.first_num_filters, 
				eps=1e-5, 
				momentum=0.997,
			)
		if self.resnet_version == 1:
			block_fn = standard_block
		else:
			block_fn = bottleneck_block

		self.stack_layers = nn.ModuleList()
		#self.stack_layers.append(stack_layer(self.first_num_filters*4,block_fn,1,self.resnet_size, self.first_num_filters))
		#self.stack_layers.append(stack_layer(self.first_num_filters*4*2,block_fn,2,self.resnet_size, self.first_num_filters*4))	
		#self.stack_layers.append(stack_layer(self.first_num_filters*4*2*2,block_fn,2,self.resnet_size, self.first_num_filters*4*2))
		self.stack_layers.append(stack_layer(self.first_num_filters*4,block_fn,1,6, self.first_num_filters))
		self.stack_layers.append(stack_layer(self.first_num_filters*4*2,block_fn,2,6, self.first_num_filters*4))	
		self.stack_layers.append(stack_layer(self.first_num_filters*4*2*2,block_fn,2,6, self.first_num_filters*4*2))


		self.output_layer = output_layer(32, self.resnet_version, self.num_classes)

	def forward(self, inputs):
	
		inputs=inputs.float()
		outputs = self.start_layer(inputs)
	
		"""
		if self.resnet_version == 1:
			outputs = self.batch_norm_relu_start(outputs)
		"""
	
		for i in range(3):
			outputs = self.stack_layers[i](outputs)
		outputs = self.output_layer(outputs)
		
	
		return outputs

