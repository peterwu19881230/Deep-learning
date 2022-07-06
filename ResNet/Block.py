import torch
from torch.functional import Tensor
import torch.nn as nn

#############################################################################
# Blocks building the network
#############################################################################

class standard_block(nn.Module):
	""" Creates a standard residual block for ResNet.

	Args:
		filters: A positive integer. The number of filters for the first 
			convolution.
		projection_shortcut: The function to use for projection shortcuts
			(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
		first_num_filters: An integer. The number of filters to use for the
			first block layer of the model.
	"""
	def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
		super(standard_block, self).__init__()
		### YOUR CODE HERE
		"""if self.projection_shortcut is not None:""" #this could be in the forward session
		
		#Resnet from pytorch: https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
		self.projection_shortcut=nn.Sequential(
		nn.Conv2d(first_num_filters,filters,kernel_size=1 ,stride=strides,padding='same',bias=False),
		nn.BatchNorm2d(filters)
		)
		
		self.conv1=nn.Conv2d(first_num_filters,filters,kernel_size=3 ,stride=strides,padding='same')
		self.batch_norm1=nn.BatchNorm2d(filters)
		
		self.conv2=nn.Conv2d(filters,filters,kernel_size=3 ,stride=strides,padding='same')
		self.batch_norm2=nn.BatchNorm2d(filters)
	
		self.rel_u=nn.ReLU()
		### YOUR CODE HERE

	def forward(self, inputs: Tensor) -> Tensor:
		"""
		if self.projection_shortcut is not None: #about projection shortcut (residual connection): https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8
			identity=self.projection_shortcut(inputs)
		else:
			identity=inputs
		"""
		
		identity=inputs
		output=self.conv1(inputs)
		output=self.batch_norm1(output)
		output=self.rel_u(output)
		output=self.conv2(output)
		output=self.batch_norm2(output)
		#pad_arg=(identity.shape[2]-output.shape[2])//2
		#output=nn.functional.pad(output, (pad_arg, pad_arg, pad_arg, pad_arg, 0, 0))
		output+=self.projection_shortcut(identity) #residual connection
		output=self.rel_u(output)
	
		return output

class stack_layer(nn.Module):
	""" Creates one stack of standard blocks or bottleneck blocks.

	Args:
		filters: A positive integer. The number of filters for the first
				convolution in a block.
		block_fn: 'standard_block' or 'bottleneck_block'.
		strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
		resnet_size: #residual_blocks in each stack layer
		first_num_filters: An integer. The number of filters to use for the
			first block layer of the model.
	"""
	def __init__(self, filters, block_fn, strides, resnet_size, first_num_filters) -> None:
		super(stack_layer, self).__init__()
		filters_out = filters * 4 if block_fn is bottleneck_block else filters
		# projection_shortcut = ?        
		# Only the first block per stack_layer uses projection_shortcut and strides
		self.resnet_size = resnet_size
		self.projection_shortcut = nn.Conv2d(filters,filters_out,kernel_size=1 ,stride=strides) #projection shortcut is used to downsample
		self.block1 = block_fn(filters,None,1,first_num_filters)
		self.block2 = block_fn(filters,None,1,filters)
	
	

	def forward(self, inputs: Tensor) -> Tensor:
	
		output=self.block1(inputs)
	
		for i in range(1,self.resnet_size):    
			#print('forwarding i={} layer'.format(i))       
			output=self.block2(output)
		
		return output

class batch_norm_relu_layer(nn.Module):
	""" Perform batch normalization then relu.
	"""
	def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
		super(batch_norm_relu_layer, self).__init__()
		### YOUR CODE HERE
		self.batch_norm = nn.BatchNorm2d(num_features,eps,momentum)
		self.rel_u=nn.ReLU()
		### YOUR CODE HERE
	def forward(self, inputs: Tensor) -> Tensor:
		### YOUR CODE HERE
		output=self.batch_norm(inputs)
		output=self.rel_u(output)     
		return output
		### YOUR CODE HERE
		
class bottleneck_block(nn.Module):
	""" Creates a bottleneck block for ResNet.

	Args:
		filters: A positive integer. The number of filters for the first 
			convolution. NOTE: filters_out will be 4xfilters.
		projection_shortcut: The function to use for projection shortcuts
			(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
		first_num_filters: An integer. The number of filters to use for the
			first block layer of the model.
	"""
	def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
		super(bottleneck_block, self).__init__()

		### YOUR CODE HERE
		# Hint: Different from standard lib implementation, you need pay attention to 
		# how to define in_channel of the first bn and conv of each block based on
		# Args given above.
		self.conv1=nn.Conv2d(filters,first_num_filters*2,kernel_size=1 ,stride=strides)
		self.batch_norm1=nn.BatchNorm2d(first_num_filters*2)
		self.rel_u=nn.ReLU()
		self.conv2=nn.Conv2d(first_num_filters*2,first_num_filters*2*2,kernel_size=3 ,stride=strides)
		self.batch_norm2=nn.BatchNorm2d(first_num_filters*2*2)
		self.conv3=nn.Conv2d(first_num_filters*2*2,filters,kernel_size=1 ,stride=1)
	
	
		### YOUR CODE HERE

	def forward(self, inputs: Tensor) -> Tensor:
		### YOUR CODE HERE
		# The projection shortcut should come after the first batch norm and ReLU
		# since it performs a 1x1 convolution.
		if self.projection_shortcut is not None:
			identity=self.projection_shortcut(inputs)
		
		else:
			identity=inputs
	
		output=self.conv1(inputs)
		output=self.batch_norm1(output)
		output=self.rel_u(output)
		output=self.conv2(output)
		output=self.batch_norm2(output)
		output=self.conv3(output)
	
		#output=nn.functional.pad(output, (pad_arg, pad_arg, pad_arg, pad_arg, 0, 0))
		#output+=identity
		output=self.relu(output)
	
		return output
		### YOUR CODE HERE


class output_layer(nn.Module):
	""" Implement the output layer.

	Args:
		filters: A positive integer. The number of filters.
		resnet_version: 1 or 2, If 2, use the bottleneck blocks.
		num_classes: A positive integer. Define the number of classes.
	"""
	def __init__(self, filters, resnet_version, num_classes) -> None:
		super(output_layer, self).__init__()
		# Only apply the BN and ReLU for model that does pre_activation in each
		# bottleneck block, e.g. resnet V2.
		self.resnet_version=resnet_version
	
		if (resnet_version == 2):
			self.bn_relu = batch_norm_relu_layer(filters, eps=1e-5, momentum=0.997)
	
		self.last=nn.Sequential(
			nn.Flatten(),
			nn.Linear(262144,10),
			)

	def forward(self, inputs: Tensor) -> Tensor:
		if self.resnet_version == 2:
			output=self.bn_relu(inputs)
		output=self.last(inputs)

		return output       