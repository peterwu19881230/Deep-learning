import numpy as np

""" This script implements the functions for data augmentation and preprocessing.
"""

def parse_record(record, training):
	""" Parse a record to an image and perform data preprocessing.

	Args:
		record: An array of shape [3072,]. One row of the x_* matrix.
		training: A boolean. Determine whether it is in training mode.

	Returns:
		image: An array of shape [3, 32, 32].
	"""
	# Reshape from [depth * height * width] to [depth, height, width].
	image = record.reshape((3, 32, 32))
	image = preprocess_image(image, training)
	return image

def preprocess_image(image, training):
	""" Preprocess a single image of shape [depth, height, width].

	Args:
		image: An array of shape [3, 32, 32].
		training: A boolean. Determine whether it is in training mode.

	Returns:
		image: An array of shape [3, 32, 32].
	"""

	# Resize the image to add four extra pixels on each side.     
	#should consider RGB, so should be: 3072= 3 (R-G-B in sequence) X 32 (rows) X 32 (columns) => 3 X (4+32+4) X (4+32+4)        
	if training:
		pass
		"""
		### YOUR CODE HERE       
		def pad_with(vector, pad_width, iaxis, kwargs):
			pad_value = kwargs.get('padder', 0)
			vector[:pad_width[0]] = pad_value
			vector[-pad_width[1]:] = pad_value
	
		#need to consider RGB
		padded_image=[]
		for i in range(len(image)): #for R, G, B, respectively
			padded_image.append(np.pad(image[i],4,pad_with))
		padded_image=np.array(padded_image)
		image=padded_image        
		### YOUR CODE HERE

		### YOUR CODE HERE
		# Randomly crop a [32, 32] section of the image.
		# HINT: randomly generate the upper left point of the image
		cropped_image=[]
		for i in range(len(image)):
			upper=(np.random.choice(9),np.random.choice(9)) #can move 9 cells for row/column
			cropped_image.append(image[i][upper[0]:(upper[0]+32), upper[1]:(upper[1]+32)]) 
		cropped_image=np.array(cropped_image)    
		image=cropped_image                  
		### YOUR CODE HERE

		### YOUR CODE HERE
		# Randomly flip the image horizontally.
		if np.random.choice(2): #random boolean 
			flipped_image=[]
			for i in range(len(image)):
				flipped_image.append(np.flip(image[i],axis=1))
			flipped_image=np.array(flipped_image)    
			image=flipped_image              
		### YOUR CODE HERE
		"""

	### YOUR CODE HERE
	# Subtract off the mean and divide by the standard deviation of the pixels.
	normalized_image=[]
	for i in range(len(image)):
		normalized_image.append((image[i]-np.mean(image[i]))/np.std(image[i]))    
	normalized_image=np.array(normalized_image)    
	image=normalized_image 
	### YOUR CODE HERE

	return image