from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, MaxPool2D

img_width, img_height = 128, 128

#建立VGG网络
model = Sequential
model.add(ZeroPadding2D((1,1), batch_input_shape=(1, 3, img_width, img_height)))
first_layer = model.layers[-1]
input_img = first_layer.input

model.add(Conv2D(64,3,3,
				 activation='relu',
				 name='conv1_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64,3,3,
				 activation='relu',
				 name='conv1_2'))
model.add(MaxPool2D((2,2),
					strides=(2,2)))


model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128,3,3,
				 activation='relu',
				 name='conv2_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128,3,3,
				 activation='relu',
				 name='conv2_2'))
model.add(MaxPool2D((2,2),
					strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256,3,3,activation='relu',name='conv3_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256,3,3,activation='relu',name='conv3_2'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256,3,3,activation='relu',name='conv3_3'))
model.add(MaxPool2D((2,2), strides=(2,2)))


model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,3,3,activation='relu',name='conv4_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,3,3,activation='relu',name='conv4_2'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,3,3,activation='relu',name='conv4_3'))
model.add(MaxPool2D((2,2), strides=(2,2)))


model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,3,3,activation='relu',name='conv5_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,3,3,activation='relu',name='conv5_2'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,3,3,activation='relu',name='conv5_3'))
model.add(MaxPool2D((2,2), strides=(2,2)))

layer_dict = dict([(layer.name, layer) for layer in model.layers])



import h5py

weights_path = 'vgg16_weigths.h5'

f=h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
	if k >= len(model.layers):
		break
	g = f['layer_{}'.format(k)]
	weights = g['param_{}'.format(p) for p in range(g.attrs['nb_params'])]
	model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')

from 