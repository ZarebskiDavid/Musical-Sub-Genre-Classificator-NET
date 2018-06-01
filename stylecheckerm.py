from keras import models
from skimage import io, transform
import sys
import numpy as np


def file2matrix(f):
	print('Converting file to a 200*200*3 matrix')
	target_size = 200

	try:

		img = io.imread(f)

		h , w, z = img.shape[0], img.shape[1], img.shape[2]
		if h > w:
			crop_size = round((h - w)/2)
			img = img[crop_size: h -crop_size, 0:w]

			print('crop size ', crop_size)

		elif h < w:
			crop_size = round((w - h)/2)
			img = img[0:h, crop_size:w-crop_size]
			print('crop size ', crop_size)

		H_ratio, W_ratio  = target_size/img.shape[0] , target_size/img.shape[1]

		img = transform.resize(img, (target_size,target_size), mode = 'edge')

		return(img)
	except: 
		print('print provide a colored, RBG styled picture')

def predictStyle(p):
	
	styles = {0:"Black Metal", 1:"Death Metal", 2:"Doom Metal", 3:"Folk Metal", 4:"Gothic Metal", 5:"Groove", 6:"Heavy Metal", 7:"Metalcore", 8:"Power Metal", 9:"Progressive Metal", 10:"Stoner", 11:"Symphonic", 12:"Thrash Metal"}
	model = models.load_model('modelBand.h5', custom_objects=None, compile=True)

	p = np.expand_dims(p,0)

	prediction = np.squeeze(model.predict(p))
	style = styles[np.argmax(prediction)]
	confidence =  round(np.amax(prediction)*100,2)

	print('==========> style: ', style, ' confidence: ', confidence, ' %')


def stylecheckerm():
	try:
		picture = str(sys.argv[1])
		i=file2matrix(picture)
		predictStyle(i)


	except:
		print('something went wrong')

	print('Done')

if __name__ == '__main__':
	stylecheckerm()
