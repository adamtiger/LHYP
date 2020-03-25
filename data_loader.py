from con_reader import CONreaderVM
from dicom_reader import DCMreaderVM
from con2img import draw_contourmtcs2image as draw
import numpy as np
from patient import Patient
import pickle
import os

class DataLoader:
	def __init__(self):
		self.fileLocation = '/userhome/student/lhyp/sample/'


	def calc_area(self, con):
		area = 0
		for i in range(len(con)-1):
			area += np.cross(con[i], con[i+1])
		area += np.cross(con[-1], con[0])
		return area

	def sort_cons(self, directory):
		cr = CONreaderVM(directory + 'sa/contours.con')
		contours = cr.get_hierarchical_contours()
		dr =  DCMreaderVM(directory + 'sa/images')
		_, __, weight, height, gender = cr.get_volume_data()
		num = [3,7,11]
		
		frm_num = list(contours[3])

		if(self.calc_area(contours[3][frm_num[0]]['ln']) 
			>= self.calc_area(contours[3][frm_num[1]]['ln'])):
			dyastole_fr = frm_num[0]
			systole_fr = frm_num[1]
		else:
			dyastole_fr = frm_num[1]
			systole_fr = frm_num[0]
		
		dy_images = []
		dy_contours = []
		sy_images = []
		sy_contours = []
		for i in num:
			dy_images.append(dr.get_image(i, dyastole_fr))
			dy_contours.append( contours[i][dyastole_fr]['ln'])
			sy_images.append(dr.get_image(i, systole_fr))
			sy_contours.append(contours[i][systole_fr]['ln'])

		return  weight, height, gender, np.array(dy_images, dtype=np.uint8), dy_contours, np.array(sy_images, dtype=np.uint8), sy_contours 


	def picklePatient(self, directory, id):
		weight, height, gender, dy_images, dy_contours, sy_images, sy_contours = self.sort_cons(directory)

		
		patient = Patient(gender, weight, height, dy_images,
						 dy_contours, sy_images, sy_contours)

		output = 'patient'+str(id)
		with open(output,'wb') as outfile:
			pickle.dump(patient,outfile)

	def readAllData(self):
		rootdir = self.fileLocation
		directories = next(os.walk(rootdir))[1]
		for i, directory in enumerate(directories):
			self.picklePatient(rootdir + directory, i)
    		
