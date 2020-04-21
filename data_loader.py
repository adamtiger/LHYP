from con_reader import CONreaderVM
from dicom_reader import DCMreaderVM
from con2img import draw_contourmtcs2image as draw
import numpy as np
from patient import Patient
import pickle
import os

class DataLoader:
	def __init__(self):
		self.fileLocation = '/mnt/c/cucc/sample/'
		#self.fileLocation = 'C:\\Users\\Sonrisa\\Desktop\\lhyp\\sample\\'

	def calc_area(self, con):
		area = 0
		for i in range(len(con)-1):
			area += np.cross(con[i], con[i+1])
		area += np.cross(con[-1], con[0])
		return area

	def sort_cons(self, directory):
		cr = CONreaderVM(directory + '/sa/contours.con')
		#cr = CONreaderVM(directory + '\\sa\\contours.con')
		contours = cr.get_hierarchical_contours()
		dr =  DCMreaderVM(directory + '/sa/images')
		#dr =  DCMreaderVM(directory + '\\sa\\images')
		_, __, weight, height, gender = cr.get_volume_data()

		num = []
		temp_num = []
		frm_num = []
		for i in contours.keys():
			
			for j in list(contours[i]):
				if('ln' in contours[i][j].keys()):
					temp_num.append(i)
					frm_num.append(j)
			

		gap = len(temp_num)//3

		num.append(temp_num[1]+1)
		for k in range(2):
			num.append(temp_num[1]+1+k*gap)

		with open(directory+"/meta.txt", "r") as meta:
			pathology = meta.readline().split(' ')[1]
		# with open(directory+"\\meta.txt", "r") as meta:
		# 	pathology = meta.readline().split(' ')[1]

		

		if(self.calc_area(contours[num[0]][frm_num[0]]['ln']) 
			>= self.calc_area(contours[num[0]][frm_num[1]]['ln'])):
			dyastole_fr = frm_num[0]
			systole_fr = frm_num[1]
		else:
			dyastole_fr = frm_num[1]
			systole_fr = frm_num[0]
		
		dy_images = []
		sy_images = []

		for i in num:
			dy_images.append(dr.get_image(i, dyastole_fr))
			sy_images.append(dr.get_image(i, systole_fr))

		return  pathology, weight, height, gender, np.array(dy_images, dtype=np.uint8), np.array(sy_images, dtype=np.uint8)


	def picklePatient(self, directory, id):
		pathology, weight, height, gender, dy_images, sy_images = self.sort_cons(directory)

		
		patient = Patient(pathology, gender, weight, height, dy_images, sy_images)

		output = 'patient'+str(id)
		with open(output,'wb') as outfile:
			pickle.dump(patient,outfile)

	def readAllData(self):
		rootdir = self.fileLocation
		directories = next(os.walk(rootdir))[1]
		for i, directory in enumerate(directories):
			self.picklePatient(rootdir + directory, i)

	def unpicklePatients(self, directory):
		patients = []

		for root, dirs, files in os.walk(directory):
			for f in files:
				with open(directory +'/'+ f, 'rb') as infile:
					patients.append(pickle.load(infile, encoding='bytes'))
			break

		return patients

def main():
	dl = DataLoader()
	dl.readAllData()

if __name__ == "__main__":
	main()