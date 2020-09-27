from con_reader import CONreaderVM
from dicom_reader import DCMreaderVM
from con2img import draw_contourmtcs2image as draw
import numpy as np
from patient_v2 import Patient
import pickle
import os
import sys
import getopt


class DataLoader:
    def __init__(self, inputdir, outputdir):
        self.fileLocation = inputdir
        self.outputdir = outputdir

    def sort_cons(self, directory):
        cr = CONreaderVM(directory + '/sa/contours.con')
        dr = DCMreaderVM(directory + '/sa/images')
        _, __, weight, height, gender = cr.get_volume_data()

        num = []
        slice_num = dr.num_slices
        frm_num = dr.num_frames

        gap = slice_num//3

        num.append(gap//2 + 1)

        for k in range(2):
            num.append(num[k] + gap)

        with open(directory+"/meta.txt", "r") as meta:
            pathology = meta.readline().split(' ')[1]

        images = []

        for i in num:
            for j in range(frm_num):
                images.append(dr.get_image(i, j))

        return pathology, weight, height, gender, np.array(images, dtype=np.uint8)

    def picklePatient(self, directory, id):
        pathology, weight, height, gender, images = self.sort_cons(directory)

        patient = Patient(pathology, gender, weight, height, images)

        output = self.outputdir + '/' + str(directory.split('/')[-1])
        with open(output, 'wb') as outfile:
            pickle.dump(patient, outfile)

    def readAllData(self):
        rootdir = self.fileLocation

        directories = next(os.walk(rootdir))[1]
        for i, directory in enumerate(directories):
            if not os.path.exists(self.outputdir + '/' + str(directory.split('/')[-1])):
                self.picklePatient(rootdir + directory, i)

    def unpicklePatients(self, directory):
        patients = []

        for root, dirs, files in os.walk(directory):
            for f in files:
                with open(directory + '/' + f, 'rb') as infile:
                    patients.append(pickle.load(infile, encoding='bytes'))
            break

        return patients


def main(argv):
    dl = DataLoader(argv[0], argv[1])
    dl.readAllData()


if __name__ == "__main__":
    main(sys.argv[1:])
