import pickle
from data_loader import DataLoader
from sklearn.model_selection import train_test_split

def loadPatients(dirpath, num_of_classes):
    data = []
    dl = DataLoader('x', 'y')
    patients = dl.unpicklePatients(dirpath)

    if num_of_classes == 2:
        diagnosys = {'Normal': 0, 'Other': 1}
    elif num_of_classes == 3:
        diagnosys = {'Normal': 0, 'HCM': 1, 'Other': 2}

    train, val = train_test_split(
        patients, test_size=0.2, random_state=42, shuffle=True)

    val, test = train_test_split(
        val, test_size=0.5, random_state=42, shuffle=True)

    data = []
    for patient in train:
        if patient.pathology in diagnosys:
            label = diagnosys[patient.pathology]
        else:
            label =  diagnosys['Other']
        for i in range(len(patient.dy_images)):
            data.append({'img': Image.fromarray(
                patient.dy_images[i]), 'label': label})
            data.append({'img': Image.fromarray(
                patient.sy_images[i]), 'label': label})
    print(data)
    with open('../data/pickle/train'+num_of_classes, 'wb') as pik:
        pickle.dump(data, pik)

    data = []
    for patient in val:
        if patient.pathology in diagnosys:
            label = diagnosys[patient.pathology]
        else:
            label =  diagnosys['Other']
        for i in range(len(patient.dy_images)):
            data.append({'img': Image.fromarray(
                patient.dy_images[i]), 'label': label})
            data.append({'img': Image.fromarray(
                patient.sy_images[i]), 'label': label})
    print(data)
    with open('../data/pickle/val'+num_of_classes, 'wb') as pik:
        pickle.dump(data, pik)

    data = []
    for patient in test:
        if patient.pathology in diagnosys:
            label = diagnosys[patient.pathology]
        else:
            label =  diagnosys['Other']
        for i in range(len(patient.dy_images)):
            data.append({'img': Image.fromarray(
                patient.dy_images[i]), 'label': label})
            data.append({'img': Image.fromarray(
                patient.sy_images[i]), 'label': label})
    print(data)
    with open('../data/pickle/test'+num_of_classes, 'wb') as pik:
        pickle.dump(data, pik)