import os
import numpy as np
import tensorflow as tf
from numpy import argmin
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2


class faceclassmodel():
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = keras.models.load_model("./facemodel/facenet_keras.h5")
        # self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        print("Model Loaded. Compiling Manually ...")
        self.model.compile()
        # print(self.model.summary())

    def gimme_embeddings(self, imagedata):
        if type(imagedata) == str:
            imagedata = cv2.imread(imagedata)
        img = cv2.resize(imagedata, (160, 160), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        m = img.mean()
        s = img.std()
        x = np.expand_dims((img - m) / s, axis=0)
        images = np.vstack([x])
        emb = self.model.predict(images)
        return emb

    def face_embeddings(self, imagedata, showfaces=True):
        if type(imagedata) == str:
            imagedata = cv2.imread(imagedata)
        img = cv2.resize(imagedata, (512, 512), interpolation=cv2.INTER_AREA)
        imgcopy = np.copy(img)
        plt.imshow(img)
        plt.show()
        faces = self.detect_face(img)
        print("faces", faces)
        embs = []
        if faces==():
            print("No faces found")
            embs.append(self.gimme_embeddings(imgcopy))
        else:
            for (x, y, w, h) in faces:
                embs.append(self.gimme_embeddings(imgcopy[y:y + h, x:x + w]))
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if showfaces:
            plt.figure()
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return img, faces, embs

    def detect_face(self, imagedata):
        if type(imagedata) == str:
            imagedata = cv2.imread(imagedata)
        img = cv2.resize(imagedata, (512, 512), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, 0)
        return faces

    def plot_faces(self, imagedata):
        if type(imagedata) == str:
            imagedata = cv2.imread(imagedata)
        img = cv2.resize(imagedata, (512, 512), interpolation=cv2.INTER_AREA)
        faces = self.detect_face(img)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # cv2_imshow(img)

    def train_model(self, train_data, validation_data, epochs):
        history = self.model.fit(
            train_data,
            epochs=epochs,
            verbose=1,
            validation_data=validation_data)
        return history


class Comparer():
    def __init__(self):
        self.facemodel = faceclassmodel()
        self.threshold = 12.0
        dataset_path = "./dataset"
        self.savedembs = []
        for dirs in os.listdir(dataset_path):
            for file in os.listdir(dataset_path + "/" + dirs):
                if file.endswith("emb1.npy"):
                    emb1 = np.load(dataset_path + "/" + dirs + "/" + file)
                    self.savedembs.append((dirs, emb1[0]))

    def compare_embeddings(self, emb1, emb2):
        return np.linalg.norm(emb1 - emb2)

    def find_match(self, imagedata):
        if type(imagedata) == str:
            imagedata = plt.imread(imagedata)
        _, _, thisemb = self.facemodel.face_embeddings(imagedata, showfaces=False)
        alldifferences = []
        for first in self.savedembs:
            first_name = first[0]
            differences = self.compare_embeddings(thisemb, first[1])
            alldifferences.append(differences)
            if differences < self.threshold:
                return first_name, differences
        return "Unknown, closest match: "+self.savedembs[argmin(alldifferences)][0], min(alldifferences)


