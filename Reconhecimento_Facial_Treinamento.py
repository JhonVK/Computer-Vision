import cv2
from PIL import Image
import os
import zipfile
import numpy as np

path='C:/Users/joaov/OneDrive/Área de Trabalho/Python/yalefaces.zip'
zip_object=zipfile.ZipFile(file=path, mode='r')
zip_object.extractall('C:/Users/joaov/OneDrive/Área de Trabalho/Python')
zip_object.close()

def dados_imagem():
    caminhos=[os.path.join('C:/Users/joaov/OneDrive/Área de Trabalho/Python/yalefaces/train', f)for f in os.listdir('C:/Users/joaov/OneDrive/Área de Trabalho/Python/yalefaces/train')]
    faces=[]
    ids=[]
    for caminho in caminhos: 
        imagem=Image.open(caminho).convert('L')#L é um canal de cores 
        imagem_np=np.array(imagem, 'uint8')
        id=int(os.path.split(caminho)[1].split('.')[0].replace('subject', ''))
        ids.append(id)
        faces.append(imagem_np)
    return np.array(ids), faces

ids, faces= dados_imagem()
print(ids)
print(faces)

lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.train(faces, ids)
lbph.write('C:/New folder/classificadorLBPH.yml')
