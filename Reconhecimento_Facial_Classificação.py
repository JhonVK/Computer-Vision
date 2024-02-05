import cv2
from PIL import Image
import os
import zipfile
import numpy as np

reconhecedor= cv2.face.LBPHFaceRecognizer_create()

reconhecedor.read('C:/New folder/classificadorLBPH.yml')

imagem_teste= 'c:/Users/joaov/OneDrive/Área de Trabalho/Python/yalefaces/test/subject10.sad.gif'

imagem=Image.open(imagem_teste).convert('L')
imagem_np=np.array(imagem, 'uint8')
print(imagem_np)

id_previsto, _=reconhecedor.predict(imagem_np)
print(id_previsto)

idcorreto = int(os.path.split(imagem_teste)[1].split('.')[0].replace('subject', ''))
print(idcorreto)
x=10
y=10
cv2.putText(imagem_np, 'P: ' + str(id_previsto), (x,y + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
cv2.putText(imagem_np, 'C: ' + str(idcorreto), (x,y + 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
cv2.imshow('img', imagem_np)
cv2.waitKey()
cv2.destroyAllWindows()
