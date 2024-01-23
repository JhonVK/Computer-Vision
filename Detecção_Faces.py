import cv2 # openCV

imagem=cv2.imread('C:/Users/joaov/Downloads/download.png')

cv2.imshow('Imagem 1', imagem)
cv2.waitKey()
cv2.destroyAllWindows()

detector_face = cv2.CascadeClassifier('C:/Users/joaov/OneDrive/Área de Trabalho/Python/haarcascade_frontalface_default.xml')

imagem_cinza=cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

cv2.imshow('Imagem CINZA', imagem_cinza)
cv2.waitKey()
cv2.destroyAllWindows()
