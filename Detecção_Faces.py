import cv2 # openCV

imagem=cv2.imread('C:/Users/joaov/Downloads/download.png')

cv2.imshow('Imagem 1', imagem)
cv2.waitKey()
cv2.destroyAllWindows()

detector_face = cv2.CascadeClassifier('C:/Users/joaov/Downloads/haarcascade_frontalface_default (1).xml')

imagem_cinza=cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

cv2.imshow('Imagem CINZA', imagem_cinza)
cv2.waitKey()
cv2.destroyAllWindows()

deteccoes=detector_face.detectMultiScale(imagem_cinza, scaleFactor=1.3, minSize=(30,30))

print(deteccoes) #x-y-balding box-balding box

for (x, y, l, a) in deteccoes:
  cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,255,0), 2)
cv2.imshow('imagem', imagem)
cv2.waitKey()
cv2.destroyAllWindows()
