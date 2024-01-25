import cv2

imagem=cv2.imread('C:/Users/joaov/Downloads/pessoas.jpg')

cv2.imshow('1', imagem)
cv2.waitKey()
cv2.destroyAllWindows()

detector=cv2.CascadeClassifier('C:/Users/joaov/Downloads/fullbody.xml')

imagem_cinza=cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

cv2.imshow('1', imagem_cinza)
cv2.waitKey()
cv2.destroyAllWindows()

deteccoes=detector.detectMultiScale(imagem_cinza, scaleFactor=1.01, minSize=(50,50))

print(deteccoes)

for (x, y, l, a) in deteccoes:
  cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,255,0), 1)
cv2.imshow('imagem', imagem)
cv2.waitKey()
cv2.destroyAllWindows()
