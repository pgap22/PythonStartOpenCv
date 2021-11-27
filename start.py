from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
# Cargar la imagen y leerla
pixeles = imread('test1.jpg')

# Cargar el archivo de IA entrenado
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

# Detectar la cara
caras = classifier.detectMultiScale(pixeles)

# por cada cara detectada dibujar un rectangulo
for rectangulo in caras:
	# sacar los pixeles para despues dibujarlos
	x, y, width, height = rectangulo
	x2, y2 = x + width, y + height
	# Dibujar los cuadritos
	rectangle(pixeles, (x, y), (x2, y2), (212,255,255), 1)

# mostrar imagen
imshow('face detection', pixeles)

# cerrar la ventana
waitKey(0)
destroyAllWindows()
