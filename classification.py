# USAGE
# python encode_faces.py --dataset dataset --encodings encodings.pickle


from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# constrói e analisa os argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="caminho do diretório de faces + imagens")
ap.add_argument("-e", "--encodings", required=True,
	help="caminho para as faces codificadas")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="modelo de detecção de face a ser utilizado: `hog` ou `cnn`")
args = vars(ap.parse_args())

# pega o caminho das imagens inseridas no banco de imagens
print("[INFO] processando rostos...")
imagePaths = list(paths.list_images(args["dataset"]))

# inicializa uma lista de codificações e nomes conhecidos
knownEncodings = []
knownNames = []

# loop sobre o caminho das imagens
for (i, imagePath) in enumerate(imagePaths):
	# pega o nome da pessoa do caminho da imagem
	print("[i] processando imagem {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# carrega a imagem e converte do RGB do OpenCV para o RGB do dlib
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detecta a coordenada (x,y) das caixas delimitadoras correspondentes 
	# a cada face na imagem inserida
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	# computa cada face incorporada
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop nas codificações
	for encoding in encodings:
		# adiciona cada imagem codificada e o nome para a 
		# lista de nomes e codificações conhecidas
		knownEncodings.append(encoding)
		knownNames.append(name)

# grava as codificações das imagens e os nomes no disco
print("[INFO] Serializando codificacoes...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
