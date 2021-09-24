# -*- coding: utf-8 -*-
import face_recognition
import argparse
import pickle
import cv2
from bs4 import BeautifulSoup
import subprocess as sp
tmp = sp.call('clear',shell=True)

camera_port = 0

camera = cv2.VideoCapture(camera_port)

emLoop= True
while(emLoop):
	retval, img = camera.read()
	img = cv2.blur(img, (1,1)) 
	cv2.imshow('Foto',img)
	k = cv2.waitKey(100)
	if k == ord('s'):
		emLoop= False
		cv2.destroyAllWindows()
		camera.release()
		ap = argparse.ArgumentParser()
		ap.add_argument("-e", "--encodings", required=True)
		ap.add_argument("-d", "--detection-method", type=str, default="cnn")
		args = vars(ap.parse_args())

		print("Carregando banco de dados de imagens...")
		data = pickle.loads(open(args["encodings"], "rb").read())

		image = img
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		print("Reconhecendo pessoa desaparecida. Aguarde...")
		boxes = face_recognition.face_locations(rgb,
			model=args["detection_method"])
		encodings = face_recognition.face_encodings(rgb, boxes)

		names = []

		for encoding in encodings:
			matches = face_recognition.compare_faces(data["encodings"],
				encoding)
			name = "unknown"

			if True in matches:
				matchedIdxs = [i for (i, b) in enumerate(matches) if b]
				counts = {}

				for i in matchedIdxs:
					name = data["names"][i]
					counts[name] = counts.get(name, 0) + 1

				name = max(counts, key=counts.get)
			
			names.append(name)
		achou = 0
		for name in names:
			achou = 1
			s = open('banco.xml')
			
			soup = BeautifulSoup(s, 'lxml')

			item_tags = soup.find_all('item')

			for item in item_tags:
				if item.id.text == name:
					print('='*60)
					print("Nome: {}".format( item.nome.text))
					print("Desaparecido em: {}".format( item.data.text))
					print("Cidade desaparecimento: {}".format( item.cidade.text))
					print("Contato: {}".format( item.contato.text))
					print('='*60)

		if achou == 0:
			print ("não reconhecido")


cv2.destroyAllWindows()