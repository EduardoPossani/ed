import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
import os

classificador = Sequential()
#Classificando o tamanho da imagem a serem incluidas(128x128,3 == RGB)
classificador.add(InputLayer(shape = (128, 128, 3)))
classificador.add(Conv2D(32, (3, 3), activation = 'relu'))

#BatchNormalization para normalizar os dados, diminuindo o valor dos dados
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))

classificador.add(Flatten())
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units =  8, activation = 'softmax'))

classificador.summary()

classificador.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale=1/255, rotation_range=7, 
                                         horizontal_flip = True, shear_range=0.2, 
                                         height_shift_range= 0.07, zoom_range=0.2)

gerador_teste = ImageDataGenerator(rescale=1./255)

# Gerando os dados a partir da pasta 'train_set'
base_treinamento = gerador_treinamento.flow_from_directory(
    'C:/Users/eduar/OneDrive/Documentos/archive (2)/skin-disease-datasaet/train_set',  # caminho para a pasta de treino
    target_size=(128, 128),  # o tamanho para o qual as imagens serão redimensionadas
    batch_size=32,  # tamanho do lote de imagens
    class_mode='categorical'  # 'categorical' para classificação multiclasse
)

base_teste = gerador_teste.flow_from_directory(
    'C:/Users/eduar/OneDrive/Documentos/archive (2)/skin-disease-datasaet/test_set',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

classificador.fit(base_treinamento, epochs = 300, validation_data = base_teste)

# Carregar e processar a imagem
caminho_imagem = 'C:/Users/eduar/OneDrive/Documentos/archive (2)/skin-disease-datasaet/test_set/FU-ringworm/40_FU-ringworm (17).jpg'
img = image.load_img(caminho_imagem, target_size=(128, 128))  # Carrega a imagem no tamanho que a rede espera
img_array = image.img_to_array(img)  # Converte para array NumPy
img_array = np.expand_dims(img_array, axis=0)  # Adiciona uma dimensão extra (batch size)
img_array = img_array / 255.0  # Normaliza os pixels

# Fazer a previsão usando o classificador treinado
previsoes = classificador.predict(img_array)

# Exibir a imagem
plt.imshow(img)

 
plt.axis('off')
plt.show()

# Classes do modelo (supondo que sejam 8, como no seu código)
classes = ['Celulite', 'Impetigo', 'Pé de Atleta', 'Fungos nas unhas', 'Micose', 'Larva mirgans cutânea', 'Catapora', 'Cobreiro']

# Printar as porcentagens para cada classe
for i, probabilidade in enumerate(previsoes[0]):
    print(f'{classes[i]}: {probabilidade * 100:.2f}%')

# Identificar a classe com a maior probabilidade
classe_predita = classes[np.argmax(previsoes)]
print(f'\nClasse predita: {classe_predita} com {np.max(previsoes[0]) * 100:.2f}% de confiança.')
