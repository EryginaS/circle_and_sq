import numpy as np 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import cv2
import random
np.random.seed(42)
img_rows, img_cols = 128, 128 #размер изображения
#загружаем данные

#создаем фон
color = (0, 255, 255)
# просто фон 3-ёх канальный 128 на 128
img = np.zeros((128, 128, 3), np.uint8)
# фон красит в голубой
img[:, 0:1 * 128] = (255, 140, 0)
#на выходе изображение круга
def circle(img,n):
    a= random.randint(20, 108)
    b= random.randint(20,108)
    cv2.circle(img, (a, b), 20, color, -1)
    n=0
    return img,n
# на выходе изображение квадрата
def sq(img,n):
    a= random.randint(20, 108)
    b= random.randint(20,108)
    cv2.rectangle(img, a,b ,a+20,b+20, color, thickness=2,lineType=8,shift=0)
    n=1
    return img, n
#функция для нормализации данных
def normal(img):
    img=img.reshape(img.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    img=img.astype('float32')
    img /=255;
input_shape = (img_rows, img_cols, 1)
# Создаем последовательную модель
model = Sequential()

model.add(Conv2D(75, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(100, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
# Компилируем модель
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

# Обучаем сеть
i=0
n=0
while (i!=1000):
    if (i<500):
        img = circle(img, n)
        img = normal(img)
        model.fit(img, n, batch_size=200, epochs=10, validation_split=0.2, verbose=2)
    else:
        img = sq(img, n)
        img = normal(img)
        model.fit(img, n, batch_size=200, epochs=10, validation_split=0.2, verbose=2)

i=0
# Оцениваем качество обучения сети на тестовых данных

while (i!=100):
    if(i<50):
        img = circle(img, n)
        img = normal(img)
        scores = model.evaluate(img, n, verbose=0)
        print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))
    else:
        img = sq(img, n)
        img = normal(img)
        scores = model.evaluate(img, n, verbose=0)
        print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))




