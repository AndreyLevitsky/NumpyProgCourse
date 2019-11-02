import numpy as np
import math as m
import time as tm
import matplotlib.pyplot as plt
from imageio import imread, imsave
from PIL import Image

class StudentsRating:
    def __init__(self, studentsToGrade):
        self.studentsToGrade = studentsToGrade

    def __add__(self, other):
        self.studentsToGrade.update(other)
        return self

    def __sub__(self, other):
        self.studentsToGrade.pop(other, None)
        return self

    def get_greater_than(self, treshold):
        return [student for student, grades in self.studentsToGrade.items() if sum(grades) > treshold]

    def __str__(self):
        return ''.join([student.ljust(30) + ' '.join(list(map(str, grades))) + '\n' for student, grades in self.studentsToGrade.items()])


'''
print("Таблица студентов:")
students = {"Иванов Петр Николавич": (3, 3, 5, 4), "Петров Иван Константинович": (5, 4, 5, 4)}
sr = StudentsRating(students)
print(sr)

print("Добавление нового студента:")
sr += {"Сидоров Емельян Егорович": (4, 3, 4, 5)}
print(sr)

print("Вывод списка студентов с рейтингом выше чем пороговый:")
print(sr.get_greater_than(15))
print()

print("Удаление студента:")
sr -= "Иванов Петр Николавич"
print(sr)
'''

'''
np.random.seed(0)
matrixInt32 = np.array(np.random.randint(100, size=(10, 10)), dtype=np.int32)
matrixUint32 = np.copy(matrixInt32).astype(dtype=np.uint32)
matrixInt64 = np.copy(matrixInt32).astype(dtype=np.int64)
matrixUint64 = np.copy(matrixInt32).astype(dtype=np.uint64)
matrixFloat = np.copy(matrixInt32).astype(dtype=np.float)

print("matrixInt32:")
print("ndim: ", matrixInt32.ndim)
print("shape:", matrixInt32.shape)
print("size: ", matrixInt32.size)
print("dtype:", matrixInt32.dtype)
print("itemsize:", matrixInt32.itemsize, "bytes")
print("nbytes:", matrixInt32.nbytes, "bytes")
print()

print("matrixUint32:")
print("ndim: ", matrixUint32.ndim)
print("shape:", matrixUint32.shape)
print("size: ", matrixUint32.size)
print("dtype:", matrixUint32.dtype)
print("itemsize:", matrixUint32.itemsize, "bytes")
print("nbytes:", matrixUint32.nbytes, "bytes")
print()

print("matrixInt64:")
print("ndim: ", matrixInt64.ndim)
print("shape:", matrixInt64.shape)
print("size: ", matrixInt64.size)
print("dtype:", matrixInt64.dtype)
print("itemsize:", matrixInt64.itemsize, "bytes")
print("nbytes:", matrixInt64.nbytes, "bytes")
print()

print("matrixUint64:")
print("ndim: ", matrixUint64.ndim)
print("shape:", matrixUint64.shape)
print("size: ", matrixUint64.size)
print("dtype:", matrixUint64.dtype)
print("itemsize:", matrixUint64.itemsize, "bytes")
print("nbytes:", matrixUint64.nbytes, "bytes")
print()

print("matrixFloat:")
print("ndim: ", matrixFloat.ndim)
print("shape:", matrixFloat.shape)
print("size: ", matrixFloat.size)
print("dtype:", matrixFloat.dtype)
print("itemsize:", matrixFloat.itemsize, "bytes")
print("nbytes:", matrixFloat.nbytes, "bytes")
'''
'''
np.random.seed(0)
matrix = np.array(np.random.randint(-100, 100, size=(8, 8)))
print("Матрица:")
print(matrix)
print()

print("Четные:")
bool_idx_even = (matrix % 2 == 0)
print(bool_idx_even)
print()

print("В диапазоне от 10 до 50:")
bool_idx_range = (matrix >= 10) & (matrix <= 50)
print(bool_idx_range)
print()

print("Четные и в диапазоне от 10 до 50:")
print(bool_idx_even & bool_idx_range)
print()

print("Нечетные и модуль в диапазоне от 10 до 50:")
bool_idx_uneven_absrange = (matrix % 2 != 0) & (np.absolute(matrix) >= 10) & (np.absolute(matrix) <= 50)
print(bool_idx_uneven_absrange)
print()

print("Нечетные или отрицательные:")
bool_idx_uneven_neg = (matrix % 2 != 0) | (matrix < 0)
print(bool_idx_uneven_neg)
print()
'''

'''
np.random.seed(0)
matrix = np.array(np.random.randint(-100, 100, size=(8, 8)))
print("Исходная матрица:")
print(matrix)
print()

# 1 2 -> 3 4
# 4 3 -> 2 1

quadrant_1 = np.vsplit(np.hsplit(matrix, 2)[0], 2)[0]
quadrant_2 = np.vsplit(np.hsplit(matrix, 2)[1], 2)[0]
quadrant_3 = np.vsplit(np.hsplit(matrix, 2)[1], 2)[1]
quadrant_4 = np.vsplit(np.hsplit(matrix, 2)[0], 2)[1]

quadrant_34 = np.hstack([quadrant_3, quadrant_4])
quadrant_21 = np.hstack([quadrant_2, quadrant_1])
newMatrix = np.vstack([quadrant_34, quadrant_21])

print("Преобразованная матрица:")
print(newMatrix)
print()
'''
'''
np.random.seed(0)
# Нет
# (3,) -> (1, 1, 3) -> (100, 100, 3)
#                      (100, 100, 4)
# 3 != 4 -> нельзя
matrix = np.array(np.random.randint(100, size=(100, 100, 4)))
arr = np.arange(3)
print(matrix + arr)

# Да
# (7, 1, 5) -> (1, 7, 1, 5)  -> (15, 7, 6, 5)
#              (15, 1, 6, 5) -> (15, 7, 6, 5)
matrix1 = np.array(np.random.randint(100, size=(15, 1, 6, 5)))
matrix2 = np.array(np.random.randint(100, size=(7, 1, 5)))
print(matrix1 + matrix2)

# Да
# (3, 1) -> (1, 3, 1) -> (12, 3, 18)
#                        (12, 3, 18)
matrix1 = np.array(np.random.randint(100, size=(12, 3, 18)))
matrix2 = np.array(np.random.randint(100, size=(3, 1)))
print(matrix1 + matrix2)

# Нет
# (3, 1, 6) -> (1, 3, 1, 6) -> (10, 3, 1, 6)
#                              (10, 4, 1, 6)
# 3 != 4 -> нельзя
matrix1 = np.array(np.random.randint(100, size=(10, 4, 1, 6)))
matrix2 = np.array(np.random.randint(100, size=(3, 1, 6)))
print(matrix1 + matrix2)

# Да
# (3, 1) -> (1, 3, 1) -> (12, 3, 18)
#                        (12, 3, 18)
matrix1 = np.array(np.random.randint(100, size=(12, 3, 18)))
matrix2 = np.array(np.random.randint(100, size=(3, 1)))
print(matrix1 + matrix2)
'''


class ImprovedStudentsRating(StudentsRating):

    def __init__(self, studentsToGrade):
        super(ImprovedStudentsRating, self).__init__(studentsToGrade)
        self.Grades = np.array(list(studentsToGrade.values()))

    def average(self):
        return np.mean(self.Grades, axis=0)

    def min(self):
        return np.min(self.Grades, axis=0)

    def max(self):
        return np.max(self.Grades, axis=0)

'''
students = {"Иванов Петр Николавич": (3, 3, 5, 4), "Петров Иван Константинович": (5, 4, 5, 4), "Сидоров Емельян Егорович": (4, 3, 4, 5)}
isr = ImprovedStudentsRating(students)
print(isr)

print(isr.average())
print(isr.min())
print(isr.max())
'''


def Function1(vector):
    return 1 / (1 + np.exp(-vector))


def Function1S(vector):
    return np.array([1 / (1 + m.exp(-x)) for x in vector])


def Function2(vector):
    return np.maximum(vector, np.zeros_like(x))


def Function2S(vector):
    return np.array([max(0, x) for x in vector])


def LinearTransform(M, x, b):
    return np.dot(M, x) + b


def LinearTransformS(M, x, b):
    return np.array([sum([j * y for j, y in zip(i, x)]) + b for i in M])


def Transform(M1, M2, x, b1, b2):
    return Function2(LinearTransform(M2.T, Function1(LinearTransform(M1, x, b1)), b2))


def TransformS(M1, M2, x, b1, b2):
    return Function2S(LinearTransformS(M2.T, Function1S(LinearTransformS(M1, x, b1)), b2))


'''
np.random.seed(0)
M1 = np.random.randint(0, 5, size=[10000, 10000])
M2 = np.random.randint(0, 5, size=[10000, 10000])
x = np.array(np.random.randint(0, 5, size=10000))
b1 = 1
b2 = 2

print("Function1, numpy: ")
start_time = tm.time()
Function1(x)
end_time = tm.time()
print("--- %s seconds ---" % (end_time - start_time))

print("Function1S, vanilla python: ")
start_time = tm.time()
Function1S(x)
end_time = tm.time()
print("--- %s seconds ---" % (end_time - start_time))


print("Function2, numpy: ")
start_time = tm.time()
Function2(x)
end_time = tm.time()
print("--- %s seconds ---" % (end_time - start_time))

print("Function2S, vanilla python: ")
start_time = tm.time()
Function2S(x)
end_time = tm.time()
print("--- %s seconds ---" % (end_time - start_time))


print("LinearTransform, numpy: ")
start_time = tm.time()
LinearTransform(M1, x, b1)
end_time = tm.time()
print("--- %s seconds ---" % (end_time - start_time))

print("LinearTransformS, vanilla python: ")
start_time = tm.time()
LinearTransformS(M1, x, b1)
end_time = tm.time()
print("--- %s seconds ---" % (end_time - start_time))


M1 = np.random.randint(0, 3, size=[10000, 7000])
M2 = np.random.randint(0, 5, size=[10000, 7000])
x = np.array(np.random.randint(0, 5, size=7000))


print("Transform, numpy: ")
start_time = tm.time()
Transform(M1, M2, x, b1, b2)
end_time = tm.time()
print("--- %s seconds ---" % (end_time - start_time))

print("TransformS, vanilla python: ")
start_time = tm.time()
TransformS(M1, M2, x, b1, b2)
end_time = tm.time()
print("--- %s seconds ---" % (end_time - start_time))
'''

'''
x = np.arange(-np.pi, np.pi, 0.1)
x_tan = np.arange(-5*np.pi/12, 5*np.pi/12, 0.1)
x_cotan = np.arange(np.pi/12, 11*np.pi/12, 0.1)

y_sin = np.sin(x)
y_cos = np.cos(x)
y_tan = np.tan(x_tan)
y_cotan = 1 / np.tan(x_cotan)

plt.plot(x, y_sin)
plt.plot(x, y_cos)
#plt.plot(x_tan, y_tan)
#plt.plot(x_cotan, y_cotan)
plt.grid(color='b', linestyle='-.', linewidth=0.25)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine, Cosine, Tangent and Cotangent')
plt.legend(['Sine', 'Cosine', 'Tangent', 'Cotangent'])
plt.show()

x = np.arange(-np.pi, np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
y_tan = np.tan(x)
y_cotan = 1 / y_tan

plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.plot(x, y_tan)
plt.plot(x, y_cotan)
plt.grid(color='b', linestyle='-.', linewidth=0.25)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine, Cosine, Tangent and Cotangent')
plt.legend(['Sine', 'Cosine', 'Tangent', 'Cotangent'])

# Из-за того, что тангенс и котангенс имеют точки разрыва 2 порядка в -pi/2, pi/2 и -pi, 0, pi соотвественно,
# график не репрезентативен, поэтому можно взять другой отрезок x

x_tan = np.arange(-5*np.pi/12, 5*np.pi/12, 0.1)
y_tan = np.tan(x_tan)

plt.plot(x_tan, y_tan)
plt.grid(color='b', linestyle='-.', linewidth=0.25)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Tangent')
plt.legend(['Tangent'])


x_cotan = np.arange(np.pi/12, 11*np.pi/12, 0.1)
y_cotan = 1 / np.tan(x_cotan)

plt.plot(x_cotan, y_cotan)
plt.grid(color='b', linestyle='-.', linewidth=0.25)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Cotangent')
plt.legend(['Cotangent'])
'''

'''
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)

y_1 = 2 * x ** 2
plt.subplot(2, 2, 1)
plt.plot(x, y_1)
plt.title('y = 2 * x ** 2')
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.grid(color='b', linestyle='-.', linewidth=0.25)

y_2 = -2 * x ** 2
plt.subplot(2, 2, 2)
plt.plot(x, y_2)
plt.title('y = -2 * x ** 2')
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.grid(color='b', linestyle='-.', linewidth=0.25)

x_1 = 2 * y ** 2
plt.subplot(2, 2, 3)
plt.plot(y, x_1)
plt.title('x = 2 * y ** 2')
plt.xlabel('y axis label')
plt.ylabel('x axis label')
plt.grid(color='b', linestyle='-.', linewidth=0.25)

x_2 = -2 * y ** 2
plt.subplot(2, 2, 4)
plt.plot(y, x_2)
plt.title('x = -2 * y ** 2')
plt.xlabel('y axis label')
plt.ylabel('x axis label')
plt.grid(color='b', linestyle='-.', linewidth=0.25)

plt.show()
'''

'''
np.random.seed(0)
# В качестве распределения возмем нормальное стандартное распределение
mu, sigma = 0, 1.0
h = np.random.normal(mu, sigma, 1000)
# Формируем bins в соответствии с правилом трех сигм
bins = np.arange(-3 * sigma, 3 * sigma, 0.05)

plt.subplot(2, 1, 1)
plt.hist(h, bins=bins, density=False)
plt.title('Гистограмма частот норм. стандартного распр.')
plt.xlabel('Значения')
plt.ylabel('Частота')

plt.subplot(2, 1, 2)
plt.hist(h, bins=bins, density=True)
plt.title('Гистограмма отн. частот норм. стандартного распр.')
plt.xlabel('Значения')
plt.ylabel('Частота')

plt.show()
'''

'''
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3)
plt.colorbar()  # Отображаем цветовую шкалу
plt.show()

rng = np.random.RandomState(0)
x = np.random.laplace(size=100)
y = np.random.laplace(size=100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3)
plt.colorbar()  # Отображаем цветовую шкалу
plt.show()
'''
'''
# Читаем изображение из файла JPEG в массив numpy.
#im = Image.open('cat.jpg', mode='r')
img = imread('cat.jpg')

print(img.dtype, img.shape)  # "uint8 (400, 248, 3)"

# Мы можем изменить оттенок изображения, масштабируя каждый из цветовых каналов
# с помощью константы. Изображение имеет форму (400, 248, 3);
# мы умножаем его на массив [1, 0.95, 0.9] формы (3);
# Трансляция массивов numpy привеодит к тому, что красный канал не изменит значение,
# а зеленый и синий каналы будут умножены на 0,95 и 0,9 соответственно.
img_tinted = np.array(img * [1, 0.95, 0.9]).astype(dtype=np.uint8)

img_tinted = Image.fromarray(obj=img_tinted)

# Изменим размер изображения на 300x300.
#img_tinted = imresize(img_tinted, (300, 300))
img_tinted = img_tinted.resize((300, 300))

# Сохраним на диск обработанное изображение.
imsave('cat_tinted.jpg', img_tinted)
'''
'''
img = imread('cat.jpg')
img_tinted = img * [1, 0, 0.9]

fig=plt.figure(figsize=(16, 8)) # Установим размер фигуры для отрисовки графиков.

# Отобразим исходное изображение
plt.subplot(1, 2, 1)
plt.imshow(img)

# Отобразим измененное изображение
plt.subplot(1, 2, 2)

# Небольшая проблема с imshow заключается в том, что функция может дать странные результаты
# если данные не являются uint8. Чтобы избежать этого, нужно привести изображение к типу
# uint8 перед его отображением.
plt.imshow(np.uint8(img_tinted))
plt.show()
'''

'''
img1 = imread('Lenna.png')
img2 = imread('Lenna.png')

r = img1[..., 0]
g = img1[..., 1]
b = img1[..., 2]

plt.subplot(311)
plt.hist(r, bins=np.arange(256))
plt.title('Red')
plt.xlabel('Значения')
plt.ylabel('Частота')

plt.subplot(312)
plt.hist(g, bins=np.arange(256))
plt.title('Green')
plt.xlabel('Значения')
plt.ylabel('Частота')

plt.subplot(313)
plt.hist(b, bins=np.arange(256))
plt.title('Blue')
plt.xlabel('Значения')
plt.ylabel('Частота')

plt.show()
'''

'''
img1 = imread('Lenna.png')
img2 = imread('Lenna.png')

img1[..., 0] *= 0
img2[..., 1] *= 0
img2[..., 2] *= 0

fig = plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(np.uint8(img1))

plt.subplot(1, 2, 2)
plt.imshow(np.uint8(img2))

plt.show()
'''
