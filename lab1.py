import numpy as np
from math import floor
from PIL import Image

ing_mat = np.zeros((1000, 1000, 3), dtype = np.uint8)

ing_mat+=0



def draw_line(x0:np.uint8, y0:np.uint8, x1:np.uint8, y1:np.uint8, image: np.ndarray) -> None:
    """Рисует линию между (x0, y0) и (x1, y1) через алгоритм DDA"""
    L = max(abs(floor(x0) - floor(x1)), abs(floor(y0) - floor(y1))) + 1

    if L == 1:
        image[y0, x0] = 0
    else:
        x,y = x0,y0

        for _ in range(L):
            image[floor(y),floor(x)] = [0, 255, 0]
            x += ((x1-x0)/L)
            y += ((y1-y0)/L)
        

#draw_line(10, 10, 1000, 900, ing_mat)

file = open("model.obj")

vertex = []
faces = []

for s in file:
    sd = s.split()
    
    if sd[0] == 'v':
        vertex.append([float(x) for x in sd[1:]])
    elif sd[0] == 'f':
        faces.append([int(x.split('/')[0]) for x in sd[1:]])


file.close()

for point in vertex:
    ing_mat[floor(-point[1]*5000+500), floor(point[0]*5000+500)] = [0, 255, 0]

print(len(vertex))
print(faces[0])

print(vertex[0])
for face in faces:
    for i in range(len(face)):
        p1 = int(face[i%len(face)])
        p2 = int(face[(i+1)%len(face)])

        point1 = vertex[p1-1]
        point2 = vertex[p2-1]
        
        draw_line(floor(point1[0]*5000+500),floor(-point1[1]*5000+500),
                  floor(point2[0]*5000+500),floor(-point2[1]*5000+500), ing_mat)


img = Image.fromarray(ing_mat)
img.save('img.png')
