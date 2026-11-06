import numpy as np
import random
from math import floor
from PIL import Image
from pathlib import Path

H = 1000
W = 1000

ing_mat = np.zeros((W, H, 3), dtype = np.uint8)
Z_Buffer = np.full((W, H),np.inf, dtype = np.float32)

def baricentcricCoord(x:np.uint8,y:np.unit8,
                      x0:np.float64, y0:np.float64,
                      x1:np.float64, y1:np.float64,
                      x2:np.float64, y2:np.float64) -> tuple:
    """Считает барометрические координаты относительно 3 координат"""
    denominator = ((x0-x2)*(y1-y2))-((x1-x2)*(y0-y2))
    if denominator == 0 : return(-1,-1,1)
    lambda0 = (((x-x2)*(y1-y2))-((x1-x2)*(y-y2)))/denominator
    lambda1 = (((x0-x2)*(y-y2))-((x-x2)*(y0-y2)))/denominator
    lambda2 = 1.0 - lambda0 - lambda1

    return (lambda0, lambda1, lambda2)


def drawPoint(x:np.uint8,y:np.uint8,image: np.ndarray) -> None:
    image[y, x] = [0, 255, 0]

def drawTriangle(image: np.ndarray, 
            x0:np.float64, y0:np.float64,z0:np.float64,
            x1:np.float64, y1:np.float64,z1:np.float64,
            x2:np.float64, y2:np.float64,z2:np.float64, 
            ZBuffer: np.ndarray,color: int
                 ) -> None:
    MaxHeight = image.shape[0]
    MaxWidth = image.shape[1]
    xmin = floor(min(x0,x1,x2)) if min(x0,x1,x2) > 0 else 0
    xmax = floor(max(x0,x1,x2)) if max(x0,x1,x2) <= MaxWidth else MaxWidth

    ymin = floor(min(y0,y1,y2)) if min(y0,y1,y2) > 0 else 0
    ymax = floor(max(y0,y1,y2)) if max(y0,y1,y2) <= MaxHeight else MaxHeight

    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            lambds = baricentcricCoord(i+0.5,j+0.5,x0,y0,x1,y1,x2,y2)
            if(lambds[0]>=0 and lambds[1]>=0 and lambds[2]>=0):
                Z=lambds[0]*z0 + lambds[1]*z1 + lambds[2]*z2
                if Z < ZBuffer[j ,i]:
                    image[j,i] = color
                    ZBuffer[j ,i] = Z

def draw_line(x0:np.uint8, y0:np.uint8, x1:np.uint8, y1:np.uint8, image: np.ndarray) -> None:
    L = max(abs(floor(x0) - floor(x1)), abs(floor(y0) - floor(y1))) + 1
    if L == 1:
        image[y0, x0] = 0
    else:
        x,y = x0,y0
        for _ in range(L):
            image[floor(y),floor(x)] = [0, 255, 0]
            x += ((x1-x0)/L)
            y += ((y1-y0)/L)

def normal(x0:np.float64, y0:np.float64,z0:np.float64,
            x1:np.float64, y1:np.float64,z1:np.float64,
            x2:np.float64, y2:np.float64,z2:np.float64
            )-> np.ndarray:
    vec1 = np.array([x1-x2,y1-y2,z1-z2])
    vec2 = np.array([x1-x0,y1-y0,z1-z0])
    return  np.cross(vec1, vec2)

def cosin(norm: np.ndarray)-> np.float64:
    l = np.array([0,0,1])
    normal = norm
    return np.dot(normal, l)/(np.linalg.norm(normal)*np.linalg.norm(l))


path = Path(__file__).resolve().parent.parent / "model.obj"
file = open(path)

vertex = []
faces = []

for s in file:
    sd = s.split()
    if sd[0] == 'v':
        vertex.append([float(x) for x in sd[1:]])
    elif sd[0] == 'f':
        faces.append([int(x.split('/')[0]) for x in sd[1:]])

file.close()

for face in faces:
    p1 = vertex[face[0]-1]
    p2 = vertex[face[1]-1]
    p3 = vertex[face[2]-1]
    cos = cosin(normal(p1[0], p1[1], p1[2], 
                    p2[0], p2[1], p2[2], 
                    p3[0], p3[1], p3[2]))
    if(cos < 0):
        color = [-255*cos,20*cos,100*cos ]
        drawTriangle(ing_mat,
                    floor(p1[0]*5000+500),floor(-p1[1]*5000+500),floor(p1[2]*5000+500),
                    floor(p2[0]*5000+500),floor(-p2[1]*5000+500),floor(p2[2]*5000+500),
                    floor(p3[0]*5000+500),floor(-p3[1]*5000+500),floor(p3[2]*5000+500),
                    color=color, ZBuffer=Z_Buffer)
img = Image.fromarray(ing_mat)
img.save('img2.png')
