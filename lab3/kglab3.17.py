import numpy as np
from math import floor
from PIL import Image, ImageTk
from numba import njit
import tkinter as tk
import random


@njit(cache=True)
def baricentcricCoord(x:np.uint8,y:np.uint8,
                      x0:np.float64, y0:np.float64,
                      x1:np.float64, y1:np.float64,
                      x2:np.float64, y2:np.float64) -> np.array:
    """Считает барометрические координаты относительно 3 координат"""
    denominator = ((x0-x2)*(y1-y2))-((x1-x2)*(y0-y2))
    if denominator == 0 : return np.array([-1.0,-1.0,1.0])

    lambda0 = (((x-x2)*(y1-y2))-((x1-x2)*(y-y2)))/denominator
    lambda1 = (((x0-x2)*(y-y2))-((x-x2)*(y0-y2)))/denominator
    lambda2 = 1.0 - lambda0 - lambda1

    return np.array([lambda0, lambda1, lambda2])

@njit(cache=True)
def cross(vec1: np.ndarray, vec2:np.ndarray) -> np.array:
    vec = [vec1[1]*vec2[2]-vec1[2]*vec2[1],
           -(vec1[0]*vec2[2]-vec1[2]*vec2[0]),
           vec1[0]*vec2[1]-vec1[1]*vec2[0]]
    return np.array(vec)

@njit(cache=True)
def dot(vec1: np.ndarray, vec2:np.ndarray) -> np.float64:
    return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2] 

@njit(cache=True)
def norm(vec: np.ndarray) -> np.float64:
    return  np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    

@njit(cache=True)
def draw_line(x0:np.uint8, y0:np.uint8, x1:np.uint8, y1:np.uint8, image: np.ndarray) -> np.ndarray:
    L = max(abs(floor(x0) - floor(x1)), abs(floor(y0) - floor(y1))) + 1
    if L == 1:
        image[y0, x0] = 0
    else:
        x,y = x0,y0
        for _ in range(L):
            image[floor(y),floor(x)] = [0,255,0]
            x += ((x1-x0)/L)
            y += ((y1-y0)/L)
    return np.array()

@njit(cache=True)
def normal(x0:np.float64, y0:np.float64,z0:np.float64,
            x1:np.float64, y1:np.float64,z1:np.float64,
            x2:np.float64, y2:np.float64,z2:np.float64
            )-> np.ndarray:
    vec1 = np.array([x1-x2,y1-y2,z1-z2])
    vec2 = np.array([x1-x0,y1-y0,z1-z0])
    return cross(vec1, vec2)

@njit(cache=True)
def cosin(normal: np.ndarray)-> np.float64:
    l = np.array([0,0,1])
    return dot(normal, l)/(norm(normal)*norm(l))

@njit(cache=True)
def drawTriangle(image: np.ndarray, 
            x0:np.float64, y0:np.float64,z0:np.float64,
            x1:np.float64, y1:np.float64,z1:np.float64,
            x2:np.float64, y2:np.float64,z2:np.float64, 
            ZBuffer: np.ndarray,color: int
                 ) -> np.ndarray:
    MaxHeight = image.shape[0]
    MaxWidth = image.shape[1]
    xmin = floor(min(x0,min(x1,x2))) if min(x0,min(x1,x2)) > 0 else 0
    xmax = floor(max(x0,max(x1,x2))) if max(x0,max(x1,x2)) <= MaxWidth else MaxWidth

    ymin = floor(min(y0,min(y1,y2))) if min(y0,min(y1,y2)) > 0 else 0
    ymax = floor(max(y0,max(y1,y2))) if max(y0,max(y1,y2)) <= MaxHeight else MaxHeight

    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            lambds = baricentcricCoord(i+0.5,j+0.5,x0,y0,x1,y1,x2,y2)
            if(lambds[0]>=0 and lambds[1]>=0 and lambds[2]>=0):
                Z=lambds[0]*z0 + lambds[1]*z1 + lambds[2]*z2
                if Z < ZBuffer[j ,i]:
                    image[j,i] = color
                    ZBuffer[j ,i] = Z
    return np.array(0)


@njit(cache=True)
def renderScene(vertex: np.ndarray, faces: np.ndarray,
                   scale: int, shift: int,
                   invertY: int,  ZBuffer: np.ndarray,
                   ingMat: np.ndarray,
                   colorR: np.uint8, colorG: np.uint8,
                   colorB: np.uint8) -> np.ndarray:
    for face in faces:
        x0 = vertex[face[0]-1][0]
        y0 = vertex[face[0]-1][1]
        z0 = vertex[face[0]-1][2]

        x1 = vertex[face[1]-1][0]
        y1 = vertex[face[1]-1][1]
        z1 = vertex[face[1]-1][2]

        x2 = vertex[face[2]-1][0]
        y2 = vertex[face[2]-1][1]
        z2 = vertex[face[2]-1][2]

        px0 = floor(x0*scale + shift)
        py0 = floor(invertY*y0*scale + shift)

        px1 = floor(x1*scale + shift)
        py1 = floor(invertY*y1*scale + shift)

        px2 = floor(x2*scale + shift)
        py2 = floor(invertY*y2*scale + shift)

        cos = cosin(normal(x0,y0,z0, 
                           x1,y1,z1, 
                           x2,y2,z2,))
        if(cos < 0):
            color = np.array([colorR*cos,
                              colorG*cos,
                              colorB*cos])
            drawTriangle(ingMat,
                        px0,py0,z0,
                        px1,py1,z1,
                        px2,py2,z2,
                        color=color, ZBuffer=ZBuffer)
    return np.array(0)



class Scene:
    def __init__(self, fileName: str):
        file = open(fileName)

        vertex = []
        faces = []

        for s in file:
            sd = s.split()
            if sd[0] == 'v':
                vertex.append([float(x) for x in sd[1:]])
            elif sd[0] == 'f':
                faces.append([int(x.split('/')[0]) for x in sd[1:]])

        file.close()

        self.vertex = np.array(vertex)
        self.faces = np.array(faces)


class Renderer:
    def __init__(self, W, H, scale, shift):
        self.H = H
        self.W = W
        self.scale = scale
        self.invertY = -1
        self.shift = shift

        self.R = 51
        self.G = 0
        self.B = 102

        self.img  = np.zeros((H, W, 3), dtype=np.uint8)
        self.zbuf = np.full((H, W), np.inf, dtype=np.float32)

    def render(self,scene: Scene):
        renderScene(scene.vertex,scene.faces,
                    self.scale,self.shift,
                    self.invertY,self.zbuf,
                    self.img, self.R, self.G, self.B)
    def clearScene(self):
        self.img  = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        self.zbuf = np.full((self.H, self.W), np.inf, dtype=np.float32)

class APP:
    def __init__(self, scene, renderer, win_w = 900, win_h = 900):
            self.scene = scene
            self.renderer = renderer
    
            # Оставляем как есть
            # Окно верхнего уровня
            self.root = tk.Tk()
            # Размеры окна
            self.root.geometry(f"{win_w}x{win_h}")
            # Область для рисования внутри окна
            self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
            self.canvas.pack(fill='both', expand=True)
            self.canvas.focus_set()
            # Обновление изображения при изменении размеров окна
            self.canvas.bind('<Configure>', lambda e: self._refit())
            # Привязка обработчика клавиш
            self.root.bind('<Key>', self.on_key)
    
            # Изображение внутри окна
            self.photo = None
            self.full_image = None
    
            self.render()
    
    def on_key(self, event):
        k = event.keysym

        if k == 'b':
            self.renderer.scale *= 1.1
        elif k == 's':
            self.renderer.scale /= 1.1
        elif k == 'a':
            self.renderer.shift -= 50
        elif k == 'd':
            self.renderer.shift += 50
        elif k == 'r':
            self.renderer.R = random.randint(0, 255)
            self.renderer.G += random.randint(0, 255)
            self.renderer.B += random.randint(0, 255)
        elif k == 'i':
            self.renderer.invertY *= -1
        elif k == 'space':
            Image.fromarray(self.renderer.img).save('rabbit.png')
            return 'break'
        else:
            return

        self.render()
        # Не обрабатываем дальше – return 'break'
        return 'break'

    def render(self):
        self.renderer.clearScene()
        self.renderer.render(self.scene)
        self.full_image = Image.fromarray(self.renderer.img)
        self._refit()

    # скопировать как есть
    def _refit(self):
        # Если ещё не отрендерилось, пропускаем
        if self.full_image is None: return
        # Ширина и высота области для рисования
        ww = self.canvas.winfo_width()
        wh = self.canvas.winfo_height()
        # Если вдруг сделали область слишком маленькой
        if ww < 10 or wh < 10: return
        # Вписать с сохранением пропорций
        s = min(ww / self.renderer.W, wh / self.renderer.H)
        dw = max(1, int(self.renderer.W * s))
        dh = max(1, int(self.renderer.H * s))
        disp = self.full_image.resize((dw, dh), Image.BILINEAR)
        # Очень важно сохранить ImageTk.PhotoImage(disp) в поле класса
        # Если оно будет локальным, то удалится после выхода сборщиком мусора
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.delete('all')
        self.canvas.create_image(ww // 2, wh // 2, image=self.photo)   

    # скопировать как есть
    def run(self):
        self.root.mainloop()
    
if __name__ == '__main__':
    scene = Scene("model.obj")
    renderer = Renderer(1000, 1000, 5000, 500)
    renderer.render(scene)

    APP(scene, renderer, 500, 500).run()

    
