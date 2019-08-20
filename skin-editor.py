import pygame, sys, PIL, numpy as np, random as rng
from pygame.locals import *

import os ; os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import load_model

GENERATOR = load_model('output/generator_final.h5')
NOISE = np.random.normal(0, 1, [1, GENERATOR.get_input_shape_at(0)[-1]])

def map(value, _MIN, _MAX, start, stop):
    return start + (stop - start) * ((value - _MIN) / (_MAX - _MIN))

def gen_pic(windowSurface):
    global GENERATOR, NOISE
    arr = GENERATOR.predict(NOISE)
    arr = arr.reshape(int(arr.shape[-1]/256), 64, 4) * 255
    pic = PIL.Image.fromarray(arr.astype('uint8'))
    pic.save('output/pic.png')
    surface = pygame.image.fromstring(pic.tobytes(), pic.size, 'RGBA')
    surface = pygame.transform.scale(surface, (500, 500))
    windowSurface.blit(surface, (0, 0))

pygame.init()
windowSurface = pygame.display.set_mode((800, 500), 0, 32)
pygame.display.set_caption('Skin Editor')



windowSurface.fill((128, 128, 150))


dx = 300 // 10
dy = 500 // 10
def draw_cells():
    global NOISE
    MIN, MAX = np.amin(NOISE), np.amax(NOISE)
    print(MIN, MAX)
    index = 0
    for y in range(0, 500, dy):
        for x in range(500, 800, dx):
            col = map(NOISE[0,index], MIN, MAX, 0, 255)
            index += 1
            pygame.draw.rect(windowSurface, (col, col, col),Rect(x, y, dx -1, dy -1))

draw_cells()
gen_pic(windowSurface)
pygame.display.update()

while True:
    for event in pygame.event.get():

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            cellpos = ((pos[0] - 500)//dx, pos[1]//dy)
            if cellpos[0] >= 0 and cellpos[1] >= 0:
                index = cellpos[0] + cellpos[1] * 10
                NOISE[0,index] = rng.randint(-3, 3)
                windowSurface.fill((0, 0, 0))
                gen_pic(windowSurface)
                draw_cells()
                pygame.display.update()
                
        elif event.type == QUIT:
            pygame.quit()
            sys.exit()


            