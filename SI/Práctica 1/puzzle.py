import pygame
import random

def printTabla():
    print("1. Rojo")
    print("2. Verde")
    print("3. Azul")
    print("4. Cian")
    print("5. Rosa")
    print("6. Amarillo")
    print("7. Naranja")
    print("8. Morado")

 
def newTablero(x, y):
    tablero = []
    for i in range(x):
        tablero.append([])
        for j in range(y):
            tablero[i].append(0)

    return tablero

NEGRO = (0, 0 ,0)
BLANCO = (255, 255, 255)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
AZUL = (0, 0, 255)
VIOLETA = (98, 0, 255)

cols = [(150, 150, 150),
        (255,   0,   0),
        (  0, 255,   0),
        (  0,   0, 255),
        (  3, 223, 252),
        (255,   0, 247),
        (229, 255, 0),
        (255, 162,   0),
        (106,   0, 255),
        NEGRO]


tablero = newTablero(9,8)
dimensiones = [len(tablero)*100,len(tablero[0])*100]
res = 100
printTabla()
n = 1
select = False



pygame.init()
pantalla = pygame.display.set_mode(dimensiones) 
pygame.display.set_caption("Puzzle")
hecho = False
reloj = pygame.time.Clock()

while not hecho:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT: 
            hecho = True
        
        if evento.type == pygame.MOUSEBUTTONDOWN:
            select = True

        if evento.type == pygame.KEYDOWN:
            if evento.key == pygame.K_r:
                tablero = newTablero()
            
            if evento.key == pygame.K_0:
                n = 0
            elif evento.key == pygame.K_1:
                n = 1
            elif evento.key == pygame.K_2:
                n = 2
            elif evento.key == pygame.K_3:
                n = 3
            elif evento.key == pygame.K_4:
                n = 4
            elif evento.key == pygame.K_5:
                n = 5
            elif evento.key == pygame.K_6:
                n = 6
            elif evento.key == pygame.K_7:
                n = 7
            elif evento.key == pygame.K_8:
                n = 8
            elif evento.key == pygame.K_9:
                n = 9

    # ---------------------------------------------------LÓGICA---------------------------------------------------
    x, y = pygame.mouse.get_pos()
    if(select):
        xi = int(x/100)
        yi = int(y/100)
        tablero[xi][yi] = n
        select = False
    # ---------------------------------------------------DIBUJO---------------------------------------------------

    pantalla.fill(BLANCO)

    for i in range(len(tablero)):
        for j in range(len(tablero[0])):
            pygame.draw.rect(pantalla, cols[tablero[i][j]], [i*res, j*res, res, res])
    
    '''for i in range(len(tablero)):
        for j in range(len(tablero[0])):
            if(i<len(tablero)):
                pygame.draw.line(pantalla, NEGRO, [i*res+res, j* res], [i*res+res, j* res+res], 2)
            if():
                pygame.draw.line(pantalla, NEGRO, [i*res, j*res+res], [i*res+res, j* res+res], 2)
                
                
    for i in range(8):
        for j in range(9):
            if(i<7 and tablero[i][j] != tablero[i+1][j]):
                pygame.draw.line(pantalla, NEGRO, [i*res+res, j* res], [i*res+res, j* res+res], 2)

            if(j<8 and tablero[i][j] != tablero[i][j+1]):
                pygame.draw.line(pantalla, NEGRO, [i*res, j*res+res], [i*res+res, j* res+res], 2)

    '''
    pygame.display.flip()
    reloj.tick(60)

pygame.quit()
