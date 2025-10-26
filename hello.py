import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Inicialização
pygame.init()
largura, altura = 800, 600
pygame.display.set_mode((largura, altura), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Terreno OpenGL Básico - Amarelo")

# Configuração da Perspectiva (3D)
gluPerspective(45, (largura / altura), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5) # Move a "câmera" para trás

# Coordenadas do Vértices do Quadrado (o "Terreno" Básico)
# Um quadrado simples (ou plano)
vertices = (
    (1, -1, 0),
    (1, 1, 0),
    (-1, 1, 0),
    (-1, -1, 0)
)

# A Ordem de Desenho dos Vértices
quad_faces = (0, 1, 2, 3)

# Função para Desenhar o Quadrado
def Quadrado():
    glColor3f(1.0, 1.0, 0.0)  # Define a cor para Amarelo (RGB: 1.0, 1.0, 0.0)
    glBegin(GL_QUADS)         # Começa a desenhar um conjunto de Quadrados
    for vertex in quad_faces:
        glVertex3fv(vertices[vertex]) # Desenha os vértices na ordem definida
    glEnd()                   # Termina o desenho

# Loop Principal do Jogo
a_correr = True
while a_correr:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            a_correr = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                a_correr = False

    # Lógica de Rotação (para o ver em 3D)
    # Roda o mundo em torno do eixo X e depois Y, 0.5 graus a cada frame
    glRotatef(0.5, 1, 1, 0)

    # Limpa o ecrã
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Desenha o objeto
    Quadrado()

    # Atualiza o ecrã
    pygame.display.flip()
    pygame.time.wait(10) # Pequena pausa para controlo do FPS

pygame.quit()