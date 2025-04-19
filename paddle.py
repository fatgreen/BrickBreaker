import pygame
from variables import screen_width, screen_height, cols, paddle_col, paddle_outline

class paddle:
    def __init__(self, height=10, width=None, speed=5, radius=5, border=2):
        self.height = height
        self.width = screen_width // cols if width is None else width
        self.speed = speed
        self.radius = radius
        self.border = border
        self.reset()

    def draw(self, screen):
        pygame.draw.rect(screen, paddle_col, self.rect)

    def reset(self):
        self.x = (screen_width - self.width) // 2
        self.y = screen_height - (self.height * 2)
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def move(self, direction):
        """ 控制球拍移動，並確保不超出螢幕範圍 """
        self.rect.x += direction * self.speed
        self.rect.x = max(0, min(self.rect.x, screen_width - self.width))  # 限制範圍
