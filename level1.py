from pygame import *
from variables import *

class level1:
    def __init__(self):
        self.width = (screen_width // cols) - 4
        self.height = 15
        self.blocks = []

    def create_wall(self):
        self.blocks = []
        pattern = [
            [1, 2, 0, 2, 1],
            [0, 1, 3, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        for row_idx, row in enumerate(pattern):
            block_row = []
            for col_idx, strength in enumerate(row):
                if strength != 0:
                    block_x = col_idx * (self.width + 4)
                    block_y = row_idx * (self.height + 4) + 30
                    rect = pygame.Rect(block_x, block_y, self.width, self.height)
                    block_individual = [rect, strength]
                    block_row.append(block_individual)
            self.blocks.append(block_row)

    def draw_wall(self):
        for row in self.blocks:
            for block in row:
                if block is not None:
                    if block[1] == 3:
                        block_col = block_red
                    elif block[1] == 2:
                        block_col = block_green
                    elif block[1] == 1:
                        block_col = block_blue
                    pygame.draw.rect(screen, block_col, block[0])
                    pygame.draw.rect(screen, bg, block[0], 1)