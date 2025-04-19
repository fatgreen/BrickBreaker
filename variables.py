import pygame.font
import pygame.display
import pygame.time

pygame.font.init()

# define font
font = pygame.font.SysFont('Constantia', 15)  # 縮小字體大小

# 螢幕大小調整為 200x200
screen_width = 192
screen_height = 192

screen = pygame.display.set_mode((screen_width, screen_height))

# define colours
bg = (0, 0, 0)  # 背景改為純黑

# block colours
block_red = (255, 255, 0)  # 紅色磚塊
block_green = (0, 255, 255)  # 綠色磚塊
block_blue = (255, 0, 255)  # 藍色磚塊
block_purple = (128, 0, 128)  # 紫色磚塊（未使用）
block_black = (0, 0, 0)  # 黑色磚塊（未使用）

# paddle colours
paddle_col = (125, 125, 125)  # 球拍顏色
paddle_outline = (0, 0, 0)  # 球拍邊框顏色

# text colour
text_col = (78, 81, 139)  # 文字顏色
score_text_color = (255, 255, 0)  # 分數文字顏色（黃色）
outline_color = (0, 0, 0)  # 文字邊框顏色（黑色）

# define game variables
cols = 5  # 磚塊列數減少
rows = 4  # 磚塊行數減少
clock = pygame.time.Clock()
fps = 30  # 遊戲幀率
live_ball = False

# Game state
game_state = 'start'  # 遊戲狀態
current_level = 1
max_levels = 1  # 只有一關

# Game ball speed variables
Coll_variable_speed = 2  # 球的速度減慢
variable_self_speed_x = 4
variable_self_speed_y = -4

lives = 3 # 生命值
score = 0  # 分數
power_up = 5  # 未使用的變數