from pygame import Rect, Vector2
from variables import *
from pygame import Rect, Vector2
from variables import *
import random
class game_ball:
    def __init__(self, x, y, ball_size=7):
        self.ball_rad = ball_size
        self.speed_max = Coll_variable_speed
        self.min_speed = 2  # 最小速度
        self.max_speed = 6  # 最大速度
        self.rect = Rect(x - self.ball_rad, y, self.ball_rad * 2, self.ball_rad * 2)
        self.previous_rect = self.rect.copy()  # 記錄前一幀位置
        self.speed = Vector2( random.choice([-2,-1,0, 1, 2]),variable_self_speed_ｙ)

    def move(self):
        """ 更新小球位置，同時記錄上幀位置 """
        self.previous_rect = self.rect.copy()  # 儲存之前的位置
        self.rect.move_ip(self.speed)

        # 碰到左右邊界反彈
        if self.rect.left < 0 or self.rect.right > screen_width:
            self.speed.x *= -1

        # 碰到上邊界反彈
        if self.rect.top < 0:
            self.speed.y *= -1

    def paddle_bounce(self, paddle):
        """ 當小球撞到 `paddle` 時，依據 `paddle` 速度改變反彈角度 """
        if self.rect.colliderect(paddle.rect):
            offset = (self.rect.centerx - paddle.rect.centerx) / (paddle.width / 2)
            self.speed.x = offset * self.max_speed  # 根據碰撞點計算新 x 速度
            self.speed.y *= -1  # 反轉 y 方向

            # 限制速度，避免球變太快或太慢
            self.speed.x = max(-self.max_speed, min(self.speed.x, self.max_speed))
            self.speed.y = max(-self.max_speed, min(self.speed.y, -self.min_speed))  # 確保 y 速度不會過慢
    def draw(self):
        pygame.draw.circle(screen, paddle_col, self.rect.center, self.ball_rad)
    def reset(self, x, y):
        """ 重置小球 """
        self.rect.topleft = (x - self.ball_rad, y)
        self.speed = Vector2(random.choice([-2,-1,0, 1, 2]),variable_self_speed_ｙ)

