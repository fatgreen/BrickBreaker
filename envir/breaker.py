import pygame
from pygame.locals import QUIT
from envir.variables import *
from envir.paddle import paddle
from envir.ball import game_ball
from envir.level1 import level1
import numpy as np
class BreakoutEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Breakout')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """重置遊戲狀態，並返回初始畫面"""
        self.player_paddle = paddle()
        self.ball = game_ball(self.player_paddle.x + (self.player_paddle.width // 2), self.player_paddle.y - self.player_paddle.height)
        self.level_wall = level1()
        self.level_wall.create_wall()
        self.lives = lives
        self.score = 0
    
        self.game_state = 'playing'  # 初始狀態為 'start'
        self.dead = False
        return self._get_frame()

    def step(self, action):
        
        # reward = -1
        reward = 0
        done = False
        
        # 處理動作（0: 不動, 1: 向左, 2: 向右）
        if action == 1:
            self.player_paddle.move(-1)  # 向左移動
        elif action == 2:
            self.player_paddle.move(1)  # 向右移動

        # 如果遊戲狀態是 'start'，按下空白鍵開始遊戲
        if self.game_state == 'start':
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.game_state = 'playing'

        # 更新遊戲狀態
        if self.game_state == 'playing':
            self.ball.move()
            reward = self._handle_collisions(reward)
    

        if self.lives == 0 :
            done = False
            self.dead = True
        if self._is_level_complete():
            done = True
            reward = self.score
        # 獲取當前畫面
        frame = self._get_frame()
        
        return frame, reward, done, self.dead

    def _get_frame(self):
        """獲取當前遊戲畫面（修正旋轉 90 度問題）"""
        self.screen.fill(bg)
        self.level_wall.draw_wall()
        self.player_paddle.draw(self.screen)
        self.ball.draw()
        pygame.display.flip()
        
        frame = pygame.surfarray.array3d(self.screen)  # 取得畫面 (width, height, 3)
        frame = np.transpose(frame, (1, 0, 2))  # 轉換為 (height, width, 3)
    
        return frame

    def _handle_collisions(self,reward):
        """處理碰撞邏輯"""
        if self.ball.rect.colliderect(self.player_paddle.rect):
            self.ball.paddle_bounce(self.player_paddle)  # 改為使用 `paddle_bounce()`
            reward = 10
        
        for row in self.level_wall.blocks:
            for block in row:
                if block is not None and self.ball.rect.colliderect(block[0]):
                    if self.ball.previous_rect.bottom <= block[0].top:  
                    # 從上方撞擊 → 反轉 y 軸
                        self.ball.speed.y *= -1
                    elif self.ball.previous_rect.top >= block[0].bottom:  
                        # 從下方撞擊 → 反轉 y 軸
                        self.ball.speed.y *= -1
                    elif self.ball.previous_rect.right <= block[0].left:  
                        # 從左側撞擊 → 反轉 x 軸
                        self.ball.speed.x *= -1
                    elif self.ball.previous_rect.left >= block[0].right:  
                        # 從右側撞擊 → 反轉 x 軸
                        self.ball.speed.x *= -1
                    block[1] -= 1
                    if block[1] == 0:
                        row.remove(block)
                    
                    self.score += 10
                    reward = 30
                    break
        
        if self.ball.rect.bottom > screen_height:
            self.lives -= 1
            self.dead = True
            reward = -50
            if self.lives == 0:
                reward = -50
                self.game_state = 'game_over'
            else:
                self.ball.reset(self.player_paddle.x + (self.player_paddle.width // 2), self.player_paddle.y - self.player_paddle.height)
                self.dead = False
                self.player_paddle = paddle()
        return reward



    def _is_level_complete(self):
        """檢查是否完成關卡"""
        for row in self.level_wall.blocks:
            if any(block is not None for block in row):
                return False
        return True

    def close(self):
        """關閉遊戲"""
        pygame.quit()

# 測試環境
if __name__ == "__main__":
    env = BreakoutEnv()
    state = env.reset()
    done = False

    while not done:
        # 處理事件（例如關閉視窗）
        for event in pygame.event.get():
            if event.type == QUIT:
                done = True

        # 獲取鍵盤輸入
        action = 0  # 0: 不動, 1: 向左, 2: 向右
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 1
            print(f"Reward: {reward}, Done: {done}")
        elif keys[pygame.K_RIGHT]:
            action = 2

        # 執行動作並更新遊戲狀態
        next_state, reward, done, _ = env.step(action)

        # 渲染畫面
        #env._get_frame()  # 獲取並顯示當前畫面

        # 控制遊戲幀率
        env.clock.tick(fps)

        # 輸出獎勵和遊戲狀態
        

    # 關閉遊戲
    env.close()