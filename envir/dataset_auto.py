import numpy as np
import cv2
import random
from envir.breaker import BreakoutEnv  # 導入打磚塊遊戲環境
from envir.variables import screen_width, screen_height, fps
import pygame
from pygame.locals import QUIT
# 初始化打磚塊遊戲環境
env = BreakoutEnv()

# 定義資料集存儲變數
current_frames = []
actions = []
next_frames = []

def get_frame(env):
    """獲取當前遊戲畫面，轉換為灰階並縮放"""
    frame = env._get_frame()  # 獲取當前畫面 (RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 轉為灰階
    frame = cv2.resize(frame, (screen_width, screen_height))  # 確保解析度一致
    frame = frame.astype(np.uint8)  # **確保數據類型一致**
    return frame

def press_space_to_start(env):
    """模擬按下空白鍵，使遊戲進入 'playing' 狀態"""
    while env.game_state != "playing":
        env.step(0)  # 執行不動作，模擬等待空白鍵
        cv2.imshow("Breakout Frame", get_frame(env))  # 顯示畫面
        cv2.waitKey(1)  # 等待畫面刷新

def collect_data(env, steps):
    """收集打磚塊遊戲的資料集，並即時顯示畫面"""
    press_space_to_start(env)  # 開始遊戲
    prev_frame_1 = get_frame(env)  # 確保前一幀不是 None
    prev_frame_2 = get_frame(env)  # 確保前兩幀不是 None

    for i in range(steps):
        # 控制遊戲運行速度
        env.clock.tick(fps)  # 限制 FPS
        
        # 獲取當前畫面
        current_frame = get_frame(env)

        # 隨機選擇動作（0: 不動, 1: 向左, 2: 向右）
        for event in pygame.event.get():
            if event.type == QUIT:
                done = True

        # 獲取鍵盤輸入
        action = 0  # 0: 不動, 1: 向左, 2: 向右
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 1
            #print(f"Reward: {reward}, Done: {done}")
        elif keys[pygame.K_RIGHT]:
            action = 2

        next_frame, reward, done, _ = env.step(action)
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)  # 轉為灰階
        next_frame = cv2.resize(next_frame, (screen_width, screen_height))  # 確保解析度一致
        next_frame = next_frame.astype(np.uint8)  # **確保數據類型一致**

        # **解決 NumPy 存儲問題，確保形狀一致**
        try:
            stacked_frames = np.stack([prev_frame_2, prev_frame_1], axis=0)  # (2, H, W)
            current_frames.append(stacked_frames)
            actions.append(action)
            next_frames.append(next_frame)
        except ValueError as e:
            print(f"❌ 發生錯誤：{e}")
            print(f"prev_frame_2.shape: {prev_frame_2.shape}, prev_frame_1.shape: {prev_frame_1.shape}")
            continue  # 跳過這一幀，繼續收集數據

        # 更新前兩幀
        prev_frame_2 = prev_frame_1
        prev_frame_1 = current_frame

        # 如果遊戲結束，重置環境
        if done:
            env.reset()
            prev_frame_1 = get_frame(env)  # **確保重置後的幀不是 None**
            prev_frame_2 = get_frame(env)
            press_space_to_start(env)  # 重新按下空白鍵

        # 顯示畫面
        cv2.imshow("Breakout Frame", next_frame)

        # 按 ESC 退出
        if cv2.waitKey(1) == 27:
            break

        # 每 100 步輸出進度
        if i % 100 == 0:
            print(f"✅ 已收集 {i} 筆資料")

    # 關閉顯示窗口
    cv2.destroyAllWindows()

# 收集 3000 筆資料
collect_data(env, 5000)

# **確保 NumPy 陣列可以正常存儲**
try:
    np.savez("breakout_dataset.npz", 
             current_frames=np.array(current_frames),  # 確保 (N, 2, H, W)
             actions=np.array(actions), 
             next_frames=np.array(next_frames))
    print("🎉 資料集已成功儲存！")
except ValueError as e:
    print(f"❌ 儲存失敗：{e}")
    print(f"current_frames 長度：{len(current_frames)}")
    print(f"actions 長度：{len(actions)}")
    print(f"next_frames 長度：{len(next_frames)}")
