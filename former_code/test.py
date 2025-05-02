import torch
import cv2
import numpy as np
from breaker import BreakoutEnv
from encoder import Autoencoder

class AutoencoderTester:
    def __init__(self, model_path):
        self.env = BreakoutEnv()  # 替換 Maze 為 Breakout 環境
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加載 Autoencoder 模型
        self.model = Autoencoder().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.key_actions = {
            ord('a'): 1,  # 左
            ord('d'): 2,  # 右
            ord(' '): 0,  # 不動
        }

        self.prev_frame_1 = None  # t-1 幀
        self.prev_frame_2 = None  # t-2 幀

    def preprocess_frame(self, frame):
        """轉換影像為 Autoencoder 模型輸入格式"""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 轉灰階
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=0)  # (H, W) → (1, H, W)
        frame = np.expand_dims(frame, axis=0)  # (1, H, W) → (1, 1, H, W)
        return torch.tensor(frame, device=self.device)

    def postprocess_frame(self, tensor):
        """將 Autoencoder 輸出轉換回可視化的影像格式"""
        frame = tensor.cpu().detach().numpy()[0, 0]  # (1, 1, H, W) → (H, W)
        frame = (frame * 255).astype(np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # 轉回 BGR 以便顯示

    def run(self):
        print("控制方式:")
        print("A: 左移  D: 右移  空白鍵: 不動")
        print("ESC: 退出")

        self.env.reset()
        self.prev_frame_1 = self.preprocess_frame(self.env._get_frame())
        self.prev_frame_2 = self.preprocess_frame(self.env._get_frame())
        while self.env.game_state != "playing":
            self.env.step(0)  # 執行不動作，模擬等待空白鍵
            
        while True:
            # 當前畫面
            current_frame = self.env._get_frame()
            current_tensor = self.preprocess_frame(current_frame)
            cv2.imshow("Breakout Frame", current_frame)
            cv2.waitKey(1)
            # 創建顯示畫布
            display = np.zeros((current_frame.shape[0], current_frame.shape[1] * 3, 3), dtype=np.uint8)

            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC 退出
                break

            if key in self.key_actions:
                action = self.key_actions[key]

                # **執行動作並獲取下一幀**
                _, _, done, _ = self.env.step(action)
                next_frame = self.env._get_frame()
                next_tensor = self.preprocess_frame(next_frame)

                # **預測前一幀 (t-1)**
                with torch.no_grad():
                    action_tensor = torch.tensor([action], device=self.device)
                    frame = torch.cat((self.prev_frame_2,self.prev_frame_1), dim=1)
                    predicted_prev_frame = self.model(frame, action_tensor)

                # **更新前兩幀**
                self.prev_frame_2 = self.prev_frame_1
                self.prev_frame_1 = current_tensor

                # **處理影像**
                predicted_prev_frame = self.postprocess_frame(predicted_prev_frame)
                true_prev_frame = self.postprocess_frame(self.prev_frame_1)
                next_frame = self.postprocess_frame(next_tensor)

                # **顯示影像**
                display[:, :current_frame.shape[1]] = true_prev_frame  # 真實前一幀
                display[:, current_frame.shape[1]:current_frame.shape[1] * 2] = predicted_prev_frame  # 預測的前一幀
                display[:, current_frame.shape[1] * 2:] = next_frame  # 當前幀（實際的下一幀）

                cv2.imshow('Autoencoder Breakout Test', display)

                if done:
                    self.env.reset()
                    self.prev_frame_1 = self.preprocess_frame(self.env._get_frame())
                    self.prev_frame_2 = self.preprocess_frame(self.env._get_frame())

        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = "maze_models/best_model_manual.pth"
    tester = AutoencoderTester(model_path)
    tester.run()
