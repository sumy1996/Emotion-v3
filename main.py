import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import pandas as pd
import requests
import base64
import time
import os

# 设置API和认证头部
API_URL = "https://api-inference.huggingface.co/models/dima806/facial_emotions_image_detection"

class EmotionAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("Emotion Analyzer")
        master.geometry('400x250')  # 设置窗口大小

        # Frame for API Token input
        self.api_frame = tk.Frame(master)
        self.api_frame.pack(padx=10, pady=5)

        # Entry for API Token
        self.lbl_api_token = tk.Label(self.api_frame, text="API Token:")
        self.lbl_api_token.pack(side=tk.LEFT)
        self.entry_api_token = tk.Entry(self.api_frame, width=40)
        self.entry_api_token.pack(side=tk.LEFT)

        # Button to save API Token
        self.btn_save_token = tk.Button(self.api_frame, text="保存Token", command=self.save_token)
        self.btn_save_token.pack(side=tk.LEFT, padx=5)

        # Frame for other controls
        self.frame = tk.Frame(master)
        self.frame.pack(padx=10, pady=10)

        # Button: select video file
        self.btn_select_video = tk.Button(self.frame, text="选择视频文件", command=self.select_video)
        self.btn_select_video.pack(fill=tk.X, pady=5)

        # Button: set frame rate
        self.btn_frame_rate = tk.Button(self.frame, text="设置抽帧频率", command=self.set_frame_rate)
        self.btn_frame_rate.pack(fill=tk.X, pady=5)

        # Button: select save path
        self.btn_save_path = tk.Button(self.frame, text="选择保存路径", command=self.select_save_path)
        self.btn_save_path.pack(fill=tk.X, pady=5)

        # Button: start analysis
        self.btn_start_analysis = tk.Button(self.frame, text="开始分析", command=self.start_analysis)
        self.btn_start_analysis.pack(fill=tk.X, pady=5)

        self.video_path = None
        self.save_path = None
        self.frame_rate = 30
        self.api_token = None

    def save_token(self):
        self.api_token = self.entry_api_token.get()
        if self.api_token:
            messagebox.showinfo("Token Saved", "API Token已保存")
            print("API Token:", self.api_token)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            print("选择的视频路径:", self.video_path)

    def set_frame_rate(self):
        self.frame_rate = simpledialog.askinteger("抽帧频率", "请输入抽帧频率（默认为30）:", minvalue=1, initialvalue=30)

    def select_save_path(self):
        self.save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if self.save_path:
            print("保存路径:", self.save_path)

    def start_analysis(self):
        if not self.video_path or not self.save_path or not self.api_token:
            messagebox.showerror("错误", "请确保已选择视频文件和保存路径并输入API Token")
            return
        headers = {"Authorization": f"Bearer {self.api_token}"}
        results = process_video(self.video_path, self.frame_rate, headers)
        save_results_to_excel(results, self.save_path)
        messagebox.showinfo("完成", "处理完成，结果已保存")


def query(frame, headers, retry_delay=20, max_retries=3):
    try:
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return {"error": "图像编码失败"}

        img_base64 = base64.b64encode(buffer).decode('utf-8')
        payload = {'inputs': img_base64}
        attempts = 0

        while attempts < max_retries:
            response = requests.post(API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503 and "currently loading" in response.text:
                print(f"模型正在加载，将在{retry_delay}秒后重试...")
                time.sleep(retry_delay)  # 等待模型加载完成
                attempts += 1
            else:
                print(f"API调用失败，状态码: {response.status_code}, 响应: {response.text}")
                return {"error": response.text}
        return {"error": "达到最大重试次数，API仍不可用"}
    except Exception as e:
        print(f"请求API时发生异常: {e}")
        return {"exception": str(e)}

def process_video(video_path, frame_rate, headers):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []
    print("总帧数:", frame_count)
    for i in range(0, frame_count, frame_rate):
        ret, frame = cap.read()
        if not ret:
            break
        print(f"正在处理第 {i} 帧...")
        result = query(frame, headers)  # 确保这里使用headers参数
        results.append((i, result))
        print(f"第 {i} 帧的结果: {result}")
    cap.release()
    return results

def save_results_to_excel(results, file_name):
    emotions = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']
    formatted_results = []
    for frame_index, emotion_data in results:
        frame_results = {'Frame': frame_index}
        for emotion in emotions:
            frame_results[emotion] = 0.0
        for item in emotion_data:
            if item['label'] in frame_results:
                frame_results[item['label']] = item['score']
        formatted_results.append(frame_results)
    df = pd.DataFrame(formatted_results)
    df = df[['Frame'] + emotions]
    try:
        df.to_excel(file_name, index=False)
        print("文件保存成功，路径为:", file_name)
    except Exception as e:
        print("保存文件失败:", e)

def main():
    root = tk.Tk()
    app = EmotionAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
