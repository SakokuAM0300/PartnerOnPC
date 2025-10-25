# gui_app.py
import tkinter as tk
import asyncio
import threading
from core_logic import (
    setup_whisper_model, 
    record_and_transcribe, 
    generate_gemini_stream, 
    generate_and_play_tts, 
    async_generator_from_string
)
# 起動時チェック用のインポート
import requests
from core_logic import VOICEVOX_ENGINE_URL, GEMINI_API_KEY # core_logicから設定値をインポート

class PartnerChatBotApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Partner Chatbot")
        self.geometry("600x450")
        
        # 1. 状態管理変数
        self.conversation_history = [] 
        self.is_listening = False  # 連続会話モードの状態
        
        # 2. GUIコンポーネントの初期化
        self._create_widgets()
        
        # 3. 非同期実行環境 (asyncio) のセットアップ
        self._setup_asyncio_loop()

    def _create_widgets(self):
        # ログ表示エリア (履歴表示)
        self.log_text = tk.Text(self, state='disabled', wrap='word', height=15)
        self.log_text.pack(pady=10, padx=10, fill='x')
        
        # ステータス表示ラベル
        self.status_label = tk.Label(self, text="準備中...", font=("Helvetica", 12))
        self.status_label.pack(pady=5)
        
        # マイクボタン (会話開始/停止のトグル)
        self.mic_button = tk.Button(
            self, 
            text="▶️ 会話を開始", 
            font=("Helvetica", 16, "bold"), 
            command=self.toggle_continuous_listening, # 機能をトグルに変更
            height=3,
            width=20
        )
        self.mic_button.pack(pady=20)
        
        # 終了ボタン
        self.exit_button = tk.Button(self, text="Exit", command=self.on_closing)
        self.exit_button.pack(pady=5)
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _setup_asyncio_loop(self):
        # Tkinterがメインスレッドを占有するため、asyncioを別スレッドで実行
        self.loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(target=self._run_asyncio_loop, daemon=True)
        self.async_thread.start()

    def _run_asyncio_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # --- ログとステータス更新ヘルパー ---
    def update_status(self, text, color="black"):
        # Tkinterコンポーネントの更新はメインスレッドで行う必要があるため after() を使用
        self.after(0, lambda: self.status_label.config(text=text, fg=color))
        
    def append_log(self, role, text):
        def _update():
            self.log_text.config(state='normal')
            self.log_text.insert(tk.END, f"{role}: {text}\n", role)
            self.log_text.tag_config("User", foreground="blue")
            self.log_text.tag_config("Assistant", foreground="green")
            self.log_text.see(tk.END) 
            self.log_text.config(state='disabled')
        self.after(0, _update)

    # --- 連続会話モードのトグル ---
    def toggle_continuous_listening(self):
        if not self.is_listening:
            # 開始処理
            self.is_listening = True
            self.mic_button.config(text="⏹️ 会話を停止", fg="white", bg="red")
            self.update_status("連続会話モード開始。", color="blue")
            # 非同期タスクとしてメインループを開始
            asyncio.run_coroutine_threadsafe(self._continuous_conversation_loop(), self.loop)
        else:
            # 停止処理
            self.is_listening = False
            self.mic_button.config(text="▶️ 会話を開始", fg="black", bg="#f0f0f0")
            self.update_status("会話モード停止。", color="darkred")

    # --- 連続会話のメインループ ---
    async def _continuous_conversation_loop(self):
        self.update_status("話しかけてください (連続モード)", color="green")
        
        while self.is_listening:
            try:
                # 録音中はボタンを無効化（誤操作防止）
                self.after(0, lambda: self.mic_button.config(text="👂 録音待機中...", state=tk.DISABLED))

                # 1. 録音と転写
                user_text = await record_and_transcribe()
                
                # ユーザーがモードを停止した場合のチェック
                if not self.is_listening:
                    break
                
                if not user_text:
                    self.update_status("発話が検出されませんでした。待機中...", color="gray")
                    await asyncio.sleep(0.5) # CPU負荷軽減
                    continue

                self.append_log("User", user_text)

                # 2. 終了トリガーチェック
                EXIT_COMMANDS = ["さようなら", "さよなら"]
                if any(cmd in user_text for cmd in EXIT_COMMANDS):
                    exit_response = "またお話しできるのを楽しみにしてるよ。またね！"
                    await generate_and_play_tts(async_generator_from_string(exit_response))
                    self.update_status("プログラムを終了します。", color="darkred")
                    self.is_listening = False 
                    self.after(2000, self.on_closing) # 2秒後にウィンドウを閉じる
                    break 
                
                # 3. Geminiで応答を生成 (ストリーミング)
                self.update_status("Gemini応答生成中...", color="orange")
                gpt_text_stream = generate_gemini_stream(user_text, self.conversation_history)
                
                # 4. TTSで再生 (ストリーミング)
                self.update_status("応答音声合成・再生中...", color="purple")
                full_assistant_response = await generate_and_play_tts(gpt_text_stream)
                
                # 5. ログと履歴更新
                self.append_log("Assistant", full_assistant_response)
                
                # 応答完了後、自動で次の録音待機状態へループ
                self.update_status("話しかけてください (連続モード)", color="green")
                self.after(0, lambda: self.mic_button.config(text="👂 録音待機中...", state=tk.NORMAL))


            except Exception as e:
                error_msg = f"致命的なエラーが発生しました: {e}"
                self.update_status(error_msg, color="red")
                self.append_log("System Error", error_msg)
                await asyncio.sleep(1) # エラー時の無限ループ防止

        # ループが終了した場合の後処理
        self.after(0, lambda: self.mic_button.config(
            state=tk.NORMAL, 
            text="▶️ 会話を開始", 
            fg="black", 
            bg="#f0f0f0"
        ))
        if self.winfo_exists():
            self.update_status("会話モード停止。", color="darkred")


    def on_closing(self):
        # プログラム終了処理
        self.is_listening = False # ループを停止させる
        if hasattr(self, 'loop') and self.loop.is_running():
            self.loop.stop()
        self.destroy()

if __name__ == "__main__":
    
    # 起動前の初期化チェック
    if not GEMINI_API_KEY:
        print("🔴 エラー: GEMINI_API_KEYが.envファイルに設定されていません。")
        exit()
    try:
        requests.get(f"{VOICEVOX_ENGINE_URL}/version")
    except requests.exceptions.ConnectionError:
        print("🔴 エラー: VOICEVOX Engineが起動していません。アプリケーションを起動してください。")
        exit()
    
    # Whisperモデルの初期ロード
    if not setup_whisper_model():
        print("🔴 エラー: Whisperモデルのロードに失敗しました。")
        exit()

    app = PartnerChatBotApp()
    app.update_status("開始ボタンを押してください", color="black")
    app.mainloop()