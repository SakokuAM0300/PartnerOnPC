import os
import asyncio
import requests
import json
import numpy as np
import sounddevice as sd
import whisper
import time
import webrtcvad
from google import genai 
from google.genai.types import Content, Part
from dotenv import load_dotenv
import urllib.parse
import wavio # 録音保存用


# --- 環境設定 ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- グローバル設定 ---
VOICEVOX_ENGINE_URL = "http://127.0.0.1:50021" 
VOICEVOX_SPEAKER_ID = 14  
RATE = 48000 
VOICEVOX_SPEED_SCALE = 1.13 # 1.0が標準速度。1.2〜1.5程度が自然
#VOICEVOX_PITCH_SCALE = 1.0 #
CHANNELS = 1
SILENCE_TIMEOUT_SECONDS = 1.5 # VADの無音検出タイムアウト
FRAME_DURATION_MS = 30 # 処理する音声フレームの長さ (VAD設定)
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000) 
VAD_AGGRESSIVENESS = 3 # VADの積極性
WHISPER_MODEL_NAME = "small" # 使用するモデル名
WHISPER_MODEL = None        # モデル格納用変数
# --- 終了コマンド設定 ---
EXIT_COMMANDS = ["さようなら", "さよなら"]

# --- クライアント初期化 ---
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
WHISPER_MODEL = None 

async def async_generator_from_string(text):
    """シンプルな文字列を非同期ジェネレーターとして返す"""
    yield text

# ====================================================================
# I. 音声入力 (VAD & ローカルWhisper)
# ====================================================================

def setup_whisper_model():
    """Whisperモデルをロードし、グローバル変数に設定する"""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print(f"👂 Whisper: ローカルモデル '{WHISPER_MODEL_NAME}' をロード中...")
        try:
            import torch
            device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"👂 Whisper: 検出されたデバイス: {device_to_use}")

            WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_NAME, device=device_to_use)
        except Exception as e:
            print(f"❌ Whisperモデルロードエラー: {e}")
            return False
        print("✅ Whisper: モデルロード完了。")
    return True

async def record_and_transcribe():
    """VADで録音し、ローカルWhisperで転写する"""
    print("\n🎤 録音待機中... 話し始めてください。")
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    stream = sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype='int16')
    stream.start()
    
    recorded_frames = []
    speaking_start_time = None
    silence_frame_count = 0
    
    # === VADによる自動録音ループ ===
    while True:
        # NOTE: sd.InputStream.read() はブロッキング関数だが、このタスクはIO待ちのため問題ない
        audio_frame, overflowed = stream.read(FRAME_SIZE)
        is_speech = vad.is_speech(audio_frame.tobytes(), RATE)
        
        if is_speech:
            if speaking_start_time is None:
                speaking_start_time = time.time()
                print("🗣️ 発話開始を検出。")
            
            silence_frame_count = 0
            recorded_frames.append(audio_frame.flatten())
        else:
            if speaking_start_time is not None:
                silence_frame_count += 1
                recorded_frames.append(audio_frame.flatten()) 
                
                if silence_frame_count * FRAME_DURATION_MS / 1000 > SILENCE_TIMEOUT_SECONDS:
                    print(f"🤫 {SILENCE_TIMEOUT_SECONDS}秒間の無音を検出しました。")
                    break 
            
        if speaking_start_time is not None and time.time() - speaking_start_time > 30:
            print("🕒 最大録音時間（30秒）を超過しました。")
            break

    stream.stop()
    stream.close()
    
    if not recorded_frames:
        print("🤷‍♂️ 発話が検出されませんでした。")
        return None
    
    recording = np.concatenate(recorded_frames, axis=0)
    print(f"✅ 録音終了。合計録音時間: {len(recording) / RATE:.2f}秒")
    
    # 一時ファイルとして保存
    temp_wav_path = "temp_input.wav"
    wavio.write(temp_wav_path, recording, RATE, sampwidth=2)

    # === Whisperによる転写 ===
    print("👂 Whisper: 転写処理を開始します...")
    try:
        # 転写処理は非同期スレッドで実行
        result = await asyncio.to_thread(
            WHISPER_MODEL.transcribe, temp_wav_path, language="ja"
        )
        os.remove(temp_wav_path) # 一時ファイルを削除
        transcript = result["text"].strip()
        print(f"✅ Whisper: 完了。テキスト: \"{transcript}\"")
        return transcript
    except Exception as e:
        print(f"❌ Whisper転写エラー: {e}")
        os.remove(temp_wav_path)
        return None

# ====================================================================
# II. 応答生成 (Gemini: ストリーミング) - 履歴管理機能を追加
# ====================================================================

async def generate_gemini_stream(prompt_text, conversation_history):
    """Gemini APIを使用して応答をストリーミングで生成する"""
    print("🤖 Gemini: 応答生成を開始します...")
    
    # プロンプト設定
    system_instruction = "あなたはユーザーの作業中の話し相手となる、知識が豊富で受動的な女性アシスタントの「こよみ」です。質問には簡潔に、親しみやすいトーンで答えてください。ゆったりした話し方で、ため口での会話をしてください。応答に特殊文字や絵文字を含めないでください。箇条書きでの回答を控えてください。事実の回答を除いて断言を控えてください。AI側から積極的に話題を振らないでください。"
    
    # 最新のユーザー入力を履歴に追加 (Gemini APIに渡すためのmessagesリストを作成)
    messages_to_send = []
    for item in conversation_history + [{"role": "user", "content": prompt_text}]:
        messages_to_send.append(
            Content(
                role=item["role"] if item["role"] != "assistant" else "model",
                parts=[Part(text=item["content"])]
            )
        )

    try:
        response_stream = await asyncio.to_thread(
            gemini_client.models.generate_content_stream,
            model="gemini-2.5-flash",
            contents=messages_to_send, # 履歴と最新の入力を渡す
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )

        full_response = ""
        for chunk in response_stream:
            content = chunk.text
            if content:
                yield content 
                full_response += content
                
        # 応答をグローバルな履歴に追加 (メインループで参照される)
        conversation_history.append({"role": "user", "content": prompt_text})
        conversation_history.append({"role": "assistant", "content": full_response})
        
        print("✅ Gemini: 全ての応答テキストを生成しました。")
        
        # 履歴が長くなりすぎたら古いものを削除 (トークンコスト削減のため)
        # 例: 履歴を直近の10ターン分に制限
        if len(conversation_history) > 10:
            conversation_history[:] = conversation_history[-10:]

    except Exception as e:
        print(f"❌ Gemini APIエラー: {e}")
        yield "API接続でエラーが発生しました。時間を置いて再度お話しください。"

# ====================================================================
# III. 音声出力 (TTS: VOICEVOX) - 変更なし
# ====================================================================

async def generate_and_play_tts(text_stream):
    # ... (VOICEVOXのコードは、gpt_tts_pipeline.pyから変更なくここに貼り付けます) ...
    full_text = ""
    sentence_buffer = "" 
    
    play_stream = sd.OutputStream(samplerate=RATE, channels=CHANNELS, dtype='float32')
    play_stream.start()
    
    print("🔈 VOICEVOX: 合成・再生を開始します...")

    async for chunk in text_stream:
        full_text += chunk
        sentence_buffer += chunk
        
        if any(p in sentence_buffer for p in ['。', '！', '？', '\n']):
            audio_data = await _call_voicevox_api(sentence_buffer)
            
            if audio_data is not None:
                audio_data_float = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                play_stream.write(audio_data_float)
            
            sentence_buffer = "" 

    if sentence_buffer.strip():
        audio_data = await _call_voicevox_api(sentence_buffer)
        if audio_data is not None:
            audio_data_float = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            play_stream.write(audio_data_float)

    play_stream.stop()
    play_stream.close()
    print("✅ VOICEVOX: 再生完了。")
    return full_text


async def _call_voicevox_api(text):
    """VOICEVOXのAPIを呼び出して音声データを取得する内部関数"""
    if not text or not text.strip():
        return None
        
    try:
        # 1. 音素とアクセントの取得 (Audio Query)
        encoded_text = urllib.parse.quote(text) 
        response = await asyncio.to_thread(
            requests.post,
            f"{VOICEVOX_ENGINE_URL}/audio_query?speaker={VOICEVOX_SPEAKER_ID}&text={encoded_text}"
        )
        response.raise_for_status() 
        
        # 2. 音声合成 (Synthesis)
        query = response.json()
        query['speedScale'] = VOICEVOX_SPEED_SCALE
        #query['pitchScale'] = VOICEVOX_PITCH_SCALE
        query['outputSamplingRate'] = RATE
        response_audio = await asyncio.to_thread(
            requests.post,
            f"{VOICEVOX_ENGINE_URL}/synthesis",
            params={'speaker': VOICEVOX_SPEAKER_ID},
            json=query
        )
        response_audio.raise_for_status()
        
        return response_audio.content 
        
    except requests.exceptions.RequestException as e:
        print(f"❌ VOICEVOX APIエラー: {e} (テキスト: \"{text}\")")
        return None