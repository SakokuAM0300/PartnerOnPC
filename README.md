PartnerOnPC
APIを利用したデスクトップ向け対話型エージェント・インターフェース

1. プロジェクト概要
本プロジェクトは、PC上での作業中にシームレスに対話を行い、ユーザーの思考整理やタスク支援を行うことを目的としたチャットボット・アプリケーションです。
既存のブラウザベースのチャットUIではなく、デスクトップ環境に特化した軽量なインターフェースを提供し、API連携による柔軟な応答生成を実現しています。

2. 技術スタック
Deep Learning Framework: PyTorch v2.9.0 (with CUDA 12.6 support)
Parallel Computing: NVIDIA CUDA v12.9 / NVCC V12.9.86
LLM API: Google Gemini API
Speech-to-Text: OpenAI Whisper (Local, GPU Accelerated)
Text-to-Speech: Voicevox
Python: v3.10+ (venv推奨)

3. システム構成
User Interface: ユーザー入力を受け取り、チャットログをリアルタイムにレンダリング。
API Handler: APIキーを環境変数利用し、非同期通信を用いてレスポンスを取得。
Context Management: 短期的な対話履歴を保持し、文脈に沿った応答を生成。

4. 主な機能
Real-time Interaction: APIとの非同期通信による、遅延を最小限に抑えた対話。

History Persistence: セッション内での対話履歴の保持。

Responsive UI: デスクトップ上での作業を妨げない、シンプルで視認性の高いデザイン。

5. 課題と解決策
課題: API呼び出し時のレイテンシ遅延による会話テンポの遅延。
解決: ローディングインジケーターの実装と、非同期処理（async/await）の適切な制御により、待機中のストレスを軽減。

課題: 長文入力時のレイアウト崩れ。
解決: CSS Flexbox/Gridを用いた動的なボックスのリサイズと、スクロール制御の最適化。

6. 今後の展望と課題
現状のシステムには以下の改善の余地があり、順次実装を検討しています。

現状分の環境でのみ稼働するため、Dockerによるパッケージ化を急ぎます。
ユーザーの入力傾向を分析し、よりパーソナライズされた応答を生成するよう自動的な最適化。
ベクトルデータベースを用いたRAGの統合による、過去の対話内容に基づいた回答生成。
フロントエンドからのAPIキー完全秘匿化のためのプロキシサーバー構築によるセキュリティ強化。
