#!/bin/bash
cd /Users/Zhuanz/ai-projects/ai-data-agent
echo "🦢 白天鹅智能管家启动中..."
export DEEPSEEK_API_KEY="你的DeepSeek密钥"
export HF_ENDPOINT="https://hf-mirror.com"
python3 scripts/step4_serve.py &
sleep 4
npx localtunnel --port 8000 &
sleep 5
echo "✅ 已启动"
wait
