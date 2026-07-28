#!/bin/bash
cd "$(dirname "$0")"
echo "🦢 白天鹅智能管家启动中..."
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
export HF_ENDPOINT="https://hf-mirror.com"
python3 scripts/step4_serve.py &
sleep 4
npx localtunnel --port 8000 &
sleep 5
echo "✅ 已启动"
wait
