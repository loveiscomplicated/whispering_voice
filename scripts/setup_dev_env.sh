#!/bin/bash

echo "🚀 개발 환경 설정 시작..."

# 1. 가상환경 생성
echo "📦 가상환경 생성..."
python3 -m venv venv

# 2. 가상환경 활성화
echo "🔄 가상환경 활성화..."
source venv/bin/activate

# 3. pip 업그레이드
pip install --upgrade pip

# 4. 의존성 설치
echo "📥 의존성 설치..."
pip install -r requirements-dev.txt

# 5. Whisper 기본 모델 다운로드 (선택사항)
echo "🤖 Whisper 기본 모델 다운로드 (이 부분은 시간이 걸릴 수 있습니다)..."
python -c "import whisper; whisper.load_model('base')" 2>/dev/null || echo "⚠ 모델 다운로드 건너뜀 (나중에 수동으로 실행 가능)"

echo ""
echo "✅ 개발 환경 설정 완료!"
echo ""
echo "다음 단계:"
echo "1. 가상환경 활성화: source venv/bin/activate"
echo "2. 파일럿 테스트: python scripts/run_pilot_test.py"
