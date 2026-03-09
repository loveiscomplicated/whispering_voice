#!/bin/bash

echo "🧪 테스트 실행..."

# 전체 테스트
pytest backend/tests/ -v --cov=backend/stt_core --cov-report=html

echo ""
echo "✅ 테스트 완료!"
echo "📊 커버리지 보고서: htmlcov/index.html"
