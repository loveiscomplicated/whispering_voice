from setuptools import setup, find_packages

setup(
    name="whisper-voice-conversion",
    version="0.1.0",
    description="속삭임 감지 음성 변환 시스템",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "librosa>=0.10.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "openai-whisper>=20230314",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
        ],
    },
    python_requires=">=3.8",
)
