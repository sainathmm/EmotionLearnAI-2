@echo off
echo Installing dependencies...
pip install flask flask-cors opencv-python deepface numpy pandas scikit-learn Pillow tf-keras
echo.
echo Starting EmotionLearnAI...
python app.py
pause
