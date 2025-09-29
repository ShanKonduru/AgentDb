@echo off
REM Activate venv if present, then run Streamlit app
IF EXIST .\.venv\Scripts\activate (
  call .\.venv\Scripts\activate
)

streamlit run app.py