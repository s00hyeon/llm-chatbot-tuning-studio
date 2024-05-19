import streamlit as st
import openai
import pandas as pd
from datetime import datetime
import base64

# Streamlit 페이지 설정
st.set_page_config(page_title="AI Chatbot Tester", layout="wide")

# 세션 상태 초기화
def initialize_session_state():
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = 'gpt-4'
    if 'instructions' not in st.session_state:
        st.session_state.instructions = ''
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = 150
    if 'top_p' not in st.session_state:
        st.session_state.top_p = 1.0
    if 'feedback' not in st.session_state:
        st.session_state.feedback = ''

initialize_session_state()

# 사이드바 설정
st.sidebar.title("Settings")
st.sidebar.subheader("LLM Model Selection")
llm_model = st.sidebar.selectbox("Choose the LLM Model", ('gpt-3.5-turbo', 'gpt-4', 'llama2', 'gemini'), index=0)
if 'gpt' not in llm_model:
    st.sidebar.warning('gpt 외 모델은 현재 업데이트 중입니다. gpt 모델만 선택하세요.')
api_key = st.sidebar.text_input("Enter API Key", type="password")
instructions = st.sidebar.text_area("Instructions", value=st.session_state.instructions)

st.sidebar.subheader("Response Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, st.session_state.temperature)
max_tokens = st.sidebar.number_input("Max Tokens", 1, 1000, st.session_state.max_tokens)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, st.session_state.top_p)

st.sidebar.subheader("Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a document for reference", type=['txt', 'pdf', 'docx'])

if st.sidebar.button("Save Settings"):
    st.session_state.api_key = api_key
    st.session_state.llm_model = llm_model
    st.session_state.instructions = instructions
    st.session_state.temperature = temperature
    st.session_state.max_tokens = max_tokens
    st.session_state.top_p = top_p

# Main 화면 설정
st.title("AI Chatbot Tester")
st.subheader("Chat with the AI")
user_input = st.text_input("Your message to the AI:")

if st.button("Reset Chat"):
    st.session_state.chat_history = []

if st.button("Send"):
    if user_input:
        response = openai.ChatCompletion.create(
            model=st.session_state.llm_model,
            messages=[{"role": "system", "content": st.session_state.instructions},
                      {"role": "user", "content": user_input}],
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens,
            top_p=st.session_state.top_p,
            api_key=st.session_state.api_key
        )
        response_message = response.choices[0].message['content']
        st.session_state.chat_history.append((user_input, response_message))
        st.write(f"AI: {response_message}")

# 피드백 입력
feedback = st.text_area("Enter your feedback on the conversation:")

# CSV 파일 생성 및 다운로드 기능
def create_csv_file():
    df = pd.DataFrame(st.session_state.chat_history, columns=['User', 'AI'])
    if df.empty:
        df = pd.DataFrame([["No conversation yet", "No conversation yet"]], columns=['User', 'AI'])
    df['Feedback'] = feedback
    df['Model'] = st.session_state.llm_model
    df['Temperature'] = st.session_state.temperature
    df['Max Tokens'] = st.session_state.max_tokens
    df['Top P'] = st.session_state.top_p
    return df.to_csv(index=False).encode('utf-8')

csv_file = create_csv_file()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
btn = st.download_button(
    label="Download Chat History and Feedback",
    data=csv_file,
    file_name=f"chat_history_{timestamp}.csv",
    mime="text/csv"
)

if st.session_state.chat_history:
    st.write("Chat History:")
    for user_msg, ai_msg in st.session_state.chat_history:
        st.write(f"You: {user_msg}")
        st.write(f"AI: {ai_msg}")
