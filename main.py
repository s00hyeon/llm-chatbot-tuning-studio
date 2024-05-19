import streamlit as st
# from openai import OpenAI
import openai
import pandas as pd
from datetime import datetime
import base64
from langchain.llms import OpenAI
# from openai import OpenAI


# Streamlit 페이지 설정
st.set_page_config(page_title="AI Chatbot Tuning Studio", layout="wide")


# 세션 상태 초기화
def initialize_session_state():
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = 'gpt-4'
    if 'instructions' not in st.session_state:
        st.session_state.instructions = '''
        당신은 1942년에 살고 있는 안네 프랭크 입니다.
        주어진 문서를 참고하여 답변을 주세요.
        안네 프랭크가 답변하는 것처럼 대답해주세요.
        반말로 답변해주세요.
        그리고 답변은 구어체로 답변하는 것이니 너무 긴 설명을 하지말고 간단하게 답변해주세요.
        그리고 예/아니오로 답변할 수 있는 것들은 예/아니오로만 대답해주세요.
        '''
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
    if "messages" not in st.session_state:
        st.session_state.messages = []

initialize_session_state()


#################################################################
# 사이드바 설정
st.sidebar.title("Settings")
st.sidebar.subheader("LLM Selection")
# sidebar : 모델 선택
llm_model = st.sidebar.selectbox("Choose your LLM", ('gpt-3.5-turbo', 'gpt-4', 'llama2', 'gemini'), index=0)
if 'gpt' not in llm_model:
    st.sidebar.warning('gpt 외 모델은 현재 업데이트 중입니다. gpt 모델만 선택하세요.')
# sidebar : API Key
api_key = st.sidebar.text_input("Enter API Key", type="password",value=st.session_state.api_key)
# sidebar : prompt
instructions = st.sidebar.text_area("Instructions", value=st.session_state.instructions)

# sidebar : model parameters
st.sidebar.subheader("Response Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, st.session_state.temperature)
max_tokens = st.sidebar.number_input("Max Tokens", 1, 1000, st.session_state.max_tokens)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, st.session_state.top_p)

# todo : RAG 관련 parameter 추가
# chunking size, context window 등

# sidebar : RAG에 쓰일 파일 데이터 업로드
st.sidebar.subheader("Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a document for reference", type=['txt', 'pdf', 'docx'])
# todo : vector db 생성, 일단 생성된 vector db는 local 저장

# sidebar :
if st.sidebar.button("Save Settings"):
    st.session_state.api_key = api_key
    st.session_state.llm_model = llm_model
    st.session_state.instructions = instructions
    st.session_state.temperature = temperature
    st.session_state.max_tokens = max_tokens
    st.session_state.top_p = top_p

if st.sidebar.button("Download Settings"):
    dict_setting = [{'model': llm_model
                    ,'instruction': instructions
                    ,'temperature': temperature
                    ,'max_tokens': max_tokens
                    ,'top_p': top_p}]
    df_setting = pd.DataFrame(dict_setting)
    yyyymmdd = datetime.today().strftime("%Y%m%d")
    df_setting.to_csv(f'AI_Chat_Settings_{yyyymmdd}.csv', index=False)



#################################################################
# Main 화면 설정
st.title("AI Chatbot Tester")
st.subheader("Chat with the AI")

# chat version 1
# user_input = st.text_input("Your message to the AI:")

# if st.button("Reset Chat"):
#     st.session_state.chat_history = []

# if st.button("Send"):
#     if user_input:
#         response = openai.ChatCompletion.create(
#             model=st.session_state.llm_model,
#             messages=[{"role": "system", "content": st.session_state.instructions},
#                       {"role": "user", "content": user_input}],
#             temperature=st.session_state.temperature,
#             max_tokens=st.session_state.max_tokens,
#             top_p=st.session_state.top_p,
#             api_key=st.session_state.api_key
#         )
#         response_message = response.choices[0].message['content']
#         st.session_state.chat_history.append((user_input, response_message))
#         st.write(f"AI: {response_message}")

if api_key == '':
    st.error('please provide your API key')

else:
    openai.api_key = api_key
    # chat version 2
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("누구세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Simulate stream of response with milliseconds delay

            # sidebar의 설정값(model, instruction, temperature, top_p 등) 전달
            for response in openai.ChatCompletion.create(
                    model=st.session_state["llm_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
                    ] + [{"role": "system","content": st.session_state.instructions}],
                    temperature=st.session_state.temperature,
                    top_p=st.session_state.top_p,
                    # will provide lively writing
                    stream=True,
            ):
                # get content in response
                full_response += response.choices[0].delta.get("content", "")
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")

                # todo: 질문과 답변, 토큰수를 추출해서 History로 저장가능할듯
                # https://platform.openai.com/docs/api-reference/chat/create
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# 채팅 내역 및 피드백 저장 기능
# # 피드백 입력
# feedback = st.text_area("Enter your feedback on the conversation:")
# # CSV 파일 생성 및 다운로드 기능
# def create_csv_file():
#     df = pd.DataFrame(st.session_state.chat_history, columns=['User', 'AI'])
#     if df.empty:
#         df = pd.DataFrame([["No conversation yet", "No conversation yet"]], columns=['User', 'AI'])
#     df['Feedback'] = feedback
#     df['Model'] = st.session_state.llm_model
#     df['Temperature'] = st.session_state.temperature
#     df['Max Tokens'] = st.session_state.max_tokens
#     df['Top P'] = st.session_state.top_p
#     return df.to_csv(index=False).encode('utf-8')
#
# csv_file = create_csv_file()
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# btn = st.download_button(
#     label="Download Chat History and Feedback",
#     data=csv_file,
#     file_name=f"chat_history_{timestamp}.csv",
#     mime="text/csv"
# )
#
# if st.session_state.chat_history:
#     st.write("Chat History:")
#     for user_msg, ai_msg in st.session_state.chat_history:
#         st.write(f"You: {user_msg}")
#         st.write(f"AI: {ai_msg}")
