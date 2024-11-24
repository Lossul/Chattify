##Run the code using streamlit run app.py command in the terminal
##To log in, please check the existing login credentials in user_database.json file or else create your own using the signup window

import streamlit as st
import pandas as pd
import os
import json
import time
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import uuid #Universally Unique Identifier
# Initialize the model

CSV_FILE = "chat_history.csv"
# Simulated user database
USER_DATABASE_FILE = "user_database.json"

def initialize_model(api_key):
    repo_options = {
        "QWEN by ALIBABA CLOUD": "Qwen/Qwen2-1.5B-Instruct",
        "MISTRAL-7B": "mistralai/Mistral-7B-Instruct-v0.2",
        "BLENDERBOT by FACEBOOK": "facebook/blenderbot-3B",
        "LLAMA-3 by META": "meta-llama/Meta-Llama-3-8B-Instruct",
        "PHI by MICROSOFT": "microsoft/Phi-3-mini-4k-instruct",
        "ZEPHYR-7B by HUGGINGFACE": "HuggingFaceH4/zephyr-7b-alpha"
    }
    st.sidebar.markdown(":orange[STEP 1]")
    st.sidebar.header(":green[**HuggingFace models**]")
    repo_name = st.sidebar.selectbox("", tuple(
        repo_options.keys()), index=None, placeholder="Choose your HF model")
    if repo_name:
        repo_id = repo_options[repo_name]
        return HuggingFaceHub(huggingfacehub_api_token=api_key, repo_id=repo_id)

def get_button_label(chat_df, chat_id):
    first_message = chat_df[(chat_df["ChatID"] == chat_id) & (chat_df["Role"] == "User")].iloc[0]["Content"]
    return f"{' '.join(first_message.split()[:5])}..." # A summary like text is formed for our button

def save_chat_history(chat_df):
    chat_df.to_csv(CSV_FILE, index=False)


def load_chat_history():
    try:
        if os.path.exists(CSV_FILE):
            return pd.read_csv(CSV_FILE)  # Return DataFrame directly
        else:
            return pd.DataFrame(columns=["ChatID", "Role", "Content"])
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        return pd.DataFrame(columns=["ChatID", "Role", "Content"])
    
def save_user_database(user_database):
    with open(USER_DATABASE_FILE, "w") as file:
        json.dump(user_database, file, indent=4)

def load_user_database():
    try:
        if os.path.exists(USER_DATABASE_FILE):
            with open(USER_DATABASE_FILE, "r") as file:
                return json.load(file)
        else:
            return {}
    except Exception as e:
        st.error(f"Error loading user database: {e}")
        return {}

def history():
    if "history" not in st.session_state:
        st.session_state.history = []

    if "input" not in st.session_state:
        st.session_state.input = ""
        
    load_dotenv()
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    model = initialize_model(api_key)

    st.sidebar.write("**********************************************************************")
    st.sidebar.markdown(":orange[STEP 3]")
    st.sidebar.header(":green[**Chit Chats from Before**]")
    chat_history_df = load_chat_history()

    for chat_id in chat_history_df["ChatID"].unique():
        button_label = get_button_label(chat_history_df, chat_id)
        if st.sidebar.button(button_label, key = chat_id):
            
            #st.session_state.current_chat_id  = chat_id
            loaded_chat = chat_history_df[chat_history_df["ChatID"] == chat_id]
            loaded_chat_string = "\n\n".join(f"{'You' if row['Role'] == 'User' else 'Bot'}: {row['Content']}" 
                                             for _, row in loaded_chat.iterrows())
            #_ this represents index which we dont use in the loop body          
            st.text_area(":green[**Conversation Thread**]", value=loaded_chat_string, height=200)


    col1, col2 = st.sidebar.columns([1, 2.2])
    
    with open('trash-fill.svg', 'r') as f:
        svg_icon = f.read()
    clear_button = col1.button(":red[Clear All]", key="clear_all")
    col2.markdown(svg_icon.replace('<svg', '<svg width="32" height="32"'),unsafe_allow_html=True)

    if clear_button:
        st.session_state.history.clear()
        save_chat_history(pd.DataFrame(columns=["ChatID", "Role", "Content"]))
        st.sidebar.error("*Chat history is currently empty.*")
        st.experimental_rerun()
    
    st.markdown(":orange[STEP 2]")
    # Create a form for user input
    with st.form(key="input_form"):
        with st.chat_message("user"):
            user_input = st.text_input("You:", value=st.session_state.input, key="input")

        col1, col2, col3, col4 = st.columns([1, 6, 1.7, 0.6])
        with col1:
                st.form_submit_button(":green[Send]")

        with col2:                                       
            with open('send.svg', 'r') as f:
                svg_icon = f.read()
            st.markdown(svg_icon.replace('<svg', '<svg width="30" height="30"'), unsafe_allow_html=True)

        with col3:
                st.form_submit_button(":orange[Chat History]")

        with col4:
            with open('chat-text.svg', 'r') as f:
                svg_icon = f.read()
            st.markdown(svg_icon.replace('<svg', '<svg width="32" height="32"'), unsafe_allow_html=True)

       # buttons history_button and send button need not be explicitly called as in st.form, each button formed returns True
    if user_input:
        # Generate response using the context-aware prompt
        # try:

            if model is None:
                st.error("Please select a model from the sidebar")
                
            else:
                        # Add user's input to history
                        st.session_state.history.append({"role": "user", "content": user_input})

                        # Build context string from recent conversation (e.g., last 10 entries)
                        context = "\n".join([entry["content"] for entry in st.session_state.history[-10:]])

                        # Update prompt template to include context
                        prompt_template = PromptTemplate.from_template(f"{context}\n{user_input}\nBot:")
                
                        chain = LLMChain(llm=model, prompt=prompt_template, llm_kwargs={"max_new_tokens": 300})
                        response = chain.predict(input=user_input)
                        content = response.strip().split("Bot:")[-1].strip()
                        placeholder = st.empty()
                        streamed_response = ""

                        
                        for word in content.split():
                                streamed_response += word + " "
                                placeholder.success(streamed_response)
                                try:
                                    time.sleep(0.05)  # Adjust sleep time for faster/slower streaming
                                except KeyboardInterrupt:
                                    break  # Exit loop if user interrupts
                            
                        # with st.chat_message("assistant"):
                        #     st.success(content)

                        # Add bot's response to history
                        st.session_state.history.append({"role": "bot", "content": content})

                        # Generate a unique ChatID for each session
                        chat_id = uuid.uuid4().hex

                        # Load the current chat history DataFrame
                        chat_history_df = load_chat_history()

                        # Append the new conversation to the DataFrame
                        new_data = pd.DataFrame([
                            {"ChatID": chat_id, "Role": "User", "Content": user_input},
                            {"ChatID": chat_id, "Role": "AI", "Content": content}
                        ])

                        chat_history_df = pd.concat([chat_history_df, new_data], ignore_index=True)

                        # Save the updated chat history DataFrame
                        save_chat_history(chat_history_df)
        # except Exception as e:
        #             st.error("An error occurred while processing. Please try again.")
# Streamlit app
def main():
    # Initialize session state variables
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None

    user_database = load_user_database()

    def login(username, password):
        if user_database.get(username) == password:
            st.session_state.logged_in = True
            st.session_state.username = username
        else:
            st.error("Invalid username or password")

    def signup(username, password):
        if username in user_database:
            st.error("Username already exists")
        else:
            user_database[username] = password
            save_user_database(user_database)
            st.success("User registered successfully. Please log in.")

    def logout():
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("Login to interact with :orange[Chattify]")
        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit_button = st.form_submit_button("**Login**")
                if submit_button:
                    login(username, password)

        with tab2:
            with st.form("signup_form"):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                signup_button = st.form_submit_button("**Sign Up**")
                if signup_button:
                    signup(new_username, new_password)
    else:
        st.sidebar.button("**Logout**", on_click=logout)
        st.title(f"Welcome, {st.session_state.username}!")
        left_co, cent_co, last_co = st.columns(3)
        with cent_co:
            st.image("CHATTIFY.jpg", width=250)
        st.title(":orange[LangChain] Chatbot with :green[Streamlit]")
        history()
        footer = """<style>.footer {position: fixed;left: 0;bottom: 0;width: 100%;background-color: #000;color: white;text-align: center;}
        </style><div class='footer'><p>Developed with ❤️ using Streamlit and Langchain</p></div>"""
        st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
