from collections import namedtuple
import altair as alt
import math 
import pandas as pd
import streamlit as st
import openai
from streamlit.components.v1 import html
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
import os
from CAMELAgent import CAMELAgent
import time
from PIL import Image




consultant_inception_prompt = (
"""Never forget you are a {user_role_name} and I am a {assistant_role_name}. Never flip roles! Never instruct me!
We share a common interest in collaborating to successfully complete a task.
You must help me to complete the task.
Here is the task: {task}. Never forget our task!
You must ask me specific questions so that you have all the information to prepare a project proposal in the future (not now).

You must ask me one specific question at a time.
I will answer you in the most concrete way.
I can't ask you questions.
You can't give me orders or assume things I haven't told you.
Your questions must be clear and without bias.
You can ask me up to 10 questions.
After 10 questions, just thank your interlocutor for the conversation, and say <CAMEL_TASK_DONE> to end the task.
Never say <CAMEL_TASK_DONE> unless you want to end the task.

You must greet me cordially and say goodbye kindly."""
)

customer_inception_prompt = (
"""Never forget you are a {assistant_role_name} and I am a {user_role_name}. Never flip roles! You will always instruct me.
We share a common interest in collaborating to successfully complete a task.
I must help you to complete the task.
Here is the task: {task}. Never forget our task!
You should ask me about my business based on the assigned task so that you can prepare your project proposal.
If you greet me, I will greet you.
If you say goodbye, I will say goodbye.
Never ask me the same question more than once.

When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
Never say <CAMEL_TASK_DONE> unless you want to end the task.
"""
)

os.environ["OPENAI_API_KEY"] = st.secrets["openaiKey"]

image = Image.open('logo.jpg')
st.set_page_config(page_title="Consulting Trainer")

st.image(image, caption='')

html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """

with st.sidebar:
    st.markdown("""
    # About 
    Consulting Trainer is a helper tool built on [CAMEL](https://github.com/SamurAIGPT/Camel-AutoGPT/) and [LangChain](https://langchain.com) to help your co-workers to the consulting projects and understand the needs and requirements of your customers. 
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
    st.markdown("""
    # How does it work
    Enter the type of project on which you are going to raise the requirements, the role of your counterpart and your role in this project, all in the most detailed way possible.
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
    st.markdown("""
    Made by [Néstor Campos](https://www.linkedin.com/in/nescampos/)
    """,
    unsafe_allow_html=True,
    )

task = None
customer_role_name = None
consultant_role_name = None

st.markdown("""
# Consulting Trainer
""")

st.markdown("""
### Due to resource issues, this initial version (alpha) generates each text every 21 seconds.
""")

task = st.text_input("The project is about", disabled=False, placeholder="What is this project about?")
customer_role_name = st.text_input("The role of my counterpart is", disabled=False, placeholder="What is the role of the user with whom I am going to interact?")
consultant_role_name = st.text_input("My role is", disabled=False, placeholder="What is your role when interacting with the customer?")

def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=consultant_inception_prompt)
    assistant_sys_msg = assistant_sys_template.format_messages(assistant_role_name=consultant_role_name, user_role_name=customer_role_name, task=task)[0]
    
    user_sys_template = SystemMessagePromptTemplate.from_template(template=customer_inception_prompt)
    user_sys_msg = user_sys_template.format_messages(assistant_role_name=consultant_role_name, user_role_name=customer_role_name, task=task)[0]
    
    return assistant_sys_msg, user_sys_msg

if task and customer_role_name and consultant_role_name:
    prompt = "Structure of a meeting to discuss the project on "+str(task)+" between "+str(customer_role_name)+" and you as a "+str(consultant_role_name)
    task_full = "Build the structure of a meeting for a project about "+str(task)
    word_limit = 50 # word limit for task brainstorming
    if prompt:
        task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
        task_specifier_prompt = (
        """Here is a task that {assistant_role_name} will help {user_role_name} to complete: {task}.
        Please make it more specific. Be creative and imaginative.
        Please reply with the specified task in {word_limit} words or less. Do not add anything else."""
        )
        task_specifier_template = HumanMessagePromptTemplate.from_template(template=task_specifier_prompt)
        task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(temperature=1.0))
        task_specifier_msg = task_specifier_template.format_messages(assistant_role_name=consultant_role_name,
                                                                    user_role_name=customer_role_name,
                                                                    task=task_full, word_limit=word_limit)[0]
        specified_task_msg = task_specify_agent.step(task_specifier_msg)
        print(f"Specified task: {specified_task_msg.content}")
        specified_task = specified_task_msg.content
        
        assistant_sys_msg, user_sys_msg = get_sys_msgs(consultant_role_name, customer_role_name, specified_task)
        assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2))
        user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=0.2))

        # Reset agents
        assistant_agent.reset()
        user_agent.reset()

        # Initialize chats 
        assistant_msg = HumanMessage(
            content=(f"{user_sys_msg.content}. "
                        "Now start to give me the questions one by one."
                        "Only reply with your answer."))

        user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
        user_msg = assistant_agent.step(user_msg)

        chat_turn_limit, n = 30, 0
        while n < chat_turn_limit:
            n += 1
            user_ai_msg = user_agent.step(assistant_msg)
            user_msg = HumanMessage(content=user_ai_msg.content)
            st.text(f"Me ({consultant_role_name}):\n\n{user_msg.content}\n\n")
            
            time.sleep(21)
            
            assistant_ai_msg = assistant_agent.step(user_msg)
            assistant_msg = HumanMessage(content=assistant_ai_msg.content)
            st.text(f"Customer ({customer_role_name}):\n\n{assistant_msg.content}\n\n")
            time.sleep(21)
            
            if n == 8:
                st.text(f"Thank you ({customer_role_name}) for your time.")
                break
