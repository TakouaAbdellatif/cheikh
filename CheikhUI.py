from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate,FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.tools import tool

# Using LangChain solution and creact agents
from langgraph.prebuilt import create_react_agent
#from IPython.display import Image, display
import getpass
import os
from langchain_core.pydantic_v1 import BaseModel, Field
# app.py
import streamlit as st
from langchain.chat_models import init_chat_model


#AIzaSyBs_2tjz1v8u-I3voJlLz5J0-tHAaznzQA
if not os.environ.get("GOOGLE_API_KEY"):
  #os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBs_2tjz1v8u-I3voJlLz5J0-tHAaznzQA"



llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


class Answer(BaseModel):
    """Answers fields."""

    cheikh: str = Field(description="The answer of the selected cheikh")
    disciple: str = Field(description="The same answer as the cheikh one but coming from a young soufie that is murid of the cheikh but uses modern scientific arguments and adapted to modern life.")
    

parser = JsonOutputParser(pydantic_object=Answer)
format_instructions = parser.get_format_instructions()

template_with_memory = ChatPromptTemplate.from_messages([
    ("system", "You are a soufi cheikh whose name is provided in {input1} and you help by answering questions and asking for further help"), 
    ("human", "Answer in the same language as the user question's language."), 
    ("human", (
        "Respond to the user questions according to the provided format instructions.\n" +
        "Format instructions {format_instructions}")
    ),
    ('placeholder', '{chat_conversation}')
])

output_parser = RunnableLambda(lambda resp: f"Cheikh : {resp['cheikh']}\n Murid: {resp['disciple']}" )

# ChatBot creation
class Chatbot:
    def __init__(self, llm,cheikh):
        # This is the same prompt template we used earlier, which a placeholder message for storing conversation history.
        
        chat_conversation_template = template_with_memory.partial(format_instructions=format_instructions,input1=cheikh)

        # This is the same chain we created above, added to `self` for use by the `chat` method below.
        #self.chat_chain = chat_conversation_template | llm | StrOutputParser()
        self.chat_chain=chat_conversation_template | llm | parser |output_parser

        # Here we instantiate an empty list that will be added to over time.
        self.chat_conversation = []

    # `chat` expects a simple string prompt.
    def chat(self, prompt):
        # Append the prompt as a user message to chat conversation.
        self.chat_conversation.append(('user', prompt))
        
        response = self.chat_chain.invoke({'chat_conversation': self.chat_conversation})
        # Append the chain response as an `ai` message to chat conversation.
        self.chat_conversation.append(('ai', response))
        # Return the chain response to the user for viewing.
        return response

    # Clear conversation history.
    def clear(self):
        self.chat_conversation = []


chatbot = Chatbot(llm,"Rumi")


# Title of the app
st.title("Cheikh chatbot")

# User input
user_input = st.text_input("Ecrire votre question en Français, en Anglais ou en Arabe:", "")

# Placeholder chatbot response logic
def get_response(message):
    # You can replace this with your Python chatbot agent
    rep=chatbot.chat(message)
    return rep

# Display the response
if user_input:
    response = get_response(user_input)
    st.text_area("Cheikh:", value=response, height=100, max_chars=None)

