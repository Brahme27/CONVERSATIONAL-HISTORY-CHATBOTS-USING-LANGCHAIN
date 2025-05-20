# Import required modules
import os 
from dotenv import load_dotenv

# Load environment variables from .env file (used for storing secrets like API keys)
load_dotenv()

# Get the GROQ API key from the environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Import ChatGroq from langchain_groq to interact with the Groq LLM
from langchain_groq import ChatGroq

# Initialize the Groq model with the selected model name and API key
model = ChatGroq(
    model="Gemma2-9b-It",
    groq_api_key=groq_api_key
)

# Import message classes to represent human and system messages in the conversation
from langchain_core.messages import SystemMessage, HumanMessage

# Create a human message to send to the model
humanmessage = "Hi my name is brahmesh,im an AI engineer"

# Send a single-turn message to the model and receive a response
response = model.invoke([
    # SystemMessage(content="You are a helpful assistant."),  # Optional system instruction
    HumanMessage(content=humanmessage)
])

# Print the content of the model's response
response.content

# Additional interaction with AI message in the history
from langchain_core.messages import AIMessage, HumanMessage

# Simulate multi-turn conversation history
model.invoke([
    HumanMessage(content=humanmessage),
    AIMessage(content="Hi Brahmesh, it's nice to meet you!  \n\nThat's awesome that you're an AI engineer. It's a really exciting field right now. \n\nWhat kind of AI work do you do?  Do you have any projects you're particularly proud of?\n"),
    HumanMessage(content="Hey what is my name and what do i do")
])

# Chat history management using a message history store
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Dictionary to store message histories by session ID
store = {}

# Function to get or create a message history for a given session ID
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap the model to support maintaining message history across turns
with_message_history = RunnableWithMessageHistory(model, get_session_history)

# Use session ID "chat1" to simulate continuity in conversation
config = {"configurable": {"session_id": "chat1"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi my name is brahmesh,im an AI engineer")],
    config=config
)
response.content

# Using the same session ID to see if model remembers previous messages
config1 = {"configurable": {"session_id": "chat1"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hey what is my name")],
    config=config1
)
response.content

# Using a different session ID to simulate a new chat (memory should not persist)
config2 = {"configurable": {"session_id": "chat2"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hey what is my name")],
    config=config2
)
response.content

# Prompt Template with MessagesPlaceholder to dynamically insert conversation history
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create a chat prompt template with a system message and placeholder for user messages
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="messages")
    ]
)

# Create a chain that combines prompt and model
chain = prompt | model

# Invoke the chain with a single human message
chain.invoke({"messages": [HumanMessage(content="Hi my name is brahmesh")]})

# Enable history-aware conversations using the new chain and history getter
with_message_history = RunnableWithMessageHistory(chain, get_session_history)

# Use session ID "chat3" for a new conversation thread
config = {"configurable": {"session_id": "chat3"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi my name is brahmesh")],
    config=config
)
response.content

# Continue the conversation in same session to check memory
config = {"configurable": {"session_id": "chat3"}}
response = with_message_history.invoke(
    [HumanMessage(content="heyy what is my name")],
    config=config
)
response.content

# Add more dynamic control by allowing language-specific responses
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistant.Answer in {language}"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

# Create a chain that takes language as a variable
chain = prompt | model

# Invoke with language specified as Hindi
response = chain.invoke(
    {
        "messages": [HumanMessage(content="Hi my name is brahmesh")],
        "language": "Hindi"
    }
)
response.content

# Wrap the multilingual chain with message history tracking
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)

# Start session "chat4" for multilingual conversation
config = {"configurable": {"session_id": "chat4"}}
response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="Hi my name is brahmesh")],
        "language": "Hindi"
    },
    config=config
)
response.content

# Continue in the same session to test memory in Hindi
response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="Heyy what is my name")],
        "language": "Hindi"
    },
    config=config
)
response.content

# Demonstrate trimming of messages to manage long histories (to fit within token limits)
# Import trimming utility and suppress warnings
import warnings
warnings.filterwarnings("ignore")
from langchain_core.messages import SystemMessage, trim_messages

# Define a trimmer with max token limit and strategy to keep last messages
trimmer = trim_messages(
    max_tokens=100,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

# Sample message history to test trimming
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi my name is brahmesh"),
    AIMessage(content="Hi Brahmesh, it's nice to meet you!..."),
    HumanMessage(content="Hey what is my name and what do i do"),
    AIMessage(content="Your name is Brahmesh and you are an AI engineer."),
    HumanMessage(content="what is 2+2"),
    AIMessage(content="2+2=4"),
    HumanMessage(content="what is 2+3"),
    AIMessage(content="2+3=5"),
    HumanMessage(content="enjoy talking to you"),
    AIMessage(content="I enjoy talking to you too!"),
]

# Apply trimming on the messages
trimmer.invoke(messages)

# Lower max tokens to 45 and see how history gets trimmed
trimmer = trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

# Apply trimmer with stricter limit
trimmer.invoke(messages)

# Use trimmed messages inside a functional chain
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

# Create a chain that trims messages before passing them to prompt and model
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

# Add another message and invoke the chain
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what is my name")],
        "language": "English"
    }
)
response.content

# Asking again to see if model remembers math problems (depends on token limit)
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what math problem did i ask")],
        "language": "English"
    }
)
response.content

# Finally, wrap the above chain with session-aware history handler
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)

# New session "chat5" for this memory-aware trimming example
config = {"configurable": {"session_id": "chat5"}}
response = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="what is my name")],
        "language": "English"
    },
    config=config
)
response.content

# Continue conversation in same session to check memory
response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="what math problem did i ask")],
        "language": "English"
    },
    config=config
)
response.content
