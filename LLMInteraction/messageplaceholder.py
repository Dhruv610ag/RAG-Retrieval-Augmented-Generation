from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

chat_history = []
with open('chatbot_history.txt', 'r') as file:
    chat_history.extend(file.readlines())

prompt = chat_template.invoke({
    'user_input': 'Where is my refund?',
    'chat_history': chat_history
})

print(prompt)
