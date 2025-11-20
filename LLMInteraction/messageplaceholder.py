from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
chat_template=ChatPromptTemplate.from_messages([
    'system: You are a helpful customer support assistant.',
    
])