from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3", temperature=2)

prompt = ChatPromptTemplate.from_template(
    "create a joke on {topic} ."
)

chain = prompt | llm

response = chain.invoke({
    "topic": "indian people"
})

print(response.content)