from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser , ResponseSchema
from langchain_classic.output_parsers import OutputFixingParser , PydanticOutputParser 
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

prompt1 = PromptTemplate(template="generate 3 detailed facts about {topic}",
                         input_varialbles=["topic"])

prompt2 = PromptTemplate(template="generate 3 poionts about this topic :  {text}",
                         input_varialbles=["text"])


parser=StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model |parser

result=chain.invoke({"topic":"cats"})


print(result)
