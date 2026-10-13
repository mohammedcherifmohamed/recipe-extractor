from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser , ResponseSchema
from langchain_classic.output_parsers import OutputFixingParser , PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()
# StructuredOutputParser
# this helps us to define the output format we want from the model.
#  we can add Pydantic BaseModel for datatype validation and type checking

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

class Person(BaseModel):
    name:str=Field(description="The name of the person")
    age:int=Field(gt=20,lt=160,description="The age of the person")
    city :str=Field(description="The city where the person lives")

schema=[
    ResponseSchema(name="mysql",description="The MySQL query"),
    ResponseSchema(name="table",description="The table name"),
    ResponseSchema(name="description",description="A description of the table")
]

parser = PydanticOutputParser(pydantic_object=Person)
safe_parser=OutputFixingParser.from_llm(llm=model,parser=parser)
template = PromptTemplate(
    template="""Give me three facts about {topic}. Return only
    valid JSON that follows this format:

    {response_format}""",
    input_variables=["topic"],
    partial_variables={"response_format": parser.get_format_instructions()},
)

chain=template | model | parser

result=chain.invoke({"topic":"cats"})
print(result)
