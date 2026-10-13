from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage , AIMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict , Annotated , List , Optional



from dotenv import load_dotenv

load_dotenv()

class Review(TypedDict):
    sql:Annotated[str, "SQL query generated based on the question and schema"]
    table:Annotated[str, "Table name related to the SQL query"]
    columns:Annotated[List[str],"List of columns involved in the SQL query"]
    description:Annotated[str, "Description of the SQL query"]

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

structred_model = model.with_structured_output(Review)


chat_template =ChatPromptTemplate([
( 'system',"""
You are a MySQL query generator.

Allowed operations: SELECT
Forbidden operations: INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, REVOKE

Rules:{rules} """),
( 'human',"""Schema:
  users_table(Uid, name, email, age, city, is_active, created_at)
  products(id, name, price, category, stock, created_at)
  orders(id, user_id, product_id, quantity, total, status, created_at)

Question: {question}""")
]) 

prompt =chat_template.invoke({
    'rules' : """
- If the question can be answered with an allowed operation, return a JSON object: {"sql": "...", "table": "...", "description": "..."}
- If the question references a table or column not present in the schema, return exactly: NULL
- If the question requires a forbidden operation, return exactly: NULL in sql and table fields and "this is a forbidden operation" in the description field.
- If the question is ambiguous, off-topic, or cannot be answered from the schema, return exactly: NULL
- Output only the JSON object or the word NULL. No commentary.""",

    'question' : "delete user with id 5."
})

print(prompt)
result= structred_model.invoke(prompt)
print(result)


# while True:
#     user_input = input("you: ")
#     chat_history.append(HumanMessage(content=user_input))
#     if user_input.lower() == "q":
#         break
#     result = model.invoke(chat_history)
#     chat_history.append(AIMessage(content=result.content[0]['text']))
#     print(result.content[0]['text'])





