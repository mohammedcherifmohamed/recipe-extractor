from dotenv import load_dotenv
from langchain_classic import hub

from langchain.agents import create_agent

from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def get_current_time(*args,**kwargs):
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="useful for when you need to know the current time",
    ),
]

# prompt = hub.pull("hwchase17/react")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Use the provided tools to answer questions.",
)


response = agent.invoke({
    
     "messages": [{"role": "user", "content": "what time is it?"}]
                  })

print(response["messages"][-1].content)

