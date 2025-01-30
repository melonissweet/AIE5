import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# Define Agent State
class AgentState(TypedDict):
  messages: Annotated[list, add_messages]


class AgentPipeline:
    def __init__(self) -> None:
        self.tavily_tool = TavilySearchResults(max_results=5)
        self.wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=150)
        self.wiki_tool = WikipediaQueryRun(api_wrapper=self.wiki_api_wrapper)
        self.tool_belt = [
            self.tavily_tool,
            self.wiki_tool,
            PubmedQueryRun()
        ]
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(self.tool_belt)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        tool_node = ToolNode(self.tool_belt)
        
        graph.add_node("agent", self.call_model)
        graph.add_node("action", tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", self.tool_call_or_helpful, {
            "continue": "agent",
            "action": "action",
            "end": END
        })
        graph.add_edge("action", "agent")
        return graph.compile()

    async def call_model(self, state):
        messages = state["messages"]
        response = await self.llm.ainvoke(messages) 
        return {"messages": [response]} # Ensures response is a list

    async def tool_call_or_helpful(self, state):
        if len(state["messages"]) > 10: # checks whether to end the query cycle as the first step of this function
            return "end"
        
        last_message = state["messages"][-1]
        
        if last_message.tool_calls:
            return "action"
        
        initial_query = state["messages"][0]
        final_response = state["messages"][-1]

        helpfulness_prompt_template = """\
        Given an initial query and a final response, determine if the final response is extremely helpful or not. 
        Indicate helpfulness with 'Y' and unhelpfulness as 'N'.

        Initial Query:
        {initial_query}

        Final Response:
        {final_response}"""

        prompt_template = PromptTemplate.from_template(helpfulness_prompt_template)
        helpfulness_model = ChatOpenAI(model="gpt-4o-mini")
        helpfulness_chain = prompt_template | helpfulness_model | StrOutputParser()
        helpfulness_response = await helpfulness_chain.ainvoke({
            "initial_query": initial_query.content,
            "final_response": final_response.content
        })

        return "end" if "Y" in helpfulness_response else "continue"

    async def astream(self, inputs):
        async for chunk in self.graph.astream(inputs, stream_mode="updates"):
            yield chunk


@cl.on_chat_start
async def on_chat_start():
    agent_pipeline = AgentPipeline()

    # reset the user session messages
    cl.user_session.set("agent_pipeline", agent_pipeline)

    await cl.Message(content="Agent is ready! Ask questions you want to know more about in life science :)").send()


@cl.on_message
async def main(message):
    agent_pipeline = cl.user_session.get("agent_pipeline")

    # start a fresh state for every user query
    user_messages = [HumanMessage(content=message.content)]

    inputs = {"messages": user_messages}
    msg = cl.Message(content="")

    async for chunk in agent_pipeline.astream(inputs):  
        for node, values in chunk.items():
            if node == "action":
                node_header = f"**Receiving update from node: '{node}'**\nTool Used: {values['messages'][0].name}\n"
            else:
                node_header = f"**Receiving update from node: '{node}'**\n"
            await msg.stream_token(node_header)
            
            for response in values["messages"]:
                response_header = f"{response.content}\n\n"
                await msg.stream_token(response_header) 

    await msg.send()