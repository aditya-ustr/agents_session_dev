from typing import TypedDict, Annotated, List, Optional, Dict
from PIL import Image
import json
import os
import io

from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage

from services.langchain_models import ModelServing
from services.agents import (RouterAgent,
                             TicketAgent,
                             FinalResponseAgent
                             )
from services.tool_service import ticket_creation_tools
from services.rag_service import PolicyRag



class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    route: str
    response_message: str


class SupportFlowService:

    def __init__(self):

        self.llm = ModelServing()
        self.rag = PolicyRag()

        self.graph = StateGraph(GraphState)

        self.graph.add_node("query_router", self.query_router_node)
        self.graph.add_node("ticket_creation_node", self.ticket_creation_node)
        self.graph.add_node("ticket_creation_tool_node", self.ticket_creation_tool_node)
        self.graph.add_node("context_fetch_node", self.context_fetch_node)
        self.graph.add_node("final_answer_node", self.final_answer_node)

        self.graph.add_edge(START, "query_router")
        self.graph.add_conditional_edges("query_router", self.router, {"answer_query": "context_fetch_node", "create_ticket": "ticket_creation_node"})
        self.graph.add_edge("context_fetch_node", "final_answer_node")
        self.graph.add_conditional_edges("ticket_creation_node", self.tool_router, {"tools": "ticket_creation_tool_node", "continue": "final_answer_node"})
        self.graph.add_edge("ticket_creation_tool_node", "final_answer_node")
        self.graph.add_edge("final_answer_node", END)

        self.app = self.graph.compile()

        img_bytes = self.app.get_graph(xray=False).draw_mermaid_png(background_color="white")
        img_stream = io.BytesIO(img_bytes)
        image = Image.open(img_stream)
        os.makedirs("data/output", exist_ok=True)
        image.save("data/output/graph.png","PNG")

        print("\n\n----- App is Ready -----\n\n")

    def router(self, state: GraphState):
        return state.get("route")
    
    def tool_router(self, state: GraphState):

        response = state.get("messages")[-1]
        if response.tool_calls:
            return "tools"
        return "continue"


    def query_router_node(self, state: GraphState):

        query = state.get("query", "")
        relevant_items = {"user_query": query}
        relevant_items["policy_index"] = """
        1.  Password Reset Policy
        2.  VPN Access Policy
        3.  Software Installation Policy
        4.  Laptop Troubleshooting Policy
        5.  Email Access Policy
        6.  Remote Work IT Policy
        7.  Hardware Replacement Policy
        8.  Security Policy
        9.  Printer Support Policy
        10. IT Support Contact Guidelines
        """

        messages, response_format = RouterAgent().get_prompt_format(relevant_items=relevant_items)

        response = self.llm.run_inference(messages=messages, response_format=response_format)
        answer = json.loads(response.content)

        route = answer["route"]
        reason = answer["reason"]

        print(f"Choosing Route: {route}, Reason: {reason}")

        return {"route": route}
    
    def ticket_creation_node(self, state: GraphState):

        query = state.get("query", "")
        relevant_items = {"user_query": query}

        messages, _ = TicketAgent().get_prompt_format(relevant_items=relevant_items)

        response = self.llm.run_inference_tools(messages=messages, tools=ticket_creation_tools)
        if not response.tool_calls:
            response.content = "No tool call needed"

        return {"messages": [response]}
    
    def ticket_creation_tool_node(self, state: GraphState):

        response = state.get("messages")[-1]

        tool_messages = []
        tool_by_name = {t.name: t for t in ticket_creation_tools}

        for tool in response.tool_calls:
            tool_name = tool["name"]
            tool_id = tool["id"]
            tool_args = tool["args"]
            print(f"Calling tool {tool_name}")
            content = tool_by_name[tool_name].func(**tool_args)
            tool_messages.append(ToolMessage(tool_call_id=tool_id, content=content))

        return {"messages": tool_messages}
    
    def context_fetch_node(self, state: GraphState):

        query = state.get("query", "")
        context = self.rag.retrieve(query=query)
        human_message = HumanMessage(content=context)

        return {"messages": human_message}
    
    def final_answer_node(self, state: GraphState):

        query = state.get("query", "")
        state_messages = state.get("messages", [])
        relevant_items = {"user_query": query}

        messages, response_format = FinalResponseAgent().get_prompt_format(relevant_items=relevant_items)

        response = self.llm.run_inference(messages=messages+state_messages, response_format=response_format)
        answer = json.loads(response.content)

        response_message = answer["response_message"]

        print(f"\n\n--> Final Response: {response_message}")

        return {"response_message": response_message}
    

if __name__ == "__main__":

    query = "how to change my password?"
    # query = "My laptop is overheating and shutting down"
    query = "i want to know about VPN policy"

    service = SupportFlowService()
    state = GraphState(
        messages=[query],
        query=query
    )

    results = service.app.invoke(state)

