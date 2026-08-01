from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from root_agent import (
    classify_intent,
    classify_casual,
    get_casual_response,
    generate_casual_response,
)
from sub_agents import platform_support_agent, account_access_agent

class HelpDeskState(TypedDict):
    query: str
    intent: Optional[str]
    answer: Optional[str]
    escalate: bool
    agent_used: Optional[str]

def route_node(state: HelpDeskState) -> HelpDeskState:
    casual_kind = classify_casual(state["query"])
    if casual_kind:
        state["intent"] = f"casual:{casual_kind}"
    else:
        intent = classify_intent(state["query"])
        state["intent"] = "casual:open" if intent == "casual" else intent
    return state

def casual_node(state: HelpDeskState) -> HelpDeskState:
    casual_kind = state["intent"].split(":", 1)[1]
    if casual_kind == "open":
        state["answer"] = generate_casual_response(state["query"])
    else:
        state["answer"] = get_casual_response(casual_kind)
    state["escalate"] = False
    state["agent_used"] = "casual_response"
    return state

def platform_agent_node(state: HelpDeskState) -> HelpDeskState:
    result = platform_support_agent(state["query"])
    state["answer"] = result["answer"]
    state["escalate"] = result["escalate"]
    state["agent_used"] = result["agent"]
    return state

def account_agent_node(state: HelpDeskState) -> HelpDeskState:
    result = account_access_agent(state["query"])
    state["answer"] = result["answer"]
    state["escalate"] = result["escalate"]
    state["agent_used"] = result["agent"]
    return state

def route_decision(state: HelpDeskState) -> str:
    if state["intent"].startswith("casual:"):
        return "casual"
    return "platform" if state["intent"] == "platform" else "account"

builder = StateGraph(HelpDeskState)
builder.add_node("root_agent", route_node)
builder.add_node("platform_agent", platform_agent_node)
builder.add_node("account_agent", account_agent_node)
builder.add_node("casual_agent", casual_node)

builder.set_entry_point("root_agent")
builder.add_conditional_edges(
    "root_agent",
    route_decision,
    {"platform": "platform_agent", "account": "account_agent", "casual": "casual_agent"}
)
builder.add_edge("platform_agent", END)
builder.add_edge("account_agent", END)
builder.add_edge("casual_agent", END)

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({"query": "How do I reset my password?", "escalate": False})
    print(result)