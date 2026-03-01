def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("setup",          catch_exceptions(setup_node))
    workflow.add_node("agent",          catch_exceptions(agent_node))
    workflow.add_node("tools",          catch_exceptions(tools_node))
    workflow.add_node("approval_check", catch_exceptions(approval_check_node))
    workflow.add_node("error_handler",  error_handler_node)

    # Remove push_approval and execute_push nodes entirely

    workflow.set_entry_point("setup")

    workflow.add_conditional_edges(
        "setup",
        route_on_error("agent"),
        {"agent": "agent", "error_handler": "error_handler"},
    )

    def agent_router(state: AgentState) -> str:
        if state.get("error"):
            return "error_handler"
        return should_continue(state)

    workflow.add_conditional_edges(
        "agent",
        agent_router,
        {
            "tools":          "tools",
            "approval_check": "approval_check",
            "end":            END,
            "error_handler":  "error_handler",
        },
    )

    # tools always goes back to agent
    workflow.add_conditional_edges(
        "tools",
        route_on_error("agent"),
        {"agent": "agent", "error_handler": "error_handler"},
    )

    workflow.add_conditional_edges(
        "approval_check",
        route_on_error("agent"),
        {"agent": "agent", "error_handler": "error_handler"},
    )

    workflow.add_edge("error_handler", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
