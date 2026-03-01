try:
                if interrupted:
                    # Resume from interrupt with user's response
                    print(f"[DEBUG] Resuming interrupt with: '{user_input}'")
                    try:
                        result = graph.invoke(Command(resume=user_input), config)
                        interrupted = False
                    except GraphInterrupt as e:
                        # Another interrupt fired during resume
                        interrupt_value = e.args[0] if e.args else "Approval required"
                        print(f"\n{interrupt_value}")
                        print("\nType 'push' to push, or 'cancel' to abort.\n")
                        continue
                else:
                    if first_invocation:
                        state["messages"] = [HumanMessage(content=user_input)]
                        try:
                            result = graph.invoke(state, config)
                        except GraphInterrupt as e:
                            interrupted = True
                            interrupt_value = e.args[0] if e.args else "Push approval required"
                            print(f"\n{interrupt_value}")
                            print("\nType 'push' to push, or 'cancel' to abort.\n")
                            first_invocation = False
                            continue
                        first_invocation = False
                    else:
                        try:
                            result = graph.invoke(
                                {"messages": [HumanMessage(content=user_input)]},
                                config,
                            )
                        except GraphInterrupt as e:
                            interrupted = True
                            interrupt_value = e.args[0] if e.args else "Push approval required"
                            print(f"\n{interrupt_value}")
                            print("\nType 'push' to push, or 'cancel' to abort.\n")
                            continue
                
                state = result
                
                # Print the last AI message
                for msg in reversed(state["messages"]):
                    if isinstance(msg, AIMessage):
                        if msg.content:  # Only print if non-empty
                            print(f"\nAgent: {msg.content}\n")
                        break
                
                # Show pending changes if any
                if state.get("awaiting_modification_approval") and state.get("pending_changes"):
                    print("\n" + format_changes_for_display(state["pending_changes"]))
                    print("\nType 'approve' to apply changes or describe what you'd like to change.\n")
                    
            except Exception as e:
                error_msg = str(e)
                print(f"\nError: {error_msg}\n")
                import traceback
                traceback.print_exc()
                interrupted = False
