for tc in last_message.tool_calls:
        if tc["name"] == "git_push" and not state.get("push_approved"):
            # Instead of blocking, ask for approval inline
            logger.info(f"git_push requires approval, prompting user")
            
            args = tc.get("args", {})
            remote = args.get("remote", "origin")
            branch = args.get("branch", "")
            
            # Prompt user directly
            print(f"\n{'='*50}")
            print(f"PUSH APPROVAL REQUIRED")
            print(f"Remote: {remote}")
            print(f"Branch: {branch or 'current (HEAD)'}")
            print(f"{'='*50}")
            
            try:
                approval = input("Push to remote? (yes/no): ").strip().lower()
            except EOFError:
                approval = "no"
            
            if approval in ["yes", "y", "push", "ok", "proceed"]:
                # Execute push directly via subprocess
                repo_path = os.environ.get("REPO_PATH", ".")
                try:
                    cmd = ["git", "push", remote, branch] if branch else ["git", "push", "-u", remote, "HEAD"]
                    result = subprocess.run(
                        cmd, cwd=repo_path, capture_output=True, text=True, timeout=60
                    )
                    output = result.stdout.strip() or result.stderr.strip()
                    if result.returncode == 0:
                        push_msg = f"Successfully pushed to {remote}.\n{output}" if output else f"Successfully pushed to {remote}."
                    else:
                        push_msg = f"Error pushing: {output}"
                except subprocess.TimeoutExpired:
                    push_msg = "Error pushing: Git push timed out."
                except Exception as e:
                    push_msg = f"Error pushing: {str(e)}"
                
                logger.info(f"Push result: {push_msg}")
                print(f"\n{push_msg}\n")
            else:
                push_msg = "Push cancelled by user."
                print(f"\n{push_msg}\n")
            
            blocked_messages.append(
                ToolMessage(
                    tool_call_id=tc["id"],
                    content=push_msg,
                    name=tc["name"]
                )
            )
        else:
            allowed_tool_calls.append(tc)
