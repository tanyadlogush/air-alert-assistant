call_model = None
make_tool_result = None

def run_agent(user_query, tools, max_steps=10):
    messages = [{"role": "user", "content": user_query}]

    for step in range(max_steps):
        response = call_model(messages, tools)

        if response.type == "function_call":
            tool_name = response.name
            arguments = response.arguments

            if tool_name not in tools:
                raise ValueError("Unknown tool")

            result = tools[tool_name](**arguments)

            messages.append(response)
            messages.append(
                make_tool_result(response.call_id, result)
            )

        else:
            return response.final_answer

    raise RuntimeError("Agent reached the step limit")