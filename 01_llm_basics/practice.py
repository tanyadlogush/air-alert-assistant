import os
from dotenv import load_dotenv
from openai import OpenAI

import json

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# #================== Input: plain & structured =============================
#
# response = client.responses.create(
#     model='gpt-4o',
#     # input='Що таке повітряна тривога?',
#     input=[
#         {
#             'role': 'system',
#             'content': 'Ти технічний експерт. Відповідай коротко, використовуючи технічну термінологію.'
#         },
#         {
#             'role': 'user',
#             'content': 'Що таке повітряна тривога?'
#         }
#     ]
# )
#
# print(response.output_text)

# # ====================== messages: saving message history ========================================
# messages = [
# {
#             'role': 'system',
#             'content': 'Ти технічний експерт. Відповідай коротко, використовуючи технічну термінологію.'
#         },
# ]
#
# n = 0   # контроль витрати токенів)
#
# while n != 3:
#     n += 1
#
#     # user
#     user_message = input('Your question: ')
#
#     if user_message == '+':
#         break
#
#     messages.append({'role': 'user', 'content': user_message})
#
#     # response
#     response = client.responses.create(
#         model='gpt-4o',
#         input = messages
#     )
#
#     # assistant
#     messages.append({'role': 'assistant', 'content': response.output_text})
#     print(response.output_text)


# # ========= structured output (text.format) ==========
# response = client.responses.create(
#     model="gpt-4o",
#     input="У Київській області оголошена повітряна тривога о 14:05.",
#     text={
#         "format": {
#             "type": "json_schema",
#             "name": "air_alert",
#             "schema": {
#                 "type": "object",
#                 "properties": {
#                     "region": {
#                         "type": "string"
#                     },
#                     "status": {
#                         "type": "string"
#                     },
#                     "time": {
#                         "type": "string"
#                     }
#                 },
#                 "required": ["region", "status", "time"],
#                 "additionalProperties": False
#             },
#             "strict": True
#         },
#         'verbosity': 'low'    # detail of the text result (low, medium, high)
#     }
# )
#
# print(response.output_text)

# ======== tools + instruction ============

# def get_air_alert_status(region):
#     return f'Air alert status for {region}: NO ALERT'

# def get_air_alert_status(region):
#     statuses = {
#         'Київ': 'ALERT',
#         'Львів': 'NO ALERT'
#     }
#     return f'Air alert status for {region}: {statuses.get(region, "UNKNOWN")}'


def get_air_alert_status(region):
    statuses = {
        'Київ': 'ALERT',
        'Львів': 'NO ALERT'
    }

    if region not in statuses:
        raise ValueError(f'Unknown region: {region}')

    return f'Air alert status for {region}: {statuses[region]}'



response = client.responses.create(
    model='gpt-4o',
    input='Який статус повітряної тривоги в Одесі?',
    instructions='Ти технічий експерт. Відповідай лаконічно, українською.',
    tools=[
        {
            'type': 'function',
            'name': 'get_air_alert_status',
            'description': 'Отримання поточного статусу повітряної тривоги у вказаному регіоні.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'region': {
                        'type': 'string',
                        'description': 'Region name'
                    }
                },
                'required': ['region'],
                'additionalProperties': False
            }
        }
    ]
)

# просте зіставлення назви tool → Python-функція
tool_functions = {
    'get_air_alert_status': get_air_alert_status
}

# цикл, який для кожного function_call створює окремий результат
while True:
    tool_outputs = []

    for tool_call in response.output:
        if tool_call.type == 'function_call':
            arguments = json.loads(tool_call.arguments)
            function = tool_functions[tool_call.name]
            try:
                result = function(**arguments)
            except Exception as e:
                result = f'Tool error: {e}'

            tool_outputs.append({
                'type': 'function_call_output',
                'call_id': tool_call.call_id,
                'output': result
            })

    if not tool_outputs:
        break

    response = client.responses.create(
        model='gpt-4o',
        input=tool_outputs,
        previous_response_id=response.id
    )

print(response.output_text)


