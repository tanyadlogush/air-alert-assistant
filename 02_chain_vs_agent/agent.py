import os
from dotenv import load_dotenv
from openai import OpenAI

import json

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def get_air_alert_status(region):
    statuses = {
        'Київ': 'ALERT',
        'Львів': 'NO ALERT'
    }

    return statuses.get(region, 'UNKNOWN')


def get_alert_description(status):
    descriptions = {
        'ALERT': 'Повітряна тривога оголошена.',
        'NO ALERT': 'Повітряної тривоги немає.',
        'UNKNOWN': 'Статус невідомий.'
    }

    return descriptions[status]


# functions map
available_functions = {
    'get_air_alert_status': get_air_alert_status,
    'get_alert_description': get_alert_description
}

# tools description
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_air_alert_status',
            'description': 'Отримання поточного статусу повітряної тривоги у вказаному регіоні.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'region': {'type': 'string', 'description': 'Назва регіону/міста'}
                },
                'required': ['region'],
                "additionalProperties": False,
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_alert_description',
            'description': 'Отримання опису за кодом статусу тривоги (ALERT, NO ALERT, UNKNOWN).',
            'parameters': {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string', 'description': 'Код статусу'}
                },
                'required': ['status'],
                "additionalProperties": False,
            },
        },
    },
]

# input request
input_messages = [
    {'role': 'user', 'content': 'Що там з тривогою в Києві?'}
]

# AGENT LOOP (Responses API)
current_input = input_messages

while True:
    response = client.responses.create(
        model='gpt-4o',
        input=current_input,
        instructions='Ти технічний асистент. Відповідай лаконічно, українською.',
        tools=tools,
        # tool_choice='auto'
    )

    # first generated action/message
    output_item = response.output[0]

    # if the model decided to call the tool
    if output_item.type == 'function_call':
        function_name = output_item.name
        function_args = json.loads(output_item.arguments)

        # calling a Python function
        function_to_call = available_functions[function_name]
        function_response = function_to_call(**function_args)

        # the next input for the model (with the result of the function)
        current_input = [
            {'role': 'user', 'content': f"Результат функції {function_name}: {function_response}. Що робити далі або яка відповідь користувачу?"}
        ]

    else:
        print('Фінальна відповідь агента:')
        print(output_item.text)
        break


