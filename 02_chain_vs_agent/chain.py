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


# step 1: LLM determines the region
response = client.responses.create(
    model='gpt-4o',
    input='Який статус повітряної тривоги в Києві?',
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

# step 2: Application executes the first tool
tool_call = response.output[0]
arguments = json.loads(tool_call.arguments)
status = get_air_alert_status(**arguments)

# step 3: Application executes the next chain step
description = get_alert_description(status)


# step 4: LLM generates the final answer
response = client.responses.create(
    model='gpt-4o',
    input=f'Регіон: {arguments["region"]}\nСтатус: {status}\nОпис: {description}',
    instructions='Сформуй коротку відповідь користувачу українською.'
)

print(response.output)

