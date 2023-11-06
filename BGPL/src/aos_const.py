import json

senttag2opinion = {'pos': 'great', 'neg': 'bad', 'neu': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

with open("force_tokens.json", 'r') as f:
    force_tokens = json.load(f)

task_data_list = {
    "aste": ["laptop14", "rest14", "rest15", "rest16"],
    "tasd": ['rest15', "rest16"],
    "acos": ['laptop16', "rest16"],
    "asqp": ['rest15', "rest16"],
}
force_words = {
    'aste': {
        'rest15': list(senttag2opinion.values()) + ['[SSEP]'],
        'rest16': list(senttag2opinion.values()) + ['[SSEP]'],
        'rest14': list(senttag2opinion.values()) + ['[SSEP]'],
        'laptop14': list(senttag2opinion.values()) + ['[SSEP]']
    }
}

task_data_list = {
    "aste": ["laptop14", "rest14", "rest15", "rest16"],
    "tasd": ['rest15', "rest16"],
    "acos": ['laptop16', "rest16"],
    "asqp": ['rest15', "rest16"],
}

optim_orders_all = {
            "aste": {
                "laptop14": [
                    '[A] [O] [S]'
                ],
                "rest14": [
                    '[A] [O] [S]'
                ],
                "rest15": [
                    '[A] [O] [S]'
                ],
                "rest16": [
                    '[A] [O] [S]'
                ],
            },

        }

heuristic_orders = {
    'aste': ['[A] [O] [S]'],
    'tasd': ['[A] [C] [S]'],
    'asqp': ['[A] [O] [C] [S]'],
    'acos': ['[A] [O] [C] [S]'],
}