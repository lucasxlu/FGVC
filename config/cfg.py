import json

with open('../config/configuration.json', mode='rt', encoding='utf-8') as f:
    cfg = json.load(f)
    print('loading config files...')
    print('+' * 300)
    print(cfg)
    print('+' * 300)
