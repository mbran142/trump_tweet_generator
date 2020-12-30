import json

unicode_conversion = {
    '\u2018': '\'',
    '\u2019': '\'',
    '\u201c': '\"',
    '\u201d': '\"',
    '\u2013': '-',
    '\u2014': '-',
    '\u00a0': ' ',
    '\u2026': '...',
    '&amp;': '&'
}

with open('data/data.json', encoding='utf8') as f:
    raw_data = json.load(f)

# data = [tweet['text'] for tweet in raw_data if tweet['isRetweet'] == 'f']
data = [tweet['text'] for tweet in raw_data if tweet['isRetweet'] == 'f'
                                            and 'http' not in tweet['text']
                                            and tweet['text'][0] != '\"'
                                            and tweet['text'][0] != '@'
                                            and tweet['text'][0] != '.'
                                            and tweet['text'][0:1] != b'\u']

# replace unicode characters with similar ascii ones
for key, val in unicode_conversion.items():
    data = [tweet.replace(key, val) for tweet in data]

# remove unicode strings from data
data = [tweet for tweet in data if len(tweet) == len(tweet.encode())]

with open('data/parsed_data.json', 'w', encoding='utf8') as f:
    json.dump(data, f, indent=4)

print(len(data))