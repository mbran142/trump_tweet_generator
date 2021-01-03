import json
import os.path

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

def clean_data(raw_file, parsed_file):

    if '.json' not in raw_file:
        raw_file += '.json'

    if '.json' not in parsed_file:
        parsed_file += '.json'

    if raw_file == parsed_file:
        print('Error: Input and output names must be different.')
        return False

    if not os.path.isfile('data/' + raw_file):
        print('Error: ' + raw_file + ' not found.')
        return False

    # load raw tweet data
    with open('data/' + raw_file, encoding='utf8') as f:
        raw_data = json.load(f)

    # strip data of 'bad' tweet examples
    data = [tweet['text'] for tweet in raw_data if tweet['isRetweet'] == 'f'      # ignore retweets
                                                and 'http' not in tweet['text']   # ignore tweets with hyperlinks
                                                and tweet['text'][0] != '\"'      # ignore tweets that start with quotes
                                                and '@' not in tweet['text']]     # ignore tweets with direct responses

    # replace unicode characters with similar ascii ones (seen in unicode_conversion dict)
    for key, val in unicode_conversion.items():
        data = [tweet.replace(key, val) for tweet in data]

    # remove unicode strings from data
    data = [tweet for tweet in data if len(tweet) == len(tweet.encode())]

    # store data
    with open('data/' + parsed_file, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4)

    print(raw_file + ' successfully cleaned into ' + parsed_file)
    return True