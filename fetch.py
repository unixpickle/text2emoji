"""
Download emojis from https://unicode.org/emoji/charts/full-emoji-list.html.
"""

import base64
from html.parser import HTMLParser
import json
import os
import sys

import requests

EMOJI_URL = 'https://unicode.org/emoji/charts/full-emoji-list.html'
OUTPUT_DIR = 'emoji_data'

TEXT_COLUMNS = {1: 'codepoints', 14: 'text'}
IMAGE_COLUMNS = {3: 'apple', 4: 'google', 5: 'twitter', 6: 'one', 7: 'fb', 8: 'samsung',
                 9: 'windows'}


def main():
    if os.path.exists(OUTPUT_DIR):
        sys.stderr.write('output directory already exists: ' + OUTPUT_DIR + '\n')
        sys.exit(1)
    parser = EmojiParser()
    print('Fetching data...')
    parser.feed(requests.get(EMOJI_URL).text)
    print('Dumping data...')
    os.mkdir(OUTPUT_DIR)
    for row in parser.rows:
        dirname = os.path.join(OUTPUT_DIR, row['codepoints'])
        os.mkdir(dirname)
        for platform in IMAGE_COLUMNS.values():
            if platform in row:
                image_path = os.path.join(dirname, '%s.png' % platform)
                data = base64_data(row[platform])
                with open(image_path, 'wb+') as out_file:
                    out_file.write(data)
    mapping = {row['codepoints']: row['text'] for row in parser.rows}
    with open(os.path.join(OUTPUT_DIR, 'text.json'), 'w+') as out_file:
        out_file.write(json.dumps(mapping))


def base64_data(image_url):
    """
    Get the data out of a base64 image URL.
    """
    b64 = image_url.split(',')[-1]
    return base64.b64decode(b64)


class EmojiParser(HTMLParser):
    def __init__(self):
        super(EmojiParser, self).__init__()
        self.rows = []
        self._headers_remaining = 2
        self._column_idx = 0
        self._row_info = None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self._headers_remaining -= 1
            self._column_idx = -1
            if self._headers_remaining < 0:
                self._row_info = {}
            else:
                self._row_info = None
            return

        if self._row_info is None:
            return

        if tag == 'td':
            self._column_idx += 1
        elif tag == 'img':
            if self._column_idx in IMAGE_COLUMNS:
                name = IMAGE_COLUMNS[self._column_idx]
                self._row_info[name] = dict(attrs)['src']

    def handle_endtag(self, tag):
        if tag == 'tr' and self._row_info:
            if self._column_idx == 14:
                self.rows.append(self._row_info)
            self._row_info = None

    def handle_data(self, data):
        if self._row_info is None:
            return
        field_names = {1: 'codepoints', 14: 'text'}
        if self._column_idx in field_names:
            name = field_names[self._column_idx]
            self._row_info[name] = (self._row_info.get(name, '') + ' ' + data).strip()


if __name__ == '__main__':
    main()
