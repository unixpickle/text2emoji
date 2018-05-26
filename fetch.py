"""
Download emojis from https://unicode.org/emoji/charts/full-emoji-list.html.
"""

from html.parser import HTMLParser
import requests


class EmojiParser(HTMLParser):
    def __init__(self):
        self._headers = 2
        self._column_idx = 0
        self._row_info = {}

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self._rows_remaining -= 1
            self._column_idx = -1
            self._row_info = {}
            return

        if self._headers >= 0:
            return

        if tag == 'td':
            self._column_idx += 1
        elif tag == 'img':
            if self._column_idx > 2 and self._column_idx < 10:
                self._row_info['images_%d' % self._column_idx] = attrs['src']

    def handle_endtag(self, tag):
        # TODO: this.
        pass

    def handle_data(self, data):
        if self._column_idx == 1:
            self._row_info['codepoints'] = self._row_info.get('codepoints', '') + ' ' + data
