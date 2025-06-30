import re


def split_to_words(text):
    """แบ่งคำที่ดียิ่งขึ้น รวม whitespace และ URL ทั้งคำ"""
    # URL pattern
    url_pattern = r'https?://[^\s]+'

    # หา URLs ก่อน
    words = []
    start = 0
    for match in re.finditer(url_pattern, text):
        # เพิ่มข้อความก่อน URL
        if match.start() > start:
            words.extend(re.findall(r'\S+|\s+', text[start:match.start()]))

        # เพิ่ม URL ทั้งคำ
        words.append(match.group())
        start = match.end()

    # เพิ่มข้อความที่เหลือ
    if start < len(text):
        words.extend(re.findall(r'\S+|\s+', text[start:]))

    return words
