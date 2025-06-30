def print_results(text, entities):
    """แสดงผลลัพธ์อย่างสวยงาม"""
    print("=" * 80)
    print("Input Text:")
    print(text)
    print("\n" + "-" * 80)
    print("Extracted Entities:")

    if not entities:
        print("No entities found")
        return

    for ent in entities:
        print(f"- {ent['text']} ({ent['label']}) [{ent['start']}:{ent['end']}]")

    # สร้างข้อความไฮไลท์ entities
    highlighted = list(text)
    for ent in entities:
        start, end = ent['start'], ent['end']
        if start < len(highlighted):
            highlighted[start] = f"【{highlighted[start]}"
        if end < len(highlighted):
            highlighted[end] = f"】{highlighted[end]}"
        elif end == len(highlighted):
            highlighted.append("】")

    print("\n" + "-" * 80)
    print("Highlighted Text:")
    print(''.join(highlighted))


def save_to_json(data, filename):
    """บันทึกผลลัพธ์เป็น JSON"""
    import json
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
