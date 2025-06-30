import json
import sys
from pathlib import Path

def validate_file(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    mismatches=[]
    for idx,item in enumerate(data):
        text=item['text']
        for ent in item.get('entities',[]):
            start=ent['start']; end=ent['end']; etext=ent['text']
            if text[start:end]!=etext:
                mismatches.append((idx,ent,text[start:end]))
    return mismatches

if __name__=='__main__':
    if len(sys.argv)<2:
        print('Usage: python validate_dataset.py <path_to_json>')
        sys.exit(1)
    path=Path(sys.argv[1])
    mism=validate_file(path)
    if mism:
        print(f'Found {len(mism)} mismatched entities in {path}')
        for idx,ent,extract in mism[:5]:
            print(f' Example at index {idx}: expected "{ent["text"]}", found "{extract}"')
    else:
        print(f'All entities validated for {path}')
