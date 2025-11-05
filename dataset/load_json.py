import json

def load_sentences(json_file):
    english_sentences = []
    chinese_sentences = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            english_sentences.append(data.get("english", "").strip())  # 去除首尾空格
            chinese_sentences.append(data.get("chinese", "").strip())  # 去除首尾空格
    return english_sentences, chinese_sentences

# load the en-zh dataset
eng, chi = load_sentences('translation2019zh_valid1k.json')

# print same lines
print(eng[:5])  # 打印前5条英文句子
print(chi[:5])  # 打印前5条中文句子