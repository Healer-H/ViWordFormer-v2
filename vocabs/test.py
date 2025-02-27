from utils.utils import preprocess_sentence
from utils.word_decomposation import is_Vietnamese, split_non_vietnamese_word
import json

with open(
    r"D:\Vipher_ver2\ViWordFormer-v2\data\UIT-VSFC\UIT-VSFC-test.json",
    "r",
    encoding="utf-8",
) as f:
    data = json.load(f)

vocab_analysis = []
for entry in data:
    words = preprocess_sentence(entry["sentence"])  # Tiền xử lý câu
    for word in words:
        is_vietnamese, phoneme = is_Vietnamese(word)
        if is_vietnamese:
            onset, medial, nucleus, coda, tone = phoneme
            vocab_analysis.append(
                {
                    "word": word,
                    "phoneme": {
                        "onset": onset,
                        "medial": medial,
                        "nucleus": nucleus,
                        "coda": coda,
                        "tone": tone,
                    },
                }
            )
        else:
            non_vietnamese_parts = split_non_vietnamese_word(word)
            vocab_analysis.append(
                {
                    "word": word,
                    "phoneme": "Non-Vietnamese",
                    "parts": non_vietnamese_parts,
                }
            )

with open(
    r"D:\Vipher_ver2\ViWordFormer-v2\vocab_test.json", "w", encoding="utf-8"
) as f:
    json.dump(vocab_analysis, f, ensure_ascii=False, indent=4)

print("Đã tạo và lưu file vocab.json thành công!")
