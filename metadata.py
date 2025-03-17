UIT_VSFC_METADATA = {
    "name": "UIT_VFSC",
    "task": {
        "sentiment": {
            "name": "UIT_VSFC_Dataset_Sentiment",
            "text": "sentence",
            "label": "sentiment",
            "num_label": 3,
        },
        "topic": {
            "name": "UIT_VSFC_Dataset_Topic",
            "text": "sentence",
            "label": "topic",
            "num_label": 4,
        }
    },
    "vocab_size": 230,
    "vocab_size_v2": 133,
    "data_paths": {
        "train": "data/UIT-VSFC/UIT-VSFC-train.json",
        "dev": "data/UIT-VSFC/UIT-VSFC-dev.json",
        "test": "data/UIT-VSFC/UIT-VSFC-test.json",
    }
}

UIT_ViCTSD_METADATA = {
    "name": "UIT_ViCTSD",
    "task": {
        "constructiveness": {
            "name": "UIT_ViCTSD_Dataset_Construct",
            "text": "comment",
            "label": "constructiveness",
            "num_label": 2,
        },
        "toxic": {
            "name": "UIT_ViCTSD_Dataset_Toxic",
            "text": "comment",
            "label": "toxicity",
            "num_label": 2,
        }
    },
    "vocab_size": 336,
    "vocab_size_v2": 213,
    "data_paths": {
        "train": "data/UIT-ViCTSD/train.json",
        "dev": "data/UIT-ViCTSD/dev.json",
        "test": "data/UIT-ViCTSD/test.json",
    }
}

UIT_ViOCD_METADATA = {
    "name": "UIT_ViOCD",
    "task": {
        "domain": {
            "name": "UIT_ViOCD_Dataset_Domain",
            "text": "review",
            "label": "domain",
            "num_label": 4,
        },
        "topic": {
            "name": "UIT_ViOCD_Dataset_Label",
            "text": "review",
            "label": "label",
            "num_label": 2,
        }
    },
    "vocab_size": 473,
    "vocab_size_v2": 357,
    "data_paths": {
        "train": "data/UIT-ViOCD/train.json",
        "dev": "data/UIT-ViOCD/dev.json",
        "test": "data/UIT-ViOCD/test.json",
    }
}
