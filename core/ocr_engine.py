from dataclasses import dataclass

@dataclass
class OCRConfig:
    lang: str = "en"

class OCREngine:
    def __init__(self, *args, **kwargs):
        pass

    def extract_text(self, image):
        return ""
