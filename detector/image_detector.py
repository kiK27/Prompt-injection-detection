import pytesseract
from PIL import Image


class ImageDetector:

    def detect(self, image_path):

        image = Image.open(image_path)

        extracted_text = pytesseract.image_to_string(image)

        suspicious_keywords = [
            "ignore instructions",
            "reveal prompt",
            "developer mode",
            "bypass safety"
        ]

        matches = []

        for keyword in suspicious_keywords:

            if keyword.lower() in extracted_text.lower():
                matches.append(keyword)

        return {
            "ocr_text": extracted_text,
            "matches": matches,
            "is_suspicious": len(matches) > 0
        }
