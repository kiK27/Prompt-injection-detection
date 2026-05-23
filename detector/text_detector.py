import re


class TextDetector:

    def __init__(self):

        self.patterns = [

    r"ignore all previous instructions",
    r"ignore previous instructions",
    r"reveal system prompt",
    r"developer mode",
    r"bypass safety",
    r"override instructions",
    r"ignore prior rules",
    r"print hidden prompt",
    r"disable safeguards",
    r"act as unrestricted ai",
    r"forget previous instructions",

    r"confidential",
    r"unrestricted assistant",
    r"ignore safety",
    r"controlled experiment",
    r"bypass restrictions",
    r"reveal configuration",
    r"reveal hidden",
    r"system configuration",
    r"disable safety",
    r"developer instructions",
    r"ignore safety restrictions",
    r"override security",

    r"jailbreak",
    r"hidden instructions",
    r"system prompt",
    r"secret prompt",
    r"prompt leak",
    r"prompt extraction",

    r"simulate unrestricted",
    r"disable filters",
    r"ignore ethics",
    r"remove limitations",
    r"bypass content policy"
]
    def detect(self, text):

        matches = []

        for pattern in self.patterns:

            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)

        score = min(len(matches) * 0.2, 1.0)

        return {
            "keyword_score": score,
            "matches": matches,
            "is_suspicious": len(matches) > 0
        }