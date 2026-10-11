from app.models.schemas import Domain


class IntentClassifier:
    """Keyword-based intent classification. Stateless, no side effects."""

    KEYWORDS: dict[str, dict[str, list[str]]] = {
        "government": {
            "en": [
                "NIN", "passport", "JAMB", "voter", "BVN", "CAC", "registration",
                "licence", "NIMC", "INEC", "immigration", "driver", "license",
                "government", "national ID", "voter card",
            ],
            "yo": ["ìwé ìrìnnà", "ìwé àṣẹ", "ìforúkọsílẹ̀", "ìjọba"],
            "ha": ["takarda", "rajista", "lasisi", "gwamnati"],
            "ig": ["akwụkwọ", "ndebanye aha", "gọọmentị"],
        },
        "health": {
            "en": [
                "fever", "malaria", "pregnant", "hospital", "medicine", "sick",
                "pain", "doctor", "clinic", "child", "baby", "cough", "diarrhoea",
                "diarrhea", "health", "disease", "symptom", "headache", "vomit",
            ],
            "yo": ["ibà", "àìsàn", "ọmọ", "ilé ìwòsàn", "ọ̀gangan", "oyún"],
            "ha": ["zazzabi", "ciwo", "asibiti", "magani", "yaro", "ciki"],
            "ig": ["ahụ ọhụrụ", "ọrịa", "ụlọ ọgwụ", "nwa", "ime"],
        },
        "market": {
            "en": [
                "price", "how much", "cost", "buy", "sell", "market", "bag",
                "basket", "naira", "expensive", "cheap",
            ],
            "yo": ["elo", "ọjà", "ta", "rà", "owó", "àpò"],
            "ha": ["nawa", "kasuwa", "saya", "sayarwa", "kuɗi", "jaka"],
            "ig": ["ego ole", "ahịa", "zụ", "re", "ego", "akpa"],
        },
        "agriculture": {
            "en": [
                "farm", "plant", "crop", "fertilizer", "harvest", "seed", "maize",
                "cassava", "rice", "poultry", "cattle", "goat", "pest", "irrigation",
                "soil", "planting", "farming", "livestock",
            ],
            "yo": ["oko", "àgbẹ̀", "irúgbín", "ajílẹ̀", "ikórè", "ọkà", "iresi"],
            "ha": ["gona", "noma", "shuka", "taki", "girbi", "masara", "shinkafa", "kaji"],
            "ig": ["ubi", "ọrụ ubi", "mkpụrụ", "manụre", "owuwe", "ọka", "osikapa"],
        },
    }

    def classify(self, text: str, language: str) -> Domain:
        """Classify text into a domain. Returns Domain.GENERAL if no match."""
        if not text or not text.strip():
            return Domain.GENERAL

        text_lower = text.lower()
        scores: dict[str, int] = {}

        for domain, lang_keywords in self.KEYWORDS.items():
            # Check keywords for the user's language AND English (code-switching)
            all_keywords = lang_keywords.get(language, []) + lang_keywords.get("en", [])
            score = sum(1 for kw in all_keywords if kw.lower() in text_lower)
            if score > 0:
                scores[domain] = score

        if not scores:
            return Domain.GENERAL

        return Domain(max(scores, key=scores.get))
