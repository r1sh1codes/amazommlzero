"""
Data loading and text normalization for entity resolution.
Handles business name and address cleaning, abbreviation expansion, etc.
"""

import re
import unicodedata
import pandas as pd

# ── Abbreviation maps ────────────────────────────────────────────────

NAME_ABBREVS = {
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "pvt": "private",
    "co": "company",
    "intl": "international",
    "natl": "national",
    "assoc": "association",
    "assn": "association",
    "dept": "department",
    "mfg": "manufacturing",
    "mgmt": "management",
    "svcs": "services",
    "svc": "service",
    "tech": "technology",
    "techs": "technologies",
    "grp": "group",
    "hldgs": "holdings",
    "sys": "systems",
    "soln": "solutions",
    "solns": "solutions",
    "engg": "engineering",
    "engr": "engineering",
    "eng": "engineering",
    "ent": "enterprises",
    "enterp": "enterprises",
    "indus": "industries",
    "ind": "industries",
    "bros": "brothers",
    "bro": "brother",
    "mktg": "marketing",
    "distrib": "distributors",
    "dist": "distributors",
    "pharm": "pharmaceuticals",
    "pharma": "pharmaceuticals",
    "hosp": "hospital",
    "univ": "university",
    "inst": "institute",
    "lab": "laboratory",
    "labs": "laboratories",
    "govt": "government",
    "fin": "financial",
    "fdn": "foundation",
    "fdtn": "foundation",
    "ctr": "center",
    "cntr": "center",
    "comm": "communications",
    "telecom": "telecommunications",
    "prop": "properties",
    "props": "properties",
    "dev": "development",
    "construc": "construction",
    "const": "construction",
    "infra": "infrastructure",
    "auto": "automobile",
    "agri": "agriculture",
}

ADDRESS_ABBREVS = {
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "rd": "road",
    "dr": "drive",
    "ct": "court",
    "cir": "circle",
    "ln": "lane",
    "pl": "place",
    "sq": "square",
    "pkwy": "parkway",
    "hwy": "highway",
    "fwy": "freeway",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    "rm": "room",
    "opp": "opposite",
    "nr": "near",
    "nr.": "near",
    "dist": "district",
    "sec": "sector",
    "mg": "mahatma gandhi",
    "marg": "road",
    "nagar": "nagar",
    "gali": "lane",
    "mohalla": "locality",
    "chowk": "square",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    # French
    "av": "avenue",
    "bd": "boulevard",
    "r": "rue",
    "pl": "place",
    "imp": "impasse",
    "chem": "chemin",
    "rte": "route",
    "all": "allee",
    "cdx": "cedex",
}


# ── Loading ──────────────────────────────────────────────────────────

def load_source(path: str) -> pd.DataFrame:
    """Load a TSV source file."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.fillna("")
    return df


def load_ground_truth(path: str) -> pd.DataFrame:
    """Load the ground truth TSV."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.fillna("")
    return df


# ── Normalisation helpers ────────────────────────────────────────────

def _unicode_normalize(text: str) -> str:
    """Normalize unicode to ASCII-compatible form."""
    text = unicodedata.normalize("NFKD", text)
    # Keep characters that are ASCII or common unicode letters
    return text


def _strip_punctuation(text: str) -> str:
    """Remove punctuation except hyphens within words."""
    # Replace & with 'and'
    text = text.replace("&", " and ")
    # Replace common punctuation with space
    text = re.sub(r"[^\w\s-]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _expand_abbreviations(text: str, abbrev_map: dict) -> str:
    """Expand abbreviations using a word-boundary aware approach."""
    tokens = text.split()
    expanded = []
    for tok in tokens:
        clean_tok = tok.strip(".,;:!?()-")
        if clean_tok.lower() in abbrev_map:
            expanded.append(abbrev_map[clean_tok.lower()])
        else:
            expanded.append(tok)
    return " ".join(expanded)


def normalize_name(name: str) -> str:
    """Normalize a business name for comparison."""
    if not name or pd.isna(name):
        return ""
    text = str(name).lower().strip()
    text = _unicode_normalize(text)
    text = _strip_punctuation(text)
    text = _expand_abbreviations(text, NAME_ABBREVS)
    # Remove common legal suffixes that don't help matching
    # (already expanded, so remove the expanded forms too)
    legal_suffixes = [
        "limited liability company", "limited liability partnership",
        "incorporated", "corporation", "limited", "private limited",
        "company", "private",
    ]
    for suffix in legal_suffixes:
        if text.endswith(" " + suffix):
            text = text[: -(len(suffix) + 1)].strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_address(address: str) -> str:
    """Normalize a business address for comparison."""
    if not address or pd.isna(address):
        return ""
    text = str(address).lower().strip()
    text = _unicode_normalize(text)
    text = _strip_punctuation(text)
    text = _expand_abbreviations(text, ADDRESS_ABBREVS)
    # Remove landmarks / noise phrases
    noise_phrases = [
        r"near\s+\w+\s+atm", r"near\s+\w+\s+bank",
        r"opposite\s+to\s+", r"next\s+to\s+",
        r"behind\s+", r"in\s+front\s+of\s+",
        r"beside\s+",
    ]
    for phrase in noise_phrases:
        text = re.sub(phrase, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_country(country: str) -> str:
    """Normalize country string."""
    if not country or pd.isna(country):
        return ""
    text = str(country).lower().strip()
    # Common variations
    mapping = {
        "us": "us", "usa": "us", "united states": "us",
        "united states of america": "us", "u.s.": "us", "u.s.a.": "us",
        "india": "india", "in": "india",
        "france": "france", "fr": "france",
    }
    return mapping.get(text, text)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized columns to a source dataframe."""
    df = df.copy()
    df["name_norm"] = df["business_name"].apply(normalize_name)
    df["addr_norm"] = df["business_address"].apply(normalize_address)
    df["country_norm"] = df["country"].apply(normalize_country)
    # Combined text for TF-IDF blocking
    df["combined_text"] = df["name_norm"] + " " + df["addr_norm"]
    return df


def parse_ground_truth(gt_df: pd.DataFrame) -> dict:
    """
    Parse ground truth into a dict: source1_entity_id -> set of matched ids.
    """
    gt = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        if matched and str(matched).strip():
            gt[s1_id] = set(str(matched).split(","))
        else:
            gt[s1_id] = set()
    return gt
