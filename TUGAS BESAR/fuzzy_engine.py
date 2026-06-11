import numpy as np
import pandas as pd


# =========================
# Mapping kategorikal
# =========================

travel_mapping = {
    "Personal Travel": 0,
    "Business travel": 1
}

class_mapping = {
    "Eco": 0,
    "Eco Plus": 1,
    "Business": 2
}


# =========================
# Fungsi Keanggotaan
# =========================

def trimf(x, abc):
    a, b, c = abc

    if x < a or x > c:
        return 0.0

    if b == a and x == a:
        return 1.0

    if b == c and x == c:
        return 1.0

    if x <= b:
        if a != b:
            return (x - a) / (b - a)
        return 1.0

    if b != c:
        return (c - x) / (c - b)

    return 1.0


def zmf(x, a, b):
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a)


def smf(x, a, b):
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


def eval_mf(val, mf_spec):
    mf_type = mf_spec[0]

    if mf_type == "trimf":
        return trimf(val, mf_spec[1])

    if mf_type == "zmf":
        return zmf(val, mf_spec[1], mf_spec[2])

    if mf_type == "smf":
        return smf(val, mf_spec[1], mf_spec[2])

    raise ValueError(f"Membership function tidak dikenal: {mf_type}")


# =========================
# Variable Linguistic
# =========================

MF_SERVICE = {
    "buruk": [0, 0, 2.5],
    "sedang": [0, 2.5, 5],
    "baik": [2.5, 5, 5],
}

MF_CLASS = {
    "rendah": ("zmf", 0, 1),
    "menengah": ("trimf", [0, 1, 2]),
    "tinggi": ("smf", 1, 2),
}

MF_TRAVEL = {
    "personal": ("zmf", 0, 1),
    "bisnis": ("smf", 0, 1),
}

MF_OUTPUT = {
    "tidak_puas": [0, 0, 50],
    "netral": [0, 50, 100],
    "puas": [50, 100, 100],
}

UNIVERSE_OUTPUT = np.linspace(0, 100, 101)

SUGENO_OUTPUT = {
    "tidak_puas": 25,
    "netral": 50,
    "puas": 75
}


# =========================
# Rule Base
# =========================

RULES = [
    {"id": 1, "ant": [("ob", "baik"), ("wifi", "baik"), ("ent", "baik")], "con": "puas"},
    {"id": 2, "ant": [("ob", "baik"), ("cls", "tinggi")], "con": "puas"},
    {"id": 3, "ant": [("ob", "baik"), ("ent", "baik")], "con": "puas"},
    {"id": 4, "ant": [("wifi", "baik"), ("ent", "baik"), ("cls", "tinggi")], "con": "puas"},
    {"id": 5, "ant": [("ob", "baik"), ("wifi", "sedang"), ("ent", "baik")], "con": "puas"},
    {"id": 6, "ant": [("ob", "sedang"), ("wifi", "baik"), ("cls", "tinggi")], "con": "puas"},
    {"id": 7, "ant": [("trav", "bisnis"), ("ob", "baik"), ("wifi", "baik")], "con": "puas"},
    {"id": 8, "ant": [("trav", "bisnis"), ("cls", "tinggi"), ("wifi", "baik"), ("ent", "baik")], "con": "puas"},

    {"id": 9, "ant": [("ob", "sedang"), ("wifi", "sedang"), ("ent", "sedang")], "con": "netral"},
    {"id": 10, "ant": [("ob", "sedang"), ("cls", "menengah")], "con": "netral"},
    {"id": 11, "ant": [("trav", "bisnis"), ("cls", "tinggi"), ("ob", "sedang")], "con": "netral"},
    {"id": 12, "ant": [("ob", "sedang"), ("ent", "sedang"), ("cls", "rendah")], "con": "netral"},
    {"id": 13, "ant": [("ob", "sedang"), ("wifi", "sedang"), ("cls", "menengah"), ("trav", "bisnis")], "con": "netral"},

    {"id": 14, "ant": [("ob", "buruk"), ("wifi", "buruk"), ("ent", "buruk")], "con": "tidak_puas"},
    {"id": 15, "ant": [("ob", "buruk"), ("cls", "rendah")], "con": "tidak_puas"},
    {"id": 16, "ant": [("wifi", "buruk"), ("ent", "buruk")], "con": "tidak_puas"},
    {"id": 17, "ant": [("trav", "personal"), ("cls", "rendah"), ("ob", "buruk")], "con": "tidak_puas"},
    {"id": 18, "ant": [("trav", "personal"), ("ob", "buruk"), ("ent", "buruk")], "con": "tidak_puas"},
    {"id": 19, "ant": [("ob", "buruk"), ("wifi", "sedang"), ("cls", "rendah")], "con": "tidak_puas"},
    {"id": 20, "ant": [("trav", "personal"), ("cls", "rendah"), ("wifi", "buruk"), ("ent", "buruk")], "con": "tidak_puas"},
]


# =========================
# Fuzzifikasi, Inferensi, Defuzzifikasi
# =========================

def fuzzify(row):
    ob = row["Online boarding"]
    wifi = row["Inflight wifi service"]
    ent = row["Inflight entertainment"]
    cls = row["Class"]
    trav = row["Type of Travel"]

    return {
        "ob": {
            "buruk": trimf(ob, MF_SERVICE["buruk"]),
            "sedang": trimf(ob, MF_SERVICE["sedang"]),
            "baik": trimf(ob, MF_SERVICE["baik"]),
        },
        "wifi": {
            "buruk": trimf(wifi, MF_SERVICE["buruk"]),
            "sedang": trimf(wifi, MF_SERVICE["sedang"]),
            "baik": trimf(wifi, MF_SERVICE["baik"]),
        },
        "ent": {
            "buruk": trimf(ent, MF_SERVICE["buruk"]),
            "sedang": trimf(ent, MF_SERVICE["sedang"]),
            "baik": trimf(ent, MF_SERVICE["baik"]),
        },
        "cls": {
            "rendah": eval_mf(cls, MF_CLASS["rendah"]),
            "menengah": eval_mf(cls, MF_CLASS["menengah"]),
            "tinggi": eval_mf(cls, MF_CLASS["tinggi"]),
        },
        "trav": {
            "personal": eval_mf(trav, MF_TRAVEL["personal"]),
            "bisnis": eval_mf(trav, MF_TRAVEL["bisnis"]),
        },
    }


def inferensi(fuzzified):
    results = []

    for rule in RULES:
        strengths = [
            fuzzified[var][himpunan]
            for var, himpunan in rule["ant"]
        ]

        firing_strength = min(strengths) if strengths else 0.0
        results.append((rule["id"], firing_strength, rule["con"]))

    return results


def mamdani_defuzz(inference_results):
    aggregated = []

    for x in UNIVERSE_OUTPUT:
        max_membership = 0

        for _, firing_strength, conclusion in inference_results:
            output_membership = trimf(x, MF_OUTPUT[conclusion])
            clipped = min(firing_strength, output_membership)

            if clipped > max_membership:
                max_membership = clipped

        aggregated.append(max_membership)

    numerator = 0
    denominator = 0

    for i in range(len(UNIVERSE_OUTPUT)):
        numerator += UNIVERSE_OUTPUT[i] * aggregated[i]
        denominator += aggregated[i]

    if denominator == 0:
        return 50

    return numerator / denominator


def sugeno_defuzz(inference_results):
    numerator = 0
    denominator = 0

    for _, firing_strength, conclusion in inference_results:
        numerator += firing_strength * SUGENO_OUTPUT[conclusion]
        denominator += firing_strength

    if denominator == 0:
        return 50

    return numerator / denominator


def calculate_fuzzy_scores(
    type_of_travel,
    flight_class,
    inflight_wifi_service,
    online_boarding,
    inflight_entertainment
):
    """
    Fungsi utama untuk Streamlit.
    Input masih dalam bentuk pilihan user.
    Output berupa mamdani_score dan sugeno_score.
    """

    encoded_row = {
        "Type of Travel": travel_mapping[type_of_travel],
        "Class": class_mapping[flight_class],
        "Inflight wifi service": inflight_wifi_service,
        "Online boarding": online_boarding,
        "Inflight entertainment": inflight_entertainment
    }

    fuzzified = fuzzify(encoded_row)
    inference_result = inferensi(fuzzified)

    mamdani_score = mamdani_defuzz(inference_result)
    sugeno_score = sugeno_defuzz(inference_result)

    return mamdani_score, sugeno_score, inference_result