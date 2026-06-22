import random
from zodiac_data import ZODIAC_DATA

def get_birth_sign(month, day):

    for sign_key, sign_data in ZODIAC_DATA.items():

        start_month, start_day = sign_data["date_range"]["start"]
        end_month, end_day = sign_data["date_range"]["end"]

    # Zodiac trong cùng năm
        if start_month <= end_month:

            if (
                 (month == start_month and day >= start_day)
                 or
                (month == end_month and day <= end_day)
                or
                (month > start_month and month < end_month)
            ):
                 return sign_key

    # Capricorn qua năm mới
        else:

            if (
                (month == start_month and day >= start_day)
                or
                (month == end_month and day <= end_day)
                or
                (month > start_month)
                or
                (month < end_month)
            ):
                 return sign_key

    return None


def generate_daily_advice(sign):


    data = ZODIAC_DATA[sign]

    return {
        "lucky_color": random.choice(data["lucky_colors"]),
        "recommended_food": random.choice(data["foods"]),
        "activity": random.choice(data["activities"]),
        "daily_message": random.choice(data["daily_messages"]),
    }


def get_zodiac_result(
    detected_sign,
    birth_month,
    birth_day,
    ):


    detected_sign = detected_sign.lower()

    SIGN_MAPPING = {
        "scorpius": "scorpio",
        "capricornus": "capricorn"
    }

    detected_sign = SIGN_MAPPING.get(
        detected_sign,
        detected_sign
    )

    if detected_sign not in ZODIAC_DATA:
        return None

    birth_sign = get_birth_sign(
        birth_month,
        birth_day,
    )

    if birth_sign is None:
        return None

    detected_data = ZODIAC_DATA[
        detected_sign
    ]

    advice = generate_daily_advice(
        birth_sign
    )

    result = {

    # Basic Info
    "detected_sign":
        detected_data["name"],

    "birth_sign":
        ZODIAC_DATA[birth_sign]["name"],

    "element":
        detected_data["element"],

    "traits":
        detected_data["traits"],

    # Astronomy
    "astronomy_fact":
        detected_data.get(
            "astronomy_fact",
            ""
        ),

    "brightest_star":
        detected_data.get(
            "brightest_star",
            ""
        ),

    "story":
        detected_data.get(
            "story",
            ""
        ),

    "mythology":
        detected_data.get(
            "mythology",
            ""
        ),

    "fun_fact":
        detected_data.get(
            "fun_fact",
            ""
        ),

    "seen_in_sky_msg":
        detected_data.get(
            "seen_in_sky_msg",
            ""
        ),

    # Personality
    "strengths":
        detected_data.get(
            "strengths",
            []
        ),

    "weaknesses":
        detected_data.get(
            "weaknesses",
            []
        ),

    "compatible_with":
        detected_data.get(
            "compatible_with",
            []
        ),

    # Daily Advice
    "lucky_color":
        advice["lucky_color"],

    "recommended_food":
        advice["recommended_food"],

    "activity":
        advice["activity"],

    "daily_message":
        advice["daily_message"],

    # Match Status
    "is_birth_match":
        (
            detected_sign
            ==
            birth_sign
        ),

    "birth_match_msg":
        detected_data.get(
            "birth_match_msg",
            ""
        ),

    "birth_mismatch_msg":
        detected_data.get(
            "birth_mismatch_msg",
            ""
        ),
}

    return result

