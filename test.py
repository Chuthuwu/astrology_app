from utils import get_birth_sign
from utils import get_zodiac_result

print("Birth Sign Test:")
print(get_birth_sign(10, 25))

print("\nZodiac Result Test:")

result = get_zodiac_result(
    "scorpio",
    10,
    25
)

print(result)

print("\nFormatted Output:\n")

print(
    f"""
Detected Constellation:
{result['detected_sign']}

Your Zodiac Sign:
{result['birth_sign']}

Traits:
{', '.join(result['traits'])}

Lucky Color:
{result['lucky_color']}

Recommended Food:
{result['recommended_food']}

Suggested Activity:
{result['activity']}

Daily Horoscope:
{result['daily_message']}
"""
)
