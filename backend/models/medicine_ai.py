import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# PROJECT PATHS
# ============================================================

# medicine_ai.py
#      ↓
# models/
#      ↓
# backend/
#      ↓
# MedVisionAi/
#      ↓
# .env

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(ENV_FILE)


# ============================================================
# GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        f"GEMINI_API_KEY not found in: {ENV_FILE}"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)

print("Gemini API configured successfully.")


# ============================================================
# MEDICINE ANALYZER
# ============================================================

def analyze_medicine(image_path):

    image_path = Path(image_path)

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    # --------------------------------------------------------
    # Check image extension
    # --------------------------------------------------------

    extension = image_path.suffix.lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }

    if extension not in mime_types:

        raise ValueError(
            "Supported formats: JPG, JPEG, PNG, WEBP"
        )

    mime_type = mime_types[extension]


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    with open(image_path, "rb") as file:

        image_bytes = file.read()


    # ========================================================
    # STEP 1 — IDENTIFY MEDICINE
    # ========================================================

    identification_prompt = """
Analyze this medicine package image.

Extract ONLY information that is clearly visible.

Return:

Medicine name:
Active ingredient:
Strength:
Manufacturer:

Rules:

- Do not guess.
- Do not invent information.
- If something is not visible, write:
  "Not clearly visible."
- Do not provide dosage.
- Do not provide medical advice.
- Keep the response concise.
"""


    response = client.models.generate_content(

        model="gemini-3.5-flash-lite",

        contents=[

            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            ),

            identification_prompt

        ]
    )


    medicine_info = response.text.strip()


    # ========================================================
    # STEP 2 — EXTRACT MEDICINE NAME
    # ========================================================

    medicine_name = "Unknown"


    for line in medicine_info.splitlines():

        if line.lower().startswith("medicine name:"):

            medicine_name = (
                line.split(":", 1)[1]
                .strip()
            )

            break


    # ========================================================
    # STEP 3 — ULTRA LIGHT USE
    # ========================================================

    if medicine_name.lower() in [
        "unknown",
        "not clearly visible",
        ""
    ]:

        medicine_use = (
            "Use could not be determined."
        )

    else:

        use_prompt = f"""
Medicine name: {medicine_name}

Give ONE very short sentence describing
the common general use of this medicine.

Rules:

- Maximum 15 words.
- Do not provide dosage.
- Do not provide personalized medical advice.
- Do not mention side effects.
- Do not guess.
- If the medicine is uncertain, respond:

"Use could not be determined."
"""


        use_response = client.models.generate_content(

            model="gemini-3.5-flash-lite",

            contents=use_prompt
        )


        medicine_use = (
            use_response.text.strip()
        )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "medicine_name": medicine_name,

        "medicine_info": medicine_info,

        "use": medicine_use

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Test image
    # --------------------------------------------------------

    image_path = (
        PROJECT_ROOT
        / "backend"
        / "test_imgs"
        / "2.jpeg"
        
    )


    print("\n")
    print("=" * 60)
    print("MEDVISION MEDICINE AI")
    print("=" * 60)

    print("\nImage:")
    print(image_path)


    try:

        result = analyze_medicine(
            image_path
        )


        print("\n")
        print("-" * 60)

        print("MEDICINE INFORMATION")
        print("-" * 60)

        print(
            result["medicine_info"]
        )


        print("\n")
        print("-" * 60)

        print("COMMON USE")
        print("-" * 60)

        print(
            result["use"]
        )


        print("\n")
        print("=" * 60)


    except Exception as e:

        print("\nERROR:")
        print(e)