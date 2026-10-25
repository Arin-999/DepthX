from src.fit_analyzer import analyze_fit

body = {
    "chest_cm": 96,
    "waist_cm": 80,
}

garment = {
    "chest_cm": 104,
    "waist_cm": 85,
}

result = analyze_fit(body, garment)

print("SAHAAI FIT ANALYSIS")
print("-" * 30)

for dimension, measurement in result["dimensions"].items():
    print(f"{dimension}:")
    print(f"  Body:    {measurement['body_cm']} cm")
    print(f"  Garment: {measurement['garment_cm']} cm")
    print(f"  Ease:    {measurement['ease_cm']} cm")
    print(f"  Fit:     {measurement['fit']}")

print("\nStatus:", result["status"])
print(result["note"])