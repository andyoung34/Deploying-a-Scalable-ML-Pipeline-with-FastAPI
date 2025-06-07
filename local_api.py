import requests
url = "http://127.0.0.1:8000"

# === GET request ===
response = requests.get(f"{url}/")
print("GET / response:")
print(response.status_code)
print(response.json())

# === POST request ===
sample = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

response = requests.post(f"{url}/data/", json=sample)
print("\nPOST /data/ response:")
print(response.status_code)
print(response.json())
