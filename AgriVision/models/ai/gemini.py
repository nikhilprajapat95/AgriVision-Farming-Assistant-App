import google.generativeai as genai

genai.configure(api_key="AIzaSyCwONDeXZLAkh215zv4wV7GV0Y-T3ai5DY")

for m in genai.list_models():
    print(m.name, "->", m.supported_generation_methods)
