from fastai.vision.all import *
import gradio as gr

# Snake names
snake_name = (
    'Banded Krait', 'Beaked Sea Snake', 'Bronzeback Tree Snake', 'Checkered Keelback',
    'Common Krait', 'Common Kukri Snake', 'Common Wolf Snake', 'Green Pit Viper',
    'Green Trinket Snake', 'Indian Rock Python', 'King Cobra', 'Monocled Cobra',
    'Rat Snake', "Russell's Viper", 'Saw-scaled Viper', 'Spectacled Cobra',
    'Yellow-lipped Sea Krait'
)

# Bangla names
english_to_bangla = {
    "Banded Krait": "ব্যান্ডেড ক্রাইট",
    "Beaked Sea Snake": "সমুদ্র সাপ",
    "Bronzeback Tree Snake": "গাছে ওঠা সাপ",
    "Checkered Keelback": "ধামন",
    "Common Krait": "কালাই সাপ",
    "Common Kukri Snake": "কুক্রি সাপ",
    "Common Wolf Snake": "নেকড়ে সাপ",
    "Green Pit Viper": "সবুজ পিট ভাইপার",
    "Green Trinket Snake": "সবুজ সাপ",
    "Indian Rock Python": "আজগর",
    "King Cobra": "রাজ গোখরা",
    "Monocled Cobra": "চশমা গোখরা",
    "Rat Snake": "দাড়াশ",
    "Russell's Viper": "চন্দ্রবোড়া",
    "Saw-scaled Viper": "একিস ভাইপার",
    "Spectacled Cobra": "গোখরা",
    "Yellow-lipped Sea Krait": "সমুদ্র ক্রাইট"
}

# Snake details (your detailed dictionary)
snake_details = {
    "Banded Krait": {
        "Found": "South and Southeast Asia (India, Bangladesh, Myanmar, Thailand)",
        "Venom Type": "Neurotoxic",
        "Danger": "Highly venomous; potentially fatal but generally shy"
    },
    "Beaked Sea Snake": {
        "Found": "Indian and Pacific Oceans, coastal waters of South Asia",
        "Venom Type": "Neurotoxic",
        "Danger": "Extremely venomous sea snake; can be deadly"
    },
    "Bronzeback Tree Snake": {
        "Found": "South and Southeast Asia, often in trees and bushes",
        "Venom Type": "Non-venomous",
        "Danger": "Harmless to humans"
    },
    "Checkered Keelback": {
        "Found": "South Asia, near rivers, ponds, and wetlands",
        "Venom Type": "Non-venomous",
        "Danger": "Harmless, sometimes mistaken for venomous snakes"
    },
    "Common Krait": {
        "Found": "Indian subcontinent (India, Bangladesh, Sri Lanka, Nepal)",
        "Venom Type": "Neurotoxic",
        "Danger": "Highly venomous; responsible for many snakebite deaths"
    },
    "Common Kukri Snake": {
        "Found": "South and Southeast Asia",
        "Venom Type": "Non-venomous",
        "Danger": "Harmless, but may bite defensively"
    },
    "Common Wolf Snake": {
        "Found": "South and Southeast Asia, often near human settlements",
        "Venom Type": "Non-venomous",
        "Danger": "Harmless, easily confused with kraits"
    },
    "Green Pit Viper": {
        "Found": "South and Southeast Asia, forests and plantations",
        "Venom Type": "Hemotoxic (sometimes with mild neurotoxic effects)",
        "Danger": "Venomous; bite is painful but rarely fatal"
    },
    "Green Trinket Snake": {
        "Found": "India, Bangladesh, Sri Lanka, Nepal",
        "Venom Type": "Non-venomous",
        "Danger": "Harmless, active in trees and bushes"
    },
    "Indian Rock Python": {
        "Found": "Indian subcontinent, grasslands, swamps, forests",
        "Venom Type": "Non-venomous (constrictor)",
        "Danger": "Large but harmless to humans; kills prey by constriction"
    },
    "King Cobra": {
        "Found": "India, Southeast Asia, forests and plantations",
        "Venom Type": "Neurotoxic",
        "Danger": "World’s longest venomous snake; highly dangerous"
    },
    "Monocled Cobra": {
        "Found": "South and Southeast Asia (especially Bangladesh, India, Thailand)",
        "Venom Type": "Neurotoxic",
        "Danger": "Highly venomous; responsible for many bites in Asia"
    },
    "Rat Snake": {
        "Found": "South and Southeast Asia, often near fields and villages",
        "Venom Type": "Non-venomous",
        "Danger": "Harmless, useful for controlling rodents"
    },
    "Russell's Viper": {
        "Found": "Indian subcontinent, grasslands, farmlands",
        "Venom Type": "Hemotoxic (with strong tissue-damaging effects)",
        "Danger": "Extremely dangerous; one of the 'Big Four' deadly snakes in India"
    },
    "Saw-scaled Viper": {
        "Found": "South Asia, Middle East, arid regions",
        "Venom Type": "Hemotoxic",
        "Danger": "Highly venomous; small but aggressive, responsible for many deaths"
    },
    "Spectacled Cobra": {
        "Found": "Indian subcontinent, fields, forests, villages",
        "Venom Type": "Neurotoxic",
        "Danger": "Highly venomous; iconic hooded cobra of India/Bangladesh"
    },
    "Yellow-lipped Sea Krait": {
        "Found": "Indian and Pacific Oceans, coastal Asia",
        "Venom Type": "Neurotoxic",
        "Danger": "Very venomous sea snake; rarely aggressive to humans"
    }
}

# Load model
model = load_learner("bangladeshi-snake-recognizer-v3.pkl")

# Predict function
def recognize_image(image):
    pred, idx, probs = model.predict(image)
    results = {}
    for i, prob in enumerate(probs):
        eng_name = snake_name[i]
        bangla_name = english_to_bangla[eng_name]
        details = snake_details[eng_name]
        display_text = (f"{eng_name} ({bangla_name})\n"
                        f"Found: {details['Found']}\n"
                        f"Venom Type: {details['Venom Type']}\n"
                        f"Danger: {details['Danger']}")
        results[display_text] = float(prob)
    return results

# Gradio interface
image_input = gr.Image(type="pil")
label_output = gr.Label(num_top_classes=5)

examples = [f"unknown_{i:02}.jpg" for i in range(1, 17)]

iface = gr.Interface(
    fn=recognize_image,
    inputs=image_input,
    outputs=label_output,
    examples=examples,
    live=False  # Only predict after Submit
)

iface.launch()

