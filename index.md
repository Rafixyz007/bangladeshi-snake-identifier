---
title: Home
layout: page
---

description: A guide to 17 types of snakes found in Bangladesh and Southeast Asia
---

# 🐍 Bangladeshi Snake Identifier

This project helps identify 17 snakes found in Bangladesh and Southeast Asia.  
Upload a snake photo to predict its type, or browse the slideshow in the background.

---

<div id="slideshow">
  <div class="overlay">
    <h2>Bangladeshi Snake Recognizer</h2>
    <input id="photo" type="file" />
    <div id="results"></div>
  </div>
</div>

---

## Snake Names (English → বাংলা)

| English Name               | Bangla Name             |
|-----------------------------|------------------------|
| Banded Krait                | ব্যান্ডেড ক্রাইট        |
| Beaked Sea Snake            | সমুদ্র সাপ             |
| Bronzeback Tree Snake       | গাছে ওঠা সাপ          |
| Checkered Keelback          | ধামন                   |
| Common Krait                | কালাই সাপ              |
| Common Kukri Snake          | কুক্রি সাপ             |
| Common Wolf Snake           | নেকড়ে সাপ             |
| Green Pit Viper             | সবুজ পিট ভাইপার        |
| Green Trinket Snake         | সবুজ সাপ               |
| Indian Rock Python          | আজগর                   |
| King Cobra                  | রাজ গোখরা             |
| Monocled Cobra              | চশমা গোখরা            |
| Rat Snake                   | দাড়াশ                  |
| Russell's Viper             | চন্দ্রবোড়া            |
| Saw-scaled Viper            | একিস ভাইপার           |
| Spectacled Cobra            | গোখরা                  |
| Yellow-lipped Sea Krait     | সমুদ্র ক্রাইট          |

---

## About This Project
This project aims to help identify snakes commonly found in Bangladesh.  
It provides both English and Bangla names for easier understanding and awareness. 🐍  

---

<style>
  /* Full-page slideshow */
  #slideshow {
    position: fixed;
    top: 0; left: 0;
    width: 100%;
    height: 100vh;
    overflow: hidden;
    z-index: -1;
  }

  #slideshow img {
    width: 100%;
    height: 100vh;
    object-fit: cover;
    position: absolute;
    top: 0; left: 0;
    opacity: 0;
    transition: opacity 1s ease-in-out;
  }

  #slideshow img.active {
    opacity: 1;
  }

  /* Overlay (prediction box) */
  .overlay {
    position: relative;
    z-index: 10;
    background: rgba(255,255,255,0.8);
    padding: 20px;
    border-radius: 12px;
    width: 90%;
    max-width: 600px;
    margin: 40px auto;
    text-align: center;
  }

  #results img {
    max-width: 100%;
    border-radius: 10px;
    margin-top: 10px;
  }

  #results p {
    font-weight: bold;
    font-size: 1.2em;
  }
</style>

<script type="module">
  import { client } from "https://cdn.jsdelivr.net/npm/@gradio/client@0.1.4/dist/index.min.js";

  // Background slideshow
  const images = [
    "./assets/images/snake_01.jfif",
    "./assets/images/snake_02.jfif",
    "./assets/images/snake_03.jfif"
  ];
  const slideshow = document.getElementById("slideshow");

  images.forEach((src, i) => {
    const img = document.createElement("img");
    img.src = src;
    if (i === 0) img.classList.add("active");
    slideshow.appendChild(img);
  });

  const slides = document.querySelectorAll("#slideshow img");
  let index = 0;
  setInterval(() => {
    slides[index].classList.remove("active");
    index = (index + 1) % slides.length;
    slides[index].classList.add("active");
  }, 7000); // 7 seconds per slide

  // Prediction logic
  const photo = document.getElementById("photo");
  const results = document.getElementById("results");

  async function loaded(reader) {
      const app = await client("https://rafix007-bangladeshi-snake-recognizer-02.hf.space/");
      const response = await app.predict("/predict", [reader.result]);
      const label = response?.data?.[0]?.label || response?.output?.[0] || "No label returned";

      results.innerHTML = `<br/><img src="${reader.result}" width="400"><p>${label}</p>`;
  }

  function read() {
      const reader = new FileReader();
      reader.addEventListener('load', () => loaded(reader));
      reader.readAsDataURL(photo.files[0]);
  }

  photo.addEventListener('input', read);
</script>
