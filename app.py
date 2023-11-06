import random
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2

image_path='image3.png'

def calculate_blurriness(image):
    diff = np.abs(image - cv2.blur(image, (3, 3)))
    return np.mean(diff)

def mutate(image, mutation_rate, original_image):
    image = original_image.copy()
    if random.random() < mutation_rate:
        for channel in range(3):  # Apply blur to each channel separately
            image[:, :, channel] = cv2.blur(image[:, :, channel], (3, 3))
    return image

def crossover(parent1, parent2):
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)
    assert parent1.shape == parent2.shape, "Parents must have the same shape"
    child = cv2.addWeighted(parent1, 0.5, parent2, 0.5, 0)
    return child

def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations = 1)
    return dilated_image

POPULATION_SIZE = 3000
MUTATION_RATE = 0.001
GENERATIONS = 15

img = Image.open(image_path).convert('RGB')  # Ensure the image is loaded as RGB
img_array = np.array(img)
population = [img_array.copy() for _ in range(POPULATION_SIZE)]

for generation in range(GENERATIONS):
    print(generation)
    fitnesses = [calculate_blurriness(img) for img in population]
    new_population = []
    for _ in range(POPULATION_SIZE):
        parent1, parent2 = random.choices(population, weights=fitnesses, k=2)
        child = crossover(parent1, parent2)
        child = mutate(child, MUTATION_RATE, img_array)
        new_population.append(child)
    population = new_population

best_img = max(population, key=calculate_blurriness)
best_img = Image.fromarray(best_img)
best_img = best_img.convert("RGBA")
data = best_img.getdata()

# Convert PIL Image to numpy array before dilation
best_img_np = np.array(best_img)
best_img_np = dilate(best_img_np)

# Convert numpy array back to PIL Image
best_img = Image.fromarray(best_img_np)

# Save the output image
best_img.save('output.png', "PNG")

new_data = []
for item in data:
    if item[0] < 3 and item[1] < 3 and item[2] < 3:
        new_data.append((255, 255, 255, 0))
    else:
        new_data.append(item)

best_img.putdata(new_data)

# Ensure both images are in "RGBA" mode and have the same size
img = img.convert("RGBA").resize(best_img.size)
best_img = best_img.convert("RGBA")

# Overlay the output image over the original image
result_img = Image.alpha_composite(img, best_img)

# Save the result image
result_img.save('result.png', "PNG")