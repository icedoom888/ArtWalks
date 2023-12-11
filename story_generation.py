import os
import argparse
import datetime
import random
import shutil
import nltk
import torch
from tqdm import tqdm
import json
from nltk.corpus import words
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
from utils import list_jpg_files_in_subfolders

# Initialize the WordNet lemmatizer
wnl = WordNetLemmatizer()
word_list = words.words()

# Define a function to categorize words as nouns or adjectives
def is_noun(word):
    synsets = nltk.corpus.wordnet.synsets(word)
    return any(synset.pos() == 'n' for synset in synsets)

def is_adjective(word):
    synsets = nltk.corpus.wordnet.synsets(word)
    return any(synset.pos() == 'a' for synset in synsets)

def generate_random_words():
    # Filter words to get nouns and adjectives
    nouns = [word for word in word_list if is_noun(word)]
    adjectives = [word for word in word_list if is_adjective(word)]

    # Generate one random adjective and one random noun
    random_adjective = random.choice(adjectives)
    random_noun = random.choice(nouns)

    # Combine the words into a single string
    result = f"{random_adjective}_{random_noun}"

    return result


def add_images_to_story(input_path, img_num, folder_name, move_originals, loop):
    # List all jpgs in input folder
    all_images = list_jpg_files_in_subfolders(input_path)
    random_images = random.sample(all_images, k=img_num)

    # check if there are already images
    num_existing_imgs = len([f for f in os.listdir(folder_name) if f.endswith('.jpg')])
    if num_existing_imgs != 0 and loop:
        os.remove(os.path.join(folder_name ,"%02d.jpg"%(num_existing_imgs-1)))
        num_existing_imgs -= 1

    for i, img_path in tqdm(enumerate(random_images)):
        if move_originals:
            # for loop
            if i == 0:
                shutil.copyfile(img_path, os.path.join(folder_name ,"%02d.png"%(i+num_existing_imgs)))
            else:
                shutil.copyfile(img_path, os.path.join(folder_name ,"%02d.png"%(i+num_existing_imgs)))


        else:
            shutil.copyfile(img_path, os.path.join(folder_name ,"%02d.png"%(i+num_existing_imgs)))
    
    # if loop is active, copy first image at the end as well
    if loop:
        if move_originals:
            shutil.move(random_images[0], os.path.join(folder_name ,"%02d.png"%(i+num_existing_imgs+1)))
        else:
            shutil.copyfile(random_images[0], os.path.join(folder_name ,"%02d.png"%(i+num_existing_imgs+1)))

def random_story_generator(input_path, img_num, output_path='stories', move_originals=False, loop=False):

    # Get random name
    random_name = generate_random_words()

    # Define the folder name based on the formatted datetime
    folder_name = os.path.join(output_path, random_name)
    os.makedirs(folder_name)

    # Add images to story
    add_images_to_story(input_path, img_num, folder_name, move_originals, loop)

    return folder_name


def generate_images_to_story(content_prompts, style_prompts, folder_name, h=1280, w=720):
    from kandinsky2 import get_kandinsky2
    from diffusion_models.kandinsky import prompt2image

    model = get_kandinsky2('cuda', task_type='text2img', model_version='2.2', use_flash_attention=False)

    for idx, (content_prompt, style_prompt) in enumerate(zip(content_prompts, style_prompts)):
        prompt = content_prompt + ' ' + style_prompt
        image = prompt2image(model, "", prompt, h=h, w=w, num_imgs=1, decoder_guidance_scale=5)[0]
        image.save(os.path.join(folder_name, "%02d.png"%idx))

    return


def json_story_generator(json_file_path, output_path='stories'):

    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Check if the data is a list and not empty
    if isinstance(data, list) and len(data) > 0:
        # Get the last element
        story = data[-1]

        # Remove the last element from the list
        data = data[:-1]

        # Write the updated data back to the JSON file
        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4)

    else:
        print('The JSON file is empty or not in the expected format.')
        exit()
    
    random_name = story["title"]
    description = story["story"]
    content_prompts = story["content_prompts"]
    style_prompts = story["style_prompts"]

    # Get random name
    random_name = random_name.replace(' ', '_')

    print('New story name: ', random_name)

    # Define the folder name based on the formatted datetime
    folder_name = os.path.join(output_path, random_name)
    os.makedirs(folder_name, exist_ok=True)

    # Save description
    with open(os.path.join(folder_name, 'story.txt'), 'w') as f:
        f.write(description)

    # Add images to story
    generate_images_to_story(content_prompts, style_prompts, folder_name)
    torch.cuda.empty_cache()
    return folder_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to folder with image folders", default='inputdata', type=str)
    parser.add_argument("--img_num", help="Number of images to sample", type=int, default=10)
    args = parser.parse_args()
    random_story_generator(args.input_path, args.img_num)