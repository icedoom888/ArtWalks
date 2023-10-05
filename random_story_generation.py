import os
import argparse
import datetime
import random
import shutil
import nltk
from tqdm import tqdm
import random
import nltk
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

def story_generator(input_path, img_num, output_path='stories', move_originals=False):

    # Get random name
    random_name = generate_random_words()

    # Define the folder name based on the formatted datetime
    folder_name = os.path.join(output_path, random_name)
    os.makedirs(folder_name)

    # List all jpgs in input folder
    all_images = list_jpg_files_in_subfolders(input_path)
    random_images = random.sample(all_images, k=img_num)

    for i, img_path in tqdm(enumerate(random_images)):
        if move_originals:
            shutil.move(img_path, os.path.join(folder_name ,"%02d.jpg"%i))
        else:
            shutil.copyfile(img_path, os.path.join(folder_name ,"%02d.jpg"%i))

    return folder_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to folder with image folders", default='inputdata', type=str)
    parser.add_argument("--img_num", help="Number of images to sample", type=int, default=10)
    args = parser.parse_args()
    story_generator(args.input_path, args.img_num)