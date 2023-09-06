import random
import json
import torch
import sqlite3

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

def get_answer(tag):
    cursor.execute("SELECT price FROM drinks WHERE drink=?", (tag,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        return None

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    # Check if the user has asked for the price of an item
    if "price of" in sentence:
        # Get the item name from the sentence
        item = sentence.split("price of ")[1]
        # Query the database for the price of the item
        cursor.execute("SELECT price FROM drinks WHERE drink=?", (item,))
        result = cursor.fetchone()
        if result is None:
            # If the item is not found in the database, respond accordingly
            print(f"{bot_name}: I'm sorry, I don't know the price of {item}.")
        else:
            # If the item is found in the database, give the price
            price = result[0]
            print(f"{bot_name}: The price of {item} is {price}.")
    else:
        # If the user has not asked for the price of an item, respond with a generic message
        print(f"{bot_name}: I'm sorry, I don't understand.")


    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        response = get_answer(tag)
        if response:
            print(f"{bot_name}: {response}")
        else:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")

conn.close()