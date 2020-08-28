import nltk 
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import random
import math

data = pd.read_csv("blkjckhands.csv")

class Card:
    def __init__(self, suit, rank, value=0):
        self.suit = suit
        self.rank = rank
        self.value = value

    def show(self):
        print("{} of {} ({})".format(self.rank, self.suit, self.value))


class Deck:
    def __init__(self):
       self.cards = []
       self.build()
       
    def build(self):
        suits = ['Clubs', 'Spades', 'Diamonds', 'Hearts']
        ranks = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        index = 0
        for s in suits:
            for r in ranks:
                self.cards.append(Card(s,r, values[index%len(values)]))
                index += 1

    def shuffle(self):
        for i in range(len(self.cards)-1, 0, -1):
            r = random.randint(0, i)
            self.cards[i], self.cards[r] = self.cards[r], self.cards[i]

    def show(self):
        for c in self.cards:
            c.show()


deck = Deck()
deck.shuffle()
deck.show()
