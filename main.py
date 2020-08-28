""""
import nltk 
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import math


data = pd.read_csv("blkjckhands.csv") """

import random

#setting up card
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
       
    def build(self):
        suits = ['Clubs', 'Spades', 'Diamonds', 'Hearts']
        ranks = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']
        values = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
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

#match parameters given to card in deck
def card_updater(deck, card_rank, card_suit):
    for card in deck.cards:
        if card.rank == card_rank:
            if card.suit == card_suit:
                return card

def counter(played_cards, count = 0):
    for card in played_cards.cards:
        if 2 <= card.value <= 6:
            count +=1
        elif 7 <= card.value <= 9:
            pass
        else:
            count -= 1
    print("The current hi-lo count is:", str(count))
    if count > 0:
        print("You should not draw. Count is too high.")
    elif count == 0:
        print("Do whatever, bro.")
    else:
        print("Count is low enough. Feel free to draw")
    return count


def blackjack_setup(played_cards):

   #card 1 information
    card1_rank = input("Enter your first card's rank.\n")
    card1_suit = input("Enter your first card's suit. \n")
    
    #card 2 information
    card2_rank = input("Enter your second card's rank.\n")
    card2_suit = input("Enter your second card's suit. \n")

    #dealer information
    dcard_rank = input("Enter the dealer card rank.\n")
    dcard_suit = input("Enter the dealer card suit. \n")
    
    #remove cards that have already been dealt
    card1 = card_updater(deck, card1_rank, card1_suit)
    played_cards.cards.append(card1)
    deck.cards.remove(card1)
    
    card2 = card_updater(deck, card2_rank, card2_suit)
    played_cards.cards.append(card2)
    deck.cards.remove(card2)
    
    dealer = card_updater(deck, dcard_rank, dcard_suit)
    played_cards.cards.append(dealer)
    deck.cards.remove(dealer)
    
    #print sum of cards
    print("\n")
    print("The sum of your cards are:", card1.value + card2.value)

    count2 = counter(played_cards)

    answer = input('Are you going to draw again? Enter Y or N.\n')   
    
    if answer == 'N':
        print("You have chosen to end the round.")
        answer2 = input("Are you going to play another round? Enter Y or N.\n")
        if answer2 == 'Y':
            blackjack_setup()
        else:
            print("Ok, thanks for playing.")
    else:
        newcard_rank = input("Enter the new card rank.\n")
        newcard_suit = input("Enter the new card suit. \n")
        newcard = card_updater(deck, newcard_rank, newcard_suit)
        played_cards.cards.append(newcard)
        deck.cards.remove(newcard)
        counter(played_cards, count2)
    

#making a deck
deck = Deck()
deck.build()
deck.shuffle()

played_cards = Deck()
played_cards.show()
blackjack_setup(played_cards)
played_cards.show()

