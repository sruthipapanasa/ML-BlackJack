import os
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import random


print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# column order in CSV file
column_names = ['card1', 'card2', 'card3', 'card4', 'card5', 'cardsum', 'dcard1', 'dcard2', 'dcard3', 'dcard4', 'dcard5', 'dcardsum',  'winloss']
feature_names = column_names[:-1]
label_name = column_names[-1]
print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

blackjackFile = "blkjckhands.csv"
modified_blackjack = 'modified_blackjack.csv'
with open(blackjackFile, 'r') as csvfile1:
    with open(modified_blackjack, 'w') as csvfile:
        pointreader = csv.reader(csvfile1)
        for row in pointreader:
            if row[15] == 'Win':
                row[15] = 1
            else:
                row[15] = 0
            temp = row[2:14]
            temp.append(row[15])
            row = temp
            pointwriter = csv.writer(csvfile)
            pointwriter.writerow(row)

#arrays
batch_size = 1000
train_dataset = tf.data.experimental.make_csv_dataset(
    modified_blackjack,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)
features = next(iter(train_dataset))
print(features)

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels
train_dataset = train_dataset.map(pack_features_vector)
features, labels = next(iter(train_dataset))
print(features[:5])

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(5,12)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(2)
])

predictions = model(features)
predictions[:5]
tf.nn.softmax(predictions[:5])

print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))



loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)
  return loss_object(y_true=y, y_pred=y_)

l = loss(model, features, labels, training=False)
print("Loss test: {}".format(l))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels, training=True).numpy()))

## Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    # training=True is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    epoch_accuracy.update_state(y, model(x, training=True))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 25 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

#graph
'''
plt.scatter(features['card1'],
            var,
            c=labels,
            cmap='viridis')

plt.xlabel("Card 1")
plt.ylabel("win loss")
plt.show()




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

#match parameters given to card in deck and returns card
def card_updater(deck, card_rank, card_suit):
    for card in deck.cards:
        if card.rank == card_rank:
            if card.suit == card_suit:
                return card

#hi-lo counter used in blackjack games
def counter(played_cards, count = 0):
    for card in played_cards.cards:
        if 2 <= card.value <= 6:
            count +=1
        elif 7 <= card.value <= 9:
            pass
        else:
            count -= 1
    return count

def prediction (count, player_hand):
    #print sum of cards
    print("\n")
    sum = 0
    for card in player_hand.cards:
        sum += card.value
    print("The sum of your cards are:", sum)

    print("The current hi-lo count is:", str(count))
    if sum <=12:
        print('You should draw. Value of hand is low.')
    else:
        if count > 0:
            print("You should not draw. Count is too high.")
        elif count == 0:
            print("Do whatever, bro.")
        else:
            print("Count is low enough. Feel free to draw")

def blackjack_setup(played_cards, player_hand):
   #card 1 information
    card1_rank = input("Enter your first card's rank.\n")
    card1_suit = input("Enter your first card's suit. \n")
    card1 = card_updater(deck, card1_rank, card1_suit)
    if card1 not in deck.cards:
        print("Your card cannot be found in the deck. Please try again.")
        blackjack_setup(played_cards, player_hand)
    else:
        played_cards.cards.append(card1)
        player_hand.cards.append(card1)
        deck.cards.remove(card1)
    
    #card 2 information
    card2_rank = input("Enter your second card's rank.\n")
    card2_suit = input("Enter your second card's suit. \n")
    card2 = card_updater(deck, card2_rank, card2_suit)
    while card2 not in deck.cards:
        print("Your card cannot be found in the deck. Please try again.\n")
        card2_rank = input("Enter your second card's rank.\n")
        card2_suit = input("Enter your second card's suit. \n")
        card2 = card_updater(deck, card2_rank, card2_suit)
    else:
        played_cards.cards.append(card2)
        player_hand.cards.append(card2)
        deck.cards.remove(card2)

    #dealer information
    dcard_rank = input("Enter the dealer card rank.\n")
    dcard_suit = input("Enter the dealer card suit. \n")
    dealer = card_updater(deck, dcard_rank, dcard_suit)
    while dealer not in deck.cards:
        print("Your card cannot be found in the deck. Please try again.\n")
        dcard_rank = input("Enter the dealer card rank.\n")
        dcard_suit = input("Enter the dealer card suit. \n")
        dealer = card_updater(deck, dcard_rank, dcard_suit)
    else:
        played_cards.cards.append(dealer)
        player_hand.cards.append(dealer)
        deck.cards.remove(dealer)

    count2 = counter(played_cards)
    prediction (count2, player_hand)

    answer = input('\nAre you going to draw again? Enter Y or N.\n')   

    while answer == 'Y':
        newcard_rank = input("Enter the new card rank.\n")
        newcard_suit = input("Enter the new card suit. \n")
        newcard = card_updater(deck, newcard_rank, newcard_suit)
        while newcard not in deck.cards:
            print("Your card cannot be found in the deck. Please try again.\n")
            newcard_rank = input("Enter the new card rank.\n")
            newcard_suit = input("Enter the new card suit. \n")
            newcard = card_updater(deck, newcard_rank, newcard_suit)
        else:
            played_cards.cards.append(newcard)
            player_hand.cards.append(newcard)
            deck.cards.remove(newcard)
        counter3 = counter(played_cards)
        prediction(counter3, player_hand)
        answer = input('\nAre you going to draw again? Enter Y or N.\n')
    
    if answer == 'N':
        print("\nYou have chosen to end the round.")
        answer2 = input("Are you going to play another round? Enter Y or N.\n")
        if answer2 == 'Y':
            player_hand.cards = []
            blackjack_setup(played_cards, player_hand)
        else:
            print("\n\nOk, thanks for playing.")


#making a deck
deck = Deck()
deck.build()
deck.shuffle()

played_cards = Deck()
player_hand = Deck()
blackjack_setup(played_cards, player_hand)
print("played cards:")
played_cards.show()
print("player hand")
player_hand.show()

'''