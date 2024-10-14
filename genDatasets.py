import pandas as pd
import random

# Set a specific random seed to create deterministic transactions
random.seed(0)

# Function to generate and save datasets
def generate_store_dataset(store_name, items):
    # Create an empty list to store transactions
    transactions = []

    # Generate 20 transactions with random items
    for i in range(1, 21):
        num_items = random.randint(1, 10)  # Number of items in each transaction
        transaction = random.sample(items, num_items)
        transaction_str = ', '.join(transaction)
        transactions.append([f"Trans{i}", transaction_str])

    # Create DataFrames
    df_items = pd.DataFrame(items, columns=['Item Name'])
    df_transactions = pd.DataFrame(transactions, columns=['Transaction ID', 'Transaction Items'])

    # Save the DataFrames to CSV files
    df_items.to_csv(f"{store_name}_items.csv", index=False)
    df_transactions.to_csv(f"{store_name}_transactions.csv", index=False)

    print(f"CSV files have been created for {store_name}: {store_name}_items.csv and {store_name}_transactions.csv")

# List of stores and their items
stores = {
    "Grocery Store": ['Milk', 'Bread', 'Eggs', 'Cheese', 'Apples', 'Bananas', 'Chicken', 'Rice', 'Pasta', 'Tomatoes'],
    "Electronics Store": ['Laptop', 'Smartphone', 'Headphones', 'Smartwatch', 'Tablet', 'Portable Charger', 'Camera', 'Laptop Bag', 'Monitor', 'Keyboard'],
    "Clothing Store": ['Jeans', 'T-Shirt', 'Dress', 'Jacket', 'Scarf', 'Hat', 'Sneakers', 'Socks', 'Belt', 'Sweater'],
    "Sports Equipment Store": ['Football', 'Basketball', 'Tennis Racket', 'Baseball Glove', 'Golf Balls', 'Yoga Mat', 'Running Shoes', 'Swim Goggles', 'Fitness Tracker', 'Water Bottle'],
    "Bookstore": ['Fiction Novel', 'Science Textbook', 'History Book', 'Biography', 'Poetry Collection', 'Mystery Novel', 'Science Fiction Novel', 'Cookbook', 'Art Book', 'Children\'s Book']
}

# Generate datasets for each store
for store, items in stores.items():
    generate_store_dataset(store, items)
