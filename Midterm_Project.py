#!/usr/bin/env python
# coding: utf-8

# ## Installing the necessary libraries

# In[1]:


# %pip install pandas
# %pip install mlxtend


# # Generating the Dataset

# In[2]:


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


# # Association Rule Mining Analysis
# 
# This script performs association rule mining using three different methods: a custom Brute Force approach, the Apriori algorithm, and the FP-Growth algorithm, comparing their performance on a selected dataset. First, we load the transaction data from a specified store.
# 

# In[3]:


# import libraries and load data

import pandas as pd
import itertools
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import time

def load_transactions(filename):
    """ Load and preprocess transactions from a CSV file. """
    df = pd.read_csv(filename)
    transactions = df['Transaction Items'].apply(lambda x: x.split(','))
    return transactions


# ## Brute Force Method
# 
# Implementing the brute force method to generate frequent itemsets. This method evaluates every possible combination of items to determine which sets meet the minimum support threshold.
# 

# In[4]:


# Brute Force Functino

def brute_force_frequent_itemsets(transactions, min_support):
    """ Efficiently generate frequent itemsets using the brute force method. """
    item_set = set(itertools.chain.from_iterable(transactions))
    item_list = list(item_set)
    frequent_itemsets = []
    num_transactions = len(transactions)
    transaction_sets = [set(transaction) for transaction in transactions]

    for r in range(1, len(item_list) + 1):
        for itemset in itertools.combinations(item_list, r):
            itemset_support = sum(1 for transaction in transaction_sets if set(itemset).issubset(transaction)) / num_transactions
            if itemset_support >= min_support:
                frequent_itemsets.append((itemset, itemset_support))

    return pd.DataFrame(frequent_itemsets, columns=["itemsets", "support"])


# ## Algorithm Execution
# 
# Here, we execute the Apriori and FP-Growth algorithms using the `mlxtend` library, and measure the execution time for each. We also include the Brute Force method executed previously.
# 

# In[5]:


def run_mlxtend_algorithm(transactions, min_support, algorithm):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    if algorithm == 'apriori':
        return apriori(df, min_support=min_support, use_colnames=True)
    else:
        return fpgrowth(df, min_support=min_support, use_colnames=True)

def generate_rules(frequent_itemsets, min_confidence):
    if not frequent_itemsets.empty:
        return association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return pd.DataFrame()

def generate_association_rules(frequent_itemsets, transactions, min_support, min_confidence):
    """Generate and print association rules from frequent itemsets."""
    rules = []
    transaction_list = list(map(set, transactions))
    total_transactions = len(transaction_list)

    for itemsets, support in frequent_itemsets.itertuples(index=False):
        # Get all non-empty subsets of the itemset
        for consequent in itertools.chain.from_iterable(itertools.combinations(itemsets, r) for r in range(1, len(itemsets))):
            antecedent = itemsets.difference(consequent)
            antecedent_support = get_support(antecedent, transaction_list) / total_transactions
            consequent_support = get_support(consequent, transaction_list) / total_transactions

            if antecedent_support > 0:  # Avoid division by zero
                confidence = support / antecedent_support
                if confidence >= min_confidence and support >= min_support:
                    rules.append((antecedent, consequent, support, confidence, consequent_support))

    # Print rules
    print("Generated Association Rules:")
    for rule in rules:
        print(f"Rule: {rule[0]} -> {rule[1]}, Support: {rule[2]}, Confidence: {rule[3]}, Lift: {rule[4]/rule[2]}")

def get_support(itemset, transaction_list):
    """Calculate support for an itemset in given list of transactions."""
    return sum(1 for transaction in transaction_list if itemset.issubset(transaction))


# ## Results Display
# 
# For each algorithm, we print the rules found, sorted by the lift metric to identify the most relevant associations.
# 

# In[8]:


def print_rules(rules):
    if not rules.empty:
        if(len(rules) > 20):
            print("Concating the rules to only 20..")
        rules_sorted = rules.sort_values(by='lift', ascending=False).head(20)
        for index, rule in rules_sorted.iterrows():
            print(f"Rule {index + 1}: {rule['antecedents']} -> {rule['consequents']}, "
                  f"Support: {rule['support']:.4f}, Confidence: {rule['confidence']:.4f}, Lift: {rule['lift']:.4f}")
    else:
        print("No rules generated.")

def main():
    store_options = ['Bookstore', 'Clothing Store', 'Electronics Store', 'Grocery Store', 'Sports Equipment Store']
    print("Available stores:")
    for i, option in enumerate(store_options, 1):
        print(f"{i}. {option}")

    store_index = int(input("Please select your store (1-5): ")) - 1
    transactions_file = f"{store_options[store_index]}_transactions.csv"
    transactions = load_transactions(transactions_file)

    min_support = float(input("Enter the minimum support (as a decimal, e.g., 0.01 for 1%): "))
    min_confidence = float(input("Enter the minimum confidence (as a decimal, e.g., 0.7 for 70%): "))

    # Running all algorithms
    # run_all_algorithms(transactions, min_support, min_confidence)
    start_time = time.time()
    bf_itemsets = brute_force_frequent_itemsets(transactions, min_support)
    bf_duration = time.time() - start_time
    print(f"Brute Force - Duration: {bf_duration:.2f}s, Itemsets Found: {len(bf_itemsets)}")
    if not bf_itemsets.empty:
        bf_rules = generate_rules(bf_itemsets, min_confidence)
        print_rules(bf_rules)
    else:
        print("No frequent itemsets found with brute force.")

    # Apriori
    start_time = time.time()
    ap_itemsets = run_mlxtend_algorithm(transactions, min_support, 'apriori')
    ap_duration = time.time() - start_time
    ap_rules = generate_rules(ap_itemsets, min_confidence)
    print(f"Apriori - Duration: {ap_duration:.2f}s, Rules Found: {len(ap_rules)}")
    print_rules(ap_rules)

    # FP-Growth
    start_time = time.time()
    fp_itemsets = run_mlxtend_algorithm(transactions, min_support, 'fpgrowth')
    fp_duration = time.time() - start_time
    fp_rules = generate_rules(fp_itemsets, min_confidence)
    print(f"FP-Growth - Duration: {fp_duration:.2f}s, Rules Found: {len(fp_rules)}")
    print_rules(fp_rules)

    print("Timing Performance:")
    print(f"Brute Force - Duration: {bf_duration:.2f}s")
    print(f"Apriori - Duration: {ap_duration:.2f}s")
    print(f"FP-Growth - Duration: {fp_duration:.2f}s")

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:




