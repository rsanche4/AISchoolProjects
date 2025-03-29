import pandas as pd
import numpy as np
import math

# Manually creating the DataFrame
data = {
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
    "Target": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
}

df = pd.DataFrame(data)


# Entropy Formula
def entropy(set, columnchosen):
    total = set[columnchosen].shape[0]
    uniquesvalues = set[columnchosen].unique()
    freq_list = []
    for t in uniquesvalues:
        howmany = set[set[columnchosen]==t].shape[0]
        freq_list.append(howmany)
    
    sum_count = 0
    for vals in freq_list:
        p = vals/total
        log2base = math.log(p) / math.log(2)
        sum_count += -p*log2base

    return sum_count

def split_group(subset):

    ent = entropy(subset, "Target")
    current_highest_info_gain = -1
    col_to_split = "dummy"
    if ent > 0:
        # we need splitting because we have uncertainty

        for cols in subset:
            if cols=="Target":
                break
            # lets start the splitting by each attribute
            grouped = subset.groupby(cols)

            group_entropies = []
            weighted = []
            # Iterate through each group
            for name, group in grouped:
                
                attrenttropy = entropy(group, "Target")
                group_entropies.append(attrenttropy)
                weighted.append(len(group)/len(subset))
    
            hafter = sum(value * weight for value, weight in zip(group_entropies, weighted))
            if current_highest_info_gain < ent-hafter:
                current_highest_info_gain = ent - hafter
                col_to_split = cols

        if col_to_split is None:
            return "Unknown"
        
        decisiontree = {col_to_split: {}}

        colsubgroup = subset.groupby(col_to_split)
        for name, groupsub in colsubgroup:
            new_subset = groupsub.drop(columns=[col_to_split])
            decisiontree[col_to_split][name] = split_group(new_subset) 
        
        return decisiontree

    else:
        return subset["Target"].iloc[0]
            

tree = split_group(df)

# Let's do some sample tests!
def classify(tree, input_data):
    if not isinstance(tree, dict):
        return tree # We've reached a leaf node (Yes/No)
    
    root_attr = next(iter(tree))  # Get the first key (e.g., "Outlook")
    if input_data[root_attr] not in tree[root_attr]:  
        return "Unknown"  # Handle unseen values
    
    return classify(tree[root_attr][input_data[root_attr]], input_data)

# Example Usage
new_input = {
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "High",
    "Wind": "Strong"
}

result = classify(tree, new_input)
print(f"Prediction: {result}")
