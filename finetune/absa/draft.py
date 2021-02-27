all_labels_dict = {"a, b": 1, "c, d": 2}

all_labels_dict = {
    tuple(k.strip("{}").split(", ")): v \
        for k, v in all_labels_dict.items()
}
    
print(all_labels_dict)