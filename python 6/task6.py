def check(x: str, file: str):
    processed_str = x.lower().split()
    processed_str.sort()
    freq = dict()

    for word in processed_str:
        freq[word] = processed_str.count(word)

    used_words = list()

    with open(file, "w") as f:
        for word in processed_str:
            if word not in used_words:
                f.write(f"{word} {freq[word]}\n")
                used_words.append(word)
