def find_common_sequences(str1, str2, min_length=2):
    # Convert strings to lists of words
    words1 = str1.lower().split()
    words2 = str2.lower().split()

    # Store common sequences
    common_sequences = []

    # Length of word lists
    len1, len2 = len(words1), len(words2)

    # Validate min_length
    if min_length < 1:
        min_length = 1
    elif min_length > min(len1, len2):
        min_length = min(len1, len2)

    # Iterate through all possible sequence lengths, starting from longest
    for seq_length in range(min(len1, len2), min_length - 1, -1):
        # Check sequences of current length in first string
        for i in range(len1 - seq_length + 1):
            sequence = words1[i:i + seq_length]

            # Check if this sequence exists in second string
            for j in range(len2 - seq_length + 1):
                if sequence == words2[j:j + seq_length]:
                    # Convert sequence back to string and add to results
                    common_seq = ' '.join(sequence)
                    if common_seq not in [seq[0] for seq in common_sequences]:
                        common_sequences.append((common_seq, seq_length))

    return common_sequences

def common_text_selector(text: str, common_text_dict: list):
    text = text.lower()
    most_left_text = common_text_dict[0][0].lower()
    most_left_index = text.find(most_left_text)

    for entry in common_text_dict[1:]:
        common_text: str = entry[0].lower()
        length = entry[1]

        if length > 2:
            entry_str_idx = text.find(common_text)

            if entry_str_idx < most_left_index:
                most_left_text = common_text
                most_left_index = entry_str_idx

    return most_left_text, most_left_index


def combine_overlapping_text(predicted_text: str, new_text):

    common_text = find_common_sequences(predicted_text, new_text)

    if len(common_text) == 0:
        return None


    common_text, start_index = common_text_selector(new_text, common_text)

    start_replace_index = str(predicted_text.lower()).find(common_text.lower())

    new_text = new_text[start_index:]

    predicted_text = predicted_text[:start_replace_index]

    predicted_text += new_text

    # Join the words into a single sentence
    return predicted_text, new_text[:start_index]
