def parse_file(filepath):
    all_values = []
    with open(filepath, 'r') as f:
        for line in f:
            spl_line = line.strip().split(' ')
            score, cls = spl_line[0], spl_line[1]

            all_values.append((float(score), cls))
    return all_values

def save_numbers_to_tsv(numbers, file_path):
    """
    Saves a list of numbers to a file in a single-column TSV format.
    
    Each number will be written to a new line.
    
    Args:
        numbers (list or iterable): A list of numbers (e.g., [0, 1, 1, 0, 1]).
        file_path (str): The path to the file to be created or overwritten.
    """
    try:
        # Open the file in write mode ('w'). 
        # The 'with' statement ensures the file is properly closed after writing.
        with open(file_path, 'w') as f:
            for num in numbers:
                # Convert the number to a string and write it, 
                # followed by a newline character (\n).
                f.write(f"{num}\n")
        
        print(f"Successfully saved {len(numbers)} numbers to '{file_path}'")
        
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

th = 0.936
values = parse_file('logs/results_test_b')
formatted_values = [1 if score < th else 0 for score, label in values]
save_numbers_to_tsv(formatted_values, 'test2.tsv')