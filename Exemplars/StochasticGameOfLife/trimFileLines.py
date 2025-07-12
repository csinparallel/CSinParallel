def trim_file_lines(input_filepath, output_filepath, max_length):
    """
    Reads lines from an input file, trims each line to a specified maximum length,
    and writes the trimmed lines to an output file.

    Args:
        input_filepath (str): The path to the input file.
        output_filepath (str): The path to the output file where trimmed lines will be written.
        max_length (int): The maximum desired length for each line.
    """
    try:
        with open(input_filepath, 'r') as infile:
            with open(output_filepath, 'w') as outfile:
                for line in infile:
                    # Remove trailing newline characters and other whitespace
                    trimmed_line = line.rstrip()
                    
                    # Trim the line to the specified maximum length
                    if len(trimmed_line) > max_length:
                        trimmed_line = trimmed_line[:max_length]
                    
                    # Write the trimmed line followed by a newline character
                    outfile.write(trimmed_line + '\n')
        print(f"Lines trimmed and saved to '{output_filepath}' successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
if __name__ == "__main__":
    input_file = "profile.txt"  # Replace with your input file name
    output_file = "trimmed_profile.txt" # Replace with your desired output file name
    trim_length = 110          # Replace with your desired maximum line length

    trim_file_lines(input_file, output_file, trim_length)
