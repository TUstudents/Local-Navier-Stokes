import os
import nbformat

def convert_notebooks_in_current_directory():
    """
    Iterates through all files in the current directory, identifies Jupyter
    Notebooks (.ipynb), and converts them to the current nbformat.
    """
    current_directory = os.getcwd()
    notebook_files = [
        f for f in os.listdir(current_directory) if f.endswith(".ipynb")
    ]

    if not notebook_files:
        print("No Jupyter Notebooks found in the current directory.")
        return

    print(f"Found the following notebooks: {notebook_files}")

    for filename in notebook_files:
        filepath = os.path.join(current_directory, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)  # Read notebook

            # nbformat.write will automatically convert to the current format
            with open(filepath, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)

            print(f"Successfully converted: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

convert_notebooks_in_current_directory()