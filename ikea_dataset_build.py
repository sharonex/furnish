import os
import csv
from pathlib import Path

def process_furniture_dataset(root_directory, output_csv_path):
    """
    Process furniture dataset and create CSV with Room, Category, Description, and Path columns.

    Args:
        root_directory (str): Path to the root directory containing the furniture dataset
        output_csv_path (str): Path where the output CSV file will be saved
    """

    # List to store all the data rows
    furniture_data = []

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    # Walk through the directory structure
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            # Get the full file path
            file_path = os.path.join(root, file)

            # Check if it's an image file
            if Path(file).suffix.lower() in image_extensions:
                # Get relative path from root directory
                relative_path = os.path.relpath(file_path, root_directory)

                # Split the path to extract room and category
                path_parts = relative_path.split(os.sep)

                if len(path_parts) >= 3:  # At least Room/Category/filename.jpg
                    room = path_parts[0]
                    category = path_parts[1]

                    # Use filename without extension as description
                    description = Path(file).stem

                    # Store the data
                    furniture_data.append({
                        'Room': room,
                        'Category': category,
                        'Description': description,
                        'Path': relative_path
                    })
                else:
                    # Handle cases where the path structure is different
                    print(f"Warning: Unexpected path structure for {relative_path}")

    # Write to CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Room', 'Category', 'Description', 'Path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write data rows
        for row in furniture_data:
            writer.writerow(row)

    print(f"Successfully processed {len(furniture_data)} images")
    print(f"CSV file saved to: {output_csv_path}")

    return len(furniture_data)

def main():
    """
    Main function to run the script.
    Modify the paths below according to your dataset location.
    """

    # Set your dataset root directory path here
    dataset_root = "/Users/sharonavni/Desktop/IKEADataset"

    # Set output CSV file path
    output_csv = "furniture_dataset.csv"

    # Check if the dataset directory exists
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset directory '{dataset_root}' does not exist.")
        print("Please update the 'dataset_root' variable with the correct path.")
        return

    # Process the dataset
    try:
        count = process_furniture_dataset(dataset_root, output_csv)
        print(f"\nDataset processing complete!")
        print(f"Found and processed {count} furniture images.")

        # Display first few rows as preview
        print(f"\nPreview of generated CSV:")
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i < 6:  # Show header + first 5 data rows
                    print(f"{row}")
                else:
                    break

    except Exception as e:
        print(f"Error processing dataset: {str(e)}")

if __name__ == "__main__":
    main()
