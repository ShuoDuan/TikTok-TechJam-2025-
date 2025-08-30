import json
import csv
import pandas as pd

def jsonl_to_csv(jsonl_file_path, csv_file_path):
    """
    Convert JSONL file (like review-Alabama_labeled.jsonl) to CSV format
    Extracts relevant fields from the nested structure
    Includes all entries regardless of label (including 'Error' labels)
    """
    try:
        data = []
        
        # Read JSONL file line by line
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        # Parse each line as JSON
                        entry = json.loads(line)
                        
                        # Extract data from the nested structure
                        raw_data = entry.get('raw', {})
                        
                        # Create flattened row with name, rating, text, and label as the fourth column
                        # Include all entries regardless of label (including 'Error' labels)
                        row = {
                            'name': raw_data.get('name', ''),
                            'rating': raw_data.get('rating', ''),
                            'text': raw_data.get('text', ''),
                            'label': entry.get('label', '')
                        }
                        
                        data.append(row)
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {e}")
                        continue
        
        # Create DataFrame with specific column order (label as the fourth column)
        df = pd.DataFrame(data, columns=['name', 'rating', 'text', 'label'])
        
        # Save to CSV
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
        print(f"Successfully converted {len(data)} entries from {jsonl_file_path} to {csv_file_path}")
        
        # Show preview of the data
        print("\nPreview of converted data:")
        print(df.head())
        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
        
    except Exception as e:
        print(f"Error converting JSONL to CSV: {e}")

# Example usage
if __name__ == "__main__":
    # Update these paths as needed
    jsonl_input = r"C:\Users\zhang\Desktop\review-Alabama.json\review-Alabama_labeled.jsonl"
    csv_output = r"C:\Users\zhang\Desktop\review-Alabama_labeled.csv"
    
    jsonl_to_csv(jsonl_input, csv_output)
