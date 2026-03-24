
import PyPDF2
import os
import sys

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from all pages of a PDF document.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: A single string containing all extracted text, or an error message.
    """
    if not os.path.exists(pdf_path):
        return f"Error: The file '{pdf_path}' does not exist."
    
    try:
        # Open the PDF file in binary read mode
        with open(pdf_path, 'rb') as file:
            # Create a PdfReader object
            reader = PyPDF2.PdfReader(file)
            
            # Initialize an empty string to store all extracted text
            full_text = ""
            
            # Iterate over each page and extract text
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                full_text += page.extract_text()
                
        return full_text
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    # Get the PDF file path from the command-line arguments
    if len(sys.argv) < 2:
        print("Error: Please provide the path to the PDF file as a command-line argument.")
        sys.exit(1)
        
    pdf_file_path = sys.argv[1]
    
    extracted_content = extract_text_from_pdf(pdf_file_path)
    
    if extracted_content.startswith("Error:"):
        print(extracted_content)
    else:
        # Optionally, save the extracted text to a .txt file
        output_filename = os.path.splitext(pdf_file_path)[0] + "_extracted.txt"
        try:
            with open(output_filename, 'w', encoding='utf-8') as output_file:
                output_file.write(extracted_content)
        except Exception as e:
            print(f"Could not save extracted text to file: {e}")
