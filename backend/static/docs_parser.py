# backend/static/docs_parser.py
import os
import json
from bs4 import BeautifulSoup
from typing import List, Dict
import re
from datetime import datetime

class ArduPilotDocsParser:
    def __init__(self, docs_dir: str = "."):  # Changed from "static" to "."
        self.docs_dir = docs_dir
        self.docs = []
    
    def parse_documentation(self):
        """Parse all HTML documentation files"""
        print(f"Looking for HTML files in: {os.path.abspath(self.docs_dir)}")
        for root, dirs, files in os.walk(self.docs_dir):
            print(f"Checking directory: {root}")
            for file in files:
                if file.endswith('.html'):
                    file_path = os.path.join(root, file)
                    print(f"Found HTML file: {file_path}")
                    doc = self.parse_html_file(file_path)
                    if doc:
                        self.docs.append(doc)
                        print(f"Successfully parsed: {file}")
        
        print(f"Total documents parsed: {len(self.docs)}")
        return self.docs
    
    def parse_html_file(self, file_path: str) -> Dict:
        """Parse a single HTML file, specifically for ArduPilot log message docs"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else os.path.basename(file_path)
            
            # For log message documentation, look for sections with message types
            message_sections = []
            
            # Find all sections with message type IDs (like att, gps, parm, etc.)
            sections = soup.find_all('section', id=True)
            print(f"Found {len(sections)} sections with IDs")
            
            for section in sections:
                section_id = section.get('id', '')
                h2 = section.find('h2')
                
                if h2:
                    message_type = h2.get_text().strip().split('¶')[0].strip()  # Remove ¶ symbol
                    print(f"Found message type: {message_type} (ID: {section_id})")
                    
                    # Get description paragraph
                    description_p = section.find('p')
                    description = description_p.get_text().strip() if description_p else ""
                    
                    # Extract table data if present
                    table_data = self.extract_table_data(section)
                    
                    # Get all text content from the section
                    section_text = section.get_text(separator='\n', strip=True)
                    section_text = re.sub(r'\n\s*\n', '\n\n', section_text)  # Clean up newlines
                    
                    message_sections.append({
                        "message_type": message_type,
                        "section_id": section_id,
                        "description": description,
                        "table_data": table_data,
                        "full_content": section_text
                    })
            
            # If no message sections found, fall back to general content extraction
            if not message_sections:
                print("No message sections found, falling back to general content extraction")
                main_content = soup.find('main') or soup.find('div', class_='content') or soup.find('body')
                if main_content:
                    # Remove navigation and sidebar elements
                    for element in main_content.find_all(['nav', 'aside', 'header', 'footer']):
                        element.decompose()
                    
                    text_content = main_content.get_text(separator='\n', strip=True)
                else:
                    text_content = soup.get_text(separator='\n', strip=True)
                
                # Clean up the text
                text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
                text_content = re.sub(r'[ \t]+', ' ', text_content)
                
                return {
                    "title": title_text,
                    "url": file_path,
                    "content": text_content,
                    "type": "ardupilot_docs",
                    "message_sections": []
                }
            
            print(f"Found {len(message_sections)} message sections")
            return {
                "title": title_text,
                "url": file_path,
                "type": "ardupilot_log_messages",
                "message_sections": message_sections,
                "total_messages": len(message_sections)
            }
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def extract_table_data(self, section) -> List[Dict]:
        """Extract structured data from tables in a section"""
        tables = section.find_all('table')
        table_data = []
        
        for table in tables:
            rows = table.find_all('tr')
            headers = []
            
            # Get headers from first row
            if rows:
                first_row = rows[0]
                header_cells = first_row.find_all(['th', 'td'])
                headers = [cell.get_text().strip() for cell in header_cells]
            
            # Extract data rows
            for row in rows[1:]:  # Skip header row
                cells = row.find_all(['td', 'th'])
                row_data = {}
                for i, cell in enumerate(cells):
                    if i < len(headers):
                        row_data[headers[i]] = cell.get_text().strip()
                if row_data:  # Only add non-empty rows
                    table_data.append(row_data)
        
        return table_data
    
    def create_docs_index(self):
        """Create a searchable index of documentation"""
        docs_data = {
            "total_docs": len(self.docs),
            "docs": self.docs,
            "created_at": datetime.now().isoformat()
        }
        
        with open('ardupilot_index.json', 'w') as f:
            json.dump(docs_data, f, indent=2)
        
        return docs_data

if __name__ == "__main__":
    parser = ArduPilotDocsParser()
    parser.parse_documentation()
    parser.create_docs_index()
    print('Documentation indexed successfully!')