import zipfile
import xml.etree.ElementTree as ET
import json
import os

# --- Configuration ---
zip_path = '/mnt/c/Users/faysa/Desktop/Mappar-på-skrivbordet/KTH/Datateknik/Mex-DA233x/Project/transcriptions.zip'
output_file = 'historical_corpus_full_raw.jsonl' 

def safe_int(value, default=0):
    """
    Safely converts a string to an integer. 
    Handles weird OCR glitches like scientific notation ('9.999999E6') or empty strings.
    """
    try:
        if value is None:
            return default
        # Convert to float first to handle scientific notation, then to int
        return int(float(value))
    except (ValueError, TypeError):
        return default

def extract_alto_data(xml_content):
    """
    Parses ALTO XML to extract metadata and spatially sorted text.
    """
    try:
        root = ET.fromstring(xml_content)
        
        # DYNAMIC NAMESPACE FIX
        # Riksarkivet might mix ALTO v2, v3, and v4. This grabs the correct one automatically.
        ns_uri = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
        ns = {'alto': ns_uri} if ns_uri else {'alto': ''}
        
        # Extract Metadata
        source_img = root.find('.//alto:sourceImageInformation/alto:fileName', ns)
        proc_date = root.find('.//alto:processingDateTime', ns)
        
        metadata = {
            "source_image": source_img.text if source_img is not None else "Unknown",
            "processing_date": proc_date.text if proc_date is not None else "Unknown"
        }

        # Extract and Sort Text Blocks
        blocks =[]
        for block in root.findall('.//alto:TextBlock', ns):
            # FIXED: Use safe_int instead of standard int()
            vpos = safe_int(block.get('VPOS', 0))
            hpos = safe_int(block.get('HPOS', 0))
            
            lines =[]
            for line in block.findall('.//alto:TextLine', ns):
                words =[word.get('CONTENT') for word in line.findall('.//alto:String', ns) if word.get('CONTENT')]
                if words:
                    lines.append(" ".join(words))
            
            if lines:
                blocks.append({
                    "vpos": vpos,
                    "hpos": hpos,
                    "text": "\n".join(lines)
                })

        # Sort blocks: Top-to-bottom (vpos), then Left-to-right (hpos)
        blocks.sort(key=lambda b: (b['vpos'] // 50, b['hpos']))
        
        full_text = "\n\n".join([b['text'] for b in blocks])
        return full_text, metadata

    except ET.ParseError:
        return "", {"error": "XML Parse Error"}

# --- Main Processing Loop ---

# --- NEW: RESUME LOGIC ---
processed_files = set()
if os.path.exists(output_file):
    print(f"Found existing output file. Scanning to see where we left off...")
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                doc = json.loads(line)
                processed_files.add(doc['doc_id'])
            except json.JSONDecodeError:
                pass
    print(f"Skipping {len(processed_files):,} files that are already completed.\n")
# -------------------------

print(f"Opening archive: {zip_path}...\n")

with zipfile.ZipFile(zip_path, 'r') as z:
    xml_files =[f for f in z.infolist() if f.filename.endswith('.xml')]
    total_files = len(xml_files)
    
    if total_files == 0:
        print("No XML files found in the archive.")
    else:
        print(f"Found {total_files:,} XML files. Resuming extraction...")
        
        # Open the .jsonl file in append mode (just in case you need to stop and restart later)
        with open(output_file, 'a', encoding='utf-8') as outfile:
            
            # Loop through ALL files using enumerate to keep track of the count
            for index, current_xml_file in enumerate(xml_files, start=1):
                
                doc_id = os.path.basename(current_xml_file.filename)
                
                # --- SKIP IF ALREADY PROCESSED ---
                if doc_id in processed_files:
                    continue
                
                with z.open(current_xml_file) as source:
                    xml_bytes = source.read()
                    # Extract the data
                    raw_text, metadata = extract_alto_data(xml_bytes)
                    
                    # If the OCR failed and the page is blank, do not save it!
                    if not raw_text.strip():
                        continue 
                    
                    doc_record = {
                        "doc_id": doc_id,
                        "source_image": metadata.get("source_image", "Unknown"),
                        "processed_at": metadata.get("processing_date", "Unknown"),
                        "raw_text": raw_text,
                        "word_count": len(raw_text.split())
                    }
                    
                    outfile.write(json.dumps(doc_record, ensure_ascii=False) + '\n')
                
                # Print an update every 10,000 files
                if index % 10000 == 0:
                    percent_done = (index / total_files) * 100
                    print(f"Processed {index:,} / {total_files:,} files ({percent_done:.1f}%)")

        print(f"\nExtraction Complete! Data saved to '{output_file}'.")