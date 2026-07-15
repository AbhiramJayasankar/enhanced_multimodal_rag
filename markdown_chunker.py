import json
import re

def chunk_markdown_by_headings(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = []
    lines = content.split('\n')

    current_chunk = {'heading': '', 'content': ''}

    for line in lines:
        # Check if line is a heading (starts with #)
        if re.match(r'^#+\s+', line):
            # Save previous chunk if it has content
            if current_chunk['heading'] or current_chunk['content'].strip():
                chunks.append({
                    'heading': current_chunk['heading'],
                    'content': current_chunk['content'].strip()
                })

            # Start new chunk
            current_chunk = {
                'heading': line.strip(),
                'content': ''
            }
        else:
            # Add line to current chunk content
            current_chunk['content'] += line + '\n'

    # Add the last chunk
    if current_chunk['heading'] or current_chunk['content'].strip():
        chunks.append({
            'heading': current_chunk['heading'],
            'content': current_chunk['content'].strip()
        })

    return chunks

# Read and chunk the markdown file
input_file = r'c:\Users\abhir\Desktop\projects\jina\markdown\colpali.md'
chunks = chunk_markdown_by_headings(input_file)

# Save to JSON
output_file = r'c:\Users\abhir\Desktop\projects\jina\storage\colpali_chunks.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"Chunked {len(chunks)} sections and saved to {output_file}")
