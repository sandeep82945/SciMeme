import json

def read_json(json_file_path):
    '''
    Returns 1st abstract and then the section information
    '''

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    paper_text = ''
    abstractText = ''
    if 'abstractText' in data:
        abstractText = data['abstractText']

    if 'sections' in data:
        for element in data['sections']:
            if 'heading' in element:
                heading = element['heading'].lower()
                text =element['text']
                paper_text = paper_text + " " + heading + " "+ text
                paper_text.replace('..','.')

    return abstractText, paper_text

