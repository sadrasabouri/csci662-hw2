import codecs
import argparse

def to_rot13(text):
    text, label = text.split('\t')[0], text.split('\t')[1] if '\t' in text else ''
    text = ' '.join([x[::-1] for x in codecs.encode(text, 'rot_13').split()])
    return text + ('\t' + label if label else '\n')
    # return codecs.encode(text, 'rot_13')

def from_rot13(text):
    text, label = text.split('\t')[0], text.split('\t')[1] if '\t' in text else ''
    text = ' '.join([x[::-1] for x in codecs.decode(text, 'rot_13').split()])
    return text + ('\t' + label if label else '\n')
    # return codecs.decode(text, 'rot_13')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple ROT13 text converter")
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument("output_file", help="Path to the output text file")
    parser.add_argument("--decode", action="store_true", help="Decode from ROT13 instead of encoding")

    args = parser.parse_args()

    with open(args.input_file, 'r') as infile:
        content = infile.readlines()

    with open(args.output_file, 'w') as outfile:
            if args.decode:
                for line in content:
                    converted_content = from_rot13(line)
                    outfile.write(converted_content)
            else:
                for line in content:
                    converted_content = to_rot13(line)
                    outfile.write(converted_content)
