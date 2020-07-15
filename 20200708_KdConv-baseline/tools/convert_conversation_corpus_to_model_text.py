"""
File: convert_conversation_corpus_to_model_text.py
"""
import sys
import json
import collections


def preprocessing_for_one_conversation(text):
    """
    preprocessing_for_one_conversation
    """
    conversation = json.loads(text.strip(), object_pairs_hook=collections.OrderedDict)
    knowledge = conversation["knowledge"]
    history = conversation['history']
    response = conversation['response'] if 'response' in conversation else 'null'
    history_str = '[SEP] '.join(history)
    src = knowledge + history_str
    model_text = '\t'.join([src, response])
    return model_text


def convert_conversation_corpus_to_model_text(corpus_file, text_file):
    """
    convert_conversation_corpus_to_model_text
    """
    fout_text = open(text_file, 'w')
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            model_text = preprocessing_for_one_conversation(line.strip())
            fout_text.write(model_text + '\n')
    fout_text.close()


def main():
    """
    main
    """
    convert_conversation_corpus_to_model_text(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nExited from the program ealier!')
