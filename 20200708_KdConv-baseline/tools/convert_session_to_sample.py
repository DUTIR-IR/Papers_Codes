"""
File: convert_session_to_sample.py
"""
import sys
import json
import collections
import codecs


def convert_session_to_sample(session_file, sample_file):
    """
    convert_session_to_sample
    """
    with open(sample_file, 'w') as fout:
        with codecs.open(session_file) as f:
            content = json.load(f)
        for j, session in enumerate(content):
            context = []
            knowledge = []
            for utterance in session['messages']:
                context.append(utterance['message'])
                if 'attrs' in utterance.keys():
                    attr = utterance['attrs'][-1]
                    knowledge.append('[Name]{}[AttrName]{}[AttrValue]{}'.format(attr['name'],
                                                                                attr['attrname'],
                                                                                attr['attrvalue']))
                else:
                    knowledge.append('[NoKnowledge]')

            for i in range(1, len(context)):
                sample = collections.OrderedDict()
                sample['knowledge'] = knowledge[i]
                sample['history'] = context[:i]
                sample['response'] = context[i]
                sample = json.dumps(sample, ensure_ascii=False)
                fout.write(sample + '\n')


def main():
    """
    main
    """
    convert_session_to_sample(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nExited from the program ealier!')
