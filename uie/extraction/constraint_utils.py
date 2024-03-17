from uie.extraction import constants
from uie.seq2seq.constraint_decoder.constraint_decoder import find_bracket_position
from uie.extraction.predict_parser.spotasoc_predict_parser import add_bracket, check_well_form, get_tree_str, resplit_label_span
from nltk.tree import ParentedTree

def get_valid_vocab(input_tokens):
    struct_vocab = [constants.type_start, constants.type_end, constants.span_start, constants.null_span, constants.null_label]
    schema_vocab = input_tokens[:input_tokens.index(constants.text_start)]
    schema_vocab = list(set(schema_vocab))
    schema_vocab.remove(constants.asoc_prompt)
    schema_vocab.remove(constants.spot_prompt)
    input_vocab = input_tokens[input_tokens.index(constants.text_start):]
    valid_vocab = list(set(struct_vocab+schema_vocab+input_vocab))

    return valid_vocab


def parse_generated_seq(generated_tokens):
    generated_seq = " ".join(generated_tokens[1:]) # drop first sep token 
    generated_seq = generated_seq.replace('<extra_id_0>', '【').replace('<extra_id_1>', '】')

    try:
        if not check_well_form(generated_seq):
                generated_seq = add_bracket(generated_seq)
        generated_tree = ParentedTree.fromstring(generated_seq, brackets='【】')
        generated_asocs = []
        cur_asoc = None

        if len(generated_tree) > 0:
            cur_event_tree = generated_tree[-1]
            for asoc_tree in cur_event_tree:
                if isinstance(asoc_tree, str) or len(asoc_tree) < 1:
                    continue

                asoc_label = asoc_tree.label()
                asoc_text = get_tree_str(asoc_tree)
                asoc_label, asoc_text = resplit_label_span(
                    asoc_label, asoc_text)

                if asoc_text is not None:
                    generated_asocs.append([asoc_label, asoc_text])
            cur_asoc = asoc_label

    except:
        return [], None

    return generated_asocs, cur_asoc