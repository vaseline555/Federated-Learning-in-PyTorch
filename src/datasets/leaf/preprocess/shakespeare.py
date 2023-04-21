import os
import re
import json
import logging

from collections import defaultdict

logger = logging.getLogger(__name__)



def preprocess(root):
    # raw file name
    RAW_FILENAME = '100.txt'
    
    # regular expression to capture an actors name, and line continuation
    CHARACTER_RE = re.compile(r'^  ([a-zA-Z][a-zA-Z ]*)\. (.*)')
    CONT_RE = re.compile(r'^    (.*)')

    # `The Comedy of Errors` has errors in its indentation so we need to use different regular expressions
    COE_CHARACTER_RE = re.compile(r'^([a-zA-Z][a-zA-Z ]*)\. (.*)')
    COE_CONT_RE = re.compile(r'^(.*)')
        
    def _split_into_plays(shakespeare_full):
        """Splits the full data by plays.
        """
        def _match_character_regex(line, comedy_of_errors=False):
            return (COE_CHARACTER_RE.match(line) if comedy_of_errors else CHARACTER_RE.match(line))

        def _match_continuation_regex(line, comedy_of_errors=False):
            return (COE_CONT_RE.match(line) if comedy_of_errors else CONT_RE.match(line))
    
        # List of tuples: (play_name, dict from character to list of lines)
        plays, discarded_lines = [], []  # Track discarded lines.
        slines = shakespeare_full.splitlines(True)[1:]

        # skip contents, the sonnets, and all's well that ends well
        author_count, start_i = 0, 0
        for i, l in enumerate(slines):
            if 'by William Shakespeare' in l:
                author_count += 1
            if author_count == 2:
                start_i = i - 5
                break
                
        current_character = None
        comedy_of_errors = False
        slines = slines[start_i:]
        for i, line in enumerate(slines):
            if i > 124195 - start_i: # this marks the end of the plays in the file.
                break
            if 'by William Shakespeare' in line: # this is a pretty good heuristic for detecting the start of a new play
                current_character = None
                characters = defaultdict(list)
                # the title will be 2, 3, 4, 5, 6, or 7 lines above "by William Shakespeare".
                if slines[i - 2].strip():
                    title = slines[i - 2]
                elif slines[i - 3].strip():
                    title = slines[i - 3]
                elif slines[i - 4].strip():
                    title = slines[i - 4]
                elif slines[i - 5].strip():
                    title = slines[i - 5]
                elif slines[i - 6].strip():
                    title = slines[i - 6]
                else:
                    title = slines[i - 7]
                title = title.strip()
                
                if not title:
                    err = f'Error on line {i}; expect title 2 or 3 lines above!'
                    logger.exception(err)
                    raise AssertionError(err)
                    
                comedy_of_errors = (title == 'THE COMEDY OF ERRORS')
                plays.append((title, characters)) # degenerate plays are removed at the end of the method.
                continue
                
            match = _match_character_regex(line, comedy_of_errors)
            if match:
                character, snippet = match.group(1), match.group(2)
                """
                Some character names are written with multiple casings, e.g., SIR_Toby and SIR_TOBY. 
                To normalize the character names, we uppercase each name.
                Note that this was not done in the original preprocessing and is a recent fix.
                """
                character = character.upper()
                if not (comedy_of_errors and character.startswith('ACT ')):
                    characters[character].append(snippet)
                    current_character = character
                    continue
                else:
                    current_character = None
                    continue
            elif current_character:
                match = _match_continuation_regex(line, comedy_of_errors)
                if match:
                    if comedy_of_errors and match.group(1).startswith('<'):
                        current_character = None
                        continue
                    else:
                        characters[current_character].append(match.group(1))
                        continue
            line = line.strip() # didn't consume the line.
            if line and i > 2646: # before 2646 are the sonnets, which we expect to discard.
                discarded_lines.append(f'{i}:{line}')
        return [play for play in plays if len(play[1]) > 1], discarded_lines # # remove degenerate "plays"

    def _get_all_examples(plays):
        """Collect all examples.
        """
        def _remove_nonalphanumerics(filename):
            return re.sub('\\W+', '_', filename)

        def _play_and_character(play, character):
            return _remove_nonalphanumerics((play + '_' + character).replace(' ', '_'))
    
        def _add_examples(example_dict, example_tuple_list):
            for play, character, sound_bite in example_tuple_list:
                example_dict[_play_and_character(play, character)].append(sound_bite)
    
        skipped_characters = 0
        all_examples = defaultdict(list)
        users_and_plays = {}
        for play, characters in plays:
            curr_characters = list(characters.keys())
            for c in curr_characters:
                users_and_plays[_play_and_character(play, c)] = play
            for character, sound_bites in characters.items():
                examples = [(play, character, sound_bite) for sound_bite in sound_bites]
                if len(examples) <= 2:
                    skipped_characters += 1 # skip characters with fewer than 2 lines since we need at least one train and one test line
                    continue
                _add_examples(all_examples, examples)
        return users_and_plays, all_examples

    def _write_data(examples, output_dir):
        """Writes a collection of data files composed of play & character.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for character_name, sound_bites in examples.items():
            filename = os.path.join(output_dir, f'{character_name}.txt')
            with open(filename, 'w') as outfile:
                for sound_bite in sound_bites:
                    outfile.write(sound_bite + '\n')
    
    def _convert_to_json(path):
        """Create meta-data of Shakespeare dataset.

        Args:
            data_path: path of raw data
            users_and_plays_path: path of `users_and_plays.json` 

        Returns:
            all_data: dictionary with keys - (users, num_samples, user_data)
        """
        def _to_data(txt_path, seq_length=80):
            """Parses text file in given directory into data for next-character model.

            Args:
                txt_path: path to text file
                seq_length: length of all strings in inputs

            Returns:
                (inputs, targets)
            """
            ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
            TOKEN_DICT = {char: idx for idx, char in enumerate(ALL_LETTERS)}

            raw_text = ""
            with open(txt_path, 'r') as file:
                raw_text = file.read()
            raw_text = raw_text.replace('\n', ' ')
            raw_text = re.sub(r'   *', r' ', raw_text)
            tokenized = [TOKEN_DICT[char] for char in list(raw_text)]
            
            inputs, targets = [], []
            for i in range(0, len(raw_text) - seq_length, 1):
                seq_in = tokenized[i:i + seq_length]
                seq_out = tokenized[i + seq_length]
                
                inputs.append(seq_in)
                targets.append(seq_out)
            return inputs, targets

        data_path = os.path.join(path, 'by_play_and_character')
        users_and_plays_path = os.path.join(path, 'intermediate', 'users_and_plays.json')
    
        with open(users_and_plays_path, 'r') as file:
            users_and_plays = json.load(file)

        users, hierarchies, num_samples, user_data = [], [], [], {}
        for f in os.listdir(data_path):
            user = f[:-4]
            filename = os.path.join(data_path, f)
            inputs, targets = _to_data(filename)
            if len(inputs) > 0:
                users.append(user)
                user_data[user] = {'x': inputs, 'y': targets}

                hierarchies.append(users_and_plays[user])
                num_samples.append(len(targets))

        all_data = {}
        all_data['users'] = users
        all_data['hierarchies'] = hierarchies
        all_data['num_samples'] = num_samples
        all_data['user_data'] = user_data
        
        # convert to json
        with open(os.path.join(path, 'all_data', 'all_data.json'), 'w') as outfile:
            json.dump(all_data, outfile)

    # set path
    DATASET_NAME = __file__.split('/')[-1].split('.')[0]
    path = os.path.join(os.path.expanduser(root), 'shakespeare')

    # check if preprocessing has already done
    if not os.path.exists(os.path.join(path, 'all_data')):
        os.makedirs(os.path.join(path, 'all_data'))
    else:
        return
    
    # load raw data
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Read raw data...!')
    with open(os.path.join(path, 'raw', RAW_FILENAME), 'r') as input_file:
        shakespeare_full = input_file.read()
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...loaded raw data!')    
    
    # split raw data into plays
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Split raw data into plays...!')
    plays, discarded_lines = _split_into_plays(shakespeare_full)
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...done splitting raw data into plays (discarded lines: {len(discarded_lines)})!')
    
    # get all examples by character
    ## check if path exists
    if not os.path.exists(os.path.join(path, 'intermediate')):
        os.makedirs(os.path.join(path, 'intermediate'))
    
    ## get all examples and users-plays pair
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Collect all examples...!')
    users_and_plays, all_examples = _get_all_examples(plays)
    
    ## save users-plays pair
    with open(os.path.join(path, 'intermediate', 'users_and_plays.json'), 'w') as outfile:
        json.dump(users_and_plays, outfile)
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...collected all examples!')    
    
    # save all examples
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Write as files...!')
    _write_data(all_examples, os.path.join(path, 'by_play_and_character'))
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...wrote files!')
    
    # prase and convert to json format
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Convert data to json format... (this may take several minutes)!')
    _convert_to_json(path) 
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...finished converting data to json format!')
