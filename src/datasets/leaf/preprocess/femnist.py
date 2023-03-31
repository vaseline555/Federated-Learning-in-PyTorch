import os
import json
import pickle
import logging
import hashlib

import numpy as np

logger = logging.getLogger(__name__)



def preprocess(root):
    MAX_WRITERS = 100  # max number of writers per json file.
    
    def _save_obj(obj, name):
        with open(f'{name}.pkl', 'wb') as f:
            pickle.dump(obj, f)

    def _load_obj(name):
        with open(f'{name}.pkl', 'rb') as f:
            file = pickle.load(f)
        return file
    
    def _parse_files(path):
        def _parse_class_files(path):
            class_files = []  # (class, file directory)
            class_dir = os.path.join(path, 'raw', 'by_class')
            rel_class_dir = os.path.join(path, 'raw', 'by_class')
            classes = os.listdir(class_dir)
            classes = [c for c in classes if len(c) == 2]

            for cl in classes:
                cldir = os.path.join(class_dir, cl)
                rel_cldir = os.path.join(rel_class_dir, cl)
                subcls = os.listdir(cldir)

                subcls = [s for s in subcls if (('hsf' in s) and ('mit' not in s))]

                for subcl in subcls:
                    subcldir = os.path.join(cldir, subcl)
                    rel_subcldir = os.path.join(rel_cldir, subcl)
                    images = os.listdir(subcldir)
                    image_dirs = [os.path.join(rel_subcldir, i) for i in images]
                    for image_dir in image_dirs:
                        class_files.append((cl, image_dir))
            _save_obj(class_files, os.path.join(path, 'intermediate', 'class_file_dirs'))
        
        def _parse_write_files(path):
            write_files = []  # (writer, file directory)
            write_dir = os.path.join(path, 'raw', 'by_write')
            rel_write_dir = os.path.join(path, 'raw', 'by_write')
            write_parts = os.listdir(write_dir)

            for write_part in write_parts:
                writers_dir = os.path.join(write_dir, write_part)
                rel_writers_dir = os.path.join(rel_write_dir, write_part)
                writers = os.listdir(writers_dir)

                for writer in writers:
                    writer_dir = os.path.join(writers_dir, writer)
                    rel_writer_dir = os.path.join(rel_writers_dir, writer)
                    wtypes = os.listdir(writer_dir)

                    for wtype in wtypes:
                        type_dir = os.path.join(writer_dir, wtype)
                        rel_type_dir = os.path.join(rel_writer_dir, wtype)
                        images = os.listdir(type_dir)
                        image_dirs = [os.path.join(rel_type_dir, i) for i in images]

                        for image_dir in image_dirs:
                            write_files.append((writer, image_dir))
            _save_obj(write_files, os.path.join(path, 'intermediate', 'write_file_dirs'))
        
        # parse class & write files sequentially
        _parse_class_files(path)
        _parse_write_files(path)
    
    def _get_hashes(path):
        def _get_class_hashes(path):
            cfd = os.path.join(path, 'intermediate', 'class_file_dirs')
            class_file_dirs = _load_obj(cfd)
            class_file_hashes = []

            count = 0
            for cclass, cfile in class_file_dirs:
                chash = hashlib.md5(open(cfile, 'rb').read()).hexdigest()
                class_file_hashes.append((cclass, cfile, chash))
                count += 1

            cfhd = os.path.join(path, 'intermediate', 'class_file_hashes')
            _save_obj(class_file_hashes, cfhd)

        def _get_write_hashes(path):
            wfd = os.path.join(path, 'intermediate', 'write_file_dirs')
            write_file_dirs = _load_obj(wfd)
            write_file_hashes = []

            count = 0
            for cclass, cfile in write_file_dirs:
                chash = hashlib.md5(open(cfile, 'rb').read()).hexdigest()
                write_file_hashes.append((cclass, cfile, chash))
                count += 1

            wfhd = os.path.join(path, 'intermediate', 'write_file_hashes')
            _save_obj(write_file_hashes, wfhd)
        
        # get class & write hashes sequentially
        _get_class_hashes(path)
        _get_write_hashes(path)
    
    def _match_by_hashes(path):
        # read class file hash
        cfhd = os.path.join(path, 'intermediate', 'class_file_hashes')
        class_file_hashes = _load_obj(cfhd) # each element is (class, file dir, hash)
        class_hash_dict = {}
        for i in range(len(class_file_hashes)):
            c, f, h = class_file_hashes[len(class_file_hashes)-i-1]
            class_hash_dict[h] = (c, f)
        
        # read write file hash
        wfhd = os.path.join(path, 'intermediate', 'write_file_hashes')
        write_file_hashes = _load_obj(wfhd) # each element is (writer, file dir, hash)
        
        # match
        write_classes = []
        for tup in write_file_hashes:
            w, f, h = tup
            write_classes.append((w, f, class_hash_dict[h][0]))
        wwcd = os.path.join(path, 'intermediate', 'write_with_class')
        _save_obj(write_classes, wwcd)
    
    def _group_by_write(path):
        wwcd = os.path.join(path, 'intermediate', 'write_with_class')
        write_class = _load_obj(wwcd)

        writers, cimages = [], [] # each entry is a (writer, [list of (file, class)]) tuple
        cw, _, _ = write_class[0]
        for w, f, c in write_class:
            if w != cw:
                writers.append((cw, cimages))
                cw = w
                cimages = [(f, c)]
            cimages.append((f, c))
        writers.append((cw, cimages))

        ibwd = os.path.join(path, 'intermediate', 'images_by_write')
        _save_obj(writers, ibwd)
    
    def _convert_to_json(path):
        def _relabel_femnist_class(c):
            """
            Maps hexadecimal class value (string) to a decimal number.

            Args:
                c: class indices represented by hexadecimal values

            Returns:
                - 0 through 9 for classes representing respective numbers
                - 10 through 35 for classes representing respective uppercase letters
                - 36 through 61 for classes representing respective lowercase letters
            """
            if c.isdigit() and int(c) < 40: # digit
                return (int(c) - 30)
            elif int(c, 16) <= 90: # uppercase
                return (int(c, 16) - 55)
            else: # lowercase
                return (int(c, 16) - 61)

        by_writer_dir = os.path.join(path, 'intermediate', 'images_by_write')
        writers = _load_obj(by_writer_dir)

        users, num_samples, user_data = [], [], {}
        writer_count, all_writers = 0, 0

        # assign data
        for w, l in writers:
            users.append(w)
            num_samples.append(len(l))
            user_data[w] = {'x': [], 'y': []}

            for f, c in l:
                #gray = PIL.Image.open(f).convert('L')
                #vec = 1 - np.array(gray) / 255  # scale all pixel values to between 0 and 1
                #vec = vec.tolist()

                nc = _relabel_femnist_class(c)
                user_data[w]['x'].append(str(f))
                user_data[w]['y'].append(nc)
            writer_count += 1
            all_writers += 1
        else:
            all_data = {}
            all_data['users'] = users
            all_data['num_samples'] = num_samples
            all_data['user_data'] = user_data

            file_name = f'all_data.json' 
            file_path = os.path.join(path, 'all_data', file_name)

            with open(file_path, 'w') as outfile:
                json.dump(all_data, outfile)
            
    # set path
    DATASET_NAME = __file__.split('/')[-1].split('.')[0]
    path = os.path.join(os.path.expanduser(root), DATASET_NAME)
    
    # check if preprocessing has already done
    if not os.path.exists(os.path.join(path, 'all_data')):
        os.makedirs(os.path.join(path, 'all_data'))
    else:
        return
    
    # create intermediate file directories
    if not os.path.exists(os.path.join(path, 'intermediate')):
        os.makedirs(os.path.join(path, 'intermediate'))
    
    # parse files
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Extract files of raw images...!')
    _parse_files(path)
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...finished extracting files of raw images!')
    
    # get file hashes
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Calculate image hashes...!')
    _get_hashes(path)
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...finished calculating image hashes!')
    
    # match by hashes
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Assign class labels to write images...!')
    _match_by_hashes(path)
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...finished assigning class labels to write images!')
    
    # group images by writer
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Group images by write...!')
    _group_by_write(path)
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...finished grouping images by writer!')
    
    # convert to json format
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Convert data to json format...!')
    _convert_to_json(path)
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...finished converting data to json format!')
