import os
import json
import logging

logger = logging.getLogger(__name__)



def preprocess(root):
    TARGET_NAME = 'Smiling'

    def _get_metadata(path):
        with open(os.path.join(path, 'raw', 'identity_CelebA.txt'), 'r') as f_identities:
            identities = f_identities.read().split('\n')
        with open(os.path.join(path, 'raw', 'list_attr_celeba.txt'), 'r') as f_attributes:
            attributes = f_attributes.read().split('\n')
        return identities, attributes

    def _get_celebrities_and_images(identities, min_samples=5):
        all_celebs = {}
        for line in identities:
            info = line.split()
            if len(info) < 2:
                continue
            image, celeb = info[0], info[1]
            if celeb not in all_celebs:
                all_celebs[celeb] = []
            all_celebs[celeb].append(image)
        
        # ignore all celebrities with less than `min_samples` images.
        good_celebs = {c: all_celebs[c] for c in all_celebs if len(all_celebs[c]) >= min_samples}
        return good_celebs

    def _get_celebrities_and_target(celebrities, attributes, attribute_name=TARGET_NAME):
        def _get_celebrities_by_image(identities):
            good_images = {}
            for c in identities:
                images = identities[c]
                for img in images:
                    good_images[img] = c
            return good_images
        
        col_names = attributes[1]
        col_idx = col_names.split().index(attribute_name)

        celeb_attributes = {}
        good_images = _get_celebrities_by_image(celebrities)

        for line in attributes[2:]:
            info = line.split()
            if len(info) == 0:
                continue
            image = info[0]
            if image not in good_images:
                continue
            celeb = good_images[image]
            att = (int(info[1:][col_idx]) + 1) / 2
            if celeb not in celeb_attributes:
                celeb_attributes[celeb] = []
            celeb_attributes[celeb].append(att)
        return celeb_attributes

    def _convert_to_json(path, celebrities, targets):
        all_data = {}

        celeb_keys = [c for c in celebrities]
        num_samples = [len(celebrities[c]) for c in celeb_keys]
        data = {c: {'x': celebrities[c], 'y': targets[c]} for c in celebrities}

        all_data['users'] = celeb_keys
        all_data['num_samples'] = num_samples
        all_data['user_data'] = data

        with open(os.path.join(path, 'all_data', 'all_data.json'), 'w') as outfile:
            json.dump(all_data, outfile)

    # set path
    DATASET_NAME = __file__.split('/')[-1].split('.')[0]
    path = os.path.join(os.path.expanduser(root), DATASET_NAME)
    
    # check if preprocessing has already done
    if not os.path.exists(os.path.join(path, 'all_data')):
        os.makedirs(os.path.join(path, 'all_data'))
    else:
        return
    
    # get IDs and attributes
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Get meta-data...!')
    identities, attributes = _get_metadata(path)
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...finished parsing meta-data (IDs and attributes)!')
    
    # filter out celebs
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Construct celeb-image hashmap...!')
    celebrities = _get_celebrities_and_images(identities)
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...constructed celeb-image hashmap!')
    
    # filter out targets
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Construct inputs-targets hashmap...!')
    targets = _get_celebrities_and_target(celebrities, attributes)
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...constructed inputs-targets hashmap!')
    
    # convert to json format
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Convert data to json format...!')
    _convert_to_json(path, celebrities, targets)
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...finished converting data to json format!')
