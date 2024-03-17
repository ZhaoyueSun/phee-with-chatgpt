#!/usr/bin/env python
# -*- coding:utf-8 -*-
from uie.extraction import constants
from dataclasses import dataclass
import numpy as np
from uie.extraction.utils import *
from uie.extraction.record_schema import RecordSchema
from uie.extraction.noiser.spot_asoc_noiser import SpotAsocNoiser
import random

@dataclass
class OrderedSpotAsocNoiser(SpotAsocNoiser):
    
    def __init__(self, schema, spot_noise_ratio=0.1, asoc_noise_ratio=0.1, null_span=constants.null_span, ordered_prompt=True):
        super().__init__()
        self.spot_noise_ratio = spot_noise_ratio
        self.asoc_noise_ratio = asoc_noise_ratio
        self.null_span = null_span
        self.ordered_prompt = ordered_prompt
        self.type_role_dict = schema.type_role_dict

    def insert_spot(self, spot_asoc, spot_label_list=None):
        """随机插入 Spot，类别从 spot_label_list 中自动选择

        Args:
            spot_asoc ([type]): [description]
            spot_label_list ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if spot_label_list is None or len(spot_label_list) == 0:
            return spot_asoc
        
        spot_list = list(self.type_role_dict.keys())
        if not self.ordered_prompt:
                random.shuffle(spot_list)
        record = []
        for spot_type in spot_list:
            for spot in spot_asoc:
                if spot['label'] == spot_type:
                    record.append(spot)
            if random.random() < self.spot_noise_ratio and spot_type in spot_label_list:
                record.append({
                    "span": self.null_span, "label": spot_type, 'asoc': list()
                })
        return record

    def insert_asoc(self, spot_asoc, asoc_label_list=None):
        """按顺序或随机插入 Asoc，类别从 asoc_label_list 中自动选择

        Args:
            spot_asoc ([type]): [description]
            asoc_label_list ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        record = []
        for spot in spot_asoc:
            new_spot = spot.copy()
            new_spot['asoc'] = []
            spot_type = spot['label']
            role_list = self.type_role_dict[spot_type]
            if not self.ordered_prompt:
                random.shuffle(role_list)
            for asoc_type in role_list:
                for asoc in spot['asoc']:
                    if asoc[0] == asoc_type:
                        new_spot['asoc'].append(asoc)
                if random.random() < self.asoc_noise_ratio and asoc_type in asoc_label_list:
                # if random.random() < self.asoc_noise_ratio and asoc_type not in asoc_label_list: # over insert postive args
                    new_spot['asoc'].append([asoc_type, self.null_span])
            record.append(new_spot)           
        return record

    def add_noise(self, spot_asoc, spot_label_list, asoc_label_list):
        spot_asoc = self.insert_asoc(
            spot_asoc=spot_asoc,
            asoc_label_list=asoc_label_list,
        )
        spot_asoc = self.insert_spot(
            spot_asoc=spot_asoc,
            spot_label_list=spot_label_list,
        )
        return spot_asoc


def main():
    from uie.extraction.constants import BaseStructureMarker
    structure_marker = BaseStructureMarker()
    spot_asoc = [{"span": "analyzer", "label": "generic", "asoc": []}, {"span": "`` Amorph ''", "label": "method", "asoc": []}]

    spot_asoc_noiser = SpotAsocNoiser(
        spot_noise_ratio=0.5,
        asoc_noise_ratio=0.5,
    )
    spot_asoc_noiser.add_noise(
        spot_asoc=spot_asoc,
        spot_label_list=['A', 'B', 'C'],
        asoc_label_list=['D', 'E', 'F'],
    )
    target = convert_spot_asoc(
        spot_asoc_instance=spot_asoc,
        structure_maker=structure_marker
    )

    target = convert_spot_asoc(
        spot_asoc_instance=spot_asoc,
        structure_maker=structure_marker
    )

    replace_map = {
        '<extra_id_0>': ' ( ',
        '<extra_id_1>': ' ) ',
        '<extra_id_5>': ':',
    }
    from nltk.tree import Tree
    for old, new in replace_map.items():
        target = target.replace(old, new)
    print(target)
    Tree.fromstring(target).pretty_print()


if __name__ == "__main__":
    main()
