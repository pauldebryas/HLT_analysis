'''Analyses text file containing info about HNL signal samples and stores information in a
list of SampleInfo objects
'''

import re
from collections import namedtuple

titles = [
    ('recommended', str), 
    ('process', str), 
    ('mass', float), 
    ('V2', float), 
    ('ctau_mm', float), 
    ('ctauRatioToTheory', float), 
    ('events', int), 
    ('cross_section', float), 
    ('final_state', str), 
    ('cross_section_error', float), 
    ('year', str), 
    ('directory', str), 
    ('lepton_type', str)
]

SampleInfo = namedtuple('SampleInfo', [t[0] for t in titles])

def get_sample_infos(fname='signal_samples.txt'):
    sample_infos = []

    with open(fname) as f:
        content = f.readlines()

    for line in content:
        line = line.rstrip()
        pieces = line.split()
        
        if len(pieces) < 12 or pieces[0] != '*':
            continue
        
        # replace the useless +- with final state inferred from directory name
        pieces[8] = re.search('Neutrino_(.*)_M', pieces[-1]).group(1)
        # infer lepton type
        pieces.append(pieces[-1].split('_')[-3])

        for i, t in enumerate(titles):
            pieces[i] = t[1](pieces[i]) if pieces[i] not in ['-', '?', 'not', 'available'] else -1.

        info = SampleInfo(*pieces)
        
        sample_infos.append(info)
    
    return sample_infos

def get_selected_sample_infos(fname='signal_samples.txt', max_mass=None, year=None, process=None, lepton_type=None, final_state=None, mass=None):
    infos = get_sample_infos()
    sel_infos = []
    for info in infos:
        if (max_mass and info.mass > max_mass) or \
        (year and info.year != year) or \
        (process and info.process != process) or \
        (lepton_type and info.lepton_type != lepton_type) or \
        (final_state and info.final_state != final_state) or \
        (mass and info.mass != mass):
            continue
        sel_infos.append(info)
    return sel_infos

if __name__ == "__main__":
    infos = get_sample_infos()
    sel_infos = get_selected_sample_infos(max_mass=10., process='majorana', lepton_type='mu', final_state='trilepton', year='2018')
    print('Selected', len(sel_infos), 'out of', len(infos), 'SampleInfo objects')