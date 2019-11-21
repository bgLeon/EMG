"""
This module returns the co-safe ltl specifications that we used in our experiments.
The set of propositional symbols are {a,b,c,d,e,f,g,h,n,s}:
    a: got_wood
    b: used_toolshed
    c: used_workbench
    d: got_grass
    e: used_factory
    f: got_iron
    g: used_bridge
    h: used_axe
    n: is_night
    s: at_shelter
"""

def get_sequence_of_subspecs():
    # Experiment 1: Sequences of Sub-specifications (Section 5.2 in paper)
    specifications = []
    specifications.append(_get_sequence('ab'))
    specifications.append(_get_sequence('ac'))
    specifications.append(_get_sequence('de'))
    specifications.append(_get_sequence('db'))
    specifications.append(_get_sequence('fae'))
    specifications.append(_get_sequence('abdc'))
    specifications.append(_get_sequence('acfb'))
    specifications.append(_get_sequence('acfc'))
    specifications.append(_get_sequence('faeg'))
    specifications.append(_get_sequence('acfbh'))
    return specifications

def get_interleaving_subspecs():
    # Experiment 2: Interleaving Sub-specifications (Section 5.3 in paper)
    specifications = []
    specifications.append(_get_sequence('ab'))
    specifications.append(_get_sequence('ac'))
    specifications.append(_get_sequence('de'))
    specifications.append(_get_sequence('db'))
    specifications.append(('and', _get_sequence('ae'), _get_sequence('fe')))
    specifications.append(('and', _get_sequence('dc'), _get_sequence('abc')))
    specifications.append(('and', _get_sequence('fb'), _get_sequence('acb')))
    specifications.append(('and', _get_sequence('fc'), _get_sequence('ac')))
    specifications.append(('and', _get_sequence('aeg'), _get_sequence('feg')))
    specifications.append(('and', _get_sequence('fbh'), _get_sequence('acbh')))
    return specifications

def get_safety_constraints():
    # Experiment 3: Safety Constraints (Section 5.4 in paper)
    specifications = []
    specifications.append(_get_sequence_night('ab'))
    specifications.append(_get_sequence_night('ac'))
    specifications.append(_get_sequence_night('de'))
    specifications.append(_get_sequence_night('db'))
    specifications.append(('and', _get_sequence_night('ae'), _get_sequence_night('fe')))
    specifications.append(('and', _get_sequence_night('dc'), _get_sequence_night('abc')))
    specifications.append(('and', _get_sequence_night('fb'), _get_sequence_night('acb')))
    specifications.append(('and', _get_sequence_night('fc'), _get_sequence_night('ac')))
    specifications.append(('and', _get_sequence_night('aeg'), _get_sequence_night('feg')))
    specifications.append(('and', _get_sequence_night('fbh'), _get_sequence_night('acbh')))
    return specifications 

def get_option(goal):
    return _get_sequence(goal)

def get_option_night(goal):
    return _get_sequence_night(goal)



def _snp(proposition):
    # adds the special constraint to go to the shelter for a proposition
    return ('or',('and', ('not','n'), proposition),('and','s',proposition))

def _sn():
    # returns formula to stay on the shelter
    return ('or',('not','n'),'s')

def _get_sequence(seq):
    if len(seq) == 1:
        return ('until','True',seq)
    return ('until','True', ('and', seq[0], _get_sequence(seq[1:])))

def _get_sequence_night(seq):
    if len(seq) == 1:
        return ('until',_sn(),_snp(seq))
    return ('until',_sn(), ('and', _snp(seq[0]), _get_sequence_night(seq[1:])))