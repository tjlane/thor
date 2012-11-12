#!/usr/bin/env python
"""
Get the cromer-mann parameters into a useful format. This code is never called
by ODIN, which has the data stored as a hard-coded dictionary in 
odin/src/python/data.py
"""

import re

def retrieve_cromer_mann():
    """
    Read in the Cromer-Mann parameters from the flat text file 'cromer-mann.txt',
    which contains the DABAX theoretical parameters.
    
    Recall the Cromer-Mann formula:
    
        f0[k] = c + [SUM a_i*EXP(-b_i*(k^2)) ]
                    i=1,4
        
    this function gets the a_i, b_i, and c parameters.
    
    Returns
    -------
    cromer_mann_params : dict
        A dictionary such that
        
        cromer_mann_params[(atomic_number, ionization_state)] =
            [a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4, c]
            
        where `ionization_state` is a 
          -- positive int for a cation
          -- negative int for an anion
          -- '.' for a radical    
            
        -- OR --
        
        cromer_mann_params[atomic_symbol] =
            [a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4, c]
                
        where `atomic_symbol` might be something like `Au1+`
    """

    f = open('cromer-mann.txt', 'r')

    cromer_mann_params = {}
    lines = f.readlines()

    for line in lines:
    
        if line[:2] == '#S':
            atomicZ = int(line[3:5].strip())
            symb = line[5:].strip()
        
            # default, not an ion
            ion_state = 0 
            
            # cation
            g = re.search('(\d+)\+', symb)
            if g:
                ion_state = int(g.group()[:-1])
            
            # anion
            g = re.search('(\d+)\-', symb)
            if g:
                ion_state = int(g.group()[:-1])
            
            # radical
            g = re.search('\.', symb)
            if g:
                ion_state = '.'
        
        # skip comments
        elif line[0] == '#':
            pass
            
        else:
            params = line.strip().split()
            params = [ float(p) for p in params ]
            
            # for some reason parameter 'c' is in the middle -- move it to end
            params.append( params.pop(4) )
            
            cromer_mann_params[(atomicZ, ion_state)] = params
            cromer_mann_params[symb] = params
        
    return cromer_mann_params
        

def write_better_cromer_mann_table():
    """
    Generates a more readable/parsable version of the cromer-mann.txt file.
    
    Output is a new file, 'new-cromer-mann.txt'
    """
    
    f = open('cromer-mann.txt', 'r')

    cromer_mann_params = {}
    lines = f.readlines()
    newlines = ['# Symb\tZ\tIon\ta_1\ta_2\ta_3\ta_4\tb_1\tb_2\tb_3\tb_4\tc\n']

    for line in lines:
    
        if line[:2] == '#S':
            atomicZ = int(line[3:5].strip())
            symb = line[5:].strip()
        
            # default, not an ion
            ion_state = 0 
            
            # cation
            g = re.search('(\d+)\+', symb)
            if g:
                ion_state = int(g.group()[:-1])
            
            # anion
            g = re.search('(\d+)\-', symb)
            if g:
                ion_state = int(g.group()[:-1])
            
            # radical
            g = re.search('\.', symb)
            if g:
                ion_state = '.'
        
        # skip comments
        elif line[0] == '#':
            pass
            
        else:
            params = line.strip().split()
            
            # for some reason parameter 'c' is in the middle -- move it to end
            params.append( params.pop(4) )
        
            newline = symb + '\t' + str(atomicZ) + '\t' + str(ion_state) + '\t' + '\t'.join(params) + '\n'
            newlines.append(newline)
        
    f = open('new-cromer-mann.txt', 'w')
    f.writelines(newlines)
    f.close()
    print "Wrote: new-cromer-mann.txt"
    
    return
        
    
def print_cromer_mann_dict():
    """
    Prints a 'dictionary', that can be dropped straight into a python file and
    imported. This is what is actually used in the main ODIN code.
    """
    
    cromer_mann_params = retrieve_cromer_mann()
    
    print 'cromer_mann_params = {'
    
    for key in sorted(cromer_mann_params.keys()):
        values = cromer_mann_params[key]
        
        if type(key) == str:
            key = "'" + key + "'"
        else:
            key = str(key)
        values = str(values)
        
        print '                      %s : %s,' % (key, values)
    
    print '                      }'
    
    return
    
    
if __name__ == '__main__':
    print_cromer_mann_dict()
