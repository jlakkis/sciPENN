from os.path import split, basename, isdir
from os import mkdir

format_loss = lambda loss, nbatch: round(loss/nbatch, 2)

def build_dir(dir_path):
    """ This function builds a directory if it does not exist.
    
    
    Arguments:
    ------------------------------------------------------------------
    - dir_path: `str`, The directory to build. E.g. if dir_path = 'folder1/folder2/folder3', then this function will creates directory if folder1 if it does not already exist. Then it creates folder1/folder2 if folder2 does not exist in folder1. Then it creates folder1/folder2/folder3 if folder3 does not exist in folder2.
    """
    
    subdirs = [dir_path]
    substring = dir_path

    while substring != '':
        splt_dir = split(substring)
        substring = splt_dir[0]
        subdirs.append(substring)
        
    subdirs.pop()
    subdirs = [x for x in subdirs if basename(x) != '..']
    
    for dir_ in subdirs[::-1]:
        if not isdir(dir_):
            mkdir(dir_)