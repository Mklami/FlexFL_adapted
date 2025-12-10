import os
import subprocess as sp
import glob

def d4j_path_prefix(proj, bug_num):
    bug_num = int(bug_num)
    if proj == 'Chart':
        return 'source/'
    elif proj == 'Closure':
        return 'src/'
    elif proj == 'Lang':
        if bug_num <= 35:
            return 'src/main/java/'
        else:
            return 'src/java/'
    elif proj == 'Math':
        if bug_num <= 84:
            return 'src/main/java/'
        else:
            return 'src/java/'
    elif proj == 'Mockito':
        return 'src/'
    elif proj == 'Time':
        return 'src/main/java/'
    elif proj == 'Cli':
        if bug_num <= 29:
            return 'src/java/'
        else:
            return 'src/main/java/'
    elif proj == 'Codec':
        if bug_num <= 10:
            return 'src/java/'
        else:
            return 'src/main/java/'
    elif proj == 'Collections':
        return 'src/main/java/'
    elif proj == 'Compress':
        return 'src/main/java/'
    elif proj == 'Csv':
        return 'src/main/java/'
    elif proj == 'Gson':
        return 'gson/src/main/java/'
    elif proj in ('JacksonCore', 'JacksonDatabind', 'JacksonXml'):
        return 'src/main/java/'
    elif proj == 'Jsoup':
        return 'src/main/java/'
    elif proj == 'JxPath':
        return 'src/java/'
    else:
        raise ValueError(f'Unrecognized project {proj}')

with open('./bug_list.txt') as f:
    bugs = [e.strip() for e in f.readlines()]
for bug in bugs:
    # Skip if buggy file list already exists
    buggy_file_list = f'./file_lists_buggy/{bug}.txt'
    if os.path.exists(buggy_file_list):
        print(f"Skipping {buggy_file_list} (already exists)")
    else:
        folder_path = f"./Collect_Methods/repos/{bug}_buggy/{d4j_path_prefix(bug.split('-')[0],bug.split('-')[1])}"
        file_list = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if not file.endswith('.java'):
                    continue
                file_path = os.path.join(root, file)
                file_path = '/'.join([file_path.split('/')[0]] + file_path.split('/')[2:])
                file_list.append(file_path)
        with open(buggy_file_list, 'w') as f:
            for file in file_list:
                f.writelines(file+'\n')

    # Skip if fixed file list already exists
    fixed_file_list = f'./file_lists_fixed/{bug}.txt'
    if os.path.exists(fixed_file_list):
        print(f"Skipping {fixed_file_list} (already exists)")
    else:
        folder_path = f"./Collect_Methods/repos/{bug}_fixed/{d4j_path_prefix(bug.split('-')[0],bug.split('-')[1])}"
        file_list = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if not file.endswith('.java'):
                    continue
                file_path = os.path.join(root, file)
                file_path = '/'.join([file_path.split('/')[0]] + file_path.split('/')[2:])
                file_list.append(file_path)
        with open(fixed_file_list, 'w') as f:
            for file in file_list:
                f.writelines(file+'\n')